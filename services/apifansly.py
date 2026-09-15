"""Shared API Fansly client contract.

Every call to the managed Fansly API goes through this module so payload shapes,
response unwrapping, authentication, and account-access errors remain identical
across chat sync, vault sync, PPV delivery, and purchase reconciliation.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import threading
import time
from collections import Counter, defaultdict, deque
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Iterable
from urllib.parse import urlparse

import httpx

from core.apifansly_gate import (
    # Redundant aliases mark these as deliberate re-exports: callers import
    # every API Fansly concept from this module, and the names are not used
    # inside it.
    ApiFanslyDisabledError as ApiFanslyDisabledError,
    apifansly_enabled as apifansly_enabled,
    require_apifansly_available,
    simulation_active as simulation_active,
)


# ``ApiFanslyDisabledError``, ``apifansly_enabled`` and ``simulation_active``
# are imported above rather than redefined, so callers keep importing every API
# Fansly concept from this one module as the docstring promises, while the
# switch itself has a single definition in core/apifansly_gate.py.

DEFAULT_BASE_URL = "https://v1.apifansly.com/api/fansly"
_USAGE_WINDOW_SECONDS = 24 * 60 * 60
_USAGE_EVENTS: deque[dict[str, Any]] = deque(maxlen=50_000)
_WEBHOOK_EVENTS: deque[dict[str, Any]] = deque(maxlen=200_000)
_USAGE_LOCK = threading.Lock()
_USAGE_STARTED_AT = time.time()


# ---------------------------------------------------------------------------
# Credit estimation.
#
# API Fansly bills in credits, and its own Usage dashboard is the AUTHORITY.
# Everything this module computes is an ESTIMATE from what the transport can
# observe, published so an operator can see where credits are going between
# dashboard refreshes and so background history work can be throttled before it
# surprises anybody.
#
# The published billing rules this models:
#
#   * an ordinary request costs 1 credit
#   * a standard response larger than 80 KB costs proportionally more
#   * 80 received webhook events cost 1 credit
#   * media upload/download costs 2 credits per MB transferred
#
# The proportional rule is applied as bytes / 80 KB with a floor of one credit,
# so a 240 KB chat page is estimated at three credits rather than one. This
# matters specifically for history backfill: an API Fansly chat page carries
# the full ``accountMedia`` metadata for every attachment on it, so a page of
# ten messages is routinely far larger than 80 KB and assuming "one page, one
# credit" would understate a 5,000-message import several-fold.
#
# A media transfer is charged on the bytes that moved rather than on the JSON
# envelope, because those bytes ARE what the provider meters. The ordinary
# one-credit floor still applies, so a media call is never estimated at less
# than a plain request.
# ---------------------------------------------------------------------------
CREDIT_RESPONSE_BYTES_PER_CREDIT = 80 * 1024
CREDIT_MEDIA_CREDITS_PER_MB = 2.0
CREDIT_BYTES_PER_MB = 1024 * 1024
WEBHOOK_EVENTS_PER_CREDIT = 80


# Which part of the product spent the credit. Live conversation is separated
# from every kind of background work precisely so that throttling background
# work can never be confused with throttling a fan's reply.
CATEGORY_LIVE_CHAT = "live_chat"
CATEGORY_BACKGROUND_HISTORY = "background_history"
CATEGORY_VAULT = "vault"
CATEGORY_RECONCILIATION = "reconciliation"
CATEGORY_ACCOUNT = "account"
CATEGORY_OTHER = "other"

USAGE_CATEGORIES: tuple[str, ...] = (
    CATEGORY_LIVE_CHAT,
    CATEGORY_BACKGROUND_HISTORY,
    CATEGORY_VAULT,
    CATEGORY_RECONCILIATION,
    CATEGORY_ACCOUNT,
    CATEGORY_OTHER,
)

# A call made outside any explicit scope still has to be classified, because an
# unclassified majority would make the breakdown useless. The operation name is
# the only thing every call site already carries, so it is the fallback.
_OPERATION_CATEGORY_DEFAULTS: tuple[tuple[str, str], ...] = (
    ("vault", CATEGORY_VAULT),
    ("chat listing", CATEGORY_RECONCILIATION),
    ("chat message listing", CATEGORY_RECONCILIATION),
    ("message delivery", CATEGORY_LIVE_CHAT),
    ("message deletion", CATEGORY_LIVE_CHAT),
    ("typing", CATEGORY_LIVE_CHAT),
    ("mark", CATEGORY_LIVE_CHAT),
    ("history", CATEGORY_BACKGROUND_HISTORY),
    ("connect", CATEGORY_ACCOUNT),
    ("2fa", CATEGORY_ACCOUNT),
    ("account", CATEGORY_ACCOUNT),
    ("follower", CATEGORY_ACCOUNT),
    ("subscriber", CATEGORY_ACCOUNT),
    ("list", CATEGORY_ACCOUNT),
)

_USAGE_CATEGORY: ContextVar[str | None] = ContextVar(
    "apifansly_usage_category", default=None
)

# An open ``collect_usage`` scope's sink. A ContextVar rather than a global, so
# two concurrent backfills each measure their own pages instead of each other's.
_USAGE_COLLECTOR: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "apifansly_usage_collector", default=None
)

# How many live-chat scopes are open right now, across every task in the
# process. Deep-history work reads this and yields, which is the whole reason
# the counter exists: a fan waiting on a reply must never queue behind a
# hundred background history pages.
_live_scope_depth = 0

# Set on a response object once its cost has been recorded, so a call site that
# both raises through ``raise_for_response`` and reports its own media bytes
# cannot be billed twice.
_RECORDED_MARKER = "_apifansly_usage_recorded"


class ApiFanslyConfigurationError(RuntimeError):
    """The deployment does not have a usable API Fansly configuration."""


class ApiFanslyAccountAccessError(RuntimeError):
    """The configured key cannot access the creator's stored account connection."""


class ApiFanslyProtocolError(RuntimeError):
    """The upstream response was successful HTTP but not the documented shape."""


class ApiFanslyTransientError(RuntimeError):
    """Upstream refused this request for a reason that is expected to pass.

    Raised only when the server answered with a rate-limit or unavailability
    status, which means the request was definitively *not* processed. That is a
    different fact from an authentication failure (which will fail identically
    until a human reconnects the account) and from a network timeout (where the
    platform may have accepted the request and we cannot know).

    Callers on an exactly-once path must treat a timeout as ambiguous even
    though it is also "transient"; only this class states that nothing
    happened upstream.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = int(status_code)
        self.retry_after_seconds = retry_after_seconds


# Statuses where the server answered and told us it did not process the request.
TRANSIENT_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})

# ---------------------------------------------------------------------------
# PERF-006 — one connection pool for the whole process.
#
# Every helper below used to fall back to ``httpx.AsyncClient()`` when no client
# was injected, so a single ``list_chat_messages`` or ``send_message`` paid for
# a fresh TCP connection and TLS handshake and then threw the pool away. The
# client is now created once and reused, which is safe because nothing about it
# is per-creator: authentication is a request header built by ``headers()`` from
# one deployment-wide key, and httpx carries no cross-request state between
# concurrent callers.
#
# ``follow_redirects`` stays False, exactly as the per-call clients had it, so a
# redirect cannot carry the x-api-key header to another host. The two helpers
# that legitimately follow redirects ask for it per request.
# ---------------------------------------------------------------------------

_SHARED_CLIENT_LOCK = threading.Lock()
_shared_client: httpx.AsyncClient | None = None

# Sized for the vault and chat reconciliation fan-out without letting one
# process open an unbounded number of sockets against the provider.
SHARED_CLIENT_LIMITS = httpx.Limits(
    max_connections=64,
    max_keepalive_connections=32,
    keepalive_expiry=60.0,
)


def shared_client() -> httpx.AsyncClient:
    """Return the process-wide pooled client, creating it on first use."""

    global _shared_client
    client = _shared_client
    if client is not None and not client.is_closed:
        return client
    with _SHARED_CLIENT_LOCK:
        if _shared_client is None or _shared_client.is_closed:
            _shared_client = httpx.AsyncClient(
                limits=SHARED_CLIENT_LIMITS,
                follow_redirects=False,
            )
        return _shared_client


def set_shared_client(client: httpx.AsyncClient | None) -> None:
    """Install a client for tests, or clear it so the next call rebuilds one.

    Does not close whatever was there: the caller owns anything it installed.
    """

    global _shared_client
    with _SHARED_CLIENT_LOCK:
        _shared_client = client


async def close_shared_client() -> None:
    """Close the pooled client. Called once from the application's shutdown."""

    global _shared_client
    with _SHARED_CLIENT_LOCK:
        client = _shared_client
        _shared_client = None
    if client is not None and not client.is_closed:
        await client.aclose()


@asynccontextmanager
async def client_scope() -> AsyncIterator[httpx.AsyncClient]:
    """Yield the shared client for a block that used to own a private one.

    Deliberately does not close on exit — the pool outlives the request.
    """

    yield shared_client()


def usage_category_default(operation: str) -> str:
    """Classify a call that was made outside an explicit usage scope."""
    text = str(operation or "").strip().lower()
    for needle, category in _OPERATION_CATEGORY_DEFAULTS:
        if needle in text:
            return category
    return CATEGORY_OTHER


def current_usage_category() -> str | None:
    """The usage category of the scope this call is running inside, if any."""
    return _USAGE_CATEGORY.get()


@contextmanager
def usage_category(category: str):
    """Attribute every API Fansly call made inside this block to one category.

    The scope also carries the live/background priority signal: entering
    ``CATEGORY_LIVE_CHAT`` raises ``live_calls_in_flight()`` for as long as the
    block runs, which is what deep-history work checks before spending another
    page. Categorisation and priority are the same fact, so they are one
    mechanism rather than two that can disagree.
    """
    global _live_scope_depth

    normalized = str(category or "").strip().lower() or CATEGORY_OTHER
    if normalized not in USAGE_CATEGORIES:
        normalized = CATEGORY_OTHER
    token = _USAGE_CATEGORY.set(normalized)
    if normalized == CATEGORY_LIVE_CHAT:
        with _USAGE_LOCK:
            _live_scope_depth += 1
    try:
        yield normalized
    finally:
        if normalized == CATEGORY_LIVE_CHAT:
            with _USAGE_LOCK:
                _live_scope_depth = max(0, _live_scope_depth - 1)
        _USAGE_CATEGORY.reset(token)


def live_calls_in_flight() -> int:
    """How many live-chat scopes are currently open in this process."""
    with _USAGE_LOCK:
        return _live_scope_depth


@contextmanager
def collect_usage(category: str):
    """Run inside a usage category AND capture what the calls in it cost.

    Yields a list that receives one event dict per provider call made inside
    the block, so a caller can report the exact calls, bytes and estimated
    credits that one unit of work consumed — which is what makes per-fan
    history cost visible instead of being averaged into a process total.

    Scoped to the calling task: two backfills running at once each measure
    their own pages.
    """
    sink: list[dict[str, Any]] = []
    token = _USAGE_COLLECTOR.set(sink)
    try:
        with usage_category(category):
            yield sink
    finally:
        _USAGE_COLLECTOR.reset(token)


def summarize_usage_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Calls, bytes and estimated credits for one collected batch."""
    return {
        "calls": len(events),
        "response_bytes": sum(int(event.get("response_bytes") or 0) for event in events),
        "media_bytes": sum(int(event.get("media_bytes") or 0) for event in events),
        "estimated_credits": round(
            sum(float(event.get("credits") or 0.0) for event in events), 3
        ),
    }


# How recently a live-chat call must have happened for deep-history work to
# stand aside. Deliberately short: this is "a conversation is happening right
# now", not "a conversation happened today".
LIVE_ACTIVITY_WINDOW_SECONDS = 30.0


def live_activity_recent(within_seconds: float = LIVE_ACTIVITY_WINDOW_SECONDS) -> bool:
    """Whether a live-chat provider call happened in the last few seconds.

    The companion to ``live_calls_in_flight()``, and the one that does the real
    work. A reply's typing indicator and its message delivery are both recorded
    as live-chat calls, so this sees an active conversation without every
    delivery path having to wrap itself in a scope — and a scope that spans a
    whole model-generating turn would be both hard to place correctly and
    coarser than this.
    """
    cutoff = time.time() - max(0.0, float(within_seconds))
    with _USAGE_LOCK:
        for event in reversed(_USAGE_EVENTS):
            if event["at"] < cutoff:
                return False
            if event.get("category") == CATEGORY_LIVE_CHAT:
                return True
    return False


def live_work_in_progress() -> bool:
    """Whether anything live is happening that background work must yield to.

    Live conversation, delivery and purchase correctness have priority over
    optional history work, always. This is the single question deep-history
    code asks before spending another provider call.
    """
    return live_calls_in_flight() > 0 or live_activity_recent()


def estimate_call_credits(
    *,
    response_bytes: int = 0,
    media_bytes: int = 0,
) -> float:
    """Estimate the credits one provider call consumed.

    See the credit-model comment at the top of this module. Media transfer is
    metered on bytes moved; everything else on response size, with the ordinary
    one-credit floor that every request pays.
    """
    if media_bytes and media_bytes > 0:
        media = (
            CREDIT_MEDIA_CREDITS_PER_MB
            * float(media_bytes)
            / float(CREDIT_BYTES_PER_MB)
        )
        return max(1.0, media)
    if response_bytes and response_bytes > CREDIT_RESPONSE_BYTES_PER_CREDIT:
        return float(response_bytes) / float(CREDIT_RESPONSE_BYTES_PER_CREDIT)
    return 1.0


def estimate_webhook_credits(events: int) -> float:
    """Estimate the credits ``events`` received webhook deliveries consumed."""
    if events <= 0:
        return 0.0
    return float(events) / float(WEBHOOK_EVENTS_PER_CREDIT)


def _prune_locked(now: float) -> None:
    """Drop events older than the rolling window. Caller holds _USAGE_LOCK."""
    cutoff = now - _USAGE_WINDOW_SECONDS
    while _USAGE_EVENTS and _USAGE_EVENTS[0]["at"] < cutoff:
        _USAGE_EVENTS.popleft()
    while _WEBHOOK_EVENTS and _WEBHOOK_EVENTS[0]["at"] < cutoff:
        _WEBHOOK_EVENTS.popleft()


def _append_usage_event(event: dict[str, Any]) -> int:
    with _USAGE_LOCK:
        _USAGE_EVENTS.append(event)
        _prune_locked(event["at"])
        return len(_USAGE_EVENTS)


def record_usage_event(
    *,
    operation: str,
    account_id: str | None,
    method: str = "",
    path: str = "",
    status: int = 0,
    response_bytes: int = 0,
    media_bytes: int = 0,
    category: str | None = None,
) -> dict[str, Any]:
    """Record one provider call with its estimated credit cost.

    Public because not every API Fansly call goes through ``request()``. The
    typing indicator, mark-as-read, account connection and 2FA verification are
    raw httpx calls that still cost credits, and a usage report that silently
    omits them is worse than none.
    """
    now = time.time()
    resolved_category = (
        str(category).strip().lower()
        if category
        else (_USAGE_CATEGORY.get() or usage_category_default(operation))
    )
    if resolved_category not in USAGE_CATEGORIES:
        resolved_category = CATEGORY_OTHER
    response_bytes = max(0, int(response_bytes or 0))
    media_bytes = max(0, int(media_bytes or 0))
    event = {
        "at": now,
        "operation": str(operation or "unknown"),
        "account_id": str(account_id or ""),
        "method": str(method or ""),
        "path": str(path or ""),
        "status": int(status or 0),
        "response_bytes": response_bytes,
        "media_bytes": media_bytes,
        "category": resolved_category,
        "credits": estimate_call_credits(
            response_bytes=response_bytes,
            media_bytes=media_bytes,
        ),
    }
    total_24h = _append_usage_event(event)
    sink = _USAGE_COLLECTOR.get()
    if sink is not None:
        sink.append(event)
    print(
        f"[APIFANSLY USAGE] operation={event['operation']} "
        f"account={event['account_id'] or 'none'} method={event['method']} "
        f"status={event['status']} bytes={event['response_bytes']} "
        f"media_bytes={event['media_bytes']} category={event['category']} "
        f"credits={event['credits']:.2f} calls_24h={total_24h}"
    )
    return event


def record_raw_call(
    response: httpx.Response,
    *,
    operation: str,
    account_id: str | None = None,
    media_bytes: int = 0,
    category: str | None = None,
) -> None:
    """Account for a response produced outside ``request()``.

    Idempotent per response object: a call site that both raises through
    ``raise_for_response`` and reports its own media bytes is billed once, not
    twice. The first record wins, so a site that knows its media bytes must
    report them at the point it raises rather than afterwards.
    """
    _record_usage(
        response,
        operation=operation,
        account_id=account_id,
        media_bytes=media_bytes,
        category=category,
    )


def record_webhook_event(
    event: str | None = None,
    *,
    account_id: str | None = None,
) -> None:
    """Count one received webhook delivery. 80 of them cost one credit."""
    now = time.time()
    with _USAGE_LOCK:
        _WEBHOOK_EVENTS.append(
            {
                "at": now,
                "event": str(event or "unknown"),
                "account_id": str(account_id or ""),
            }
        )
        _prune_locked(now)


def _record_usage(
    response: httpx.Response,
    *,
    operation: str,
    account_id: str | None,
    media_bytes: int = 0,
    category: str | None = None,
) -> None:
    """Record a bounded, secret-free API usage event for diagnostics."""
    if getattr(response, _RECORDED_MARKER, False):
        return
    try:
        setattr(response, _RECORDED_MARKER, True)
    except Exception:
        # A test double that refuses attributes still gets accounted; it just
        # cannot be deduplicated. Under-reporting is worse than double-counting
        # a stub.
        pass
    try:
        request = response.request
        method = str(request.method or "")
        path = str(request.url.path or "")
    except RuntimeError:
        method = ""
        path = ""
    try:
        response_bytes = len(response.content or b"")
    except Exception:
        # A streamed response that was never read has no .content. The call
        # still happened and still costs its one-credit floor.
        response_bytes = 0
    record_usage_event(
        operation=operation,
        account_id=account_id,
        method=method,
        path=path,
        status=int(response.status_code),
        response_bytes=response_bytes,
        media_bytes=media_bytes,
        category=category,
    )


def _usage_totals() -> dict[str, Any]:
    """Every accounting figure the snapshot and the throttle both need."""
    now = time.time()
    with _USAGE_LOCK:
        _prune_locked(now)
        events = list(_USAGE_EVENTS)
        webhook_events = list(_WEBHOOK_EVENTS)
    return {
        "now": now,
        "events": events,
        "webhook_events": webhook_events,
    }


def _breakdown(events: list[dict[str, Any]], key: str, *, fallback: str) -> dict[str, Any]:
    grouped: dict[str, dict[str, float]] = defaultdict(
        lambda: {"calls": 0, "response_bytes": 0, "media_bytes": 0, "estimated_credits": 0.0}
    )
    for event in events:
        bucket = grouped[str(event.get(key) or fallback)]
        bucket["calls"] += 1
        bucket["response_bytes"] += int(event.get("response_bytes") or 0)
        bucket["media_bytes"] += int(event.get("media_bytes") or 0)
        bucket["estimated_credits"] += float(event.get("credits") or 0.0)
    return {
        name: {
            "calls": int(values["calls"]),
            "response_bytes": int(values["response_bytes"]),
            "media_bytes": int(values["media_bytes"]),
            "estimated_credits": round(values["estimated_credits"], 3),
        }
        for name, values in sorted(grouped.items())
    }


def background_history_credit_budget() -> float:
    """Rolling 24h estimated-credit ceiling for deep history work.

    Zero or unset means no ceiling. This budget governs OPTIONAL backfill only.
    It is never consulted on a live conversation, a delivery, or a purchase
    reconciliation, because running out of history budget must never be able to
    stop a fan being answered or a sale being recorded.
    """
    raw = str(os.environ.get("APIFANSLY_HISTORY_CREDIT_BUDGET_24H") or "").strip()
    if not raw:
        return 0.0
    try:
        value = float(raw)
    except ValueError:
        return 0.0
    return value if value > 0 else 0.0


def background_history_budget_state() -> dict[str, Any]:
    """Whether optional deep-history work may spend another provider call."""
    totals = _usage_totals()
    spent = sum(
        float(event.get("credits") or 0.0)
        for event in totals["events"]
        if event.get("category") == CATEGORY_BACKGROUND_HISTORY
    )
    budget = background_history_credit_budget()
    remaining = max(0.0, budget - spent) if budget else None
    return {
        "budget_credits": budget or None,
        "spent_credits": round(spent, 3),
        "remaining_credits": round(remaining, 3) if remaining is not None else None,
        "exhausted": bool(budget and spent >= budget),
    }


def background_history_allowed() -> bool:
    """False only when an explicitly configured history budget is spent."""
    return not background_history_budget_state()["exhausted"]


# The operation name every vault media transfer is recorded under, so the vault
# view below can separate "we moved media bytes" from "we listed an album".
VAULT_MEDIA_DOWNLOAD_OPERATION = "vault protected media download"


def vault_classification_usage(events: list[dict[str, Any]]) -> dict[str, Any]:
    """What vault sync and classification cost, and how much of it was media.

    Derived from the SAME events every other figure in the snapshot is derived
    from — this is a view, not a second accounting system. It exists because
    the three questions an operator actually asks about vault spend ("how much
    did classification cost", "how much of that was media transfer", "which
    creator caused it") are each one filter away from the event stream and
    nobody should have to do that arithmetic by hand.
    """
    vault_events = [
        event for event in events if event.get("category") == CATEGORY_VAULT
    ]
    media_events = [
        event for event in vault_events if int(event.get("media_bytes") or 0) > 0
    ]
    media_bytes = sum(int(event.get("media_bytes") or 0) for event in media_events)
    media_credits = sum(float(event.get("credits") or 0.0) for event in media_events)
    total_credits = sum(float(event.get("credits") or 0.0) for event in vault_events)
    return {
        "calls": len(vault_events),
        "estimated_credits": round(total_credits, 3),
        # The number this sprint exists to keep small.
        "media_downloads": len(media_events),
        "media_bytes": media_bytes,
        "media_megabytes": round(media_bytes / float(CREDIT_BYTES_PER_MB), 2),
        "media_credits": round(media_credits, 3),
        # Everything that was NOT a media transfer: album listings, link
        # refreshes. This is the part that should dominate a healthy deployment.
        "metadata_credits": round(total_credits - media_credits, 3),
        "media_share": (
            round(media_credits / total_credits, 3) if total_credits else 0.0
        ),
        "by_account": _breakdown(vault_events, "account_id", fallback="unbound"),
        "media_by_account": _breakdown(
            media_events, "account_id", fallback="unbound"
        ),
        "by_operation": _breakdown(vault_events, "operation", fallback="unknown"),
    }


def usage_snapshot() -> dict[str, Any]:
    """Return the current process's rolling 24-hour API usage summary.

    Credit figures are ESTIMATES derived from observed bytes and the published
    billing rules. API Fansly's own Usage dashboard is authoritative; this
    exists so an operator can see the shape of spend between refreshes and
    attribute it to an operation, an account and a part of the product.
    """
    totals = _usage_totals()
    events = totals["events"]
    webhook_events = totals["webhook_events"]
    now = totals["now"]

    call_credits = sum(float(event.get("credits") or 0.0) for event in events)
    webhook_credits = estimate_webhook_credits(len(webhook_events))
    total_credits = call_credits + webhook_credits

    # Run-rate is extrapolated from what this process has actually observed. A
    # process that booted ten minutes ago has ten minutes of evidence, not a
    # day of it, so the divisor is the observed window rather than a flat 24h.
    observed_seconds = max(1.0, min(_USAGE_WINDOW_SECONDS, now - _USAGE_STARTED_AT))
    per_day = total_credits * (86400.0 / observed_seconds)

    by_operation = Counter(event["operation"] for event in events)
    by_account = Counter(event["account_id"] or "unbound" for event in events)
    by_status = Counter(str(event["status"]) for event in events)

    return {
        "window_hours": 24,
        "process_started_at": _USAGE_STARTED_AT,
        "observed_seconds": round(observed_seconds, 1),
        "calls": len(events),
        "response_bytes": sum(event["response_bytes"] for event in events),
        "media_bytes": sum(int(event.get("media_bytes") or 0) for event in events),
        # Retained exactly as before so existing dashboard readers keep working.
        "by_operation": dict(sorted(by_operation.items())),
        "by_account": dict(sorted(by_account.items())),
        "by_status": dict(sorted(by_status.items())),
        # The credit view.
        "estimated_credits": round(total_credits, 3),
        "estimated_call_credits": round(call_credits, 3),
        "estimated_webhook_credits": round(webhook_credits, 3),
        "webhook_events": len(webhook_events),
        "estimated_monthly_credits": round(per_day * 30.0, 1),
        "estimated_daily_credits": round(per_day, 1),
        "credits_by_operation": _breakdown(events, "operation", fallback="unknown"),
        "credits_by_account": _breakdown(events, "account_id", fallback="unbound"),
        "credits_by_category": _breakdown(events, "category", fallback=CATEGORY_OTHER),
        "webhook_events_by_type": dict(
            sorted(Counter(event["event"] for event in webhook_events).items())
        ),
        "background_history_budget": background_history_budget_state(),
        # Vault sync and classification, split into metadata and media
        # transfer, and attributed per account.
        "vault_classification": vault_classification_usage(events),
        "credit_model": {
            "request_credits": 1,
            "response_bytes_per_extra_credit": CREDIT_RESPONSE_BYTES_PER_CREDIT,
            "media_credits_per_mb": CREDIT_MEDIA_CREDITS_PER_MB,
            "webhook_events_per_credit": WEBHOOK_EVENTS_PER_CREDIT,
        },
        "note": (
            "Calls and bytes are what this backend process observed. Credit "
            "figures are ESTIMATES from the published billing rules "
            "(1/request, proportional above 80 KB, 2/MB of media, 80 webhook "
            "events per credit). API Fansly's own Usage dashboard is "
            "authoritative."
        ),
    }


def reset_usage_for_tests() -> None:
    """Clear rolling usage state. Test support only.

    The deques are process-global on purpose — they describe this process, not
    a request — which is exactly what makes them leak between tests.
    """
    global _live_scope_depth
    with _USAGE_LOCK:
        _USAGE_EVENTS.clear()
        _WEBHOOK_EVENTS.clear()
        _live_scope_depth = 0

def api_key() -> str:
    value = str(os.environ.get("APIFANSLY_API_KEY") or "").strip()
    if not value:
        raise ApiFanslyConfigurationError("APIFANSLY_API_KEY is not configured")
    return value


def base_url() -> str:
    value = str(os.environ.get("APIFANSLY_BASE_URL") or DEFAULT_BASE_URL).strip()
    value = value.rstrip("/")
    if not value.startswith("https://") or not value.endswith("/api/fansly"):
        raise ApiFanslyConfigurationError(
            "APIFANSLY_BASE_URL must end with /api/fansly"
        )
    return value


def url(path: str = "") -> str:
    suffix = str(path or "").strip("/")
    return f"{base_url()}/{suffix}" if suffix else base_url()


def headers(*, json_content: bool = False) -> dict[str, str]:
    """Build the authenticated request headers, refusing when the connector is off.

    Six call sites across main.py and services/suggestions.py drive httpx
    directly instead of going through ``request()`` (connect, verify-2fa,
    mark-as-read, media upload + its status poll, the vault media URL lookup and
    the typing indicator). Every one of them builds its auth header here,
    because no API Fansly request is possible without ``x-api-key``.

    Enforcing the switch here therefore covers the raw call sites as well as the
    helpers, and makes "zero remote calls while disabled or simulating" a
    property of the transport rather than a list of patched callers.
    """
    require_apifansly_available("API Fansly request")
    result = {"x-api-key": api_key()}
    if json_content:
        result["Content-Type"] = "application/json"
    return result


def is_fansly_cdn_url(value: str) -> bool:
    """Return whether a URL is an HTTPS Fansly CDN asset.

    Vault locations are signed and may be rejected when fetched directly from
    application infrastructure.  Only known Fansly hosts may be sent to the
    managed media-download proxy.
    """
    try:
        parsed = urlparse(str(value or "").strip())
    except ValueError:
        return False
    host = str(parsed.hostname or "").lower()
    return (
        parsed.scheme == "https"
        and (host == "fansly.com" or host.endswith(".fansly.com"))
    )


def response_message(response: httpx.Response) -> str:
    try:
        payload: Any = response.json()
    except Exception:
        payload = None
    if isinstance(payload, dict):
        for key in ("error", "message", "detail"):
            value = payload.get(key)
            if value:
                return str(value)
    return response.text[:200] or f"HTTP {response.status_code}"


def _retry_after_seconds(response: httpx.Response) -> float | None:
    """Read a Retry-After header, seconds form only."""

    raw = str(response.headers.get("retry-after") or "").strip()
    if not raw:
        return None
    try:
        seconds = float(raw)
    except ValueError:
        # HTTP-date form. Honouring it needs a clock comparison we do not need
        # here; the caller's own backoff is a safe substitute.
        return None
    return seconds if seconds >= 0 else None


def raise_for_response(
    response: httpx.Response,
    *,
    operation: str,
    account_id: str | None = None,
    media_bytes: int = 0,
    category: str | None = None,
) -> None:
    """Account for the call, then translate a failure into a typed error.

    Accounting happens FIRST and unconditionally: a refused call still reached
    the provider and still costs a credit, so a usage report that only counted
    successes would understate spend exactly when spend is going wrong.

    ``media_bytes`` is how a media transfer reports the bytes that actually
    moved, which is what the provider meters at 2 credits/MB. It is passed here
    rather than recorded afterwards because accounting is idempotent per
    response object: whoever records first decides the cost.
    """
    _record_usage(
        response,
        operation=operation,
        account_id=account_id,
        media_bytes=media_bytes,
        category=category,
    )
    if response.is_success:
        return
    message = response_message(response)
    if response.status_code in {401, 403}:
        target = f" for account {account_id}" if account_id else ""
        raise ApiFanslyAccountAccessError(
            f"API Fansly access denied{target} during {operation}: {message}. "
            "Reconnect this creator under the current APIFANSLY_API_KEY."
        )
    if response.status_code in TRANSIENT_STATUS_CODES:
        # The server answered, so the request was definitively not processed.
        # Distinguishing this from an invalid key matters: a rate limit clears
        # by itself, while a disconnected account never does, and treating them
        # alike is what freezes a fan for a passing 503.
        raise ApiFanslyTransientError(
            f"API Fansly {operation} is temporarily unavailable "
            f"(HTTP {response.status_code}): {message}",
            status_code=response.status_code,
            retry_after_seconds=_retry_after_seconds(response),
        )
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise httpx.HTTPStatusError(
            f"API Fansly {operation} failed: {message}",
            request=exc.request,
            response=exc.response,
        ) from exc


def response_data(payload: Any) -> Any:
    """Return the documented ``data.data.response`` value."""
    if not isinstance(payload, dict):
        return None
    outer = payload.get("data")
    if not isinstance(outer, dict):
        return None
    inner = outer.get("data")
    if not isinstance(inner, dict):
        return None
    return inner.get("response")


def response_cursor(payload: Any, *, response: Any | None = None) -> str | None:
    """Normalize cursors used by chat, message, follower, and vault endpoints."""
    if isinstance(response, dict):
        for key in ("cursor", "nextCursor"):
            if response.get(key) is not None and response.get(key) != "":
                return str(response[key])
    if not isinstance(payload, dict):
        return None
    outer = payload.get("data")
    if not isinstance(outer, dict):
        return None
    value = outer.get("nextCursor")
    return str(value) if value is not None and value != "" else None


def media_references(
    media_ids: Iterable[str],
    *,
    preview_ids: dict[str, str | None] | None = None,
) -> list[dict[str, str | None]]:
    """Build the documented media attachment objects for message sends."""
    seen: set[str] = set()
    references: list[dict[str, str | None]] = []
    previews = preview_ids or {}
    for raw in media_ids:
        media_id = str(raw or "").strip()
        if not media_id or media_id in seen:
            continue
        seen.add(media_id)
        references.append({
            "mediaId": media_id,
            "previewId": (
                str(previews[media_id])
                if previews.get(media_id)
                else None
            ),
        })
    return references


def message_payload(
    *,
    content: str,
    media_ids: Iterable[str] = (),
    preview_ids: dict[str, str | None] | None = None,
    price_dollars: float | None = None,
) -> dict[str, Any]:
    """Build a free-text, free-media, or locked-PPV message payload."""
    payload: dict[str, Any] = {"content": str(content or "")}
    media = media_references(media_ids, preview_ids=preview_ids)
    if media:
        payload["mediaIds"] = media
    if price_dollars is not None:
        price = round(float(price_dollars), 2)
        if not media:
            raise ValueError("PPV messages require at least one media item")
        if price < 1 or price > 500:
            raise ValueError("PPV price must be between $1 and $500")
        payload["access_type"] = ["ppv"]
        payload["price"] = price
    return payload


def sent_message_id(payload: Any) -> str | None:
    """Extract the message ID from the documented send-message response."""
    response = response_data(payload)
    if not isinstance(response, dict):
        return None
    value = response.get("id")
    return str(value) if value is not None and value != "" else None


def sent_attachment_ids(payload: Any) -> list[str]:
    """Extract the account-media bundle IDs returned by a successful send."""
    response = response_data(payload)
    if not isinstance(response, dict):
        return []
    result: list[str] = []
    for attachment in response.get("attachments") or []:
        if not isinstance(attachment, dict):
            continue
        content_id = str(attachment.get("contentId") or "").strip()
        if content_id and content_id not in result:
            result.append(content_id)
    return result


def account_media_prices(row: dict[str, Any]) -> list[float]:
    """Return every price Fansly exposes for an account-media item.

    API Fansly responses are not consistent about where the PPV price lives:
    some put it on ``accountMedia.price`` while others put it in the nested
    permission flags (occasionally inside JSON-encoded metadata).
    """
    prices: list[float] = []

    def collect(value: Any, *, price_key: bool = False) -> None:
        if isinstance(value, str):
            stripped = value.strip()
            if stripped and stripped[0:1] in {"{", "["}:
                try:
                    collect(json.loads(stripped))
                    return
                except (TypeError, ValueError, json.JSONDecodeError):
                    pass
            if price_key:
                try:
                    prices.append(float(stripped))
                except (TypeError, ValueError):
                    pass
            return
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if price_key:
                prices.append(float(value))
            return
        if isinstance(value, dict):
            for key, nested in value.items():
                collect(nested, price_key=str(key).lower() == "price")
            return
        if isinstance(value, list):
            for nested in value:
                collect(nested)

    collect({"price": row.get("price")})
    collect(row.get("permissions") or {})
    unique: list[float] = []
    for price in prices:
        if price not in unique:
            unique.append(price)
    return unique


def _first_media_location(value: Any) -> str | None:
    """The first https:// location anywhere in a nested media/variant shape."""
    if isinstance(value, dict):
        direct = value.get("location")
        if isinstance(direct, str) and direct.startswith("https://"):
            return direct
        for key in ("locations", "variants", "media", "preview"):
            found = _first_media_location(value.get(key))
            if found:
                return found
    elif isinstance(value, list):
        for nested in value:
            found = _first_media_location(nested)
            if found:
                return found
    return None


def account_media_lookup(account_media: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Index one page's ``accountMedia`` block by every id it answers to.

    This is metadata the chat-message response ALREADY carried and we have
    ALREADY paid for. Reading it costs nothing extra and downloads nothing:
    there is no media call anywhere in this function, by design. History
    backfill in particular must never fetch a binary, so the only media facts
    it can ever persist are the ones indexed here.
    """
    lookup: dict[str, dict[str, Any]] = {}
    for item in account_media:
        if not isinstance(item, dict):
            continue
        media = item.get("media") or {}
        media_url = _first_media_location(media) or _first_media_location(item)
        prices = account_media_prices(item)
        positive_prices = [price for price in prices if price > 0]
        raw_price = positive_prices[0] if positive_prices else (prices[0] if prices else 0)
        info = {
            "url": media_url,
            "price": raw_price,
            "is_ppv": bool(positive_prices),
            "purchased": bool(item.get("purchased", item.get("isPurchased", False))),
            "access": item.get("access"),
            "mimetype": (
                media.get("mimetype")
                or media.get("mimeType")
                or item.get("mimetype")
                or item.get("mimeType")
            ),
            "filename": (
                media.get("filename")
                or media.get("fileName")
                or item.get("filename")
                or item.get("fileName")
            ),
        }
        for key in (item.get("id"), item.get("mediaId")):
            if key:
                lookup[str(key)] = info
    return lookup


def message_sent_at(message: dict[str, Any]) -> str:
    """Normalize a platform ``createdAt`` into an ISO-8601 UTC timestamp."""
    created_at = message.get("createdAt")
    try:
        timestamp = float(created_at or 0)
    except (TypeError, ValueError):
        timestamp = 0
    if timestamp > 0:
        if timestamp > 1e12:
            timestamp /= 1000
        return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()
    return datetime.now(timezone.utc).isoformat()


def chat_message_row(
    message: dict[str, Any],
    *,
    fan_id: str,
    creator_id: str,
    creator_platform_id: str,
    media_lookup: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    """Turn one platform message into the ``messages`` row this product stores.

    ONE parser for every ingestion path — live reconciliation, the active-chat
    endpoint and historical backfill — so a message imported from page 300 of a
    five-year-old conversation is byte-identical to the same message imported
    live. Two parsers would eventually disagree about a role or a timestamp, and
    the unique (creator_id, fansly_message_id) index would then be deciding
    which of two wrong rows survives.

    Returns None for a message with neither text nor attachments: there is
    nothing durable to store and nothing a writer could ever use.
    """
    message_id = str(message.get("id") or "")
    if not message_id:
        return None
    content = str(message.get("content") or "")
    attachments = message.get("attachments") or []
    if not content and not attachments:
        return None

    resolved_attachments = []
    for attachment in attachments:
        if not isinstance(attachment, dict):
            continue
        content_id = str(attachment.get("contentId") or "")
        info = media_lookup.get(content_id) or {}
        resolved_attachments.append(
            {
                "contentId": content_id,
                "url": info.get("url"),
                "type": attachment.get("contentType", 1),
                "mimetype": info.get("mimetype"),
                "filename": info.get("filename"),
                "price": info.get("price"),
                "is_ppv": info.get("is_ppv"),
                "purchased": info.get("purchased"),
                "access": info.get("access"),
            }
        )

    sender_id = str(message.get("senderId") or "")
    return {
        "fan_id": fan_id,
        "creator_id": creator_id,
        "role": "creator" if sender_id == creator_platform_id else "fan",
        "content": content,
        "fansly_message_id": message_id,
        "sent_at": message_sent_at(message),
        "media_context": (
            {"attachments": resolved_attachments} if resolved_attachments else None
        ),
    }


def ppv_delivery_evidence(
    messages: Iterable[dict[str, Any]],
    account_media: Iterable[dict[str, Any]],
    *,
    message_id: str,
    expected_media_ids: Iterable[str],
    expected_price_cents: int,
) -> dict[str, Any]:
    """Validate that a sent message is actually locked by Fansly.

    A successful send response only proves that Fansly accepted the message.
    The documented chat-message read response is the authoritative place where
    the resulting ``accountMedia`` price and original ``mediaId`` are exposed.
    Prices have appeared as both dollars and cents across API Fansly responses,
    so comparison accepts either representation but never accepts zero.
    """
    sent = next(
        (
            row
            for row in messages
            if str(row.get("id") or "") == str(message_id)
        ),
        None,
    )
    if not sent:
        return {"verified": False, "reason": "message_not_visible"}

    attachment_ids = {
        str(attachment.get("contentId") or "")
        for attachment in (sent.get("attachments") or [])
        if isinstance(attachment, dict) and attachment.get("contentId")
    }
    if not attachment_ids:
        return {"verified": False, "reason": "message_has_no_media"}

    matched = [
        row
        for row in account_media
        if isinstance(row, dict)
        and (
            str(row.get("id") or "") in attachment_ids
            or str(row.get("mediaId") or "") in attachment_ids
        )
    ]
    if not matched:
        return {
            "verified": False,
            "reason": "account_media_not_visible",
            "attachment_ids": sorted(attachment_ids),
            "attachment_count": len(attachment_ids),
        }

    expected_ids = {
        str(value).strip()
        for value in expected_media_ids
        if str(value).strip()
    }
    actual_ids = {
        str(row.get("mediaId") or row.get("id") or "").strip()
        for row in matched
        if row.get("mediaId") or row.get("id")
    }
    if expected_ids and not expected_ids.issubset(actual_ids):
        return {
            "verified": False,
            "reason": "media_mismatch",
            "actual_media_ids": sorted(actual_ids),
        }

    expected_cents = int(expected_price_cents)
    required_rows = [
        row
        for row in matched
        if str(row.get("mediaId") or row.get("id") or "").strip() in expected_ids
    ] if expected_ids else matched
    raw_prices = [
        price
        for row in required_rows
        for price in account_media_prices(row)
    ]
    positive_prices_by_media = [
        [price for price in account_media_prices(row) if price > 0]
        for row in required_rows
    ]
    if (
        not required_rows
        or not positive_prices_by_media
        or any(not prices for prices in positive_prices_by_media)
    ):
        return {
            "verified": False,
            "reason": "media_is_not_payment_gated",
            "raw_prices": raw_prices,
        }

    def _matches_expected(raw_price: float) -> bool:
        return (
            abs(raw_price - expected_cents) < 0.01
            or abs((raw_price * 100) - expected_cents) < 0.01
        )

    if any(
        not any(_matches_expected(price) for price in prices)
        for prices in positive_prices_by_media
    ):
        return {
            "verified": False,
            "reason": "price_mismatch",
            "raw_prices": raw_prices,
        }

    return {
        "verified": True,
        "reason": "locked_ppv_confirmed",
        "actual_media_ids": sorted(actual_ids),
        "raw_prices": raw_prices,
    }


# A read may be repeated freely; a write may not. Automatic retry is therefore
# decided by the HTTP method, never by how transient the failure looked.
IDEMPOTENT_METHODS = frozenset({"GET", "HEAD"})

# Deliberately small. This exists to ride out a rate limit or a single bad
# gateway, not to become a generic retry framework, and every attempt is a real
# provider call that costs credits.
MAX_READ_ATTEMPTS = 3
_RETRY_BASE_SECONDS = 0.5
_RETRY_MAX_SECONDS = 8.0


def _retry_delay(attempt: int, advertised: float | None) -> float:
    if advertised is not None:
        return min(max(advertised, 0.0), _RETRY_MAX_SECONDS)
    ceiling = min(_RETRY_BASE_SECONDS * (2 ** attempt), _RETRY_MAX_SECONDS)
    return random.uniform(0.0, ceiling)


async def request(
    method: str,
    path: str,
    *,
    operation: str,
    account_id: str | None = None,
    params: dict[str, Any] | None = None,
    json: dict[str, Any] | None = None,
    files: Any = None,
    timeout: float = 30,
    client: httpx.AsyncClient | None = None,
    follow_redirects: bool | None = None,
    retry_idempotent: bool = True,
) -> dict[str, Any]:
    """Execute one API Fansly request with uniform errors and JSON validation.

    Uses the process-wide connection pool unless a client is injected, so a
    single call no longer pays for its own TLS handshake (PERF-006).

    A GET or HEAD is repeated after a transient refusal or a network failure,
    because repeating a read cannot have a side effect. Every other method is
    attempted exactly once and its error is raised to the caller, which is the
    only layer that knows whether the platform may already have accepted the
    write. Timeouts on a send are ambiguous by nature and must go through the
    delivery journal, not through a retry here.
    """

    # Stated explicitly as well as in headers(): the refusal names the operation
    # that was blocked, which headers() cannot know, and it happens before a
    # connection is taken from the pool.
    require_apifansly_available(f"API Fansly {operation}")

    active_client = client if client is not None else shared_client()
    upper_method = str(method or "").upper()
    attempts = (
        MAX_READ_ATTEMPTS
        if retry_idempotent and upper_method in IDEMPOTENT_METHODS
        else 1
    )

    for attempt in range(attempts):
        try:
            response = await active_client.request(
                method,
                url(path),
                headers=headers(json_content=json is not None),
                params=params,
                json=json,
                files=files,
                timeout=timeout,
                # None means "whatever this client was built with", so an
                # injected client keeps its own redirect policy.
                follow_redirects=(
                    httpx.USE_CLIENT_DEFAULT
                    if follow_redirects is None
                    else follow_redirects
                ),
            )
            raise_for_response(
                response,
                operation=operation,
                account_id=account_id,
            )
        except (ApiFanslyTransientError, httpx.TransportError) as exc:
            if attempt + 1 >= attempts:
                raise
            advertised = getattr(exc, "retry_after_seconds", None)
            delay = _retry_delay(attempt, advertised)
            print(
                f"[APIFANSLY RETRY] operation={operation} "
                f"attempt={attempt + 1}/{attempts} in {delay:.2f}s: {exc}"
            )
            await asyncio.sleep(delay)
            continue

        try:
            payload = response.json()
        except Exception as exc:
            raise ApiFanslyProtocolError(
                f"API Fansly {operation} returned invalid JSON"
            ) from exc
        if not isinstance(payload, dict):
            raise ApiFanslyProtocolError(
                f"API Fansly {operation} returned a non-object response"
            )
        return payload

    # Unreachable: the loop either returns or re-raises on its last attempt.
    raise ApiFanslyProtocolError(f"API Fansly {operation} produced no response")


async def download_media(
    cdn_url: str,
    *,
    client: httpx.AsyncClient | None = None,
    timeout: float = 45,
    account_id: str | None = None,
    operation: str = "protected media download",
) -> bytes:
    """Download a protected Fansly CDN asset through the documented proxy.

    Unlike the regular API helpers, this endpoint returns binary content rather
    than the usual JSON envelope.

    THE EXPENSIVE ONE. Metered at 2 credits per megabyte transferred, so a
    250 MB video is ~500 credits on one call. Callers on an automatic path must
    take a policy decision before reaching here — see
    ``services.media_cost_guard``; this function deliberately enforces no limit
    of its own, because a transport that silently truncated a download would be
    worse than one that is simply not called.

    ``account_id`` attributes the spend to a creator's account in the usage
    snapshot, which is what makes "which creator caused these credits?"
    answerable rather than a guess.
    """
    require_apifansly_available("API Fansly protected media download")
    if not is_fansly_cdn_url(cdn_url):
        raise ValueError("media download requires an HTTPS Fansly CDN URL")

    active_client = client if client is not None else shared_client()
    # No try/finally: the pool is process-wide and deliberately outlives this
    # call, and an injected client belongs to whoever injected it.
    response = await active_client.post(
        url("media/download"),
        headers=headers(json_content=True),
        json={"cdnUrl": cdn_url},
        timeout=timeout,
        follow_redirects=True,
    )
    # A media transfer is metered on bytes, not on being one request, so the
    # downloaded size is reported at the moment of accounting. Reading
    # ``response.content`` here is free: the transport already buffered it.
    raise_for_response(
        response,
        operation=operation,
        account_id=account_id,
        media_bytes=len(response.content or b""),
    )
    content_type = str(response.headers.get("content-type") or "").lower()
    if "application/json" in content_type:
        raise ApiFanslyProtocolError(
            "API Fansly media download returned JSON instead of media: "
            + response_message(response)
        )
    if len(response.content) <= 1000:
        raise ApiFanslyProtocolError(
            "API Fansly media download returned an empty or truncated file"
        )
    return bytes(response.content)


async def send_message(
    account_id: str,
    chat_id: str,
    *,
    content: str,
    media_ids: Iterable[str] = (),
    preview_ids: dict[str, str | None] | None = None,
    price_dollars: float | None = None,
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """Send one text/media/PPV message using the documented request contract."""
    return await request(
        "POST",
        f"{account_id}/chats/{chat_id}/messages",
        operation="message delivery",
        account_id=account_id,
        json=message_payload(
            content=content,
            media_ids=media_ids,
            preview_ids=preview_ids,
            price_dollars=price_dollars,
        ),
        timeout=15,
        client=client,
    )


async def delete_message(
    account_id: str,
    message_id: str,
    *,
    client: httpx.AsyncClient | None = None,
) -> bool:
    """Delete a sent message through the documented compensation endpoint."""
    await request(
        "DELETE",
        f"{account_id}/messages/{message_id}",
        operation="message deletion",
        account_id=account_id,
        timeout=15,
        client=client,
    )
    return True


async def list_chats(
    account_id: str,
    *,
    cursor: str | None = None,
    filter: str = "all",
    sort: str = "newest",
    search: str | None = None,
    subscription_tier_id: str | None = None,
    client: httpx.AsyncClient | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str | None]:
    params: dict[str, Any] = {"filter": filter, "sort": sort}
    if cursor:
        params["cursor"] = cursor
    if search:
        params["search"] = search
    if subscription_tier_id:
        params["subscriptionTierId"] = subscription_tier_id
    payload = await request(
        "GET",
        f"{account_id}/chats",
        operation="chat listing",
        account_id=account_id,
        params=params,
        client=client,
    )
    response = response_data(payload)
    if not isinstance(response, dict):
        raise ApiFanslyProtocolError("API Fansly chat listing response is invalid")
    chats = response.get("data")
    aggregation = response.get("aggregationData")
    return (
        chats if isinstance(chats, list) else [],
        aggregation.get("accounts", [])
        if isinstance(aggregation, dict)
        and isinstance(aggregation.get("accounts"), list)
        else [],
        response_cursor(payload, response=response),
    )


# ---------------------------------------------------------------------------
# DO NOT RAISE THIS. IT IS NOT A CONSERVATIVE DEFAULT.
#
# API Fansly's documented chat-messages endpoint declares `limit min=1 max=10`.
# Ten is the provider's ceiling, not ours. Asking for 50 does not return 50: it
# is silently clamped upstream (or rejected), which is how a "history import"
# that looked like it was reading 50 messages a page was in fact reading 10 and
# quietly paying five times the pages it thought it was.
#
# The consequence is structural, and it is why history backfill in this
# codebase is cursor-based and resumable rather than a loop: a 5,000-message
# conversation is 500 provider round trips, minimum, forever, until the
# UPSTREAM API changes its documented maximum. Raising this constant without
# that upstream change buys nothing and hides the real cost.
#
# If API Fansly ever publishes a larger maximum, change it HERE, in one place,
# and update docs/historical_memory_and_credits.md with the new page economics.
# ---------------------------------------------------------------------------
CHAT_MESSAGE_PAGE_MAX = 10


async def list_chat_messages(
    account_id: str,
    chat_id: str,
    *,
    cursor: str | None = None,
    limit: int = CHAT_MESSAGE_PAGE_MAX,
    client: httpx.AsyncClient | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str | None]:
    """One page of a conversation, newest first, with its accountMedia.

    ``limit`` is clamped to ``CHAT_MESSAGE_PAGE_MAX`` because that is the
    provider's documented maximum. See the constant above before changing it.
    """
    params: dict[str, Any] = {
        "limit": max(1, min(CHAT_MESSAGE_PAGE_MAX, int(limit)))
    }
    if cursor:
        params["cursor"] = cursor
    payload = await request(
        "GET",
        f"{account_id}/chats/{chat_id}/messages",
        operation="chat message listing",
        account_id=account_id,
        params=params,
        client=client,
    )
    response = response_data(payload)
    if not isinstance(response, dict):
        raise ApiFanslyProtocolError(
            "API Fansly chat message listing response is invalid"
        )
    messages = response.get("messages")
    account_media = response.get("accountMedia")
    return (
        messages if isinstance(messages, list) else [],
        account_media if isinstance(account_media, list) else [],
        response_cursor(payload, response=response),
    )


async def list_vault_albums(
    account_id: str,
    *,
    client: httpx.AsyncClient | None = None,
) -> list[dict[str, Any]]:
    payload = await request(
        "GET",
        f"{account_id}/vault/albums",
        operation="vault album listing",
        account_id=account_id,
        client=client,
    )
    response = response_data(payload)
    if not isinstance(response, dict):
        raise ApiFanslyProtocolError(
            "API Fansly vault album listing response is invalid"
        )
    albums = response.get("albums")
    return albums if isinstance(albums, list) else []


async def list_vault_album_media(
    account_id: str,
    album_id: str,
    *,
    cursor: str | None = None,
    limit: int = 50,
    client: httpx.AsyncClient | None = None,
) -> tuple[list[dict[str, Any]], str | None]:
    params: dict[str, Any] = {"limit": max(1, int(limit))}
    if cursor:
        params["cursor"] = cursor
    payload = await request(
        "GET",
        f"{account_id}/vault/albums/{album_id}/media",
        operation="vault album media listing",
        account_id=account_id,
        params=params,
        client=client,
    )
    response = response_data(payload)
    if isinstance(response, list):
        items = response
    elif isinstance(response, dict):
        raw = response.get("data") or response.get("media") or []
        items = raw if isinstance(raw, list) else []
    else:
        raise ApiFanslyProtocolError(
            "API Fansly vault album media response is invalid"
        )
    return items, response_cursor(payload, response=response)


# --- Account lists -----------------------------------------------------------
#
# Agencies build Lists (VIP, Whales, Buyers, Re-engage, ...) directly on the
# creator's Fansly account. Cleopatra mirrors them read-only; it never creates,
# renames or deletes a remote list.
#
# The path segments follow the same {accountId}/<resource> shape as every other
# endpoint in this module, and are overridable without a deploy so a
# documentation change does not require one.
_LISTS_PATH = "lists"
_LIST_ITEMS_PATH = "items"

# Remote payloads vary in which key carries the collection and which carries the
# identifier, exactly as they do for vault albums and followers above. Parsing
# stays tolerant rather than asserting one shape.
_LIST_COLLECTION_KEYS = ("lists", "items", "data", "accountLists")
_LIST_ID_KEYS = ("id", "listId", "_id")
_LIST_NAME_KEYS = ("label", "name", "title")
_LIST_ITEM_COLLECTION_KEYS = ("items", "listItems", "accounts", "data", "members")
_LIST_MEMBER_ID_KEYS = ("accountId", "itemId", "id", "userId", "followerId")


def _lists_path() -> str:
    return str(os.environ.get("APIFANSLY_LISTS_PATH") or _LISTS_PATH).strip("/")


def _list_items_path() -> str:
    return str(
        os.environ.get("APIFANSLY_LIST_ITEMS_PATH") or _LIST_ITEMS_PATH
    ).strip("/")


def _first_value(row: dict[str, Any], keys: Iterable[str]) -> str | None:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _collection(response: Any, keys: Iterable[str]) -> list[Any]:
    """Return the collection from a list-shaped or object-shaped response."""
    if isinstance(response, list):
        return response
    if not isinstance(response, dict):
        return []
    for key in keys:
        value = response.get(key)
        if isinstance(value, list):
            return value
    return []


def parse_account_lists(response: Any) -> list[dict[str, Any]]:
    """Normalize one page of remote lists into id/name/count records.

    A list with no usable remote identifier is dropped rather than mirrored:
    the remote id is the only stable mapping key, and a mirror without one would
    duplicate itself on the next sync.
    """
    parsed: list[dict[str, Any]] = []
    for row in _collection(response, _LIST_COLLECTION_KEYS):
        if not isinstance(row, dict):
            continue
        external_id = _first_value(row, _LIST_ID_KEYS)
        if not external_id:
            continue
        name = _first_value(row, _LIST_NAME_KEYS) or f"Fansly list {external_id}"
        try:
            item_count = int(row.get("itemCount") or row.get("count") or 0)
        except (TypeError, ValueError):
            item_count = 0
        parsed.append(
            {
                "external_list_id": external_id,
                "name": name,
                "item_count": max(item_count, 0),
            }
        )
    return parsed


def parse_list_member_ids(response: Any) -> list[str]:
    """Normalize one page of list membership into platform account IDs.

    Members are identified only by their Fansly account ID. Usernames and
    display names are deliberately ignored: they are mutable and not unique.
    """
    member_ids: list[str] = []
    seen: set[str] = set()
    for row in _collection(response, _LIST_ITEM_COLLECTION_KEYS):
        if isinstance(row, (str, int)):
            account_id = str(row).strip()
        elif isinstance(row, dict):
            account_id = _first_value(row, _LIST_MEMBER_ID_KEYS) or ""
        else:
            continue
        if account_id and account_id not in seen:
            seen.add(account_id)
            member_ids.append(account_id)
    return member_ids


async def list_account_lists(
    account_id: str,
    *,
    cursor: str | None = None,
    limit: int = 100,
    client: httpx.AsyncClient | None = None,
) -> tuple[list[dict[str, Any]], str | None]:
    """Return one page of the creator's own Fansly lists."""
    params: dict[str, Any] = {"limit": max(1, int(limit))}
    if cursor:
        params["cursor"] = cursor
    payload = await request(
        "GET",
        f"{account_id}/{_lists_path()}",
        operation="account list listing",
        account_id=account_id,
        params=params,
        client=client,
    )
    response = response_data(payload)
    if response is None:
        raise ApiFanslyProtocolError("API Fansly account list response is invalid")
    return parse_account_lists(response), response_cursor(payload, response=response)


async def list_account_list_members(
    account_id: str,
    list_id: str,
    *,
    cursor: str | None = None,
    limit: int = 100,
    client: httpx.AsyncClient | None = None,
) -> tuple[list[str], str | None]:
    """Return one page of Fansly account IDs belonging to a remote list."""
    params: dict[str, Any] = {"limit": max(1, int(limit))}
    if cursor:
        params["cursor"] = cursor
    payload = await request(
        "GET",
        f"{account_id}/{_lists_path()}/{list_id}/{_list_items_path()}",
        operation="account list member listing",
        account_id=account_id,
        params=params,
        client=client,
    )
    response = response_data(payload)
    if response is None:
        raise ApiFanslyProtocolError(
            "API Fansly account list member response is invalid"
        )
    return parse_list_member_ids(response), response_cursor(
        payload,
        response=response,
    )


async def list_followers(
    account_id: str,
    *,
    cursor: str | None = None,
    limit: int = 100,
    client: httpx.AsyncClient | None = None,
) -> tuple[Any, str | None]:
    params: dict[str, Any] = {"limit": max(1, int(limit))}
    if cursor:
        params["cursor"] = cursor
    payload = await request(
        "GET",
        f"{account_id}/followers",
        operation="follower listing",
        account_id=account_id,
        params=params,
        client=client,
    )
    response = response_data(payload)
    return response, response_cursor(payload, response=response)


async def list_subscribers(
    account_id: str,
    *,
    status: str = "all",
    cursor: str | None = None,
    limit: int = 100,
    search: str | None = None,
    subscription_tier_ids: str | None = None,
    client: httpx.AsyncClient | None = None,
) -> tuple[Any, str | None]:
    if status not in {"all", "active", "expired"}:
        raise ValueError("subscriber status must be all, active, or expired")
    params: dict[str, Any] = {
        "status": status,
        "limit": max(1, int(limit)),
    }
    if cursor:
        params["cursor"] = cursor
    if search:
        params["search"] = search
    if subscription_tier_ids:
        params["subscriptionTierIds"] = subscription_tier_ids
    payload = await request(
        "GET",
        f"{account_id}/subscribers",
        operation="subscriber listing",
        account_id=account_id,
        params=params,
        client=client,
    )
    response = response_data(payload)
    return response, response_cursor(payload, response=response)


async def top_supporters(
    account_id: str,
    *,
    before_ms: int | None = None,
    after_ms: int | None = None,
    client: httpx.AsyncClient | None = None,
) -> Any:
    params: dict[str, Any] = {}
    if before_ms is not None:
        params["before"] = int(before_ms)
    if after_ms is not None:
        params["after"] = int(after_ms)
    payload = await request(
        "GET",
        f"{account_id}/top-supporters",
        operation="top supporter listing",
        account_id=account_id,
        params=params,
        client=client,
    )
    return response_data(payload)


async def current_account(
    account_id: str,
    *,
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    payload = await request(
        "GET",
        f"{account_id}/me",
        operation="account profile load",
        account_id=account_id,
        client=client,
    )
    response = response_data(payload)
    if not isinstance(response, dict):
        raise ApiFanslyProtocolError("API Fansly account response is invalid")
    return response
