"""Cost-aware, restart-safe historical conversation import.

An existing fan may have thousands of messages behind them. Cleopatra has to be
able to resume that conversation naturally without ever putting the transcript
in a writer prompt and without paying for the whole archive before it can
answer. This module is the paging half of that; services/fan_history_memory.py
is the compaction half.

Four rules shape everything here.

1. TEN MESSAGES PER PAGE IS THE PROVIDER'S CEILING.
   API Fansly documents `limit min=1 max=10` on the chat-messages endpoint, so
   a 5,000-message conversation is 500 round trips and no amount of local
   cleverness changes that. Paging is therefore cursor-based, durable and
   resumable rather than a loop inside one request — see
   services.apifansly.CHAT_MESSAGE_PAGE_MAX.

2. LIVE CONVERSATION ALWAYS WINS.
   Deep history yields to live work before every single page. A returning fan
   must never wait behind hundreds of background requests, and an exhausted
   history budget must never be able to stop a reply, a delivery or a purchase
   reconciliation. The budget in this module governs OPTIONAL backfill only.

3. NO MEDIA BINARIES, EVER.
   The chat-message response already carries `accountMedia` metadata that has
   been paid for. History persists that metadata — id, type, price, purchased,
   access — and makes zero media calls. There is no download path in this file
   and there must never be one: media transfer is billed at 2 credits/MB and a
   fan's archive is not worth a gigabyte of re-downloaded video.

4. NOTHING IS EVER SENT.
   Historical import writes to the local database and nothing else. It does not
   reply, does not mark read, does not advertise typing, and does not enter the
   reply pipeline.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from typing import Any

from core.apifansly_gate import apifansly_enabled
from core.pagination import fetch_all_rows_async
from core.simulation import is_simulatable_fan
from core.supabase import get_supabase
from db.fan_history_queries import (
    ensure_backfill_state,
    get_backfill_state,
    list_unfinished_backfills,
    update_backfill_state,
)
from db.queries import (
    PLATFORM_IDENTITY_CONFLICT,
    is_missing_conflict_target,
)
from services.apifansly import (
    CATEGORY_BACKGROUND_HISTORY,
    CHAT_MESSAGE_PAGE_MAX,
    account_media_lookup,
    background_history_allowed,
    background_history_budget_state,
    chat_message_row,
    client_scope,
    collect_usage,
    estimate_call_credits,
    list_chat_messages,
    live_work_in_progress,
    summarize_usage_events,
)


# How many messages of local context count as "enough to answer naturally".
# Below this, a returning fan gets a warm resume before the reply path runs.
WARM_RESUME_MIN_LOCAL_MESSAGES = 12

# The newest pages a warm resume may fetch. Three pages is thirty messages:
# enough for the writer to see what the conversation was actually about, and
# cheap enough to sit in front of a live reply. This is the ONLY history work
# that may ever block a turn.
WARM_RESUME_MAX_PAGES = 3

# Pages one deep-history invocation may fetch before returning. Bounded so the
# scheduler stays responsive and so a single fan cannot monopolise the budget.
DEEP_BACKFILL_PAGES_PER_RUN = 20


def _int_env(name: str, default: int, *, minimum: int, maximum: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)) or default)
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def warm_resume_min_local_messages() -> int:
    return _int_env(
        "HISTORY_WARM_MIN_LOCAL_MESSAGES",
        WARM_RESUME_MIN_LOCAL_MESSAGES,
        minimum=0,
        maximum=200,
    )


def warm_resume_max_pages() -> int:
    return _int_env(
        "HISTORY_WARM_MAX_PAGES", WARM_RESUME_MAX_PAGES, minimum=1, maximum=10
    )


def deep_backfill_pages_per_run() -> int:
    return _int_env(
        "HISTORY_DEEP_PAGES_PER_RUN",
        DEEP_BACKFILL_PAGES_PER_RUN,
        minimum=1,
        maximum=200,
    )


def history_backfill_enabled() -> bool:
    """Whether the background deep-history worker may run at all.

    Off by default. Warm resume and an operator pressing Load history are NOT
    governed by this switch — they are explicit, bounded and user-visible. This
    governs only the unattended scheduler.
    """
    return os.getenv("HISTORY_BACKFILL_ENABLED", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def estimate_page_credits(response_bytes: int) -> float:
    """What one history page cost, from its ACTUAL size.

    Never assume one page is one credit. A page carries the full accountMedia
    metadata for every attachment on it, so a page of ten messages from a
    media-heavy conversation routinely exceeds the 80 KB threshold above which
    API Fansly bills proportionally. Assuming a flat credit per page is how a
    500-page import gets estimated at 500 credits and bills at 1,500.
    """
    return estimate_call_credits(response_bytes=max(0, int(response_bytes or 0)))


async def local_message_count(fan_id: str) -> int:
    """How many messages this fan already has locally."""

    def _count() -> int:
        response = (
            get_supabase()
            .table("messages")
            .select("id", count="exact")
            .eq("fan_id", fan_id)
            .limit(1)
            .execute()
        )
        count = getattr(response, "count", None)
        if count is not None:
            return int(count)
        return len(list(response.data or []))

    return await asyncio.to_thread(_count)


async def _existing_platform_ids(fan_id: str, message_ids: list[str]) -> set[str]:
    if not message_ids:
        return set()

    def _get() -> set[str]:
        response = (
            get_supabase()
            .table("messages")
            .select("fansly_message_id")
            .eq("fan_id", fan_id)
            .in_("fansly_message_id", message_ids)
            .execute()
        )
        return {
            str(row["fansly_message_id"])
            for row in (response.data or [])
            if row.get("fansly_message_id")
        }

    return await asyncio.to_thread(_get)


async def _vault_links(creator_id: str, content_ids: list[str]) -> dict[str, dict[str, Any]]:
    """Local vault metadata for media ids seen in history, when we have any.

    This is the "connect it to existing local metadata" step, and it is a read
    of rows Cleopatra already mirrors — not a provider call. A media id we have
    never vaulted simply gets no link.
    """
    unique = [value for value in dict.fromkeys(content_ids) if value]
    if not unique:
        return {}

    def _get() -> list[dict[str, Any]]:
        response = (
            get_supabase()
            .table("creator_vault_media")
            .select("id, fansly_media_id, media_id, content_category, mimetype")
            .eq("creator_id", creator_id)
            .in_("fansly_media_id", unique)
            .limit(len(unique) * 2)
            .execute()
        )
        return list(response.data or [])

    try:
        rows = await asyncio.to_thread(_get)
    except Exception as exc:
        # An enrichment failure must not fail an import. The attachment keeps
        # the platform metadata it already had.
        print(f"[FAN HISTORY] vault link lookup failed creator={creator_id}: {exc}")
        return {}

    links: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row.get("fansly_media_id") or "")
        if not key:
            continue
        links[key] = {
            "vault_media_id": str(row.get("id") or "") or None,
            "vault_content_category": row.get("content_category"),
            "vault_mimetype": row.get("mimetype"),
        }
    return links


def _attach_vault_links(
    rows: list[dict[str, Any]],
    links: dict[str, dict[str, Any]],
) -> int:
    """Fold local vault facts into each row's attachment metadata, in place."""
    linked = 0
    for row in rows:
        context = row.get("media_context") or {}
        for attachment in context.get("attachments") or []:
            link = links.get(str(attachment.get("contentId") or ""))
            if not link:
                continue
            attachment.update({key: value for key, value in link.items() if value})
            linked += 1
    return linked


async def _insert_history_rows(rows: list[dict[str, Any]]) -> int:
    """Persist historical rows idempotently on platform message identity.

    Idempotency is the database's job, not a read-then-write in Python: the
    unique (creator_id, fansly_message_id) index created by
    db/message_platform_identity_v1.sql is what makes re-running a page after a
    restart write nothing. The pre-migration fallback mirrors db/queries.py so
    an import still works if the code ships ahead of the migration.
    """
    if not rows:
        return 0

    def _write() -> int:
        db = get_supabase()
        try:
            response = (
                db.table("messages")
                .upsert(
                    rows,
                    on_conflict=PLATFORM_IDENTITY_CONFLICT,
                    ignore_duplicates=True,
                )
                .execute()
            )
        except Exception as exc:
            if not is_missing_conflict_target(exc):
                raise
            print(
                "[FAN HISTORY] platform-identity index missing — importing with "
                "check-then-insert. Apply db/message_platform_identity_v1.sql"
            )
            inserted = 0
            for row in rows:
                existing = (
                    db.table("messages")
                    .select("id")
                    .eq("creator_id", row["creator_id"])
                    .eq("fansly_message_id", row["fansly_message_id"])
                    .limit(1)
                    .execute()
                )
                if existing.data:
                    continue
                db.table("messages").insert(row).execute()
                inserted += 1
            return inserted
        return len(list(response.data or []))

    return await asyncio.to_thread(_write)


class HistoryPage:
    """One fetched page and what it cost."""

    __slots__ = ("rows", "cursor", "messages", "usage", "media_references")

    def __init__(
        self,
        *,
        rows: list[dict[str, Any]],
        cursor: str | None,
        messages: list[dict[str, Any]],
        usage: dict[str, Any],
        media_references: int,
    ) -> None:
        self.rows = rows
        self.cursor = cursor
        self.messages = messages
        self.usage = usage
        self.media_references = media_references


async def fetch_history_page(
    *,
    creator_id: str,
    fan_id: str,
    account_id: str,
    creator_platform_id: str,
    group_id: str,
    cursor: str | None,
    client: Any = None,
) -> HistoryPage:
    """Fetch and parse one page of history. No writes, no media calls."""
    with collect_usage(CATEGORY_BACKGROUND_HISTORY) as spent:
        messages, account_media, next_cursor = await list_chat_messages(
            account_id,
            group_id,
            cursor=cursor,
            limit=CHAT_MESSAGE_PAGE_MAX,
            client=client,
        )
    media_lookup = account_media_lookup(account_media)
    rows: list[dict[str, Any]] = []
    # The provider returns newest first; reversing keeps stored rows in
    # chronological order within the page, matching every other ingestion path.
    for message in reversed(messages):
        row = chat_message_row(
            message,
            fan_id=fan_id,
            creator_id=creator_id,
            creator_platform_id=creator_platform_id,
            media_lookup=media_lookup,
        )
        if row:
            rows.append(row)
    return HistoryPage(
        rows=rows,
        cursor=next_cursor,
        messages=messages,
        usage=summarize_usage_events(spent),
        media_references=len(media_lookup),
    )


async def persist_history_page(
    page: HistoryPage,
    *,
    creator_id: str,
    fan_id: str,
) -> dict[str, Any]:
    """Store one page's rows idempotently, with local vault links attached."""
    if not page.rows:
        return {"imported": 0, "seen": 0, "vault_linked": 0}

    message_ids = [str(row["fansly_message_id"]) for row in page.rows]
    known = await _existing_platform_ids(fan_id, message_ids)
    fresh = [row for row in page.rows if str(row["fansly_message_id"]) not in known]

    content_ids = [
        str(attachment.get("contentId") or "")
        for row in fresh
        for attachment in ((row.get("media_context") or {}).get("attachments") or [])
    ]
    vault_linked = _attach_vault_links(fresh, await _vault_links(creator_id, content_ids))

    imported = await _insert_history_rows(fresh)
    return {
        "imported": imported,
        "seen": len(page.rows),
        "vault_linked": vault_linked,
    }


def _page_bounds(rows: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if not rows:
        return None, None
    return rows[0], rows[-1]


def _merged_progress(
    state: dict[str, Any],
    *,
    page: HistoryPage,
    persisted: dict[str, Any],
) -> dict[str, Any]:
    """The state patch one successful page produces. Pure and testable."""
    oldest, newest = _page_bounds(page.rows)
    patch: dict[str, Any] = {
        "pages_fetched": int(state.get("pages_fetched") or 0) + 1,
        "messages_seen": int(state.get("messages_seen") or 0) + int(persisted["seen"]),
        "messages_imported": int(state.get("messages_imported") or 0)
        + int(persisted["imported"]),
        "api_calls": int(state.get("api_calls") or 0) + int(page.usage["calls"]),
        "response_bytes": int(state.get("response_bytes") or 0)
        + int(page.usage["response_bytes"]),
        "estimated_credits": round(
            float(state.get("estimated_credits") or 0.0)
            + float(page.usage["estimated_credits"]),
            3,
        ),
        "media_references_seen": int(state.get("media_references_seen") or 0)
        + int(page.media_references),
        "last_page_at": _now(),
        "last_error": None,
    }
    # The cursor advances only here, after the page has been persisted. A crash
    # between the fetch and this write re-fetches the page, which is idempotent.
    patch["page_cursor"] = page.cursor
    if not page.cursor or not page.messages:
        patch["exhausted"] = True
    if oldest is not None:
        patch["oldest_message_id"] = str(oldest["fansly_message_id"])
        patch["oldest_sent_at"] = oldest["sent_at"]
    if newest is not None and not state.get("newest_message_id"):
        patch["newest_message_id"] = str(newest["fansly_message_id"])
        patch["newest_sent_at"] = newest["sent_at"]
    return patch


async def _load_bindings(creator_id: str, fan_id: str) -> dict[str, str]:
    db = get_supabase()

    def _fan() -> dict[str, Any]:
        response = (
            db.table("fans")
            .select("fansly_group_id, platform_fan_id")
            .eq("id", fan_id)
            .eq("creator_id", creator_id)
            .limit(1)
            .execute()
        )
        rows = list(response.data or [])
        return rows[0] if rows else {}

    def _creator() -> dict[str, Any]:
        response = (
            db.table("creators")
            .select("apifansly_account_id, fansly_account_id")
            .eq("id", creator_id)
            .limit(1)
            .execute()
        )
        rows = list(response.data or [])
        return rows[0] if rows else {}

    fan, creator = await asyncio.gather(
        asyncio.to_thread(_fan), asyncio.to_thread(_creator)
    )
    return {
        "group_id": str(fan.get("fansly_group_id") or ""),
        "account_id": str(creator.get("apifansly_account_id") or ""),
        "creator_platform_id": str(creator.get("fansly_account_id") or ""),
        "platform_fan_id": str(fan.get("platform_fan_id") or ""),
    }


def _history_unavailable_reason(bindings: dict[str, str]) -> str | None:
    """Why this fan's history cannot or must not be fetched from the provider.

    Two refusals, both before any provider call:

      * the connector is off deployment-wide — every other remote path already
        answers this way, and history is the most optional of them;
      * the fan is a simulator test fan. Requirement 6's boundary is that a test
        fan never reaches the platform, and a "historical import" would be a
        real provider call against a fan that does not exist there. A stale
        fansly_group_id on a test fan is exactly the case this catches.
    """
    if not apifansly_enabled():
        return "connector_disabled"
    if is_simulatable_fan(bindings.get("platform_fan_id")):
        return "simulation_fan"
    if not bindings.get("group_id") or not bindings.get("account_id"):
        return "unbound"
    return None


async def warm_resume(
    *,
    creator_id: str,
    fan_id: str,
    client: Any = None,
) -> dict[str, Any]:
    """Make an old conversation answerable now, for a few credits.

    Called when a previously known fan becomes active. If enough recent local
    context already exists this costs nothing and returns immediately; that is
    the common case and the reason this is safe to call on every turn.

    Otherwise it fetches the newest few pages — thirty messages by default —
    persists them, and returns. It deliberately does NOT read the archive: the
    remaining history is left to the resumable deep pass, because a fan waiting
    for a reply must never wait through five hundred provider requests.
    """
    minimum = warm_resume_min_local_messages()
    local = await local_message_count(fan_id)
    if local >= minimum:
        return {
            "status": "sufficient_context",
            "local_messages": local,
            "pages": 0,
            "imported": 0,
            "estimated_credits": 0.0,
        }

    bindings = await _load_bindings(creator_id, fan_id)
    refusal = _history_unavailable_reason(bindings)
    if refusal:
        return {
            "status": refusal,
            "local_messages": local,
            "pages": 0,
            "imported": 0,
            "estimated_credits": 0.0,
        }

    state = await ensure_backfill_state(creator_id=creator_id, fan_id=fan_id) or {}
    # A backfill that has already paged owns the cursor. Warm resume then reads
    # the newest pages for continuity WITHOUT moving that cursor, so a deep pass
    # that is four hundred pages in is never rewound to the start.
    owns_cursor = int(state.get("pages_fetched") or 0) == 0

    pages = 0
    imported = 0
    calls = 0
    response_bytes = 0
    credits = 0.0
    cursor: str | None = None
    max_pages = warm_resume_max_pages()

    async def _run(active_client: Any) -> None:
        nonlocal pages, imported, calls, response_bytes, credits, cursor, state
        while pages < max_pages:
            page = await fetch_history_page(
                creator_id=creator_id,
                fan_id=fan_id,
                account_id=bindings["account_id"],
                creator_platform_id=bindings["creator_platform_id"],
                group_id=bindings["group_id"],
                cursor=cursor,
                client=active_client,
            )
            persisted = await persist_history_page(
                page, creator_id=creator_id, fan_id=fan_id
            )
            pages += 1
            imported += int(persisted["imported"])
            calls += int(page.usage["calls"])
            response_bytes += int(page.usage["response_bytes"])
            credits += float(page.usage["estimated_credits"])
            if owns_cursor:
                patch = _merged_progress(state, page=page, persisted=persisted)
                patch["status"] = "complete" if patch.get("exhausted") else "paused"
                updated = await update_backfill_state(fan_id, patch)
                state = updated or {**state, **patch}
            cursor = page.cursor
            if not cursor or not page.messages:
                break

    if client is not None:
        await _run(client)
    else:
        async with client_scope() as owned:
            await _run(owned)

    if not owns_cursor:
        # Still record the cost: these credits were spent on this fan's history
        # and an operator looking at the fan must see them.
        await update_backfill_state(
            fan_id,
            {
                "api_calls": int(state.get("api_calls") or 0) + calls,
                "response_bytes": int(state.get("response_bytes") or 0) + response_bytes,
                "estimated_credits": round(
                    float(state.get("estimated_credits") or 0.0) + credits, 3
                ),
                "last_page_at": _now(),
            },
        )

    print(
        f"[HISTORY WARM RESUME] fan={fan_id} local_before={local} pages={pages} "
        f"imported={imported} credits={credits:.2f}"
    )
    return {
        "status": "resumed",
        "local_messages": local,
        "pages": pages,
        "imported": imported,
        "calls": calls,
        "response_bytes": response_bytes,
        "estimated_credits": round(credits, 3),
        "cursor_advanced": owns_cursor,
    }


async def advance_backfill(
    *,
    creator_id: str,
    fan_id: str,
    max_pages: int | None = None,
    client: Any = None,
    respect_live_priority: bool = True,
) -> dict[str, Any]:
    """Fetch the next bounded run of deep history, resuming from the cursor.

    Returns without doing anything when live work is in progress or the
    optional history budget is spent. Both are ``paused``, not ``error``: the
    work is still wanted, just not now, and the cursor is untouched so the next
    run continues exactly where this one stopped.
    """
    state = await ensure_backfill_state(creator_id=creator_id, fan_id=fan_id)
    if state is None:
        return {"status": "unavailable", "reason": "history_state_missing", "pages": 0}
    if state.get("exhausted"):
        return {
            "status": "complete",
            "pages": 0,
            "imported": 0,
            "estimated_credits": 0.0,
            **history_progress_view(state),
        }

    budget = background_history_budget_state()
    if budget["exhausted"]:
        await update_backfill_state(fan_id, {"status": "paused"})
        return {
            "status": "paused",
            "reason": "history_credit_budget_exhausted",
            "pages": 0,
            "budget": budget,
        }
    if respect_live_priority and live_work_in_progress():
        await update_backfill_state(fan_id, {"status": "paused"})
        return {"status": "paused", "reason": "live_work_in_progress", "pages": 0}

    bindings = await _load_bindings(creator_id, fan_id)
    refusal = _history_unavailable_reason(bindings)
    if refusal == "unbound":
        await update_backfill_state(
            fan_id, {"status": "error", "last_error": "missing fan or creator binding"}
        )
        return {"status": "error", "reason": "unbound", "pages": 0}
    if refusal:
        # A disabled connector or a simulator fan is not an error state — there
        # is simply nothing to fetch — so the cursor and status are left alone.
        return {"status": "paused", "reason": refusal, "pages": 0}

    limit = max(1, int(max_pages or deep_backfill_pages_per_run()))
    patch_started = {"status": "running"}
    if not state.get("started_at"):
        patch_started["started_at"] = _now()
    state = await update_backfill_state(fan_id, patch_started) or {
        **state,
        **patch_started,
    }

    pages = 0
    imported = 0
    credits = 0.0
    stop_reason = "page_budget"

    async def _run(active_client: Any) -> None:
        nonlocal pages, imported, credits, state, stop_reason
        while pages < limit:
            # Re-asked before EVERY page, not once at the top. A reply that
            # arrives mid-run must not queue behind the rest of this batch.
            if respect_live_priority and live_work_in_progress():
                stop_reason = "live_work_in_progress"
                return
            if not background_history_allowed():
                stop_reason = "history_credit_budget_exhausted"
                return
            page = await fetch_history_page(
                creator_id=creator_id,
                fan_id=fan_id,
                account_id=bindings["account_id"],
                creator_platform_id=bindings["creator_platform_id"],
                group_id=bindings["group_id"],
                cursor=state.get("page_cursor") or None,
                client=active_client,
            )
            persisted = await persist_history_page(
                page, creator_id=creator_id, fan_id=fan_id
            )
            patch = _merged_progress(state, page=page, persisted=persisted)
            state = await update_backfill_state(fan_id, patch) or {**state, **patch}
            pages += 1
            imported += int(persisted["imported"])
            credits += float(page.usage["estimated_credits"])
            if patch.get("exhausted"):
                stop_reason = "exhausted"
                return

    try:
        if client is not None:
            await _run(client)
        else:
            async with client_scope() as owned:
                await _run(owned)
    except Exception as exc:
        await update_backfill_state(
            fan_id, {"status": "error", "last_error": str(exc)[:500]}
        )
        print(f"[HISTORY BACKFILL ERROR] fan={fan_id}: {exc}")
        return {
            "status": "error",
            "reason": str(exc)[:200],
            "pages": pages,
            "imported": imported,
            "estimated_credits": round(credits, 3),
        }

    # Anything short of an exhausted cursor is "paused", whether this run
    # stopped at its page budget, yielded to live work or hit the credit
    # ceiling. The cursor already records where to resume; the status only says
    # whether there is anything left to resume to.
    final = (
        {"status": "complete", "completed_at": _now()}
        if state.get("exhausted")
        else {"status": "paused"}
    )
    state = await update_backfill_state(fan_id, final) or {**state, **final}

    print(
        f"[HISTORY BACKFILL] fan={fan_id} pages={pages} imported={imported} "
        f"credits={credits:.2f} stop={stop_reason}"
    )
    return {
        "status": state.get("status") or "paused",
        "stop_reason": stop_reason,
        "pages": pages,
        "imported": imported,
        "estimated_credits": round(credits, 3),
        **history_progress_view(state),
    }


def history_progress_view(state: dict[str, Any] | None) -> dict[str, Any]:
    """Operator-facing progress and cost for one fan's history.

    "How much remains" is reported honestly: the provider does not tell us how
    long a conversation is until the cursor runs out, so an unfinished backfill
    reports what it has cost and what a page costs on average rather than
    inventing a total.
    """
    if not state:
        return {
            "history": {
                "status": "not_started",
                "pages_fetched": 0,
                "messages_imported": 0,
                "fully_paged": False,
                "estimated_credits": 0.0,
            }
        }
    pages = int(state.get("pages_fetched") or 0)
    credits = float(state.get("estimated_credits") or 0.0)
    exhausted = bool(state.get("exhausted"))
    return {
        "history": {
            "status": str(state.get("status") or "pending"),
            "fully_paged": exhausted,
            "pages_fetched": pages,
            "messages_seen": int(state.get("messages_seen") or 0),
            "messages_imported": int(state.get("messages_imported") or 0),
            "messages_extracted": int(state.get("messages_extracted") or 0),
            "chunks_extracted": int(state.get("chunks_extracted") or 0),
            "facts_accepted": int(state.get("facts_accepted") or 0),
            "media_references_seen": int(state.get("media_references_seen") or 0),
            "api_calls": int(state.get("api_calls") or 0),
            "response_bytes": int(state.get("response_bytes") or 0),
            "estimated_credits": round(credits, 3),
            "estimated_credits_per_page": round(credits / pages, 3) if pages else None,
            # Zero when finished, null when genuinely unknown — never a guess.
            # The provider does not reveal how long a conversation is until its
            # cursor runs out, and an invented total would be read as a promise.
            "remaining_pages": 0 if exhausted else None,
            "remaining_pages_note": (
                "complete"
                if exhausted
                else "unknown until the provider cursor is exhausted"
            ),
            "messages_per_page_max": CHAT_MESSAGE_PAGE_MAX,
            "last_error": state.get("last_error"),
            "last_page_at": state.get("last_page_at"),
        }
    }


async def fan_history_status(fan_id: str) -> dict[str, Any]:
    """Progress view for one fan, reading nothing else."""
    return history_progress_view(await get_backfill_state(fan_id))


async def backfill_scheduler_pass(*, max_fans: int = 3) -> dict[str, Any]:
    """One scheduler tick: advance the least recently touched backfills.

    Bounded in fans and, through ``advance_backfill``, in pages. Yields
    immediately and entirely when live work is happening, and does nothing at
    all unless HISTORY_BACKFILL_ENABLED is on.
    """
    if not history_backfill_enabled():
        return {"status": "disabled", "fans": 0}
    if live_work_in_progress():
        return {"status": "yielded", "reason": "live_work_in_progress", "fans": 0}
    budget = background_history_budget_state()
    if budget["exhausted"]:
        return {
            "status": "yielded",
            "reason": "history_credit_budget_exhausted",
            "fans": 0,
            "budget": budget,
        }

    rows = await list_unfinished_backfills(limit=max(1, int(max_fans)))
    advanced = 0
    credits = 0.0
    async with client_scope() as client:
        for row in rows:
            if live_work_in_progress():
                break
            result = await advance_backfill(
                creator_id=str(row.get("creator_id") or ""),
                fan_id=str(row.get("fan_id") or ""),
                client=client,
            )
            advanced += 1
            credits += float(result.get("estimated_credits") or 0.0)
    return {
        "status": "ok",
        "fans": advanced,
        "estimated_credits": round(credits, 3),
    }


async def all_local_history(fan_id: str) -> list[dict[str, Any]]:
    """Every locally stored message for a fan, oldest first.

    Used by compaction, which walks chronologically. Paginated because a
    5,000-message fan is five times PostgREST's default cap and an unranged
    read would silently return a prefix.
    """
    db = get_supabase()
    return await fetch_all_rows_async(
        lambda start, end: db.table("messages")
        .select("id, role, content, sent_at, fansly_message_id, media_context")
        .eq("fan_id", fan_id)
        .order("sent_at")
        .order("id")
        .range(start, end)
        .execute()
    )
