"""Operational health: is this deployment actually keeping up?

Three distinct questions get three distinct answers, because conflating them is
how a healthcheck turns a provider hiccup into a restart loop:

* **Liveness** — is the process and its event loop alive? If this endpoint
  answers at all, the answer is yes. Nothing external can make it no.
* **Readiness** — can this process do its job at all? Only genuinely fatal
  infrastructure state counts: the database being unreachable. A model provider
  returning 500s does *not* make the process unready; the queue is durable and
  the work waits.
* **Degraded** — is it keeping up? Queue depth, oldest pending age, scheduler
  cycle recency, and model-gate saturation. Visible to operators, never fatal to
  the platform.

Nothing here returns message content, prompts, fan names, tokens, or provider
keys. Only counts, ages, statuses and configuration limits.
"""

from __future__ import annotations

import asyncio
import os
import time
from datetime import datetime, timezone
from typing import Any

from core.action_failures import TERMINAL_PREFIX, terminal_code
from core.model_gate import MODEL_GATE
from core.vault_gate import VAULT_GATE
from services.model_availability import current_model_availability
from workers.scheduled_actions import worker_health_snapshot

# Probe results are cached briefly so a chatty healthcheck (or a dashboard
# banner polling on focus) cannot itself become a source of database load.
_CACHE_SECONDS = 5.0
_DB_PROBE_TIMEOUT_SECONDS = 1.5
_QUEUE_PROBE_TIMEOUT_SECONDS = 2.5

DEFAULT_QUEUE_MAX_AGE_SECONDS = 900
DEFAULT_QUEUE_MAX_DEPTH = 500

# The scheduler is considered stale after this many idle poll intervals without
# completing a cycle. A cycle can legitimately take a while when a full batch of
# actions is draining, so the multiplier is generous.
STALE_CYCLE_INTERVALS = 6
MIN_STALE_CYCLE_SECONDS = 120

_cache: dict[str, Any] = {"at": 0.0, "value": None}

# A freshly started process has legitimately not completed a scheduler cycle
# yet. Reporting degraded for the first few seconds of every deploy would make
# the signal noise, so the "never ran" check only applies after a grace period.
_PROCESS_STARTED_AT = time.monotonic()
SCHEDULER_START_GRACE_SECONDS = 90


def process_uptime_seconds() -> float:
    return time.monotonic() - _PROCESS_STARTED_AT


# How many terminal failures are inspected per probe. The health document is
# polled often and this is a diagnosis aid, not an accounting record: enough
# rows to identify WHICH cause is blocking work, cheap enough to read every few
# seconds.
_TERMINAL_SAMPLE_LIMIT = 200


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


def queue_max_age_seconds() -> int:
    return _env_int("HEALTH_QUEUE_MAX_AGE_SECONDS", DEFAULT_QUEUE_MAX_AGE_SECONDS)


def queue_max_depth() -> int:
    return _env_int("HEALTH_QUEUE_MAX_DEPTH", DEFAULT_QUEUE_MAX_DEPTH)


def _parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


async def probe_database() -> dict:
    """One cheap, time-bounded round trip. Never raises."""
    from core.supabase import get_supabase

    started = time.perf_counter()

    def _ping() -> None:
        (
            get_supabase().table("creators")
            .select("id")
            .limit(1)
            .execute()
        )

    try:
        await asyncio.wait_for(
            asyncio.to_thread(_ping), timeout=_DB_PROBE_TIMEOUT_SECONDS
        )
    except (asyncio.TimeoutError, TimeoutError):
        return {
            "reachable": False,
            "latency_ms": int((time.perf_counter() - started) * 1000),
            "error": "timeout",
        }
    except Exception as exc:
        return {
            "reachable": False,
            "latency_ms": int((time.perf_counter() - started) * 1000),
            # Type name only. The message can carry a connection string.
            "error": type(exc).__name__,
        }
    return {
        "reachable": True,
        "latency_ms": int((time.perf_counter() - started) * 1000),
        "error": None,
    }


async def probe_queue() -> dict:
    """Counts by status plus the age of the oldest thing that should have run."""
    from core.supabase import get_supabase

    def _read() -> dict:
        db = get_supabase()
        counts: dict[str, int] = {}
        for status in ("PENDING", "PROCESSING", "FAILED"):
            response = (
                db.table("scheduled_actions")
                .select("id", count="exact")
                .eq("status", status)
                .limit(1)
                .execute()
            )
            counts[status.lower()] = int(getattr(response, "count", 0) or 0)
        oldest = (
            db.table("scheduled_actions")
            .select("execute_at, action_type")
            .eq("status", "PENDING")
            .lte("execute_at", datetime.now(timezone.utc).isoformat())
            .order("execute_at")
            .limit(1)
            .execute()
        ).data or []
        inbound = (
            db.table("scheduled_actions")
            .select("id", count="exact")
            .eq("status", "PENDING")
            .eq("action_type", "PROCESS_INBOUND_MESSAGE")
            .limit(1)
            .execute()
        )
        # REL-005 — an operator needs to tell "the provider is having a bad ten
        # minutes" apart from "twenty fans cannot send because a binding is
        # permanently broken". Both show up as FAILED rows; only the second is
        # someone's job to fix. Terminal failures carry a machine-readable code
        # in last_error, so they can be counted per cause rather than in total.
        blocked = (
            db.table("scheduled_actions")
            .select("action_type, last_error")
            .eq("status", "FAILED")
            .like("last_error", f"{TERMINAL_PREFIX}:%")
            .order("id")
            .limit(_TERMINAL_SAMPLE_LIMIT)
            .execute()
        ).data or []
        return {
            "counts": counts,
            "oldest": oldest[0] if oldest else None,
            "pending_inbound": int(getattr(inbound, "count", 0) or 0),
            "blocked": blocked,
        }

    try:
        raw = await asyncio.wait_for(
            asyncio.to_thread(_read), timeout=_QUEUE_PROBE_TIMEOUT_SECONDS
        )
    except (asyncio.TimeoutError, TimeoutError):
        return {"available": False, "error": "timeout"}
    except Exception as exc:
        return {"available": False, "error": type(exc).__name__}

    oldest = raw["oldest"] or {}
    oldest_at = _parse_time(oldest.get("execute_at"))
    age_seconds = (
        round((datetime.now(timezone.utc) - oldest_at).total_seconds(), 1)
        if oldest_at
        else 0.0
    )
    counts = raw["counts"]
    blocked_by_reason: dict[str, int] = {}
    for row in raw.get("blocked") or []:
        code = terminal_code(row.get("last_error")) or "unknown"
        blocked_by_reason[code] = blocked_by_reason.get(code, 0) + 1

    return {
        "available": True,
        "pending": counts.get("pending", 0),
        "processing": counts.get("processing", 0),
        "failed": counts.get("failed", 0),
        "pending_inbound_messages": raw["pending_inbound"],
        "oldest_due_execute_at": oldest_at.isoformat() if oldest_at else None,
        "oldest_due_action_type": str(oldest.get("action_type") or "") or None,
        "oldest_pending_age_seconds": age_seconds,
        # Actions that will never run again without an operator changing
        # something, counted by cause. Sampled, so it is a floor, not a census —
        # the point is "which problem", not an exact total.
        "blocked_actions": sum(blocked_by_reason.values()),
        "blocked_by_reason": blocked_by_reason,
        "blocked_sample_truncated": (
            len(raw.get("blocked") or []) >= _TERMINAL_SAMPLE_LIMIT
        ),
        "error": None,
    }


def evaluate(
    *,
    database: dict,
    queue: dict,
    scheduler: dict,
    model_gate: dict,
    model_availability: dict,
) -> dict:
    """Turn the raw probes into a status plus the reasons behind it.

    ``fatal`` means the process genuinely cannot work and readiness should fail.
    Everything else is degraded: visible, alertable, and explicitly not a reason
    to restart the container.
    """
    degraded: list[str] = []
    fatal: list[str] = []

    if not database.get("reachable"):
        fatal.append(f"database_unreachable:{database.get('error') or 'unknown'}")

    max_age = queue_max_age_seconds()
    max_depth = queue_max_depth()

    if queue.get("available"):
        if queue.get("oldest_pending_age_seconds", 0) > max_age:
            degraded.append(
                f"queue_oldest_pending_age_exceeds_{max_age}s"
            )
        if queue.get("pending", 0) > max_depth:
            degraded.append(f"queue_depth_exceeds_{max_depth}")
    else:
        degraded.append("queue_unreadable")

    poll = float(scheduler.get("poll_seconds") or 60)
    stale_after = max(MIN_STALE_CYCLE_SECONDS, poll * STALE_CYCLE_INTERVALS)
    since = scheduler.get("seconds_since_last_cycle")
    if since is None:
        if (
            scheduler.get("cycles_completed", 0) == 0
            and process_uptime_seconds() > SCHEDULER_START_GRACE_SECONDS
        ):
            degraded.append("scheduler_has_not_completed_a_cycle")
    elif since > stale_after:
        degraded.append(f"scheduler_stale_for_{int(since)}s")
    if scheduler.get("last_error"):
        degraded.append("scheduler_last_cycle_errored")

    # Terminal failures are degraded rather than fatal: the deployment is
    # working, some creator's configuration is not. Named per cause so the
    # alert says what to fix.
    for code, count in sorted((queue.get("blocked_by_reason") or {}).items()):
        degraded.append(f"actions_blocked_{code}:{count}")

    limit = int(model_gate.get("limit") or 0)
    if limit and int(model_gate.get("waiting") or 0) >= limit:
        degraded.append("model_gate_saturated")

    availability_status = str(model_availability.get("status") or "unknown")
    if availability_status in {"unavailable", "misconfigured"}:
        # Provider trouble is degraded, never fatal: the queue is durable and
        # the work waits rather than being lost.
        degraded.append(f"model_{availability_status}")

    if fatal:
        status = "unhealthy"
    elif degraded:
        status = "degraded"
    else:
        status = "ok"
    return {"status": status, "degraded_reasons": degraded, "fatal_reasons": fatal}


async def collect(*, use_cache: bool = True) -> dict:
    """Assemble the full health document, cached for a few seconds."""
    now = time.monotonic()
    if use_cache and _cache["value"] is not None and now - _cache["at"] < _CACHE_SECONDS:
        return _cache["value"]

    database, queue = await asyncio.gather(probe_database(), probe_queue())
    scheduler = worker_health_snapshot()
    model_gate = MODEL_GATE.snapshot()
    # VAULT-001 — informational, never part of the verdict. A queue here is the
    # gate working as designed: vault work waiting is exactly what stops it
    # competing with chat, so it must not be reported as degraded.
    vault_gate = VAULT_GATE.snapshot()
    availability = current_model_availability()
    model_summary = {
        "status": availability.get("status"),
        "checked_at": availability.get("checked_at"),
        "gate": model_gate,
    }

    verdict = evaluate(
        database=database,
        queue=queue,
        scheduler=scheduler,
        model_gate=model_gate,
        model_availability=availability,
    )

    document = {
        "status": verdict["status"],
        "liveness": "ok",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "degraded_reasons": verdict["degraded_reasons"],
        "fatal_reasons": verdict["fatal_reasons"],
        "thresholds": {
            "queue_max_age_seconds": queue_max_age_seconds(),
            "queue_max_depth": queue_max_depth(),
        },
        "database": database,
        "queue": queue,
        "scheduler": scheduler,
        "model": model_summary,
        "vault": {"gate": vault_gate},
    }
    _cache["at"] = now
    _cache["value"] = document
    return document


def reset_cache() -> None:
    """Test-support only."""
    _cache["at"] = 0.0
    _cache["value"] = None
