"""Persistence for restart-safe historical chat backfill state.

One row per fan in ``public.fan_history_backfill``, holding the paging cursor,
the compaction cursor and the provider cost already spent. Everything here is
deliberately small and synchronous-inside-a-thread, matching the rest of db/.

Every read tolerates the table being absent, because the code may deploy before
db/fan_history_backfill_v1.sql is applied. A missing table degrades history
backfill to "unavailable", which is correct: it is optional work. It must never
degrade a live reply, and nothing on the live path calls into this module.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from core.supabase import get_supabase


# Postgres 42P01: relation does not exist. The same shape of pre-migration
# tolerance db/queries.py applies to the platform-identity index.
_MISSING_TABLE_MARKERS = (
    "42p01",
    "does not exist",
    "could not find the table",
    "relation \"fan_history_backfill\"",
)

_backfill_table_missing = False


def backfill_table_missing() -> bool:
    """Whether history state is currently believed to be un-migrated."""
    return _backfill_table_missing


def reset_backfill_table_state() -> None:
    """Test support only. This flag is process-global on purpose."""
    global _backfill_table_missing
    _backfill_table_missing = False


def _is_missing_table(error: Exception) -> bool:
    text = str(error).lower()
    return any(marker in text for marker in _MISSING_TABLE_MARKERS)


def _note_missing_table(error: Exception) -> bool:
    global _backfill_table_missing
    if not _is_missing_table(error):
        return False
    if not _backfill_table_missing:
        print(
            "[FAN HISTORY] fan_history_backfill is missing — historical "
            "backfill is disabled until db/fan_history_backfill_v1.sql is "
            "applied. Live conversation is unaffected."
        )
    _backfill_table_missing = True
    return True


def _clear_missing_table() -> None:
    global _backfill_table_missing
    if _backfill_table_missing:
        print("[FAN HISTORY] fan_history_backfill is present again")
    _backfill_table_missing = False


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def get_backfill_state(fan_id: str) -> dict[str, Any] | None:
    """The fan's backfill row, or None when it has never been started."""

    def _get() -> dict[str, Any] | None:
        try:
            response = (
                get_supabase()
                .table("fan_history_backfill")
                .select("*")
                .eq("fan_id", fan_id)
                .limit(1)
                .execute()
            )
        except Exception as exc:
            if _note_missing_table(exc):
                return None
            raise
        _clear_missing_table()
        rows = list(response.data or [])
        return rows[0] if rows else None

    return await asyncio.to_thread(_get)


async def ensure_backfill_state(
    *,
    creator_id: str,
    fan_id: str,
) -> dict[str, Any] | None:
    """Return the fan's backfill row, creating a pending one if absent.

    Idempotent: the unique (fan_id) constraint decides the winner of a race
    between a dashboard button, the scheduler and a warm resume, and the loser
    reads the row the winner wrote rather than raising.
    """
    existing = await get_backfill_state(fan_id)
    if existing is not None:
        return existing

    def _insert() -> dict[str, Any] | None:
        payload = {
            "creator_id": creator_id,
            "fan_id": fan_id,
            "status": "pending",
            "updated_at": _now(),
        }
        # The whole block is guarded, not just .execute(): before the migration
        # the failure can come from anywhere in the chain, and history being
        # unavailable must never surface as an exception on a caller's path.
        try:
            db = get_supabase()
            response = (
                db.table("fan_history_backfill")
                .upsert(payload, on_conflict="fan_id", ignore_duplicates=True)
                .execute()
            )
            _clear_missing_table()
            rows = list(response.data or [])
            if rows:
                return rows[0]
            readback = (
                db.table("fan_history_backfill")
                .select("*")
                .eq("fan_id", fan_id)
                .limit(1)
                .execute()
            )
            rows = list(readback.data or [])
            return rows[0] if rows else None
        except Exception as exc:
            if _note_missing_table(exc):
                return None
            raise

    return await asyncio.to_thread(_insert)


async def update_backfill_state(fan_id: str, patch: dict[str, Any]) -> dict[str, Any] | None:
    """Apply one patch to the fan's backfill row and return the new row."""
    payload = {**patch, "updated_at": _now()}

    def _update() -> dict[str, Any] | None:
        try:
            response = (
                get_supabase()
                .table("fan_history_backfill")
                .update(payload)
                .eq("fan_id", fan_id)
                .execute()
            )
        except Exception as exc:
            if _note_missing_table(exc):
                return None
            raise
        _clear_missing_table()
        rows = list(response.data or [])
        return rows[0] if rows else None

    return await asyncio.to_thread(_update)


async def list_unfinished_backfills(limit: int = 25) -> list[dict[str, Any]]:
    """Fans with history left to fetch or compact, least recently touched first.

    Ordered and bounded rather than unranged: PostgREST would otherwise cap the
    read at its own row limit in an undefined order, which is how a scheduler
    ends up working the same few fans forever.
    """

    def _list() -> list[dict[str, Any]]:
        try:
            response = (
                get_supabase()
                .table("fan_history_backfill")
                .select(
                    "id, creator_id, fan_id, status, page_cursor, exhausted, "
                    "pages_fetched, messages_imported, messages_seen, "
                    "extraction_cursor_sent_at, messages_extracted, "
                    "estimated_credits, api_calls, updated_at"
                )
                .in_("status", ["pending", "running", "paused", "error"])
                .order("updated_at")
                .limit(max(1, int(limit)))
                .execute()
            )
        except Exception as exc:
            if _note_missing_table(exc):
                return []
            raise
        _clear_missing_table()
        return list(response.data or [])

    return await asyncio.to_thread(_list)


async def creator_history_totals(creator_id: str) -> dict[str, Any]:
    """Rolled-up backfill cost and progress for one creator."""

    def _read() -> list[dict[str, Any]]:
        try:
            response = (
                get_supabase()
                .table("fan_history_backfill")
                .select(
                    "fan_id, status, exhausted, pages_fetched, messages_imported, "
                    "messages_extracted, api_calls, response_bytes, estimated_credits"
                )
                .eq("creator_id", creator_id)
                .order("updated_at", desc=True)
                .limit(1000)
                .execute()
            )
        except Exception as exc:
            if _note_missing_table(exc):
                return []
            raise
        _clear_missing_table()
        return list(response.data or [])

    rows = await asyncio.to_thread(_read)
    return summarize_backfill_rows(rows)


def summarize_backfill_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate backfill rows. Pure, so the arithmetic is directly testable."""
    totals = {
        "fans": len(rows),
        "complete": 0,
        "in_progress": 0,
        "errored": 0,
        "pages_fetched": 0,
        "messages_imported": 0,
        "messages_extracted": 0,
        "api_calls": 0,
        "response_bytes": 0,
        "estimated_credits": 0.0,
    }
    for row in rows:
        status = str(row.get("status") or "")
        if status == "complete":
            totals["complete"] += 1
        elif status == "error":
            totals["errored"] += 1
        else:
            totals["in_progress"] += 1
        totals["pages_fetched"] += int(row.get("pages_fetched") or 0)
        totals["messages_imported"] += int(row.get("messages_imported") or 0)
        totals["messages_extracted"] += int(row.get("messages_extracted") or 0)
        totals["api_calls"] += int(row.get("api_calls") or 0)
        totals["response_bytes"] += int(row.get("response_bytes") or 0)
        totals["estimated_credits"] += float(row.get("estimated_credits") or 0.0)
    totals["estimated_credits"] = round(totals["estimated_credits"], 3)
    return totals


async def get_history_continuity(fan_id: str) -> dict[str, Any]:
    """The fan's compact historical continuity state, or an empty mapping.

    Read on the reply path through ``get_fan_intelligence_context``, so it is
    one narrow indexed row and it never raises: a fan with no history state is
    the ordinary case, and a failure here must not cost a reply.
    """

    def _get() -> dict[str, Any]:
        try:
            response = (
                get_supabase()
                .table("fan_history_backfill")
                .select("continuity, status, exhausted, messages_extracted")
                .eq("fan_id", fan_id)
                .limit(1)
                .execute()
            )
        except Exception as exc:
            if _note_missing_table(exc):
                return {}
            print(f"[FAN HISTORY] continuity read failed fan={fan_id}: {exc}")
            return {}
        _clear_missing_table()
        rows = list(response.data or [])
        if not rows:
            return {}
        row = rows[0]
        continuity = row.get("continuity")
        if not isinstance(continuity, dict) or not continuity:
            return {}
        return {
            **continuity,
            "backfill_status": row.get("status"),
            "history_fully_paged": bool(row.get("exhausted")),
            "messages_extracted": int(row.get("messages_extracted") or 0),
        }

    return await asyncio.to_thread(_get)
