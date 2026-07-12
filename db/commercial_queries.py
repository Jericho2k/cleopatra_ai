"""Persistence for the commercial layer: per-creator policy, per-fan state, and the
scheduled-actions queue.

The queue is deliberately simple (Postgres, not Kafka) but exactly-once where it
counts: a unique dedupe_key per logical event, and claim-with-lock so multiple
Railway workers can't send the same follow-up twice.
"""
import asyncio
from datetime import datetime, timedelta, timezone

from core.supabase import get_supabase
from models.commercial import CreatorPolicy, FanCommercialState, FanStatus


# ---------------- creator policy ----------------

async def get_creator_policy(creator_id: str) -> CreatorPolicy:
    def _get():
        r = (get_supabase().table("creator_commercial_policies")
             .select("*").eq("creator_id", creator_id).execute())
        return (r.data or [None])[0]
    row = await asyncio.to_thread(_get)
    if not row:
        return CreatorPolicy()  # sane defaults if no row yet
    row.pop("creator_id", None)
    row.pop("updated_at", None)
    try:
        return CreatorPolicy(**row)
    except Exception:
        return CreatorPolicy()


# ---------------- fan commercial state ----------------

async def get_fan_state(fan_id: str) -> FanCommercialState:
    def _get():
        r = (get_supabase().table("fan_commercial_states")
             .select("*").eq("fan_id", fan_id).execute())
        return (r.data or [None])[0]
    row = await asyncio.to_thread(_get)
    if not row:
        return FanCommercialState()
    for k in ("fan_id", "creator_id", "updated_at"):
        row.pop(k, None)
    try:
        return FanCommercialState(**row)
    except Exception:
        return FanCommercialState()


async def save_fan_state(fan_id: str, creator_id: str, state: FanCommercialState) -> None:
    payload = {
        "fan_id": fan_id,
        "creator_id": creator_id,
        "status": state.status.value,
        "desired_experience": state.desired_experience,
        "preferences_snapshot": state.preferences_snapshot,
        "confirmed_budget_cents": state.confirmed_budget_cents,
        "budget_source": state.budget_source,
        "payday_raw": state.payday_raw,
        "payday_at": state.payday_at.isoformat() if state.payday_at else None,
        "payday_confidence": state.payday_confidence,
        "last_declined_price_cents": state.last_declined_price_cents,
        "teaser_messages_used": state.teaser_messages_used,
        "free_session_started_at": (
            state.free_session_started_at.isoformat()
            if state.free_session_started_at else None
        ),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    def _upsert():
        get_supabase().table("fan_commercial_states").upsert(
            payload, on_conflict="fan_id"
        ).execute()
    await asyncio.to_thread(_upsert)


# ---------------- scheduled actions ----------------

async def schedule_action(
    creator_id: str,
    fan_id: str,
    action_type: str,
    execute_at: datetime,
    payload: dict,
    dedupe_key: str,
) -> None:
    """Upsert on dedupe_key: if he says 'actually I get paid Monday', this REPLACES
    the Friday action rather than creating a second one."""
    row = {
        "creator_id": creator_id,
        "fan_id": fan_id,
        "action_type": action_type,
        "execute_at": execute_at.isoformat(),
        "payload": payload,
        "dedupe_key": dedupe_key,
        "status": "PENDING",
        "attempts": 0,
        "locked_at": None,
        "last_error": None,
    }
    def _upsert():
        get_supabase().table("scheduled_actions").upsert(
            row, on_conflict="dedupe_key"
        ).execute()
    await asyncio.to_thread(_upsert)


async def cancel_actions_for_fan(fan_id: str, action_type: str | None = None) -> None:
    """He paid early / no longer needs the follow-up."""
    def _cancel():
        q = (get_supabase().table("scheduled_actions")
             .update({"status": "CANCELLED"})
             .eq("fan_id", fan_id).eq("status", "PENDING"))
        if action_type:
            q = q.eq("action_type", action_type)
        q.execute()
    await asyncio.to_thread(_cancel)


async def claim_due_actions(limit: int = 20, stale_minutes: int = 10) -> list[dict]:
    """Claim due PENDING actions for this worker.

    Supabase's REST client can't do FOR UPDATE SKIP LOCKED, so we emulate a claim:
    flip PENDING -> PROCESSING with a locked_at stamp, and only pick rows that
    aren't already being processed. Rows stuck in PROCESSING longer than
    stale_minutes are reclaimed (a worker died mid-send).
    """
    now = datetime.now(timezone.utc)
    stale_before = (now - timedelta(minutes=stale_minutes)).isoformat()

    def _claim():
        db = get_supabase()
        due = (db.table("scheduled_actions")
               .select("*")
               .eq("status", "PENDING")
               .lte("execute_at", now.isoformat())
               .order("execute_at")
               .limit(limit).execute()).data or []

        stale = (db.table("scheduled_actions")
                 .select("*")
                 .eq("status", "PROCESSING")
                 .lt("locked_at", stale_before)
                 .limit(limit).execute()).data or []

        claimed = []
        for row in due + stale:
            # Conditional update acts as the lock: if another worker already moved
            # it out of this status, the update matches 0 rows and we skip it.
            res = (db.table("scheduled_actions")
                   .update({"status": "PROCESSING", "locked_at": now.isoformat()})
                   .eq("id", row["id"])
                   .eq("status", row["status"])
                   .execute())
            if res.data:
                claimed.append(row)
        return claimed

    return await asyncio.to_thread(_claim)


async def complete_action(action_id: str) -> None:
    def _done():
        get_supabase().table("scheduled_actions").update(
            {"status": "COMPLETED"}
        ).eq("id", action_id).execute()
    await asyncio.to_thread(_done)


async def fail_action(action_id: str, error: str, attempts: int, max_attempts: int = 3) -> None:
    status = "FAILED" if attempts + 1 >= max_attempts else "PENDING"
    def _fail():
        get_supabase().table("scheduled_actions").update({
            "status": status,
            "attempts": attempts + 1,
            "last_error": error[:500],
            "locked_at": None,
        }).eq("id", action_id).execute()
    await asyncio.to_thread(_fail)