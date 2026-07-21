"""Durable, rate-limited re-engagement for otherwise idle Full Auto chats."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from core.supabase import get_supabase
from db.commercial_queries import (
    cancel_actions_for_fan,
    get_creator_policy,
    get_fan_state,
    save_fan_state,
    schedule_action,
)
from models.commercial import CreatorPolicy, FanCommercialState, FanStatus
from services.auto_audience import resolve_auto_eligibility_for_fan
from services.followup_lifecycle import as_utc, followup_at


ACTION_TYPE = "INACTIVITY_REENGAGEMENT"
WINDOW_DAYS = 30


@dataclass(frozen=True)
class InactivityCheck:
    ok: bool
    reason: str


def _refresh_window(
    state: FanCommercialState,
    now: datetime,
) -> None:
    started = as_utc(state.inactivity_reengagement_window_started_at)
    if started is None or now >= started + timedelta(days=WINDOW_DAYS):
        state.inactivity_reengagement_window_started_at = now
        state.inactivity_reengagement_count = 0


def frequency_check(
    state: FanCommercialState,
    policy: CreatorPolicy,
    *,
    now: datetime,
) -> InactivityCheck:
    _refresh_window(state, now)
    if state.inactivity_reengagement_count >= policy.inactivity_reengagement_max_per_30_days:
        return InactivityCheck(False, "30-day inactivity follow-up limit reached")
    last = as_utc(state.last_inactivity_reengagement_at)
    if last is not None:
        available_at = last + timedelta(hours=policy.inactivity_reengagement_cooldown_hours)
        if now < available_at:
            return InactivityCheck(False, f"inactivity cooldown remains until {available_at.isoformat()}")
    return InactivityCheck(True, "frequency limits allow one inactivity follow-up")


def record_inactivity_sent(
    state: FanCommercialState,
    *,
    now: datetime,
) -> None:
    _refresh_window(state, now)
    state.inactivity_reengagement_count += 1
    state.last_inactivity_reengagement_at = now


async def _latest_message(fan_id: str) -> dict[str, Any]:
    def _load() -> dict[str, Any]:
        rows = (
            get_supabase().table("messages")
            .select("id, role, sent_at")
            .eq("fan_id", fan_id)
            .order("sent_at", desc=True)
            .limit(1)
            .execute()
        ).data or []
        return rows[0] if rows else {}

    return await asyncio.to_thread(_load)


async def schedule_inactivity_reengagement(
    *,
    creator_id: str,
    fan_id: str,
    source_message_id: str | None = None,
    now: datetime | None = None,
) -> bool:
    """Schedule one nudge for the current silence episode.

    A commercial obligation always wins. The exact last creator message is
    stored in the payload so any fan return or newer reply invalidates this job.
    """
    current = as_utc(now) or datetime.now(timezone.utc)
    policy = await get_creator_policy(creator_id)
    if not policy.inactivity_reengagement_enabled:
        return False
    state = await get_fan_state(fan_id)
    if state.status != FanStatus.IDLE or state.next_followup_type:
        return False
    frequency = frequency_check(state, policy, now=current)
    if not frequency.ok:
        return False
    eligibility = await resolve_auto_eligibility_for_fan(creator_id, fan_id)
    if not eligibility.eligible:
        return False
    latest = await _latest_message(fan_id)
    if latest.get("role") != "creator":
        return False
    latest_id = str(latest.get("id") or "")
    if source_message_id and latest_id != str(source_message_id):
        return False
    if not latest_id:
        return False
    anchor = as_utc(latest.get("sent_at")) or current
    execute_at = followup_at(anchor, policy.inactivity_reengagement_delay_hours)
    last = as_utc(state.last_inactivity_reengagement_at)
    if last is not None:
        execute_at = max(
            execute_at,
            last + timedelta(hours=policy.inactivity_reengagement_cooldown_hours),
        )
    payload = {
        "source_message_id": latest_id,
        "source_message_at": anchor.isoformat(),
        "scheduled_at": current.isoformat(),
    }
    dedupe_key = f"inactivity:{fan_id}:{latest_id}"
    state.next_followup_at = execute_at
    state.next_followup_type = ACTION_TYPE
    state.next_followup_payload = payload
    state.next_followup_dedupe_key = dedupe_key
    # Fan state is the restart-safe obligation. The worker repairs a queue write
    # that fails after this point.
    await save_fan_state(fan_id, creator_id, state)
    try:
        await cancel_actions_for_fan(fan_id, ACTION_TYPE)
        await schedule_action(
            creator_id=creator_id,
            fan_id=fan_id,
            action_type=ACTION_TYPE,
            execute_at=execute_at,
            payload=payload,
            dedupe_key=dedupe_key,
        )
    except Exception as exc:
        print(f"[INACTIVITY REENGAGEMENT] queue repair needed fan={fan_id}: {exc}")
    print(
        f"[INACTIVITY REENGAGEMENT] scheduled fan={fan_id} "
        f"at={execute_at.isoformat()} source_message={latest_id}"
    )
    return True


async def validate_inactivity_action(
    action: dict[str, Any],
    state: FanCommercialState,
    policy: CreatorPolicy,
    *,
    now: datetime | None = None,
) -> InactivityCheck:
    current = as_utc(now) or datetime.now(timezone.utc)
    if not policy.inactivity_reengagement_enabled:
        return InactivityCheck(False, "inactivity re-engagement disabled for creator")
    if state.status != FanStatus.IDLE:
        return InactivityCheck(False, f"fan entered a commercial flow (status={state.status.value})")
    frequency = frequency_check(state, policy, now=current)
    if not frequency.ok:
        return frequency
    eligibility = await resolve_auto_eligibility_for_fan(
        str(action["creator_id"]),
        str(action["fan_id"]),
    )
    if not eligibility.eligible:
        return InactivityCheck(False, f"not eligible for Full Auto ({eligibility.reason})")
    latest = await _latest_message(str(action["fan_id"]))
    source_id = str((action.get("payload") or {}).get("source_message_id") or "")
    if not source_id or str(latest.get("id") or "") != source_id:
        return InactivityCheck(False, "fan returned or a newer message replaced this silence episode")
    if latest.get("role") != "creator":
        return InactivityCheck(False, "fan has already returned")
    return InactivityCheck(True, "eligible idle chat is still inactive")
