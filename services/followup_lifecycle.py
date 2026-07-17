"""Pure helpers for durable full-auto follow-up and PPV reconciliation."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from models.commercial import CreatorPolicy, FanCommercialState, FanStatus


@dataclass(frozen=True)
class FollowupObligation:
    action_type: str
    execute_at: datetime
    payload: dict[str, Any]
    dedupe_key: str


def as_utc(value: datetime | str | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def followup_at(anchor: datetime | str, delay_hours: int) -> datetime:
    parsed = as_utc(anchor)
    if parsed is None:
        raise ValueError("A valid follow-up anchor is required")
    return parsed + timedelta(hours=max(1, int(delay_hours)))


def next_awake_time(
    value: datetime,
    *,
    sleep_start_hour: int,
    sleep_end_hour: int,
    timezone_name: str,
) -> datetime:
    """Return the same UTC time unless it falls inside creator sleep hours."""
    try:
        zone = ZoneInfo(timezone_name or "UTC")
    except ZoneInfoNotFoundError:
        zone = timezone.utc
    current = as_utc(value).astimezone(zone)
    start = max(0, min(23, int(sleep_start_hour)))
    end = max(0, min(23, int(sleep_end_hour)))
    if start == end:
        return as_utc(value)
    if start < end:
        in_sleep = start <= current.hour < end
        target = current.replace(hour=end, minute=0, second=0, microsecond=0)
    else:
        in_sleep = current.hour >= start or current.hour < end
        target = current.replace(hour=end, minute=0, second=0, microsecond=0)
        if current.hour >= start:
            target += timedelta(days=1)
    return target.astimezone(timezone.utc) if in_sleep else as_utc(value)


def payment_expires_at(
    pending: dict[str, Any],
    *,
    payment_window_hours: int,
) -> datetime | None:
    explicit = as_utc(pending.get("expires_at"))
    if explicit is not None:
        return explicit
    sent_at = as_utc(pending.get("sent_at"))
    if sent_at is None:
        return None
    return sent_at + timedelta(hours=max(1, int(payment_window_hours)))


def next_reconcile_at(
    now: datetime,
    *,
    expires_at: datetime,
    recheck_minutes: int,
) -> datetime:
    candidate = as_utc(now) + timedelta(minutes=max(5, int(recheck_minutes)))
    return min(candidate, as_utc(expires_at))


def pending_reference(pending: dict[str, Any]) -> str:
    explicit = str(pending.get("reference") or "").strip()
    if explicit:
        return explicit
    material = "|".join(
        str(pending.get(key) or "")
        for key in ("media_id", "set_id", "step_index", "price", "sent_at")
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]


def post_session_payload(
    session: dict[str, Any],
    *,
    buyer_stage: str | None = None,
) -> dict[str, Any]:
    plan = list(session.get("plan") or [])
    return {
        "session_completed_at": session.get("completed_at"),
        "commercial_package_id": session.get("commercial_package_id"),
        "set_ids": list(session.get("set_ids") or []),
        "experience": session.get("scene_key") or "the private session",
        "revenue_cents": int(session.get("revenue_cents", 0) or 0),
        "step_count": len(plan),
        "buyer_stage": buyer_stage or "UNKNOWN",
    }


def complete_session_state(
    state: FanCommercialState,
    session: dict[str, Any],
    *,
    policy: CreatorPolicy,
    fan_id: str,
    buyer_stage: str,
    now: datetime | None = None,
) -> tuple[FanCommercialState, FollowupObligation | None]:
    output = state.model_copy(deep=True)
    completed_at = as_utc(session.get("completed_at")) or as_utc(now) or datetime.now(timezone.utc)
    output.status = FanStatus.IDLE
    output.last_session_completed_at = completed_at
    output.last_session_revenue_cents = int(session.get("revenue_cents", 0) or 0)
    output.last_session_package_id = session.get("commercial_package_id")
    output.last_session_set_ids = list(session.get("set_ids") or [])
    output.last_session_experience = (
        output.desired_experience
        or session.get("scene_key")
        or "private session"
    )
    output.confirmed_budget_cents = None
    output.budget_source = None
    output.offered_packages = []
    output.selected_package_id = None
    output.selected_package_set_id = None
    output.selected_package_set_ids = []
    output.selected_package_label = None
    output.selected_package_price_cents = None

    obligation = None
    if policy.post_session_followup_enabled:
        execute_at = followup_at(completed_at, policy.post_session_followup_delay_hours)
        payload = post_session_payload(
            {**session, "scene_key": output.last_session_experience},
            buyer_stage=buyer_stage,
        )
        dedupe_key = f"post-session:{fan_id}:{completed_at.isoformat()}"
        output.next_followup_at = execute_at
        output.next_followup_type = "POST_SESSION_FOLLOWUP"
        output.next_followup_payload = payload
        output.next_followup_dedupe_key = dedupe_key
        obligation = FollowupObligation(
            action_type="POST_SESSION_FOLLOWUP",
            execute_at=execute_at,
            payload=payload,
            dedupe_key=dedupe_key,
        )
    else:
        output.next_followup_at = None
        output.next_followup_type = None
        output.next_followup_payload = {}
        output.next_followup_dedupe_key = None
    return output, obligation


def abandoned_ppv_payload(
    pending: dict[str, Any],
    *,
    desired_experience: str | None,
    selected_package_id: str | None,
) -> dict[str, Any]:
    return {
        "payment_reference": pending_reference(pending),
        "media_id": str(pending.get("media_id") or ""),
        "set_id": pending.get("set_id"),
        "price_cents": int(round(float(pending.get("price") or 0) * 100)),
        "sent_at": pending.get("sent_at"),
        "desired_experience": desired_experience or "",
        "selected_package_id": selected_package_id,
    }
