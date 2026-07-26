"""Durable callbacks for explicit, dated personal events stated by a fan.

This deliberately does not guess that ordinary conversation deserves a reminder.
A callback is created only when the current fan message contains both a supported
real-life event and an unambiguous future date expression. The scheduled worker
revalidates Full Auto eligibility, commercial state, frequency, and newer activity
before it sends anything.
"""
from __future__ import annotations

import asyncio
import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from core.supabase import get_supabase
from db.commercial_queries import (
    cancel_actions_for_fan,
    get_creator_policy,
    get_fan_state,
    schedule_action,
)
from models.commercial import CreatorPolicy, FanStatus
from services.auto_audience import resolve_auto_eligibility_for_fan
from services.followup_lifecycle import as_utc
from services.payday import resolve_payday


ACTION_TYPE = "PERSONAL_EVENT_CALLBACK"
WINDOW_DAYS = 30

_EVENT_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b(?:job\s+)?interview\b", re.I), "job interview"),
    (re.compile(r"\b(?:exam|finals?)\b", re.I), "exam"),
    (re.compile(r"\b(?:driving\s+)?test\b", re.I), "test"),
    (re.compile(r"\b(?:doctor(?:'s)?|dentist|medical)\s+(?:appointment|visit)\b", re.I), "appointment"),
    (re.compile(r"\b(?:appointment)\b", re.I), "appointment"),
    (re.compile(r"\b(?:surgery|operation|procedure)\b", re.I), "procedure"),
    (re.compile(r"\b(?:presentation|pitch)\b", re.I), "presentation"),
    (re.compile(r"\b(?:audition)\b", re.I), "audition"),
    (re.compile(r"\b(?:flight|fly(?:ing)?)\b", re.I), "flight"),
    (re.compile(r"\b(?:trip|vacation|holiday)\b", re.I), "trip"),
    (re.compile(r"\b(?:birthday)\b", re.I), "birthday"),
    (re.compile(r"\b(?:wedding|graduation)\b", re.I), "celebration"),
    (re.compile(r"\b(?:match|tournament|competition|race)\b", re.I), "competition"),
    (re.compile(r"\b(?:court\s+(?:date|hearing)|hearing)\b", re.I), "hearing"),
    (re.compile(r"\b(?:moving|move)\s+(?:house|home|apartments?)\b", re.I), "move"),
)
_FUTURE_RE = re.compile(
    r"\b(?:tomorrow|next\s+(?:week|monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"in\s+\d{1,2}\s+days?|(?:on\s+)?the\s+\d{1,2}(?:st|nd|rd|th)|today)\b",
    re.I,
)
_PAST_RE = re.compile(
    r"\b(?:yesterday|ago|last\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|week))\b",
    re.I,
)
_CANCEL_RE = re.compile(
    r"\b(?:cancelled|canceled|called\s+off|not\s+happening|postponed|rescheduled|moved)\b",
    re.I,
)
_PERSONAL_RE = re.compile(r"\b(?:i|i'm|im|i've|ive|my|we|we're|were)\b", re.I)
_PAYDAY_RE = re.compile(r"\b(?:payday|paycheck|salary|wages?|get\s+paid)\b", re.I)


@dataclass(frozen=True)
class PersonalEvent:
    summary: str
    evidence: str
    execute_at: datetime
    callback_window_start_at: datetime


@dataclass(frozen=True)
class PersonalCallbackCheck:
    ok: bool
    reason: str


def _zone(name: str | None):
    try:
        return ZoneInfo(name or "UTC")
    except ZoneInfoNotFoundError:
        return timezone.utc


def resolve_personal_event(
    fan_message: str,
    policy: CreatorPolicy,
    *,
    now: datetime | None = None,
) -> PersonalEvent | None:
    """Resolve a conservatively supported event without asking an LLM to guess."""
    text = " ".join(str(fan_message or "").split()).strip()
    if (
        not text
        or len(text) > 600
        or not _PERSONAL_RE.search(text)
        or not _FUTURE_RE.search(text)
        or _PAST_RE.search(text)
        or _PAYDAY_RE.search(text)
    ):
        return None

    summary = next(
        (label for pattern, label in _EVENT_PATTERNS if pattern.search(text)),
        "",
    )
    if not summary:
        return None

    zone = _zone(policy.timezone)
    current = as_utc(now) or datetime.now(timezone.utc)
    local_now = current.astimezone(zone)
    if re.search(r"\btoday\b", text, re.I):
        local_target = local_now.replace(
            hour=policy.personal_event_callback_send_hour_local,
            minute=0,
            second=0,
            microsecond=0,
        )
        if local_target <= local_now + timedelta(hours=1):
            return None
    else:
        local_target, confidence = resolve_payday(
            text,
            now=local_now,
            send_hour=policy.personal_event_callback_send_hour_local,
            timezone_name=policy.timezone,
        )
        if local_target is None or confidence < 0.7:
            return None

    execute_at = local_target.astimezone(timezone.utc)
    if execute_at <= current + timedelta(hours=1):
        return None
    if execute_at > current + timedelta(days=62):
        return None

    # If the conversation is active from lunchtime onward on the event day, the
    # normal reply path can ask naturally. The scheduled nudge then cancels rather
    # than duplicating that interaction in the evening.
    local_window_start = local_target.replace(hour=12, minute=0, second=0, microsecond=0)
    return PersonalEvent(
        summary=summary,
        evidence=text[:240],
        execute_at=execute_at,
        callback_window_start_at=local_window_start.astimezone(timezone.utc),
    )


async def _completed_count(fan_id: str, *, now: datetime) -> int:
    cutoff = now - timedelta(days=WINDOW_DAYS)

    def _count() -> int:
        response = (
            get_supabase().table("scheduled_actions")
            .select("id", count="exact", head=True)
            .eq("fan_id", fan_id)
            .eq("action_type", ACTION_TYPE)
            .eq("status", "COMPLETED")
            .gte("execute_at", cutoff.isoformat())
            .execute()
        )
        return int(response.count or 0)

    return await asyncio.to_thread(_count)


async def _messages_since(fan_id: str, since: datetime) -> list[dict[str, Any]]:
    def _load() -> list[dict[str, Any]]:
        return (
            get_supabase().table("messages")
            .select("id, role, content, sent_at")
            .eq("fan_id", fan_id)
            .gte("sent_at", since.isoformat())
            .order("sent_at")
            .limit(20)
            .execute()
        ).data or []

    return await asyncio.to_thread(_load)


async def schedule_personal_event_callback(
    *,
    creator_id: str,
    fan_id: str,
    fan_message: str,
    source_message_id: str | None,
    now: datetime | None = None,
) -> bool:
    """Create or replace the fan's pending explicit personal-event callback."""
    current = as_utc(now) or datetime.now(timezone.utc)
    policy = await get_creator_policy(creator_id)

    # A direct cancellation is honored even if the feature is currently off.
    if _CANCEL_RE.search(fan_message) and any(
        pattern.search(fan_message) for pattern, _label in _EVENT_PATTERNS
    ):
        await cancel_actions_for_fan(fan_id, ACTION_TYPE)

    if not policy.personal_event_callbacks_enabled:
        return False
    event = resolve_personal_event(fan_message, policy, now=current)
    if event is None:
        return False
    eligibility = await resolve_auto_eligibility_for_fan(creator_id, fan_id)
    if not eligibility.eligible:
        return False
    state = await get_fan_state(fan_id)
    if state.status != FanStatus.IDLE or state.next_followup_type:
        return False
    if await _completed_count(fan_id, now=current) >= policy.personal_event_callback_max_per_30_days:
        return False

    source = str(source_message_id or "").strip()
    if not source:
        material = f"{fan_id}|{event.evidence.casefold()}|{event.execute_at.isoformat()}"
        source = hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]
    payload = {
        "event_summary": event.summary,
        "evidence": event.evidence,
        "event_at": event.execute_at.isoformat(),
        "callback_window_start_at": event.callback_window_start_at.isoformat(),
        "source_message_id": source,
        "scheduled_at": current.isoformat(),
    }
    dedupe_key = f"personal-event:{fan_id}:{source}"
    await cancel_actions_for_fan(fan_id, ACTION_TYPE)
    await schedule_action(
        creator_id=creator_id,
        fan_id=fan_id,
        action_type=ACTION_TYPE,
        execute_at=event.execute_at,
        payload=payload,
        dedupe_key=dedupe_key,
    )
    print(
        f"[PERSONAL CALLBACK] scheduled fan={fan_id} event={event.summary} "
        f"at={event.execute_at.isoformat()} source={source}"
    )
    return True


async def validate_personal_event_action(
    action: dict[str, Any],
    state,
    policy: CreatorPolicy,
    *,
    now: datetime | None = None,
) -> PersonalCallbackCheck:
    current = as_utc(now) or datetime.now(timezone.utc)
    if not policy.personal_event_callbacks_enabled:
        return PersonalCallbackCheck(False, "personal-event callbacks disabled for creator")
    if state.status != FanStatus.IDLE or state.next_followup_type:
        return PersonalCallbackCheck(
            False,
            f"a commercial flow or higher-priority follow-up is active (status={state.status.value})",
        )
    eligibility = await resolve_auto_eligibility_for_fan(
        str(action["creator_id"]),
        str(action["fan_id"]),
    )
    if not eligibility.eligible:
        return PersonalCallbackCheck(False, f"not eligible for Full Auto ({eligibility.reason})")
    if await _completed_count(str(action["fan_id"]), now=current) >= (
        policy.personal_event_callback_max_per_30_days
    ):
        return PersonalCallbackCheck(False, "30-day personal callback limit reached")

    payload = action.get("payload") or {}
    window_start = as_utc(payload.get("callback_window_start_at"))
    event_at = as_utc(payload.get("event_at"))
    if window_start is None or event_at is None:
        return PersonalCallbackCheck(False, "personal event timing snapshot is incomplete")
    if current > event_at + timedelta(days=2):
        return PersonalCallbackCheck(False, "personal event callback is stale")
    newer = await _messages_since(str(action["fan_id"]), window_start)
    if newer:
        return PersonalCallbackCheck(
            False,
            "conversation was active during the event callback window",
        )
    return PersonalCallbackCheck(True, "explicit event remains eligible for a callback")
