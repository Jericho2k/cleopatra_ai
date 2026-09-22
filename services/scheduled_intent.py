"""Generic durable conversational obligations, created by Core v1 when due.

Cleopatra already kept several SPECIFIC promises — payday re-engagement, the
post-session follow-up, the abandoned offer and PPV chases, inactivity. What it
could not keep was a promise the conversation itself made:

    creator: "wait right there 😏"

Nothing durable existed for that, so the beat was simply dropped once the reply
was sent. This module is the general case, built on the same
``scheduled_actions`` infrastructure rather than a second scheduler.

Two rules make it safe to give a model any say in it at all:

1. **Nothing fan-facing is frozen.** What is persisted is a semantic goal. When
   the action comes due it re-enters the ordinary path — current conversation,
   GLM revalidates, Kimi writes NOW — so the fan never receives wording composed
   against a conversation that has since moved on.

2. **Application code owns the clock.** A model may ask for a relative delay
   inside deterministic bounds, or point at a time the application already
   parsed from evidence (payday). It may never state an absolute time.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from typing import Any

from db.commercial_queries import cancel_action_by_dedupe_key, schedule_action
from models.conversation_decision import (
    CANCEL_ON_ACTIVITY,
    REVALIDATE_ON_ACTIVITY,
    ScheduledIntent,
)

INTENT_ACTION = "CONVERSATIONAL_INTENT"

#: A short continuation is a beat in a live conversation; making it wait for
#: "morning" would be absurd. Anything further out than this respects the
#: creator's configured sleep hours like every other proactive message does.
SLEEP_HOURS_APPLY_ABOVE_SECONDS = 60 * 60

#: Small jitter so a whole deployment's "two minutes later" beats do not land on
#: the same second, and so the cadence does not read as a timer.
JITTER_FRACTION = 0.15

#: Exactly what may be persisted. An allowlist rather than a filter, so a future
#: field cannot smuggle prewritten copy into a durable payload.
PAYLOAD_KEYS = (
    "kind",
    "goal",
    "activity_policy",
    "source_ids",
    "created_generation",
    "created_at",
    "timing_kind",
    "requested_delay_seconds",
    "timing_reference",
)


class IntentRejected(RuntimeError):
    """The requested intention could not be normalized against evidence."""


def dedupe_key(fan_id: str, kind: str) -> str:
    """One live obligation per kind per fan, so a newer beat replaces an older."""
    return f"conv-intent:{fan_id}:{kind}"


def _jittered(seconds: float, rng: random.Random) -> float:
    spread = max(1.0, abs(seconds) * JITTER_FRACTION)
    return max(5.0, seconds + rng.uniform(-spread, spread))


def resolve_execute_at(
    intent: ScheduledIntent,
    *,
    now: datetime,
    payday_at: datetime | None = None,
    pending_offer_expires_at: datetime | None = None,
    rng: random.Random | None = None,
) -> datetime:
    """Turn a REQUESTED timing into the application's own normalized time."""
    generator = rng or random.Random()
    if intent.timing_kind == "relative":
        return now + timedelta(
            seconds=_jittered(float(intent.relative_seconds), generator)
        )
    if intent.timing_kind == "reference":
        if intent.reference == "payday":
            if payday_at is None:
                raise IntentRejected("no evidenced payday to schedule against")
            anchor = payday_at
        elif intent.reference == "pending_offer_expiry":
            if pending_offer_expires_at is None:
                raise IntentRejected("no pending offer expiry to schedule against")
            anchor = pending_offer_expires_at
        else:  # pragma: no cover - the contract already narrowed this
            raise IntentRejected(f"unknown time reference {intent.reference!r}")
        if anchor.tzinfo is None:
            anchor = anchor.replace(tzinfo=timezone.utc)
        anchor = anchor.astimezone(timezone.utc)
        # An evidenced time already in the past is a real obligation that was
        # missed, not a reason to drop it; run it shortly rather than never.
        return max(anchor, now + timedelta(minutes=1))
    raise IntentRejected("intention has no usable timing")


async def _apply_sleep_hours(
    creator_id: str, execute_at: datetime, *, now: datetime
) -> datetime:
    if (execute_at - now).total_seconds() <= SLEEP_HOURS_APPLY_ABOVE_SECONDS:
        return execute_at
    try:
        from db.commercial_queries import get_creator_policy
        from db.queries import get_creator_sleep_hours
        from services.followup_lifecycle import next_awake_time

        sleep_start, sleep_end = await get_creator_sleep_hours(creator_id)
        policy = await get_creator_policy(creator_id)
        return next_awake_time(
            execute_at,
            sleep_start_hour=sleep_start,
            sleep_end_hour=sleep_end,
            timezone_name=getattr(policy, "timezone", "UTC") or "UTC",
        )
    except Exception as exc:  # noqa: BLE001 - never lose an obligation to this
        print(f"[SCHEDULED INTENT] sleep-hour check failed creator={creator_id}: {exc}")
        return execute_at


def build_payload(
    intent: ScheduledIntent,
    *,
    conversation_generation: int,
    now: datetime,
) -> dict[str, Any]:
    """The durable record. Semantic only — there is deliberately no copy here."""
    return {
        "kind": intent.kind,
        "goal": intent.goal,
        "activity_policy": intent.activity_policy,
        "source_ids": list(intent.source_ids)[:8],
        "created_generation": int(conversation_generation),
        "created_at": now.isoformat(),
        "timing_kind": intent.timing_kind,
        "requested_delay_seconds": int(intent.relative_seconds),
        "timing_reference": intent.reference,
    }


async def persist_scheduled_intent(
    *,
    creator_id: str,
    fan_id: str,
    intent: ScheduledIntent,
    conversation_generation: int,
    payday_at: datetime | None = None,
    pending_offer_expires_at: datetime | None = None,
    now: datetime | None = None,
    rng: random.Random | None = None,
) -> dict[str, Any] | None:
    """Validate, normalize and durably record one future conversational beat."""
    if not intent.requested:
        return None
    moment = now or datetime.now(timezone.utc)
    try:
        execute_at = resolve_execute_at(
            intent,
            now=moment,
            payday_at=payday_at,
            pending_offer_expires_at=pending_offer_expires_at,
            rng=rng,
        )
    except IntentRejected as exc:
        print(
            f"[SCHEDULED INTENT REJECTED] fan={fan_id} kind={intent.kind}: {exc}"
        )
        return None

    execute_at = await _apply_sleep_hours(creator_id, execute_at, now=moment)
    payload = build_payload(
        intent, conversation_generation=conversation_generation, now=moment
    )
    key = dedupe_key(fan_id, intent.kind)
    await schedule_action(
        creator_id=creator_id,
        fan_id=fan_id,
        action_type=INTENT_ACTION,
        execute_at=execute_at,
        payload=payload,
        dedupe_key=key,
        replace_existing=True,
    )
    print(
        f"[SCHEDULED INTENT] fan={fan_id} kind={intent.kind} "
        f"policy={intent.activity_policy} generation={conversation_generation} "
        f"execute_at={execute_at.isoformat()}"
    )
    return {
        "action_type": INTENT_ACTION,
        "dedupe_key": key,
        "execute_at": execute_at.isoformat(),
        **payload,
    }


async def cancel_scheduled_intent(fan_id: str, kind: str) -> None:
    await cancel_action_by_dedupe_key(dedupe_key(fan_id, kind))


def goal_for_due_intent(payload: dict[str, Any]) -> str:
    """What the due turn is being asked to accomplish, in semantic terms only.

    This is handed to the same proactive path a payday follow-up uses, which
    re-enters GLM and then Kimi. Nothing here is fan-facing text, and the
    wording the fan eventually sees is written at this moment, not when the
    intention was created.
    """
    goal = str(payload.get("goal") or "").strip()
    kind = str(payload.get("kind") or "conversational intention")
    return (
        f"A conversational intention you recorded earlier ({kind}) has come due. "
        f"Its goal was: {goal}. "
        "Read the CURRENT conversation first. If it has moved on, been resolved, "
        "or would now be intrusive, say nothing and hold instead. If it still "
        "makes sense, continue the conversation naturally from where it actually "
        "is. Do not refer to waiting, scheduling, timers or automation, do not "
        "state a price, and do not send media."
    )


def activity_cancels(payload: dict[str, Any]) -> bool:
    policy = str(payload.get("activity_policy") or CANCEL_ON_ACTIVITY)
    return policy != REVALIDATE_ON_ACTIVITY
