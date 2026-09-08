"""Scheduled-actions worker.

Runs the future promises the commercial layer makes — chiefly the payday
re-engagement: a fan who told us he'd have money on Friday is the most qualified
lead in the system, and until now we simply dropped him.

The important part is not the sending, it's the REVALIDATION. Between scheduling
on Monday and firing on Friday, the world changes: he may have paid, been frozen
for review, gone cold, or auto mode may be off. Sending blindly is how you get an
embarrassing message in front of an agency. Every check below is a reason to skip.
"""
import asyncio
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from core.action_telemetry import ActionTimings, action_scope, emit
from db.commercial_queries import (
    action_needs_repair,
    bulk_upsert_pending_actions,
    claim_due_actions,
    complete_action,
    fail_action,
    get_action_states_by_dedupe_key,
    get_creator_policy,
    get_fan_state,
    get_followup_obligations,
    reschedule_action,
    save_fan_state,
    schedule_action,
)
from models.commercial import FanStatus

# --- Capacity configuration ------------------------------------------------
#
# Three knobs, each with a default that is safe to deploy untouched.
#
# CONCURRENCY is how many *independent fans* may be in flight at once. It is
# deliberately not larger than the model gate: an action spends most of its wall
# clock either inside a model call or inside a deliberate human-like delay, so
# more action slots than model slots just moves the queue from the worker to the
# gate without improving drain time.
#
# CLAIM_LIMIT is sized at 3x concurrency so the pool always has work queued
# behind its slots, while keeping the number of rows locked in one process small
# enough that the 10-minute stale-reclaim window is never in danger.
DEFAULT_CONCURRENCY = 8
DEFAULT_CLAIM_LIMIT = 24
DEFAULT_POLL_SECONDS = 5

# A full claim means backlog probably remains, so the next cycle starts almost
# immediately. The small floor keeps an unproductive claim (for example a stale
# PROCESSING row that fails instantly) from becoming a busy loop.
BUSY_POLL_SECONDS = 0.25
MAX_CONSECUTIVE_BUSY_CYCLES = 60

# Obligation repair is a safety net, not delivery work, so it runs on its own
# slower cadence rather than once per (now much faster) claim poll.
REPAIR_INTERVAL_SECONDS = 60

# How far ahead the repair pass looks. The durable action only has to exist
# before its execute time, and the idle poll is 5s with a 60s fallback, so five
# minutes leaves a wide margin for clock skew and a stalled cycle while removing
# the every-obligation-every-minute scan.
REPAIR_HORIZON_SECONDS = 300

# Actions whose whole purpose is bookkeeping or ingestion rather than sending a
# proactive message; the proactive revalidation gate does not apply to them.
NON_PROACTIVE_ACTIONS = {"PPV_RECONCILE", "OFFER_EXPIRY", "PROCESS_INBOUND_MESSAGE"}


def _env_int(name: str, default: int, *, low: int, high: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return max(low, min(int(raw), high))
    except ValueError:
        return default


def action_concurrency() -> int:
    return _env_int("SCHEDULED_ACTION_CONCURRENCY", DEFAULT_CONCURRENCY, low=1, high=128)


def claim_limit() -> int:
    return _env_int("SCHEDULED_ACTION_CLAIM_LIMIT", DEFAULT_CLAIM_LIMIT, low=1, high=500)


def poll_seconds() -> float:
    return float(_env_int("SCHEDULED_ACTION_POLL_SECONDS", DEFAULT_POLL_SECONDS, low=1, high=300))


POLL_SECONDS = DEFAULT_POLL_SECONDS
LAST_RUN_STARTED_AT: datetime | None = None
LAST_RUN_COMPLETED_AT: datetime | None = None
LAST_RUN_ERROR: str | None = None
LAST_RUN_SENT = 0
LAST_RUN_CLAIMED = 0
LAST_RUN_DURATION_MS = 0
LAST_RUN_BATCH_FULL = False
LAST_RUN_MAX_CONCURRENCY = 0
LAST_REPAIR_AT: datetime | None = None
LAST_REPAIR_COUNT = 0
CYCLES_COMPLETED = 0

# New work created inside this process (an inbound webhook, for example) sets
# this so the loop stops waiting out its idle poll instead of sitting on a
# message that is already durable.
_wakeup: asyncio.Event | None = None


def notify_work_available() -> None:
    """Wake the worker loop early because durable work was just enqueued."""
    event = _wakeup
    if event is not None:
        try:
            event.set()
        except RuntimeError:  # pragma: no cover - loop already closed
            pass


@dataclass
class CycleResult:
    """What one worker cycle did, for the loop and the health surface."""

    sent: int = 0
    claimed: int = 0
    processed: int = 0
    errors: int = 0
    batch_full: bool = False
    repaired: int = 0
    duration_ms: int = 0
    max_concurrency: int = 0
    completions: list[dict] = field(default_factory=list)


def worker_health_snapshot() -> dict:
    now = datetime.now(timezone.utc)
    completed = LAST_RUN_COMPLETED_AT
    return {
        "last_run_started_at": (
            LAST_RUN_STARTED_AT.isoformat() if LAST_RUN_STARTED_AT else None
        ),
        "last_run_completed_at": completed.isoformat() if completed else None,
        "seconds_since_last_cycle": (
            round((now - completed).total_seconds(), 1) if completed else None
        ),
        "last_error": LAST_RUN_ERROR,
        "last_sent": LAST_RUN_SENT,
        "last_claimed": LAST_RUN_CLAIMED,
        "last_cycle_duration_ms": LAST_RUN_DURATION_MS,
        "last_batch_full": LAST_RUN_BATCH_FULL,
        "last_max_action_concurrency": LAST_RUN_MAX_CONCURRENCY,
        "cycles_completed": CYCLES_COMPLETED,
        "last_repair_at": LAST_REPAIR_AT.isoformat() if LAST_REPAIR_AT else None,
        "last_repair_count": LAST_REPAIR_COUNT,
        "poll_seconds": poll_seconds(),
        "action_concurrency_limit": action_concurrency(),
        "claim_limit": claim_limit(),
    }


@dataclass(frozen=True)
class ActionCheck:
    ok: bool
    reason: str = ""
    retry_at: datetime | None = None


@dataclass(frozen=True)
class HandlerResult:
    sent_message: bool = False
    retry_at: datetime | None = None
    reason: str = ""


def _parse_time(value) -> datetime | None:
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


def _same_time(first, second) -> bool:
    left = _parse_time(first)
    right = _parse_time(second)
    return bool(left and right and abs((left - right).total_seconds()) < 2)


async def _creator_auto_mode_default(creator_id: str) -> bool:
    """Read creator auto mode across mixed-version/rolling deployments.

    The helper originally lived only in ``db.queries``. A worker can briefly run
    against an older imported module during a deployment, so a missing symbol must
    not permanently fail a promised follow-up.
    """
    try:
        from db import queries

        getter = getattr(queries, "get_creator_auto_mode_default", None)
        if getter is not None:
            return bool(await getter(creator_id))
    except (ImportError, AttributeError):
        pass

    from core.supabase import get_supabase

    def _get() -> bool:
        response = (
            get_supabase().table("creators")
            .select("auto_mode")
            .eq("id", creator_id)
            .single()
            .execute()
        )
        return bool((response.data or {}).get("auto_mode", False))

    return await asyncio.to_thread(_get)


async def _should_still_send(action: dict) -> ActionCheck:
    """Revalidate a proactive message immediately before delivery."""
    from db.queries import (
        get_conversation_history,
        get_creator_sleep_hours,
        get_fan_by_id,
    )
    from services.followup_lifecycle import next_awake_time

    fan_id = action["fan_id"]
    creator_id = action["creator_id"]
    action_type = str(action.get("action_type") or "")
    payload = action.get("payload") or {}

    fan = await get_fan_by_id(fan_id)
    if not fan:
        return ActionCheck(False, "fan gone")

    # Frozen for a human (crisis / whale handoff): never auto-message.
    if getattr(fan, "needs_human_review", False):
        return ActionCheck(False, "fan frozen for human review")

    if action_type != "POST_PURCHASE_REACTION":
        # Auto mode must still be on for this fan.
        fan_auto = getattr(fan, "auto_mode", None)
        if fan_auto is None:
            try:
                fan_auto = await _creator_auto_mode_default(creator_id)
            except Exception:
                fan_auto = False
        if not fan_auto:
            return ActionCheck(False, "auto mode off")

    if action_type == "AUTO_REPLY":
        from main import _creator_auto_availability

        availability = await _creator_auto_availability(creator_id)
        if not availability.get("auto_available"):
            return ActionCheck(False, "no approved sets remain for Auto mode")
        trigger_at = _parse_time(payload.get("trigger_sent_at"))
        if not trigger_at:
            return ActionCheck(False, "Auto reply trigger timestamp is missing")
        history = await get_conversation_history(fan_id, limit=10)
        if any(
            (message_at := _parse_time(getattr(message, "sent_at", None)))
            and message_at > trigger_at
            for message in history
        ):
            return ActionCheck(False, "newer conversation activity replaced Auto reply")

        sleep_start, sleep_end = await get_creator_sleep_hours(creator_id)
        now = datetime.now(timezone.utc)
        awake_at = next_awake_time(
            now,
            sleep_start_hour=sleep_start,
            sleep_end_hour=sleep_end,
            timezone_name="UTC",
        )
        if awake_at > now:
            return ActionCheck(False, "creator sleep hours are active", retry_at=awake_at)
        return ActionCheck(True)

    if action_type == "POST_PURCHASE_REACTION":
        purchase_at = _parse_time(payload.get("purchase_at"))
        if not purchase_at:
            return ActionCheck(False, "purchase reaction timestamp is missing")
        history = await get_conversation_history(fan_id, limit=10)
        if any(
            (message_at := _parse_time(getattr(message, "sent_at", None)))
            and message_at > purchase_at
            for message in history
        ):
            return ActionCheck(False, "conversation continued after purchase")
        return ActionCheck(True)

    state = await get_fan_state(fan_id)
    policy = await get_creator_policy(creator_id)

    if state.next_followup_type != action_type:
        return ActionCheck(False, "follow-up obligation was cancelled or replaced")
    expected_dedupe = str(state.next_followup_dedupe_key or "")
    if expected_dedupe and expected_dedupe != str(action.get("dedupe_key") or ""):
        return ActionCheck(False, "follow-up dedupe key was replaced")

    if action_type == "PAYDAY_REENGAGEMENT":
        if not policy.payday_reengagement_enabled:
            return ActionCheck(False, "payday re-engagement disabled for creator")
        if state.status not in (FanStatus.PAUSED_UNTIL_PAYDAY, FanStatus.PAUSED_NO_BUDGET):
            return ActionCheck(False, f"no longer paused (status={state.status.value})")
        if payload.get("payday_at") and not _same_time(payload["payday_at"], state.payday_at):
            return ActionCheck(False, "a newer payday replaced this one")
    elif action_type == "POST_SESSION_FOLLOWUP":
        if not policy.post_session_followup_enabled:
            return ActionCheck(False, "post-session follow-up disabled for creator")
        if not _same_time(payload.get("session_completed_at"), state.last_session_completed_at):
            return ActionCheck(False, "session completion snapshot is no longer current")
        if state.status != FanStatus.IDLE:
            return ActionCheck(False, f"fan entered a new flow (status={state.status.value})")
    elif action_type == "ABANDONED_PPV_FOLLOWUP":
        if not policy.abandoned_ppv_followup_enabled:
            return ActionCheck(False, "abandoned-PPV follow-up disabled for creator")
        if str(payload.get("media_id") or "") != str(state.last_abandoned_media_id or ""):
            return ActionCheck(False, "a newer PPV outcome replaced this one")
        if state.status in {FanStatus.PAYMENT_PENDING, FanStatus.PAID_SESSION_ACTIVE}:
            return ActionCheck(False, f"fan entered a new paid flow (status={state.status.value})")
    elif action_type == "ABANDONED_OFFER_FOLLOWUP":
        if not policy.abandoned_offer_followup_enabled:
            return ActionCheck(False, "abandoned-offer follow-up disabled for creator")
        if str(payload.get("offered_at") or "") != (
            state.last_offer_at.isoformat() if state.last_offer_at else ""
        ):
            return ActionCheck(False, "a newer offer replaced this one")
        if state.status != FanStatus.IDLE:
            return ActionCheck(False, f"fan entered a new flow (status={state.status.value})")
    elif action_type == "INACTIVITY_REENGAGEMENT":
        from services.inactivity_reengagement import validate_inactivity_action

        inactivity = await validate_inactivity_action(action, state, policy)
        if not inactivity.ok:
            return ActionCheck(False, inactivity.reason)
    else:
        return ActionCheck(False, f"unsupported proactive action {action_type}")

    # Recent fan activity postpones the action instead of barging in or silently
    # dropping a still-valid commercial obligation.
    try:
        history = await get_conversation_history(fan_id, limit=5)
        fan_messages = [message for message in history if getattr(message, "role", None) == "fan"]
        if fan_messages:
            ts = _parse_time(getattr(fan_messages[-1], "sent_at", None))
            if ts:
                suppress_hours = policy.followup_recent_activity_suppression_hours
                retry_at = ts + timedelta(hours=max(0, suppress_hours))
                if retry_at > datetime.now(timezone.utc):
                    return ActionCheck(
                        False,
                        f"fan active inside {suppress_hours}h suppression window",
                        retry_at=retry_at,
                    )
    except Exception:
        pass  # never block a send purely because history lookup failed

    sleep_start, sleep_end = await get_creator_sleep_hours(creator_id)
    now = datetime.now(timezone.utc)
    awake_at = next_awake_time(
        now,
        sleep_start_hour=sleep_start,
        sleep_end_hour=sleep_end,
        timezone_name=policy.timezone,
    )
    if awake_at > now:
        return ActionCheck(
            False,
            "creator sleep hours are active",
            retry_at=awake_at,
        )

    return ActionCheck(True)


async def _send_goal(action: dict, goal: str) -> HandlerResult:
    from services.proactive import send_proactive_message

    sent = await send_proactive_message(
        creator_id=action["creator_id"],
        fan_id=action["fan_id"],
        goal=goal,
        action_id=str(action["id"]),
        action_payload=action.get("payload") or {},
    )
    if not sent:
        raise RuntimeError("proactive delivery was not confirmed")
    return HandlerResult(sent_message=True)


async def _run_payday_reengagement(action: dict) -> HandlerResult:
    """Generate and send a contextual, creator-voice payday follow-up.

    Note the framing: we do NOT assert he definitely has money ("you got paid, let's
    spend it"). We reopen the door he left open.
    """
    payload = action.get("payload") or {}
    desired = payload.get("desired_experience") or "what we were talking about"

    goal = (
        f"It's the day he said his money would come in. Reopen the conversation warmly "
        f"and playfully — reference that he wanted {desired} and that you said it'd "
        f"still be here for him. Do NOT assume he definitely has money, do NOT pressure, "
        f"do NOT state a price, do NOT send media. One short message, his energy, "
        f"leaving him an easy yes."
    )

    return await _send_goal(action, goal)


async def _run_post_session_followup(action: dict) -> HandlerResult:
    payload = action.get("payload") or {}
    experience = payload.get("experience") or "the private session"
    buyer_stage = str(payload.get("buyer_stage") or "UNKNOWN")
    relationship_note = {
        "FIRST_TIME_BUYER": "Treat this as his first purchase and make him feel remembered, not sold to.",
        "REPEAT_BUYER": "He is a repeat buyer; use comfortable continuity without forcing another offer.",
        "VIP": "He is a valued regular; be warm and personal, with no generic sales language.",
    }.get(buyer_stage, "Be warm and personal, with no immediate sales pitch.")
    goal = (
        f"Follow up after the completed {experience} experience. {relationship_note} "
        "Ask one light, natural question or make one simple callback. Do NOT mention "
        "automation, scheduling, a new price, or send media. One short message."
    )
    return await _send_goal(action, goal)


async def _run_abandoned_ppv_followup(action: dict) -> HandlerResult:
    payload = action.get("payload") or {}
    desired = payload.get("desired_experience") or "what you picked"
    goal = (
        f"He selected {desired}, received the locked option, but never unlocked it. "
        "Reopen the conversation lightly without accusing him, claiming he saw it, "
        "repeating the price, discounting it, or sending media. Make it easy for him "
        "to continue or just chat. One short message."
    )
    return await _send_goal(action, goal)


async def _run_abandoned_offer_followup(action: dict) -> HandlerResult:
    payload = action.get("payload") or {}
    approved = payload.get("primary_experience") or "the private options"
    goal = (
        f"He was shown approved options around {approved}, but left before choosing one. "
        "Reopen the conversation lightly and naturally. You may reference only that approved "
        "experience, without claiming he selected it, repeating a price, discounting, pressuring, "
        "or sending media. Make it easy to resume or just chat. One short message."
    )
    return await _send_goal(action, goal)


async def _run_inactivity_reengagement(action: dict) -> HandlerResult:
    goal = (
        "The fan has been quiet after an ordinary conversation and is still eligible for Full Auto. "
        "Reopen naturally using one small callback from the recent conversation when possible. "
        "Do not mention that he disappeared, guilt him, sell, quote a price, promise content, or send "
        "media. Sound casual rather than witty or campaign-like. One short message."
    )
    return await _send_goal(action, goal)


async def _run_auto_reply(action: dict) -> HandlerResult:
    from services.suggestions import deliver_scheduled_auto_reply

    sent = await deliver_scheduled_auto_reply(action)
    if not sent:
        from core.supabase import get_supabase

        pending_approval = await asyncio.to_thread(
            lambda: get_supabase().table("ppv_approval_requests")
            .select("id")
            .eq("fan_id", action["fan_id"])
            .eq("status", "pending")
            .limit(1)
            .execute()
        )
        if pending_approval.data:
            return HandlerResult(
                sent_message=False,
                reason="durable Auto reply prepared a PPV approval",
            )
        raise RuntimeError("Auto reply completed without a confirmed message")
    return HandlerResult(sent_message=sent, reason="durable Auto reply processed")


async def _run_post_purchase_reaction(action: dict) -> HandlerResult:
    return await _send_goal(
        action,
        "A purchase was just confirmed; send the already prepared short reaction.",
    )


async def _run_offer_expiry(action: dict) -> HandlerResult:
    """Turn the exact still-pending offer into a later follow-up obligation."""
    from services.followup_lifecycle import expire_pending_offer_state

    state = await get_fan_state(action["fan_id"])
    policy = await get_creator_policy(action["creator_id"])
    expired, followup, changed = expire_pending_offer_state(
        state,
        payload=action.get("payload") or {},
        policy=policy,
        fan_id=action["fan_id"],
        now=datetime.now(timezone.utc),
    )
    if not changed:
        return HandlerResult(reason="offer already changed or returned")

    # Persist the obligation first. If scheduling fails, the repair pass recreates
    # it from fan state on the next worker tick.
    await save_fan_state(action["fan_id"], action["creator_id"], expired)
    if followup:
        try:
            await schedule_action(
                creator_id=action["creator_id"],
                fan_id=action["fan_id"],
                action_type=followup.action_type,
                execute_at=followup.execute_at,
                payload=followup.payload,
                dedupe_key=followup.dedupe_key,
            )
        except Exception as exc:
            print(
                f"[FOLLOWUP REPAIR NEEDED] fan={action['fan_id']} "
                f"type=ABANDONED_OFFER_FOLLOWUP error={exc}"
            )
    return HandlerResult(reason="pending offer expired")


async def _run_process_inbound_message(action: dict) -> HandlerResult:
    """Run the inbound-message pipeline that used to live inside the webhook.

    The webhook's job now ends once the platform message is durably persisted and
    this obligation exists. Everything expensive — media enrichment against API
    Fansly, the analyzer, the writer, Auto scheduling — happens here, where a
    failure is retried with backoff instead of turning into a webhook timeout and
    a platform redelivery.
    """
    from main import run_durable_inbound_message

    return await run_durable_inbound_message(action)


async def _run_ppv_reconcile(action: dict) -> HandlerResult:
    from services.ppv_reconciliation import (
        PPVReconcileDisposition,
        reconcile_pending_ppv,
    )

    payload = action.get("payload") or {}
    result = await reconcile_pending_ppv(
        creator_id=action["creator_id"],
        fan_id=action["fan_id"],
        expected_reference=payload.get("payment_reference"),
    )
    if result.disposition == PPVReconcileDisposition.PENDING:
        return HandlerResult(retry_at=result.retry_at, reason=result.reason)
    return HandlerResult(reason=result.reason)


HANDLERS = {
    "AUTO_REPLY": _run_auto_reply,
    "POST_PURCHASE_REACTION": _run_post_purchase_reaction,
    "PAYDAY_REENGAGEMENT": _run_payday_reengagement,
    "POST_SESSION_FOLLOWUP": _run_post_session_followup,
    "ABANDONED_PPV_FOLLOWUP": _run_abandoned_ppv_followup,
    "OFFER_EXPIRY": _run_offer_expiry,
    "ABANDONED_OFFER_FOLLOWUP": _run_abandoned_offer_followup,
    "INACTIVITY_REENGAGEMENT": _run_inactivity_reengagement,
    "PPV_RECONCILE": _run_ppv_reconcile,
    "PROCESS_INBOUND_MESSAGE": _run_process_inbound_message,
}


async def _record_message_action_resolution(action: dict, *, sent: bool) -> None:
    action_type = str(action.get("action_type") or "")
    if action_type in {
        "AUTO_REPLY",
        "POST_PURCHASE_REACTION",
        "PPV_RECONCILE",
        "PROCESS_INBOUND_MESSAGE",
    }:
        return
    state = await get_fan_state(action["fan_id"])
    current_dedupe = str(state.next_followup_dedupe_key or "")
    action_dedupe = str(action.get("dedupe_key") or "")
    if (
        state.next_followup_type == action_type
        and (not current_dedupe or current_dedupe == action_dedupe)
    ):
        state.next_followup_at = None
        state.next_followup_type = None
        state.next_followup_payload = {}
        state.next_followup_dedupe_key = None
    if sent:
        sent_at = datetime.now(timezone.utc)
        state.last_followup_at = sent_at
        if action_type == "INACTIVITY_REENGAGEMENT":
            from services.inactivity_reengagement import record_inactivity_sent

            record_inactivity_sent(state, now=sent_at)
    await save_fan_state(action["fan_id"], action["creator_id"], state)


async def _record_followup_postponed(action: dict, retry_at: datetime) -> None:
    state = await get_fan_state(action["fan_id"])
    current_dedupe = str(state.next_followup_dedupe_key or "")
    action_dedupe = str(action.get("dedupe_key") or "")
    if (
        state.next_followup_type == action.get("action_type")
        and (not current_dedupe or current_dedupe == action_dedupe)
    ):
        state.next_followup_at = retry_at
        await save_fan_state(action["fan_id"], action["creator_id"], state)


async def repair_followup_obligations(
    *,
    horizon_seconds: int = REPAIR_HORIZON_SECONDS,
) -> int:
    """Recreate missing durable actions from the fan-state obligation record.

    This is a safety net for the case where commercial state says a follow-up is
    owed but its durable ``scheduled_actions`` row is missing or terminal. The
    invariant is unchanged; only the cost is.

    Previously this scanned every outstanding obligation — including ones due
    next week — and spent a SELECT plus a conditional write per row, so 200
    obligations cost 401 round trips on every cycle. Now it reads only the
    obligations that could fire within the horizon, resolves their existing
    action states in one bounded query per 200 keys, decides in memory, and
    writes the repairs in one bulk upsert. Three round trips covers a typical
    pass regardless of how many obligations exist.
    """
    horizon = datetime.now(timezone.utc) + timedelta(seconds=max(0, horizon_seconds))
    obligations = await get_followup_obligations(due_before=horizon)

    candidates: list[dict] = []
    for row in obligations:
        execute_at = _parse_time(row.get("next_followup_at"))
        action_type = str(row.get("next_followup_type") or "")
        dedupe_key = str(row.get("next_followup_dedupe_key") or "")
        if not execute_at or not action_type or not dedupe_key:
            continue
        candidates.append(
            {
                "creator_id": str(row["creator_id"]),
                "fan_id": str(row["fan_id"]),
                "action_type": action_type,
                "execute_at": execute_at.isoformat(),
                "payload": row.get("next_followup_payload") or {},
                "dedupe_key": dedupe_key,
                "status": "PENDING",
                "attempts": 0,
                "locked_at": None,
                "last_error": None,
            }
        )
    if not candidates:
        return 0

    existing = await get_action_states_by_dedupe_key(
        [row["dedupe_key"] for row in candidates]
    )
    repairs = [
        row for row in candidates
        if action_needs_repair(existing.get(row["dedupe_key"]))
    ]
    if not repairs:
        return 0

    await bulk_upsert_pending_actions(repairs)
    print(f"[SCHEDULED] repaired {len(repairs)} follow-up obligation(s)")
    return len(repairs)


async def _resolve_action(action: dict, *, sent_counter: list[int]) -> str:
    """Run one claimed action to a terminal state. Never raises.

    Extracted from the old inline loop body unchanged in behaviour: the same
    revalidation gate, the same reschedule/complete/fail transitions, and the
    same "freeze rather than duplicate" handling when the external send
    succeeded but local persistence did not.
    """
    aid = action["id"]
    action_type = str(action.get("action_type") or "")
    timings = ActionTimings(
        action_id=str(aid),
        action_type=action_type,
        fan_id=str(action.get("fan_id") or ""),
        creator_id=str(action.get("creator_id") or ""),
        queue_wait_ms=_queue_wait_ms(action),
    )
    with action_scope(timings):
        try:
            handler = HANDLERS.get(action_type)
            if not handler:
                await fail_action(
                    aid,
                    f"no handler for {action_type}",
                    action.get("attempts", 0),
                )
                timings.outcome = "no_handler"
                return "no_handler"

            if action_type not in NON_PROACTIVE_ACTIONS:
                started = time.perf_counter()
                check = await _should_still_send(action)
                timings.add("revalidation_ms", (time.perf_counter() - started) * 1000)
                if not check.ok:
                    if check.retry_at:
                        await _record_followup_postponed(action, check.retry_at)
                        await reschedule_action(aid, check.retry_at)
                        print(
                            f"[SCHEDULED] postponed {action_type} "
                            f"fan={action['fan_id']} until={check.retry_at.isoformat()}: {check.reason}"
                        )
                        timings.outcome = "postponed"
                        return "postponed"
                    await complete_action(aid)
                    await _record_message_action_resolution(action, sent=False)
                    print(
                        f"[SCHEDULED] skip {action_type} "
                        f"fan={action['fan_id']}: {check.reason}"
                    )
                    timings.outcome = "skipped"
                    return "skipped"

            result = await handler(action)
            if result.retry_at:
                await reschedule_action(aid, result.retry_at)
                print(
                    f"[SCHEDULED] retry {action_type} fan={action['fan_id']} "
                    f"at={result.retry_at.isoformat()}: {result.reason}"
                )
                timings.outcome = "retry"
                return "retry"
            if result.sent_message:
                try:
                    started = time.perf_counter()
                    await _record_message_action_resolution(action, sent=True)
                    timings.add("persistence_ms", (time.perf_counter() - started) * 1000)
                except Exception as exc:
                    # The external send already happened. Never turn a local
                    # persistence failure into a duplicate proactive message.
                    from db.queries import freeze_fan_for_review

                    await freeze_fan_for_review(
                        action["fan_id"],
                        "followup_sent_but_resolution_not_persisted",
                    )
                    await complete_action(aid)
                    sent_counter[0] += 1
                    print(
                        f"[SCHEDULED PERSIST ERROR] {action_type} "
                        f"fan={action['fan_id']}: {exc}"
                    )
                    timings.outcome = "sent_persist_error"
                    return "sent_persist_error"
            else:
                started = time.perf_counter()
                await _record_message_action_resolution(action, sent=False)
                timings.add("persistence_ms", (time.perf_counter() - started) * 1000)
            await complete_action(aid)
            if result.sent_message:
                sent_counter[0] += 1
            print(
                f"[SCHEDULED] completed {action_type} fan={action['fan_id']} "
                f"sent={result.sent_message} reason={result.reason}"
            )
            timings.outcome = "sent" if result.sent_message else "completed"
            return timings.outcome
        except asyncio.CancelledError:
            timings.outcome = "cancelled"
            raise
        except Exception as e:
            print(f"[SCHEDULED ERROR] {action_type} fan={action.get('fan_id')}: {e}")
            await fail_action(
                aid,
                str(e),
                action.get("attempts", 0),
                max_attempts=(50 if action_type == "PPV_RECONCILE" else 8),
            )
            timings.outcome = "failed"
            return "failed"
        finally:
            emit(timings)


def _queue_wait_ms(action: dict) -> int:
    """How long this action sat past its execute time before being picked up."""
    execute_at = _parse_time(action.get("execute_at"))
    if not execute_at:
        return 0
    delta = (datetime.now(timezone.utc) - execute_at).total_seconds() * 1000
    return int(max(0.0, delta))


def group_actions_by_fan(actions: list[dict]) -> list[list[dict]]:
    """Split a claimed batch into per-fan chains, preserving claim order.

    This is the whole of the per-fan safety story for concurrency. Actions for
    different fans are independent and run in parallel; actions for the SAME fan
    stay in one chain and run strictly one after another, so two conflicting
    sends for one conversation can never be in flight together. It needs no lock,
    no registry, and no distributed coordination — the grouping is the guarantee.

    All the existing per-fan protections (the AUTO_REPLY dedupe key,
    ``_pending_auto_replies``, ``cancel_actions_for_fan``, the PROCESSING status
    CAS, ``_should_still_send``, the expected trigger timestamp, the
    post-generation history re-check, and the PPV/proactive delivery journals)
    remain in force underneath it.
    """
    chains: dict[str, list[dict]] = {}
    order: list[str] = []
    for action in actions:
        # An action with no fan is still serialised against itself only.
        key = str(action.get("fan_id") or f"__action__{action.get('id')}")
        if key not in chains:
            chains[key] = []
            order.append(key)
        chains[key].append(action)
    return [chains[key] for key in order]


async def process_cycle(
    *,
    run_repair: bool = True,
    concurrency: int | None = None,
    limit: int | None = None,
) -> CycleResult:
    """Claim one batch of due actions and run it with bounded concurrency."""
    global LAST_RUN_STARTED_AT, LAST_RUN_COMPLETED_AT, LAST_RUN_ERROR, LAST_RUN_SENT
    global LAST_RUN_CLAIMED, LAST_RUN_DURATION_MS, LAST_RUN_BATCH_FULL
    global LAST_RUN_MAX_CONCURRENCY, LAST_REPAIR_AT, LAST_REPAIR_COUNT, CYCLES_COMPLETED

    LAST_RUN_STARTED_AT = datetime.now(timezone.utc)
    LAST_RUN_ERROR = None
    started_at = time.perf_counter()
    result = CycleResult()
    sent_counter = [0]

    try:
        if run_repair:
            result.repaired = await repair_followup_obligations()
            LAST_REPAIR_AT = datetime.now(timezone.utc)
            LAST_REPAIR_COUNT = result.repaired

        batch_size = claim_limit() if limit is None else max(1, int(limit))
        actions = await claim_due_actions(limit=batch_size)
        result.claimed = len(actions)
        result.batch_full = len(actions) >= batch_size

        slots = action_concurrency() if concurrency is None else max(1, int(concurrency))
        gate = asyncio.Semaphore(slots)
        inflight = [0]
        peak = [0]

        async def _run_chain(chain: list[dict]) -> None:
            for action in chain:
                async with gate:
                    inflight[0] += 1
                    peak[0] = max(peak[0], inflight[0])
                    try:
                        outcome = await _resolve_action(action, sent_counter=sent_counter)
                    finally:
                        inflight[0] -= 1
                    result.processed += 1
                    if outcome == "failed":
                        result.errors += 1

        chains = group_actions_by_fan(actions)
        if chains:
            # return_exceptions keeps one pathological chain from abandoning the
            # rest of the batch mid-flight; _resolve_action already swallows
            # handler errors, so anything arriving here is a genuine defect.
            outcomes = await asyncio.gather(
                *(_run_chain(chain) for chain in chains),
                return_exceptions=True,
            )
            for outcome in outcomes:
                if isinstance(outcome, BaseException) and not isinstance(
                    outcome, asyncio.CancelledError
                ):
                    result.errors += 1
                    print(f"[SCHEDULED CHAIN ERROR] {outcome}")

        result.sent = sent_counter[0]
        result.max_concurrency = peak[0]
        LAST_RUN_SENT = result.sent
        LAST_RUN_CLAIMED = result.claimed
        LAST_RUN_BATCH_FULL = result.batch_full
        LAST_RUN_MAX_CONCURRENCY = result.max_concurrency
        CYCLES_COMPLETED += 1
        return result
    except Exception as exc:
        LAST_RUN_ERROR = str(exc)[:500]
        raise
    finally:
        result.duration_ms = int((time.perf_counter() - started_at) * 1000)
        LAST_RUN_DURATION_MS = result.duration_ms
        LAST_RUN_COMPLETED_AT = datetime.now(timezone.utc)


async def process_once() -> int:
    """Backwards-compatible single cycle returning the number of messages sent."""
    return (await process_cycle()).sent


async def scheduled_actions_loop() -> None:
    """Poll, drain, and immediately re-poll while a full batch keeps coming back.

    The old loop slept a flat 60 seconds after every cycle, full batch or not, so
    a backlog could only ever drain one batch per minute. Now a full claim is
    treated as evidence that more work is waiting and the next cycle starts
    almost immediately; only a short claim falls back to the idle poll. The idle
    path waits on an event as well as a timeout, so work enqueued in this process
    is picked up without waiting out the interval.
    """
    global _wakeup
    print("[SCHEDULED] worker started")
    _wakeup = asyncio.Event()
    last_repair = 0.0
    consecutive_busy = 0
    while True:
        now = time.monotonic()
        run_repair = (now - last_repair) >= REPAIR_INTERVAL_SECONDS
        batch_full = False
        try:
            cycle = await process_cycle(run_repair=run_repair)
            batch_full = cycle.batch_full
            if cycle.sent:
                print(f"[CRON] scheduled actions: sent {cycle.sent}")
        except Exception as e:
            print(f"[SCHEDULED LOOP ERROR] {e}")
        if run_repair:
            last_repair = now

        if batch_full and consecutive_busy < MAX_CONSECUTIVE_BUSY_CYCLES:
            consecutive_busy += 1
            await asyncio.sleep(BUSY_POLL_SECONDS)
            continue

        consecutive_busy = 0
        _wakeup.clear()
        try:
            await asyncio.wait_for(_wakeup.wait(), timeout=poll_seconds())
        except (asyncio.TimeoutError, TimeoutError):
            pass


if __name__ == "__main__":
    asyncio.run(scheduled_actions_loop())
