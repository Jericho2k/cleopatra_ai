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
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from db.commercial_queries import (
    claim_due_actions,
    complete_action,
    ensure_action_pending,
    fail_action,
    get_creator_policy,
    get_fan_state,
    get_followup_obligations,
    reschedule_action,
    save_fan_state,
    schedule_action,
)
from models.commercial import FanStatus

POLL_SECONDS = 60


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

    fan = await get_fan_by_id(fan_id)
    if not fan:
        return ActionCheck(False, "fan gone")

    # Frozen for a human (crisis / whale handoff): never auto-message.
    if getattr(fan, "needs_human_review", False):
        return ActionCheck(False, "fan frozen for human review")

    # Auto mode must still be on for this fan.
    fan_auto = getattr(fan, "auto_mode", None)
    if fan_auto is None:
        try:
            fan_auto = await _creator_auto_mode_default(creator_id)
        except Exception:
            fan_auto = False
    if not fan_auto:
        return ActionCheck(False, "auto mode off")

    state = await get_fan_state(fan_id)
    policy = await get_creator_policy(creator_id)
    action_type = str(action.get("action_type") or "")
    payload = action.get("payload") or {}

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
    "PAYDAY_REENGAGEMENT": _run_payday_reengagement,
    "POST_SESSION_FOLLOWUP": _run_post_session_followup,
    "ABANDONED_PPV_FOLLOWUP": _run_abandoned_ppv_followup,
    "OFFER_EXPIRY": _run_offer_expiry,
    "ABANDONED_OFFER_FOLLOWUP": _run_abandoned_offer_followup,
    "INACTIVITY_REENGAGEMENT": _run_inactivity_reengagement,
    "PPV_RECONCILE": _run_ppv_reconcile,
}


async def _record_message_action_resolution(action: dict, *, sent: bool) -> None:
    action_type = str(action.get("action_type") or "")
    if action_type == "PPV_RECONCILE":
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


async def repair_followup_obligations() -> int:
    """Recreate missing durable actions from the fan-state obligation record."""
    repaired = 0
    for row in await get_followup_obligations():
        execute_at = _parse_time(row.get("next_followup_at"))
        action_type = str(row.get("next_followup_type") or "")
        dedupe_key = str(row.get("next_followup_dedupe_key") or "")
        if not execute_at or not action_type or not dedupe_key:
            continue
        await ensure_action_pending(
            creator_id=str(row["creator_id"]),
            fan_id=str(row["fan_id"]),
            action_type=action_type,
            execute_at=execute_at,
            payload=row.get("next_followup_payload") or {},
            dedupe_key=dedupe_key,
        )
        repaired += 1
    return repaired


async def process_once() -> int:
    await repair_followup_obligations()
    actions = await claim_due_actions()
    sent = 0
    for action in actions:
        aid = action["id"]
        try:
            handler = HANDLERS.get(action["action_type"])
            if not handler:
                await fail_action(aid, f"no handler for {action['action_type']}",
                                  action.get("attempts", 0))
                continue

            if action["action_type"] not in {"PPV_RECONCILE", "OFFER_EXPIRY"}:
                check = await _should_still_send(action)
                if not check.ok:
                    if check.retry_at:
                        await _record_followup_postponed(action, check.retry_at)
                        await reschedule_action(aid, check.retry_at)
                        print(
                            f"[SCHEDULED] postponed {action['action_type']} "
                            f"fan={action['fan_id']} until={check.retry_at.isoformat()}: {check.reason}"
                        )
                    else:
                        await complete_action(aid)
                        await _record_message_action_resolution(action, sent=False)
                        print(
                            f"[SCHEDULED] skip {action['action_type']} "
                            f"fan={action['fan_id']}: {check.reason}"
                        )
                    continue

            result = await handler(action)
            if result.retry_at:
                await reschedule_action(aid, result.retry_at)
                print(
                    f"[SCHEDULED] retry {action['action_type']} fan={action['fan_id']} "
                    f"at={result.retry_at.isoformat()}: {result.reason}"
                )
                continue
            if result.sent_message:
                try:
                    await _record_message_action_resolution(action, sent=True)
                except Exception as exc:
                    # The external send already happened. Never turn a local
                    # persistence failure into a duplicate proactive message.
                    from db.queries import freeze_fan_for_review

                    await freeze_fan_for_review(
                        action["fan_id"],
                        "followup_sent_but_resolution_not_persisted",
                    )
                    await complete_action(aid)
                    sent += 1
                    print(
                        f"[SCHEDULED PERSIST ERROR] {action['action_type']} "
                        f"fan={action['fan_id']}: {exc}"
                    )
                    continue
            else:
                await _record_message_action_resolution(action, sent=False)
            await complete_action(aid)
            if result.sent_message:
                sent += 1
            print(
                f"[SCHEDULED] completed {action['action_type']} fan={action['fan_id']} "
                f"sent={result.sent_message} reason={result.reason}"
            )
        except Exception as e:
            print(f"[SCHEDULED ERROR] {action.get('action_type')} fan={action.get('fan_id')}: {e}")
            await fail_action(
                aid,
                str(e),
                action.get("attempts", 0),
                max_attempts=(50 if action.get("action_type") == "PPV_RECONCILE" else 3),
            )
    return sent


async def scheduled_actions_loop() -> None:
    print("[SCHEDULED] worker started")
    while True:
        try:
            await process_once()
        except Exception as e:
            print(f"[SCHEDULED LOOP ERROR] {e}")
        await asyncio.sleep(POLL_SECONDS)


if __name__ == "__main__":
    asyncio.run(scheduled_actions_loop())
