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
from datetime import datetime, timedelta, timezone

from db.commercial_queries import (
    claim_due_actions,
    complete_action,
    fail_action,
    get_creator_policy,
    get_fan_state,
)
from models.commercial import FanStatus

POLL_SECONDS = 60


async def _should_still_send(action: dict) -> tuple[bool, str]:
    """Revalidate at send time. Returns (ok, reason_if_not)."""
    from db.queries import get_fan_by_id, get_conversation_history

    fan_id = action["fan_id"]
    creator_id = action["creator_id"]

    fan = await get_fan_by_id(fan_id)
    if not fan:
        return False, "fan gone"

    # Frozen for a human (crisis / whale handoff): never auto-message.
    if getattr(fan, "needs_human_review", False):
        return False, "fan frozen for human review"

    # Auto mode must still be on for this fan.
    from db.queries import get_creator_auto_mode_default
    fan_auto = getattr(fan, "auto_mode", None)
    if fan_auto is None:
        try:
            fan_auto = await get_creator_auto_mode_default(creator_id)
        except Exception:
            fan_auto = False
    if not fan_auto:
        return False, "auto mode off"

    state = await get_fan_state(fan_id)

    # He already resolved it himself (bought, or told us money arrived).
    if state.status not in (FanStatus.PAUSED_UNTIL_PAYDAY, FanStatus.PAUSED_NO_BUDGET):
        return False, f"no longer paused (status={state.status.value})"

    # He's been talking to us since — don't barge in with a scripted follow-up.
    try:
        history = await get_conversation_history(fan_id, limit=5)
        if history:
            last = history[-1]
            ts = getattr(last, "created_at", None) or getattr(last, "sent_at", None)
            if ts:
                if isinstance(ts, str):
                    ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                if ts > datetime.now(timezone.utc) - timedelta(hours=6):
                    return False, "fan active in the last 6h"
    except Exception:
        pass  # never block a send purely because history lookup failed

    # Creator sleep hours.
    policy = await get_creator_policy(creator_id)
    if policy.payday_reengagement_enabled is False:
        return False, "payday re-engagement disabled for creator"

    return True, ""


async def _run_payday_reengagement(action: dict) -> None:
    """Generate and send a contextual, creator-voice payday follow-up.

    Note the framing: we do NOT assert he definitely has money ("you got paid, let's
    spend it"). We reopen the door he left open.
    """
    from services.proactive import send_proactive_message

    payload = action.get("payload") or {}
    desired = payload.get("desired_experience") or "what we were talking about"

    goal = (
        f"It's the day he said his money would come in. Reopen the conversation warmly "
        f"and playfully — reference that he wanted {desired} and that you said it'd "
        f"still be here for him. Do NOT assume he definitely has money, do NOT pressure, "
        f"do NOT state a price, do NOT send media. One short message, his energy, "
        f"leaving him an easy yes."
    )

    await send_proactive_message(
        creator_id=action["creator_id"],
        fan_id=action["fan_id"],
        goal=goal,
    )


HANDLERS = {
    "PAYDAY_REENGAGEMENT": _run_payday_reengagement,
}


async def process_once() -> int:
    actions = await claim_due_actions()
    sent = 0
    for action in actions:
        aid = action["id"]
        try:
            ok, reason = await _should_still_send(action)
            if not ok:
                print(f"[SCHEDULED] skip {action['action_type']} fan={action['fan_id']}: {reason}")
                await complete_action(aid)  # resolved, not failed — don't retry
                continue

            handler = HANDLERS.get(action["action_type"])
            if not handler:
                await fail_action(aid, f"no handler for {action['action_type']}",
                                  action.get("attempts", 0))
                continue

            await handler(action)
            await complete_action(aid)
            sent += 1
            print(f"[SCHEDULED] sent {action['action_type']} fan={action['fan_id']}")
        except Exception as e:
            print(f"[SCHEDULED ERROR] {action.get('action_type')} fan={action.get('fan_id')}: {e}")
            await fail_action(aid, str(e), action.get("attempts", 0))
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