"""Deterministic durable lifecycle updates for a presented offer.

This module owns no conversational policy.  Both orchestration generations use
it only after a presentation decision has already been made and delivered.
"""

from datetime import datetime

from db.commercial_queries import cancel_actions_for_fan, schedule_action
from models.commercial import FanStatus
from services.followup_lifecycle import pending_offer_expiry_obligation


def _clear_followup_obligation(state) -> None:
    state.next_followup_at = None
    state.next_followup_type = None
    state.next_followup_payload = {}
    state.next_followup_dedupe_key = None


async def sync_pending_offer_expiry(
    *,
    creator_id: str,
    fan_id: str,
    state,
    policy,
    anchor: datetime,
    cancel_action=cancel_actions_for_fan,
    schedule=schedule_action,
) -> None:
    """Make fan state and the durable queue agree about one pending offer."""
    if state.status != FanStatus.OFFER_PENDING or state.pending_offer is None:
        try:
            await cancel_action(fan_id, "OFFER_EXPIRY")
        except Exception as exc:
            print(f"[OFFER EXPIRY] cancellation failed fan={fan_id}: {exc}")
        if state.next_followup_type == "OFFER_EXPIRY":
            _clear_followup_obligation(state)
        return

    previous_type = state.next_followup_type
    state.last_offer_at = anchor
    obligation = pending_offer_expiry_obligation(
        state,
        policy=policy,
        fan_id=fan_id,
    )
    if obligation is None:
        return

    if previous_type and previous_type != "OFFER_EXPIRY":
        try:
            await cancel_action(fan_id, previous_type)
        except Exception as exc:
            print(
                f"[OFFER EXPIRY] superseded action cancellation failed "
                f"fan={fan_id} type={previous_type}: {exc}"
            )
    try:
        await cancel_action(fan_id, "OFFER_EXPIRY")
        await schedule(
            creator_id=creator_id,
            fan_id=fan_id,
            action_type=obligation.action_type,
            execute_at=obligation.execute_at,
            payload=obligation.payload,
            dedupe_key=obligation.dedupe_key,
        )
    except Exception as exc:
        # The state obligation is persisted by the caller and repaired by the
        # worker, so a queue write failure cannot lose the expiry promise.
        print(f"[OFFER EXPIRY] scheduling repair needed fan={fan_id}: {exc}")
    state.next_followup_at = obligation.execute_at
    state.next_followup_type = obligation.action_type
    state.next_followup_payload = obligation.payload
    state.next_followup_dedupe_key = obligation.dedupe_key
