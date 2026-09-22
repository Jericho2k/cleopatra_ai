"""Commercial state that must land exactly when the fan actually sees the words.

``_commit_presented_offer`` used to run immediately after the orchestrator sent
the reply inline, so "an offer is pending" and "the fan was shown an offer" were
one event. Durable timed delivery splits them: the words are planned now and
leave later, and they may never leave at all if the fan interrupts.

Committing at plan time would leave a fan state claiming a pending offer nobody
was ever shown — which the next turn would treat as acceptable to deliver
against, and which the abandoned-offer follow-up would chase. So the commit
moves to the first bubble's delivery receipt, and only runs once.

This module deliberately imports no orchestration. It is the settlement half of
delivery and nothing else, which is what keeps it callable from the worker.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from db.commercial_queries import get_creator_policy, get_fan_state, save_fan_state
from db.queries import freeze_fan_for_review
from models.commercial import FanStatus, Offer

#: Metadata key the orchestrator writes onto an outbound sequence, naming the
#: commercial settlement its first delivered bubble owes.
POST_SEND_KEY = "post_send_operation"


def present_offer_instruction(offer_view: dict[str, Any] | None) -> dict[str, Any]:
    """The durable, price-free-of-model-opinion record of an offer presentation."""
    if not offer_view:
        return {}
    return {
        "kind": "present_offer",
        "offer": {
            key: offer_view.get(key)
            for key in (
                "offer_id",
                "set_id",
                "label",
                "price_cents",
                "asset_type",
                "media_count",
                "legal_description",
            )
        },
    }


def check_payment_instruction(pending: dict[str, Any] | None) -> dict[str, Any]:
    if not pending:
        return {}
    return {"kind": "check_payment_claim", "pending_payment": dict(pending)}


def _offer_from(record: dict[str, Any]) -> Offer | None:
    try:
        return Offer(
            offer_id=str(record["offer_id"]),
            set_id=str(record["set_id"]),
            label=str(record.get("label") or ""),
            price_cents=int(record.get("price_cents") or 0),
            asset_type=str(record.get("asset_type") or "photo_set"),
            media_count=int(record.get("media_count") or 0),
            legal_description=record.get("legal_description"),
        )
    except (KeyError, TypeError, ValueError):
        return None


async def _commit_presented_offer(
    *, creator_id: str, fan_id: str, record: dict[str, Any]
) -> str:
    offer = _offer_from(record)
    if offer is None:
        return "offer_record_unreadable"
    from services.offer_lifecycle import sync_pending_offer_expiry

    state = await get_fan_state(fan_id)
    if state.pending_offer and state.pending_offer.offer_id != offer.offer_id:
        # A different offer appeared after this reply was planned. Never
        # overwrite it because an older presentation happened to land later.
        await freeze_fan_for_review(fan_id, "outbound_offer_state_changed")
        return "offer_state_changed"
    if (
        state.pending_offer
        and state.pending_offer.offer_id == offer.offer_id
        and state.status == FanStatus.OFFER_PENDING
    ):
        return "already_recorded"

    policy = await get_creator_policy(creator_id)
    state.pending_offer = offer
    state.status = FanStatus.OFFER_PENDING
    state.last_offer_at = datetime.now(timezone.utc)
    state.accepted_offer_id = None
    state.accepted_offer_set_id = None
    state.accepted_offer_label = None
    state.accepted_offer_price_cents = None
    try:
        await save_fan_state(fan_id, creator_id, state)
        await sync_pending_offer_expiry(
            creator_id=creator_id,
            fan_id=fan_id,
            state=state,
            policy=policy,
            anchor=state.last_offer_at,
        )
        await save_fan_state(fan_id, creator_id, state)
    except Exception as exc:  # noqa: BLE001
        await freeze_fan_for_review(fan_id, "outbound_offer_sent_not_recorded")
        print(f"[OUTBOUND SETTLEMENT ERROR] fan={fan_id}: {exc}")
        return "offer_not_recorded"
    return "offer_recorded"


async def settle_after_first_part(
    *, creator_id: str, fan_id: str, instruction: dict[str, Any] | None
) -> str:
    """Run the commercial settlement this delivered reply owes, at most once."""
    record = dict(instruction or {})
    kind = str(record.get("kind") or "")
    if not kind:
        return "none"
    if kind == "present_offer":
        return await _commit_presented_offer(
            creator_id=creator_id, fan_id=fan_id, record=record.get("offer") or {}
        )
    if kind == "check_payment_claim":
        from services.payment_claims import verify_ppv_purchase

        await verify_ppv_purchase(
            fan_id, creator_id, dict(record.get("pending_payment") or {})
        )
        return "payment_claim_checked"
    return "unknown_settlement"
