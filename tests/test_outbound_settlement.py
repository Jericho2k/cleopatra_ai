"""Commercial state lands when the fan sees the words, not when they are planned.

Durable timed delivery splits "the reply was authorized" from "the reply was
delivered", and a planned reply may never leave at all. Committing an offer
presentation at plan time would therefore leave fan state claiming a pending
offer nobody was ever shown — which the next turn would treat as deliverable
against, and which the abandoned-offer chase would then pursue.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from models.commercial import CreatorPolicy, FanCommercialState, FanStatus, Offer
from services import outbound_settlement


def run(coro):
    return asyncio.run(coro)


def offer(offer_id: str = "offer-1", cents: int = 2500) -> Offer:
    return Offer(
        offer_id=offer_id,
        set_id=f"set-{offer_id}",
        label="approved set",
        price_cents=cents,
        media_count=3,
        asset_type="photo_set",
    )


def offer_view(record: Offer) -> dict:
    return {
        "offer_id": record.offer_id,
        "set_id": record.set_id,
        "label": record.label,
        "price_cents": record.price_cents,
        "asset_type": record.asset_type,
        "media_count": record.media_count,
        "legal_description": record.legal_description,
    }


@pytest.fixture
def commercial(monkeypatch):
    state = {"value": FanCommercialState(status=FanStatus.IDLE)}
    frozen: list = []
    expiries: list = []

    async def _get_state(_fan_id):
        return state["value"]

    async def _save_state(_fan_id, _creator_id, new_state):
        state["value"] = new_state

    async def _policy(_creator_id):
        return CreatorPolicy()

    async def _freeze(fan_id, reason):
        frozen.append((fan_id, reason))

    async def _sync(**kwargs):
        expiries.append(kwargs)

    monkeypatch.setattr(outbound_settlement, "get_fan_state", _get_state)
    monkeypatch.setattr(outbound_settlement, "save_fan_state", _save_state)
    monkeypatch.setattr(outbound_settlement, "get_creator_policy", _policy)
    monkeypatch.setattr(outbound_settlement, "freeze_fan_for_review", _freeze)
    monkeypatch.setattr(
        "services.offer_lifecycle.sync_pending_offer_expiry", _sync
    )
    return SimpleNamespace(state=state, frozen=frozen, expiries=expiries)


def settle(instruction):
    return run(
        outbound_settlement.settle_after_first_part(
            creator_id="creator-1", fan_id="fan-1", instruction=instruction
        )
    )


def test_an_ordinary_reply_settles_nothing(commercial):
    assert settle(None) == "none"
    assert commercial.state["value"].status is FanStatus.IDLE


def test_the_offer_is_recorded_when_the_first_bubble_lands(commercial):
    result = settle(outbound_settlement.present_offer_instruction(offer_view(offer())))

    assert result == "offer_recorded"
    assert commercial.state["value"].status is FanStatus.OFFER_PENDING
    assert commercial.state["value"].pending_offer.offer_id == "offer-1"
    assert commercial.expiries, "an expiry obligation must exist for it"


def test_settling_twice_does_not_double_record(commercial):
    instruction = outbound_settlement.present_offer_instruction(offer_view(offer()))
    settle(instruction)

    assert settle(instruction) == "already_recorded"


def test_an_offer_that_changed_while_the_reply_waited_is_never_overwritten(commercial):
    """Case 14: inventory moved on while the delayed reply sat in the queue."""
    newer = FanCommercialState(status=FanStatus.OFFER_PENDING)
    newer.pending_offer = offer("offer-2", 4000)
    newer.last_offer_at = datetime.now(timezone.utc)
    commercial.state["value"] = newer

    result = settle(outbound_settlement.present_offer_instruction(offer_view(offer())))

    assert result == "offer_state_changed"
    assert commercial.state["value"].pending_offer.offer_id == "offer-2"
    assert commercial.frozen == [("fan-1", "outbound_offer_state_changed")]


def test_an_unreadable_offer_record_never_invents_one(commercial):
    assert settle({"kind": "present_offer", "offer": {}}) == "offer_record_unreadable"
    assert commercial.state["value"].status is FanStatus.IDLE


def test_a_payment_claim_check_runs_the_authoritative_path(commercial, monkeypatch):
    checked: list = []

    async def _verify(fan_id, creator_id, pending):
        checked.append((fan_id, creator_id, pending))

    monkeypatch.setattr("services.payment_claims.verify_ppv_purchase", _verify)

    result = settle(
        outbound_settlement.check_payment_instruction({"reference": "ref-1"})
    )

    assert result == "payment_claim_checked"
    assert checked == [("fan-1", "creator-1", {"reference": "ref-1"})]


def test_an_empty_pending_payment_produces_no_instruction():
    assert outbound_settlement.check_payment_instruction(None) == {}
    assert outbound_settlement.present_offer_instruction(None) == {}
