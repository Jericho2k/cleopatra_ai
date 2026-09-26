"""Unit coverage for Core v2's deterministic pieces and its Assisted path."""

from __future__ import annotations

import json

import pytest

from models.commercial import CreatorPolicy
from models.conversational_session import (
    ContentLifecycle,
    ConversationalSessionState,
    SessionStatus,
)
from models.live_orchestration import EvidenceSnapshot, TurnTrigger
from services import conversational_session
from services import live_orchestration as lo
from services.conversational_session_contract import parse_session_decision
from services.media_packages import build_candidate_offers, build_next_offer
from tests.conversational_v2_harness import (
    CREATOR_ID,
    FAN_ID,
    Item,
    V2World,
    install,
    owner_json,
)


def snapshot(identity: str = "m1", text: str = "hello", **extra) -> EvidenceSnapshot:
    return EvidenceSnapshot(
        creator_id="c",
        fan_id="f",
        trigger=TurnTrigger("fan_message", identity, text),
        state_revision="r",
        **extra,
    )


def vault_rows():
    photos = [
        {"id": f"p{index}", "title": f"cafe {index}", "location": "cafe", "outfit": "blue",
         "explicit_min": index, "explicit_max": index, "suggested_price": 15 + 5 * index,
         "media_ids": [f"p{index}-1"], "tags": ["nude_photo"]}
        for index in range(1, 5)
    ]
    video = {"id": "v1", "title": "cafe clip", "description": "A private clip.", "location": "cafe",
             "explicit_min": 2, "explicit_max": 2, "base_price_cents": 4500, "min_price_cents": 4000,
             "max_price_cents": 6000, "media_ids": ["v1-1"], "tags": ["video", "individual_video"]}
    return [*photos, video]


# --- content candidates -------------------------------------------------------


def test_candidates_start_with_the_unchanged_next_offer_and_stay_bounded():
    rows = vault_rows()
    next_offer = build_next_offer(rows, CreatorPolicy())
    candidates = build_candidate_offers(rows, next_offer=next_offer, max_candidates=4)
    assert candidates[0] is next_offer
    assert len(candidates) <= 4
    assert len({offer.set_id for offer in candidates}) == len(candidates)
    # A redirect toward a clip can be served from real inventory.
    assert any(offer.asset_type == "video" for offer in candidates)
    for offer in candidates:
        assert offer.content_floor_cents <= offer.price_cents <= offer.content_ceiling_cents


def test_candidates_respect_an_explicit_ceiling():
    rows = vault_rows()
    ceiling = 2500
    next_offer = build_next_offer(rows, CreatorPolicy(), hard_ceiling_cents=ceiling)
    candidates = build_candidate_offers(
        rows, next_offer=next_offer, hard_ceiling_cents=ceiling, max_candidates=4
    )
    assert candidates
    assert all(offer.price_cents <= ceiling for offer in candidates)


# --- contract -------------------------------------------------------------------


def test_v2_contract_reads_the_move_and_the_session_delta():
    result = parse_session_decision(
        json.dumps(
            owner_json(move="linger", session_delta={"tempo": "linger"})
        )
    )
    assert result.usable
    assert result.decision.source == "conversational_decision_v2"
    assert result.next_experience_move.kind.value == "linger"
    assert result.session_delta == {"tempo": "linger"}


def test_v2_contract_degrades_optional_fields_and_keeps_v1_refusals():
    missing = parse_session_decision(json.dumps({"disposition": "reply", "response_goal": "talk"}))
    assert missing.usable
    assert missing.next_experience_move.kind.value == "converse"
    assert "next_experience_move" in missing.degradations

    prose = parse_session_decision(
        json.dumps(owner_json(session_delta={"trajectory": {"beats": [{"kind": "conversation", "reply": "hi!"}]}}))
    )
    assert prose.usable and prose.session_delta == {}
    assert "wording" in prose.degradations["session_delta"]

    copy = parse_session_decision(json.dumps({**owner_json(), "reply": "hey you"}))
    assert not copy.usable


# --- lifecycle & constraints ------------------------------------------------------


def test_status_transitions_are_validated_and_new_sessions_archive_the_old():
    refused = conversational_session.validate_session_delta(
        ConversationalSessionState(), {"status": "completed"}, snapshot=snapshot()
    )
    assert "cannot move" in refused.rejected_fields["status"]

    active = conversational_session.validate_session_delta(
        ConversationalSessionState(), {"status": "active", "interaction_goal": "talk"}, snapshot=snapshot("m1")
    ).state_after
    first_id = active.session.session_id
    assert first_id
    closed = conversational_session.validate_session_delta(
        active, {"status": "completed"}, snapshot=snapshot("m2")
    ).state_after
    assert closed.session.status is SessionStatus.COMPLETED
    reopened = conversational_session.validate_session_delta(
        closed, {"status": "proposed"}, snapshot=snapshot("m3")
    ).state_after
    assert reopened.session.status is SessionStatus.PROPOSED
    assert reopened.session.session_id != first_id
    assert [row.session_id for row in reopened.previous_sessions] == [first_id]


def test_spending_limit_counts_only_purchases_after_it_was_stated():
    before_limit = snapshot(
        "m1",
        confirmed_deliveries=({"set_id": "A", "platform_message_id": "p1"},),
        confirmed_purchases=({"set_id": "A", "reference": "buy-a", "price_cents": 4000},),
    )
    state, _ = conversational_session.reconcile_with_authority(
        ConversationalSessionState(), snapshot=before_limit
    )
    limited = conversational_session.validate_session_delta(
        state,
        {"add_constraints": [{"constraint_id": "l", "kind": "spending_limit", "statement": "60 tonight",
                              "amount_cents": 6000, "source_refs": ["m2"]}]},
        snapshot=snapshot("m2", "ok i can do $60 tonight"),
    ).state_after
    # The earlier purchase of A does not count against a limit stated later.
    assert conversational_session.remaining_spending_cents(limited) == 6000
    after = snapshot(
        "m3",
        confirmed_deliveries=(
            {"set_id": "A", "platform_message_id": "p1"},
            {"set_id": "C", "platform_message_id": "p2"},
        ),
        confirmed_purchases=(
            {"set_id": "A", "reference": "buy-a", "price_cents": 4000},
            {"set_id": "C", "reference": "buy-c", "price_cents": 3000},
        ),
    )
    reconciled, _ = conversational_session.reconcile_with_authority(limited, snapshot=after)
    assert reconciled.session.lifecycle_of("C") is ContentLifecycle.PURCHASED
    assert conversational_session.remaining_spending_cents(reconciled) == 3000


def test_prices_and_completed_transactions_never_enter_goals():
    result = conversational_session.validate_session_delta(
        ConversationalSessionState(),
        {"interaction_goal": "sell content B for $75", "content_direction": "he has already paid for more"},
        snapshot=snapshot(),
    )
    assert "price" in result.rejected_fields["interaction_goal"]
    assert "ledger" in result.rejected_fields["content_direction"]
    ok = conversational_session.validate_session_delta(
        ConversationalSessionState(),
        {"interaction_goal": "learn his limit before planning a longer paid evening"},
        snapshot=snapshot(),
    )
    assert "interaction_goal" in ok.accepted_fields


# --- Assisted ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_v2_assisted_draft_rebinds_to_the_exact_candidate_on_approval(monkeypatch):
    world = install(
        monkeypatch,
        V2World(
            catalog={
                "A": Item("A", 4000, "content A: the opening"),
                "B": Item("B", 3500, "content B: what he asked for"),
            }
        ),
    )
    stored = {}

    async def remember(provenance):
        stored["provenance"] = provenance
        return "token-1"

    monkeypatch.setattr(lo, "remember_assisted_provenance", remember)
    world.fan_says("could i get content B?")
    world.queue(
        lambda payload: owner_json(
            move="use_content",
            operation={
                "kind": "present_offer",
                "subject": "content B",
                "because": "direct request",
                "candidate_handle": world.handle_for(payload, "B"),
            },
        ),
        ["yes, that one", "of course", "it's yours if you want it"],
    )
    response = await lo.get_assisted_suggestions(
        creator_id=CREATOR_ID,
        fan_id=FAN_ID,
        fan_message="could i get content B?",
        save_fan_message=False,
        conversation_core="conversational_v2",
    )
    assert response.conversation_core == "conversational_v2"
    provenance = stored["provenance"]
    assert provenance.decision["semantic_set_id"] == "B"
    # Drafting presents nothing and persists nothing.
    assert world.pending_offer is None
    assert world.session_row() is None

    prepared = await lo.prepare_assisted_approval(
        provenance, creator_id=CREATOR_ID, fan_id=FAN_ID
    )
    assert prepared.execution.operation == "present_offer"
    assert prepared.execution.offer["set_id"] == "B"
    await lo.finalize_assisted_plain_approval(prepared)
    assert world.pending_offer is not None and world.pending_offer.set_id == "B"
    assert world.session_row() is not None
