from __future__ import annotations

import json

from services.conversational_decision_contract import parse_semantic_decision


def parse(**overrides):
    payload = {
        "turn_id": "turn-1",
        "conversation_revision": "revision-1",
        "disposition": "reply",
        "response_goal": "continue the active subject",
        "must_address": [{"need": "answer the correction", "source_ids": ["m-1"]}],
        "contribution_goal": "add a creator opinion",
        "relevant_thread_ids": ["thread-1"],
        "initiative": "creator",
        "pacing": "redirect",
        "evidence_requests": [],
        "operation_proposal": {"kind": "none"},
        "state_delta": {"current_direction": "the corrected topic"},
        "memory_candidates": [
            {
                "claim": "the fan prefers tea",
                "source_type": "explicit_fan_statement",
                "source_refs": ["m-1"],
            }
        ],
        "confidence": 0.7,
    }
    payload.update(overrides)
    return parse_semantic_decision(json.dumps(payload))


def test_semantic_contract_carries_behavior_without_wording():
    result = parse()
    assert result.usable
    assert result.decision.response_goal == "continue the active subject"
    assert result.decision.contribution_goal == "add a creator opinion"
    assert result.decision.initiative == "creator"
    assert result.decision.pacing == "redirect"
    assert result.decision.must_address == ("answer the correction",)
    assert result.decision.supporting_messages == ("m-1",)
    assert result.state_delta == {"current_direction": "the corrected topic"}


def test_any_fan_facing_copy_field_invalidates_glm_output():
    for key in ("reply", "caption", "rewrite", "phrasing"):
        result = parse(**{key: "language the fan might see"})
        assert not result.usable
        assert "fan-facing prose" in result.failure


def test_evidence_requests_are_typed_and_bounded():
    result = parse(
        evidence_requests=[
            {"category": "inventory"},
            {"category": "memory"},
            {"category": "arbitrary_storage"},
        ]
    )
    assert result.decision.evidence_requests == ("inventory", "memory")
    assert "evidence_request:arbitrary_storage" in result.degradations


def test_operation_uses_opaque_candidate_handle_not_price_or_inventory_ids():
    result = parse(
        operation_proposal={
            "kind": "present_offer",
            "subject": "the media they asked about",
            "candidate_handle": "offer_candidate_1",
            "price": 25,
            "offer_id": "private-offer-id",
        }
    )
    operation = result.decision.proposed_operation
    assert operation.candidate_handle == "offer_candidate_1"
    assert operation.offer_id == ""
    assert not hasattr(operation, "price")

