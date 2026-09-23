"""The paid-platform world model, action affordances, and grounding backstops."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from ai.writer_style import MODE_AUTO
from models.commercial import CreatorPolicy, FanCommercialState, FanStatus, Offer
from models.conversation_decision import (
    ConversationDecision,
    IntimacyContext,
    OperationKind,
    ProposedOperation,
    ResponseDisposition,
)
from models.conversational_core import ConversationalWorkingState
from models.live_orchestration import (
    ApprovedExecution,
    EvidenceFact,
    EvidenceSnapshot,
    TurnTrigger,
    ValidationResult,
)
from models.schemas import Fan, Persona
from services import conversation_signals, live_orchestration
from services.context_packet import ContextPacket
from services.platform_operating_model import (
    CLEOPATRA_MISSION,
    PLATFORM_OPERATING_MODEL,
    operation_affordances,
    platform_context,
    prepared_execution_reality,
)


def _message(text: str, message_id: str = "m-1") -> dict:
    return {"message_id": message_id, "speaker": "fan", "text": text, "at": None}


def _offer() -> Offer:
    return Offer(
        offer_id="offer-1",
        set_id="set-1",
        label="private content",
        price_cents=2500,
        legal_description="private content",
    )


def _loaded(
    latest: str = "hey",
    *,
    offer: Offer | None = None,
    creator_facts: tuple[EvidenceFact, ...] = (),
) -> live_orchestration.LoadedEvidence:
    burst = (_message(latest),)
    snapshot = EvidenceSnapshot(
        creator_id="creator-1",
        fan_id="fan-1",
        trigger=TurnTrigger("fan_message", "m-1", latest_message=latest),
        state_revision="rev-1",
        creator_facts=creator_facts,
        latest_fan_burst=burst,
        recent_messages=burst,
        approved_inventory=(
            {
                "candidate_handle": "offer_candidate_1",
                "asset_type": offer.asset_type,
                "legal_description": offer.legal_description,
            },
        )
        if offer
        else (),
        commercial_opportunity={
            **conversation_signals.purchase_intent(burst),
            "unsent_approved_inventory_exists": bool(offer),
            "offer_already_presented": False,
        },
        media_request=conversation_signals.media_request(burst),
        platform_context=platform_context(),
        publication_evidence=conversation_signals.publication_evidence(creator_facts),
        purchase_claim=conversation_signals.purchase_claim(burst),
    )
    target = SimpleNamespace(model="glm-test", provider="test", name="glm-test")
    spec = SimpleNamespace(
        primary_target=lambda: target,
        fallback_target=lambda: None,
        max_tokens=4096,
    )
    record = live_orchestration.LoadedEvidence(
        snapshot=snapshot,
        packet=ContextPacket(),
        history=[],
        fan=Fan(
            id="fan-1",
            display_name="Fan",
            platform_fan_id="test_fan_1",
            auto_mode=True,
        ),
        persona=Persona(),
        commercial_state=FanCommercialState(status=FanStatus.IDLE),
        policy=CreatorPolicy(),
        next_offer=offer,
        active_session=None,
        pending_payment=None,
        sent_ppv=[],
        within_daily_caps=True,
        stack=SimpleNamespace(
            profile_id="cleo_v3",
            profile=SimpleNamespace(stage=lambda _stage: spec),
        ),
    )
    if offer:
        record.candidate_handles = {"offer_candidate_1": offer}
    return record


def _reasons(
    text: str,
    loaded: live_orchestration.LoadedEvidence,
    execution: ApprovedExecution | None = None,
) -> list[str]:
    return live_orchestration.writer_contract_reasons(
        [text],
        loaded,
        execution or ApprovedExecution(validation=ValidationResult(True, "none")),
        mode=MODE_AUTO,
    )


def test_one_operating_model_and_application_owned_surface_reach_both_models():
    context = platform_context()

    assert context["surface"] == "private_creator_fan_chat"
    assert context["fan_is_already_in_private_chat"] is True
    assert context["paid_media_delivery_surface"] == "this_chat"
    assert context["can_see_fan"] is False
    assert PLATFORM_OPERATING_MODEL in live_orchestration.CONVERSATIONAL_V1_SYSTEM
    assert CLEOPATRA_MISSION in live_orchestration.CONVERSATIONAL_V1_SYSTEM

    prompt = live_orchestration.build_conversational_writer_prompt(
        _loaded(),
        ConversationDecision(),
        ApprovedExecution(validation=ValidationResult(True, "none")),
        ConversationalWorkingState(),
        mode=MODE_AUTO,
    )
    assert PLATFORM_OPERATING_MODEL in prompt[0]["content"]
    assert CLEOPATRA_MISSION in prompt[0]["content"]
    assert json.loads(prompt[1]["content"])["platform_context"] == context


@pytest.mark.parametrize(
    "copy",
    [
        "come find me in my dms",
        "come to my DMs",
        "message me in my dms",
        "go to my DMs",
        "find me in messages",
        "message me somewhere else",
    ],
)
def test_an_already_present_fan_cannot_be_redirected_to_another_chat(copy):
    assert "unsupported_delivery_route" in _reasons(copy, _loaded())


@pytest.mark.parametrize(
    "copy",
    [
        "we're already in the DMs",
        "you're in the right place",
        "message me when you're ready",
        "you found the right place",
    ],
)
def test_current_chat_references_remain_legal(copy):
    assert "unsupported_delivery_route" not in _reasons(copy, _loaded())


def test_narration_does_not_create_delivery_but_here_goes_alone_is_legal():
    loaded = _loaded("show me")

    assert "false_delivery_claim" in _reasons("I just sent it", loaded)
    assert "false_delivery_claim" in _reasons("look at this", loaded)
    assert "false_delivery_claim" not in _reasons("here goes", loaded)


def test_shared_imagination_requires_hypothetical_scope_not_just_scene_mode():
    loaded = _loaded("keep going")

    assert _reasons("if you were here i'd be unclipping it slowly", loaded) == []
    assert "unsupported_present_creator_action" in _reasons(
        "already unclipping it", loaded
    )


def test_unsupported_present_action_is_locally_reframed_inside_shared_roleplay():
    loaded = _loaded("keep going")
    decision = ConversationDecision(
        disposition=ResponseDisposition.REPLY,
        intimacy_context=IntimacyContext(active=True, scene_mode="shared_imagined"),
    )

    settled, execution, replies, repaired = (
        live_orchestration._validate_conversational_v1_reply(
            decision,
            ["already unclipping it slowly"],
            ApprovedExecution(validation=ValidationResult(True, "none")),
            loaded,
            mode=MODE_AUTO,
        )
    )

    assert repaired is True
    assert settled.disposition is ResponseDisposition.REPLY
    assert execution.operation == "none"
    assert replies == ["imagine me slowly unclipping it slowly"]
    assert _reasons(replies[0], loaded) == []


def test_sourced_current_creator_action_remains_legal():
    fact = EvidenceFact(
        value="already unclipping it",
        source_ref="creator_live_status:123",
        certainty="confirmed",
    )

    assert "unsupported_present_creator_action" not in _reasons(
        "already unclipping it", _loaded(creator_facts=(fact,))
    )


@pytest.mark.asyncio
async def test_glm_receives_media_request_and_product_effects_not_enum_names_only():
    loaded = _loaded("show me more", offer=_offer())
    calls: list[dict] = []

    async def complete(target, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            text=json.dumps(
                {
                    "turn_id": "m-1",
                    "conversation_revision": "rev-1",
                    "disposition": "reply",
                    "response_goal": "respond to the direct request",
                    "operation_proposal": {"kind": "none"},
                }
            ),
            target=target,
            upstream_provider="test",
            served_model="glm-test",
            latency_ms=1,
            usage=None,
            reported_cost_usd=None,
        )

    await live_orchestration.decide_conversational_v1(
        loaded, ConversationalWorkingState(), owner_complete=complete
    )
    payload = json.loads(calls[0]["messages"][0]["content"])

    assert payload["evidence_snapshot"]["media_request"] == {
        "present": True,
        "strength": "direct",
        "source_ids": ["m-1"],
        "rule": loaded.snapshot.media_request["rule"],
    }
    assert payload["legal_operations"]["present_offer"]["legal"] is True
    assert "does not" in payload["legal_operations"]["present_offer"]["effect"]
    assert payload["legal_operations"]["send_locked_paid_message"]["legal"] is True
    assert (
        "same-chat media delivery"
        in payload["legal_operations"]["send_locked_paid_message"]["effect"]
    )


def test_media_request_without_inventory_cannot_invent_a_media_action():
    loaded = _loaded("I really need to see more of you")
    affordances = operation_affordances(live_orchestration.legal_operations(loaded))

    assert loaded.snapshot.media_request["strength"] == "direct"
    assert affordances["present_offer"]["legal"] is False
    assert affordances["send_locked_paid_message"]["legal"] is False
    assert "false_delivery_claim" in _reasons("look at this", loaded)


def test_offer_and_locked_delivery_have_distinct_prepared_truth():
    offered = prepared_execution_reality(
        {"operation": "present_offer", "offer": {"price_cents": 2500}}
    )
    delivered = prepared_execution_reality(
        {
            "operation": "send_locked_paid_message",
            "delivery": {"price_cents": 2500},
            "approval_required": False,
        }
    )

    assert offered["external_product_action_happened"] is True
    assert offered["new_media_attached_to_this_message"] is False
    assert "No media has been delivered" in offered["effect"]
    assert delivered["new_media_attached_to_this_message"] is True
    assert "same outgoing locked message" in delivered["effect"]

    awaiting_approval = prepared_execution_reality(
        {
            "operation": "send_locked_paid_message",
            "delivery": None,
            "approval_required": True,
        }
    )
    assert awaiting_approval["external_product_action_happened"] is False
    assert awaiting_approval["new_media_attached_to_this_message"] is False
    assert "No approved delivery" in awaiting_approval["effect"]


@pytest.mark.asyncio
async def test_rejected_operation_is_removed_before_kimi_sees_execution_reality():
    loaded = _loaded("show me")
    proposed = ConversationDecision(
        disposition=ResponseDisposition.REPLY,
        response_goal="answer without fabricating media",
        proposed_operation=ProposedOperation(
            kind=OperationKind.PRESENT_OFFER,
            subject="content that is not authorized",
            candidate_handle="invented",
        ),
    )

    settled = await live_orchestration._authorize_conversational_v1_operation(
        loaded, decision=proposed, execute_operations=True
    )
    prompt = live_orchestration.build_conversational_writer_prompt(
        loaded,
        settled.decision,
        settled.execution,
        ConversationalWorkingState(),
        mode=MODE_AUTO,
    )
    payload = json.loads(prompt[1]["content"])

    assert settled.proposed_operation == "present_offer"
    assert settled.operation_rejected is True
    assert settled.decision.proposed_operation.kind is OperationKind.NONE
    assert payload["prepared_operation_facts"]["operation"] == "none"
    assert payload["execution_reality"]["external_product_action_happened"] is False
    assert "rejected proposal" in payload["execution_reality"]["rule"]
