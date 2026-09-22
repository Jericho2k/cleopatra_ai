"""Failure boundaries for the GLM-decision -> Kimi-writer Core v1 path."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from models.conversation_decision import OperationKind, ResponseDisposition
from models.conversation_decision import ConversationDecision
from models.conversational_core import ConversationalWorkingState
from models.live_orchestration import EvidenceSnapshot, TurnTrigger
from models.model_runtime import ModelResponseDiagnostics, ModelTarget, ModelUsage
from models.schemas import Persona
from services import live_orchestration

OWNER_TARGET = ModelTarget(
    name="test-glm",
    provider="openrouter",
    model="z-ai/glm-5.3-flash",
    metadata={"reasoning_enabled": True},
)
KIMI_TARGET = ModelTarget(
    name="test-kimi",
    provider="openrouter",
    model="moonshotai/kimi-k2.6",
)


def snapshot() -> EvidenceSnapshot:
    return EvidenceSnapshot(
        creator_id="creator-1",
        fan_id="fan-1",
        trigger=TurnTrigger(
            kind="fan_message", identity="msg-1", latest_message="mm"
        ),
        state_revision="revision-1",
        latest_fan_burst=(
            {"message_id": "msg-1", "speaker": "fan", "text": "mm", "at": None},
        ),
        recent_messages=(
            {
                "message_id": "c-1",
                "speaker": "creator",
                "text": "the rain just started",
                "at": None,
            },
            {"message_id": "msg-1", "speaker": "fan", "text": "mm", "at": None},
        ),
        creator_voice={"communication_style": "lowercase, dry, playful"},
    )


def loaded():
    owner_spec = SimpleNamespace(
        primary_target=lambda: OWNER_TARGET,
        fallback_target=lambda: None,
        resolved_max_tokens=lambda: 2048,
        prompt_version="conversational_decision_v1",
    )
    writer_spec = SimpleNamespace(
        primary_target=lambda: KIMI_TARGET,
        fallback_target=lambda: None,
        prompt_version="conversational_writer_v1",
    )
    return SimpleNamespace(
        snapshot=snapshot(),
        packet=None,
        history=[],
        fan=SimpleNamespace(id="fan-1", needs_human_review=False, auto_mode=True),
        persona=Persona(),
        commercial_state=SimpleNamespace(
            pending_offer=None, last_offer_at=None, desired_experience=None
        ),
        policy=SimpleNamespace(
            require_operator_ppv_approval=False, pending_offer_expiry_hours=24
        ),
        next_offer=None,
        active_session=None,
        pending_payment=None,
        sent_ppv=[],
        within_daily_caps=True,
        candidate_handles={},
        hermes_examples=[],
        stack=SimpleNamespace(
            profile_id="cleo_v3",
            profile=SimpleNamespace(
                stage=lambda name: writer_spec
                if name == live_orchestration.STAGE_CONVERSATIONAL_WRITER
                else owner_spec
            ),
        ),
    )


def decision_payload(**overrides) -> str:
    payload = {
        "turn_id": "msg-1",
        "conversation_revision": "revision-1",
        "disposition": "reply",
        "response_goal": "continue the rain scene after the fan's acknowledgement",
        "contribution_goal": "take creator initiative",
        "initiative": "creator",
        "pacing": "continue",
        "operation_proposal": {"kind": "none"},
        "state_delta": {"initiative_holder": "creator"},
        "confidence": 0.8,
    }
    payload.update(overrides)
    return json.dumps(payload)


def owner_response(text: str, *, latency_ms: int = 10):
    return SimpleNamespace(
        text=text,
        target=OWNER_TARGET,
        served_model="z-ai/glm-5.3-flash",
        upstream_provider="z-ai",
        latency_ms=latency_ms,
        usage=ModelUsage(input_tokens=100, output_tokens=20),
        reported_cost_usd=0.001,
        diagnostics=ModelResponseDiagnostics(
            provider="openrouter",
            model=OWNER_TARGET.model,
            served_model=OWNER_TARGET.model,
            content_chars=len(text),
            latency_ms=latency_ms,
        ),
    )


def scripted(*responses):
    calls = []

    async def complete(_target, **kwargs):
        calls.append(kwargs)
        return responses[len(calls) - 1]

    complete.calls = calls
    return complete


@pytest.mark.asyncio
async def test_glm_never_returns_fan_facing_copy():
    transport = scripted(owner_response(decision_payload()))
    decision, replies, trace, delta = await live_orchestration.decide_conversational_v1(
        loaded(), ConversationalWorkingState(), owner_complete=transport
    )
    assert replies == []
    assert decision.response_goal.startswith("continue the rain")
    assert delta == {"initiative_holder": "creator"}
    assert trace.role == "conversational_decision"
    assert len(transport.calls) == 1


@pytest.mark.asyncio
async def test_glm_copy_field_is_rejected_then_repaired_once():
    transport = scripted(
        owner_response(decision_payload(reply="forbidden copy")),
        owner_response(decision_payload()),
    )
    decision, replies, trace, _delta = await live_orchestration.decide_conversational_v1(
        loaded(), ConversationalWorkingState(), owner_complete=transport
    )
    assert decision.disposition is ResponseDisposition.REPLY
    assert replies == []
    assert len(transport.calls) == 2
    assert trace.repair_attempted is True
    assert trace.repaired is True


@pytest.mark.asyncio
async def test_malformed_glm_decision_fails_after_one_repair():
    transport = scripted(owner_response("not json"), owner_response("still not json"))
    decision, replies, trace, _delta = await live_orchestration.decide_conversational_v1(
        loaded(), ConversationalWorkingState(), owner_complete=transport
    )
    assert decision.disposition is ResponseDisposition.SILENCE
    assert replies == []
    assert len(transport.calls) == 2
    assert trace.failure_reason.startswith("the conversational owner did not answer")


@pytest.mark.asyncio
async def test_kimi_is_the_only_writer_and_receives_no_model_fallback(monkeypatch):
    observed = {}

    async def generate(prompt, _persona, **kwargs):
        observed["prompt"] = prompt
        observed["target"] = kwargs["target_override"]
        observed["fallback"] = kwargs["fallback_target_override"]
        trace = kwargs["trace"]
        trace.record_request(
            primary_target=KIMI_TARGET,
            fallback_target=None,
            profile="cleo_v3",
            policy="test",
            deadline_seconds=1,
        )
        trace.record_success(
            target=KIMI_TARGET,
            role="pinned",
            attempt_index=1,
            upstream_provider="moonshot",
            outcome="success",
            attempts=1,
            pinned_attempts=1,
            alternate_attempts=0,
            elapsed_ms=5,
            served_model=KIMI_TARGET.model,
        )
        return ["stay under the awning a little longer"]

    monkeypatch.setattr(live_orchestration, "generate_replies", generate)
    decision = live_orchestration.parse_semantic_decision(decision_payload()).decision
    replies, trace = await live_orchestration._write_conversational_v1_turn(
        loaded(),
        decision,
        live_orchestration.ApprovedExecution(),
        ConversationalWorkingState(),
        mode="auto",
    )
    assert replies == ["stay under the awning a little longer"]
    assert observed["target"].model == "moonshotai/kimi-k2.6"
    assert observed["fallback"] is None
    assert trace.role == "fan_facing_writer"
    prompt_text = json.loads(observed["prompt"][1]["content"])
    assert (
        prompt_text["raw_conversation"]["latest_fan_message_burst"][0]["text"]
        == "mm"
    )
    assert prompt_text["semantic_decision"]["intimacy_context"] == {
        "active": False,
        "content_register": "none",
        "scene_mode": "none",
        "direction": "continue",
        "last_beat": "",
        "boundaries": [],
    }


@pytest.mark.asyncio
async def test_kimi_failure_never_calls_glm_as_writer(monkeypatch):
    calls = []

    async def generate(_prompt, _persona, **kwargs):
        calls.append(kwargs["target_override"].model)
        kwargs["trace"].failure_reason = "writer unavailable"
        return []

    monkeypatch.setattr(live_orchestration, "generate_replies", generate)
    decision = live_orchestration.parse_semantic_decision(decision_payload()).decision
    replies, trace = await live_orchestration._write_conversational_v1_turn(
        loaded(),
        decision,
        live_orchestration.ApprovedExecution(),
        ConversationalWorkingState(),
        mode="auto",
    )
    assert replies == []
    assert calls and set(calls) == {"moonshotai/kimi-k2.6"}
    assert "glm" not in " ".join(calls).lower()
    assert trace.failure_reason


@pytest.mark.asyncio
async def test_non_kimi_served_identity_is_suppressed(monkeypatch):
    async def generate(_prompt, _persona, **kwargs):
        trace = kwargs["trace"]
        trace.record_request(
            primary_target=KIMI_TARGET,
            fallback_target=None,
            profile="cleo_v3",
            policy="test",
            deadline_seconds=1,
        )
        trace.record_success(
            target=KIMI_TARGET,
            role="pinned",
            attempt_index=1,
            upstream_provider="router",
            outcome="success",
            attempts=1,
            pinned_attempts=1,
            alternate_attempts=0,
            elapsed_ms=5,
            served_model="other/model",
        )
        return ["unsafe routed copy"]

    monkeypatch.setattr(live_orchestration, "generate_replies", generate)
    decision = live_orchestration.parse_semantic_decision(decision_payload()).decision
    replies, trace = await live_orchestration._write_conversational_v1_turn(
        loaded(),
        decision,
        live_orchestration.ApprovedExecution(),
        ConversationalWorkingState(),
        mode="auto",
    )
    assert replies == []
    assert "routing_mismatch" in trace.failure_reason


@pytest.mark.asyncio
async def test_bad_operation_does_not_erase_kimi_reply():
    evidence = loaded()
    decision = live_orchestration.parse_semantic_decision(
        decision_payload(
            operation_proposal={
                "kind": "present_offer",
                "subject": "the requested photos",
                "candidate_handle": "missing",
            }
        )
    ).decision
    authorized = await live_orchestration._authorize_conversational_v1_operation(
        evidence, decision=decision, execute_operations=False
    )
    settlement = await live_orchestration.settle_conversational_v1_turn(
        evidence,
        decision=authorized.decision,
        replies=["i'm still right here with you"],
        mode="auto",
        execute_operations=False,
    )
    assert settlement.execution.operation == "none"
    assert settlement.replies == ["i'm still right here with you"]
    assert authorized.proposed_operation == OperationKind.PRESENT_OFFER.value


@pytest.mark.asyncio
async def test_missing_evidence_is_refreshed_in_the_same_fan_turn(monkeypatch):
    first = loaded()
    second = loaded()
    second.snapshot = EvidenceSnapshot(
        **{
            **second.snapshot.__dict__,
            "approved_inventory": (
                {"candidate_handle": "offer_candidate_1", "asset_type": "photo"},
            ),
        }
    )
    loads = [first, second]

    async def load_evidence(**_kwargs):
        return loads.pop(0)

    decisions = [
        ConversationDecision(
            response_goal="decide after inventory is refreshed",
            evidence_requests=("inventory",),
            source="conversational_decision_v1",
        ),
        ConversationDecision(
            response_goal="continue naturally with the refreshed facts",
            source="conversational_decision_v1",
        ),
    ]

    async def decide(_loaded, _state):
        return decisions.pop(0), [], live_orchestration.GenerationTrace(), {}

    async def write(*_args, **_kwargs):
        return ["now i know exactly which moment you mean"], live_orchestration.GenerationTrace()

    async def state(*_args, **_kwargs):
        return ConversationalWorkingState()

    monkeypatch.setattr(live_orchestration, "load_evidence", load_evidence)
    monkeypatch.setattr(live_orchestration, "load_working_state", state)
    monkeypatch.setattr(live_orchestration, "decide_conversational_v1", decide)
    monkeypatch.setattr(live_orchestration, "_write_conversational_v1_turn", write)

    prepared = await live_orchestration.prepare_turn(
        creator_id="creator-1",
        fan_id="fan-1",
        trigger_kind="fan_message",
        trigger_identity="msg-1",
        latest_message="mm",
        execute_operations=False,
        conversation_core="conversational_v1",
    )
    assert prepared.replies == ["now i know exactly which moment you mean"]
    assert prepared.decision.response_goal.startswith("continue naturally")
    assert loads == []



@pytest.mark.asyncio
async def test_intimate_context_reaches_kimi_and_survives_operation_rejection(monkeypatch):
    observed = {}

    async def generate(prompt, _persona, **kwargs):
        observed["payload"] = json.loads(prompt[1]["content"])
        trace = kwargs["trace"]
        trace.record_request(
            primary_target=KIMI_TARGET,
            fallback_target=None,
            profile="cleo_v3",
            policy="test",
            deadline_seconds=1,
        )
        trace.record_success(
            target=KIMI_TARGET,
            role="pinned",
            attempt_index=0,
            upstream_provider="moonshot",
            outcome="success",
            attempts=1,
            pinned_attempts=1,
            alternate_attempts=0,
            elapsed_ms=5,
            served_model=KIMI_TARGET.model,
        )
        return ["keep the same moment going"]

    monkeypatch.setattr(live_orchestration, "generate_replies", generate)
    evidence = loaded()
    decision = live_orchestration.parse_semantic_decision(
        decision_payload(
            intimacy_context={
                "active": True,
                "content_register": "explicit",
                "scene_mode": "shared_imagined",
                "direction": "hold",
                "last_beat": "stay with the same shared premise",
                "boundaries": ["do not rush"],
            },
            operation_proposal={
                "kind": "present_offer",
                "subject": "the requested content",
                "candidate_handle": "missing",
            },
        )
    ).decision

    authorized = await live_orchestration._authorize_conversational_v1_operation(
        evidence, decision=decision, execute_operations=False
    )
    assert authorized.execution.operation == "none"
    assert authorized.decision.intimacy_context == decision.intimacy_context

    replies, _trace = await live_orchestration._write_conversational_v1_turn(
        evidence,
        authorized.decision,
        authorized.execution,
        ConversationalWorkingState(),
        mode="auto",
    )

    assert replies == ["keep the same moment going"]
    assert observed["payload"]["semantic_decision"]["intimacy_context"] == {
        "active": True,
        "content_register": "explicit",
        "scene_mode": "shared_imagined",
        "direction": "hold",
        "last_beat": "stay with the same shared premise",
        "boundaries": ["do not rush"],
    }
