"""The Conversational Core v1 failure model: what must NOT kill a turn.

Every test here is a production failure this runtime has already had. The shape
of the suite is the shape of the rule it enforces:

    A recoverable model-format, state, or operation mistake must not kill an
    otherwise valid conversation.

so the assertions are almost always "the reply still went out, and the thing
that was wrong was recorded as rejected". The few that assert a failure assert
that it is CLEAN — a named category, no legacy fallback, and no second repair.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from models.conversation_decision import (
    HoldReason,
    OperationKind,
    ResponseDisposition,
)
from models.conversational_core import ConversationalWorkingState
from models.live_orchestration import EvidenceFact, EvidenceSnapshot, TurnTrigger
from models.model_runtime import (
    FAILURE_EMPTY_TRUNCATED,
    FAILURE_NO_JSON,
    FAILURE_PROVIDER_ERROR,
    FAILURE_TIMEOUT,
    ModelResponseDiagnostics,
    ModelTarget,
    ModelUsage,
)
from services import conversational_core, live_orchestration, owner_contract


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


OWNER_TARGET = ModelTarget(
    name="test-owner",
    provider="openrouter",
    model="owner-model",
    metadata={"reasoning_enabled": True, "reasoning_effort": "low"},
    timeout_seconds=60.0,
)


def snapshot(
    identity: str = "msg-1",
    *,
    latest: str = "i keep thinking about that unfinished story",
    pending_payment: dict | None = None,
    purchases: tuple[dict, ...] = (),
) -> EvidenceSnapshot:
    return EvidenceSnapshot(
        creator_id="creator-1",
        fan_id="fan-1",
        trigger=TurnTrigger(
            kind="fan_message", identity=identity, latest_message=latest
        ),
        state_revision="authoritative-revision",
        creator_facts=(
            EvidenceFact(
                value="favourite season: late autumn",
                source_ref="creator_legend:favourite_season",
                certainty="creator_confirmed",
            ),
        ),
        pending_payment=pending_payment,
        confirmed_purchases=purchases,
    )


def loaded_evidence(snap: EvidenceSnapshot | None = None, *, max_tokens: int = 8192):
    """A LoadedEvidence-shaped stand-in with no sellable offer and no payment.

    Barren on purpose: every operation the owner proposes here is refused by the
    real validator, which is the case these tests are about.
    """
    spec = SimpleNamespace(
        primary_target=lambda: OWNER_TARGET,
        fallback_target=lambda: None,
        max_tokens=max_tokens,
        resolved_max_tokens=lambda: max_tokens,
    )
    return SimpleNamespace(
        snapshot=snap or snapshot(),
        packet=None,
        history=[],
        fan=SimpleNamespace(id="fan-1", needs_human_review=False, auto_mode=True),
        persona=None,
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
        stack=SimpleNamespace(
            profile_id="cleo_v3",
            profile=SimpleNamespace(stage=lambda _name: spec),
        ),
    )


def owner_response(
    text: str,
    *,
    latency_ms: int = 900,
    finish_reason: str = "stop",
    content_is_null: bool = False,
    reasoning_tokens: int = 220,
    completion_tokens: int = 500,
):
    return SimpleNamespace(
        text=text,
        target=OWNER_TARGET,
        usage=ModelUsage(input_tokens=1_000, output_tokens=completion_tokens),
        latency_ms=latency_ms,
        raw_response_id="gen-test",
        upstream_provider="z-ai",
        reported_cost_usd=0.0001,
        diagnostics=ModelResponseDiagnostics(
            provider="openrouter",
            model="owner-model",
            upstream_provider="z-ai",
            response_id="gen-test",
            latency_ms=latency_ms,
            response_format_requested="json_object",
            reasoning_requested="on,max_tokens=1024",
            max_tokens_requested=8192,
            choice_count=1,
            finish_reason=finish_reason,
            message_fields=(("reasoning",) if content_is_null else ("content", "reasoning")),
            content_is_null=content_is_null,
            content_chars=len(text),
            reasoning_present=True,
            reasoning_chars=reasoning_tokens * 4,
            prompt_tokens=1_000,
            completion_tokens=completion_tokens,
            reasoning_tokens=reasoning_tokens,
        ),
    )


def scripted_owner(*responses):
    """A transport that returns each response in turn and counts its calls."""
    calls: list[dict] = []

    async def complete(target, **kwargs):
        index = min(len(calls), len(responses) - 1)
        calls.append({"target": target, **kwargs})
        return responses[index]

    complete.calls = calls  # type: ignore[attr-defined]
    return complete


REPLY = "that story stopping mid-sentence has been bothering me all week"


def valid_payload(**overrides):
    payload = {
        "reply": REPLY,
        "response_intent": "ordinary_conversation",
        "disposition": "reply",
        "operation": "none",
        "hold": "none",
        "confidence": 0.7,
    }
    payload.update(overrides)
    return json.dumps(payload, ensure_ascii=False)


async def settle(loaded, decision, replies):
    return await live_orchestration.settle_conversational_v1_turn(
        loaded,
        decision=decision,
        replies=replies,
        mode="auto",
        execute_operations=False,
    )


# ---------------------------------------------------------------------------
# 1. valid reply + invalid operation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_invalid_operation_does_not_erase_a_good_reply():
    loaded = loaded_evidence()
    owner = scripted_owner(
        owner_response(
            valid_payload(
                operation="present_offer",
                operation_subject="the set they asked about",
                operation_offer_id="offer-that-does-not-exist",
                operation_set_id="set-that-does-not-exist",
            )
        )
    )
    decision, replies, trace, _delta = await live_orchestration.decide_conversational_v1(
        loaded, ConversationalWorkingState(), owner_complete=owner
    )
    settlement = await settle(loaded, decision, replies)

    assert settlement.replies == [REPLY]
    assert settlement.decision.disposition is ResponseDisposition.REPLY
    assert settlement.proposed_operation == "present_offer"
    assert settlement.operation_rejected is True
    assert settlement.execution.operation == "none"
    assert trace.failure_reason == ""


@pytest.mark.asyncio
async def test_price_text_in_operation_prose_is_repaired_not_silenced():
    """PR #65's case: a rejected proposal used to convert the whole turn to no_send."""
    loaded = loaded_evidence()
    owner = scripted_owner(
        owner_response(
            valid_payload(
                operation="check_payment_claim",
                operation_subject="the $24 set they mentioned",
                operation_because="they said $24 was fine",
                operation_payment_reference="pay-1",
            )
        )
    )
    decision, replies, _trace, _delta = await live_orchestration.decide_conversational_v1(
        loaded, ConversationalWorkingState(), owner_complete=owner
    )
    settlement = await settle(loaded, decision, replies)

    assert settlement.replies == [REPLY]
    assert settlement.execution.operation == "none"
    assert settlement.locally_repaired is True


@pytest.mark.asyncio
async def test_an_unreadable_confidence_disqualifies_the_operation_only():
    """A number the parser had to invent may never carry an external effect."""
    result = owner_contract.extract_owner_result(
        valid_payload(
            confidence=1.8,
            operation="check_payment_claim",
            operation_payment_reference="pay-1",
        ),
        source="conversational_owner_v1",
    )
    assert result.usable is True
    assert result.reply == REPLY
    assert result.decision.confidence == 0.0
    assert result.decision.proposed_operation.kind is OperationKind.NONE
    assert "confidence" in result.operation_discarded


# ---------------------------------------------------------------------------
# 2. valid reply + invalid state delta
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_malformed_state_delta_does_not_erase_a_good_reply():
    loaded = loaded_evidence()
    owner = scripted_owner(
        owner_response(valid_payload(state_delta="not an object at all"))
    )
    decision, replies, trace, raw_delta = (
        await live_orchestration.decide_conversational_v1(
            loaded, ConversationalWorkingState(), owner_complete=owner
        )
    )
    validation = conversational_core.validate_and_apply_delta(
        ConversationalWorkingState(),
        raw_delta,
        snapshot=loaded.snapshot,
    )

    assert replies == [REPLY]
    assert trace.failure_reason == ""
    assert "state_delta" in validation.rejected_fields
    assert validation.state_after.revision == 0


@pytest.mark.asyncio
async def test_an_unevidenced_state_element_is_rejected_without_touching_the_reply():
    loaded = loaded_evidence()
    owner = scripted_owner(
        owner_response(
            valid_payload(
                state_delta={
                    "current_focus": "the unfinished story",
                    "add_established_elements": [
                        {
                            "element_id": "invented",
                            "claim": "they said they live in Lisbon",
                            "source_type": "explicit_fan_statement",
                            "source_refs": ["message-that-does-not-exist"],
                            "world_scope": "conversation",
                        }
                    ],
                }
            )
        )
    )
    _decision, replies, _trace, raw_delta = (
        await live_orchestration.decide_conversational_v1(
            loaded, ConversationalWorkingState(), owner_complete=owner
        )
    )
    validation = conversational_core.validate_and_apply_delta(
        ConversationalWorkingState(),
        raw_delta,
        snapshot=loaded.snapshot,
    )

    assert replies == [REPLY]
    assert "current_focus" in validation.accepted_fields
    assert "unknown evidence reference" in validation.rejected_fields[
        "add_established_elements[0]"
    ]


@pytest.mark.asyncio
async def test_a_state_delta_that_cannot_be_read_at_all_never_reaches_the_reply(
    monkeypatch,
):
    """Even an unexpected exception in the delta validator is not a lost turn."""

    def explode(*_args, **_kwargs):
        raise RuntimeError("delta validator met a shape it has never seen")

    monkeypatch.setattr(live_orchestration, "validate_and_apply_delta", explode)
    monkeypatch.setattr(
        live_orchestration,
        "load_working_state",
        _async_value(ConversationalWorkingState()),
    )
    loaded = loaded_evidence()
    monkeypatch.setattr(live_orchestration, "load_evidence", _async_value(loaded))
    monkeypatch.setattr(
        live_orchestration,
        "decide_conversational_v1",
        _async_value(
            (
                owner_contract.extract_owner_result(
                    valid_payload(), source="conversational_owner_v1"
                ).decision,
                [REPLY],
                live_orchestration.GenerationTrace(),
                {"current_focus": "the unfinished story"},
            )
        ),
    )
    prepared = await live_orchestration.prepare_turn(
        creator_id="creator-1",
        fan_id="fan-1",
        trigger_kind="fan_message",
        trigger_identity="msg-1",
        latest_message="hi",
        execute_operations=False,
        conversation_core="conversational_v1",
    )
    assert prepared.replies == [REPLY]
    assert "state_delta" in prepared.state_delta_validation.rejected_fields


def _async_value(value):
    async def _call(*_args, **_kwargs):
        return value

    return _call


# ---------------------------------------------------------------------------
# 3 & 4. malformed initial response, with and without a working repair
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_malformed_json_is_repaired_in_exactly_one_extra_call():
    loaded = loaded_evidence()
    owner = scripted_owner(
        owner_response("I think the best move is to keep the scene going."),
        owner_response(valid_payload()),
    )
    decision, replies, trace, _delta = await live_orchestration.decide_conversational_v1(
        loaded, ConversationalWorkingState(), owner_complete=owner
    )

    assert len(owner.calls) == 2
    assert replies == [REPLY]
    assert decision.disposition is ResponseDisposition.REPLY
    assert trace.repair_attempted is True
    assert trace.repaired is True
    assert trace.outcome == "conversational_v1_repair_success"
    assert trace.failure_reason == ""
    assert trace.attempts == 2
    assert [row["attempt"] for row in trace.owner_attempts] == ["initial", "repair"]


@pytest.mark.asyncio
async def test_a_failed_repair_fails_clearly_and_is_never_retried_again():
    loaded = loaded_evidence()
    owner = scripted_owner(
        owner_response("no object here"),
        owner_response("still no object here"),
    )
    decision, replies, trace, _delta = await live_orchestration.decide_conversational_v1(
        loaded, ConversationalWorkingState(), owner_complete=owner
    )
    settlement = await settle(loaded, decision, replies)

    assert len(owner.calls) == 2, "the repair budget is exactly one extra call"
    assert replies == []
    assert trace.repair_attempted is True
    assert trace.repaired is False
    assert trace.outcome == "conversational_v1_owner_invalid"
    assert FAILURE_NO_JSON in trace.failure_reason
    # No legacy controller, no writer, no pretend reply.
    assert settlement.replies == []
    assert settlement.execution.operation == "none"
    assert decision.hold is HoldReason.INSUFFICIENT_EVIDENCE


@pytest.mark.asyncio
async def test_an_owner_failure_is_reported_as_owner_failed_and_sends_nothing(
    monkeypatch,
):
    loaded = loaded_evidence()
    trace = live_orchestration.GenerationTrace()
    trace.record_attempt(
        label="initial",
        diagnostics=ModelResponseDiagnostics(
            model="owner-model", finish_reason="length", content_chars=0
        ),
        extra={"failure_category": FAILURE_EMPTY_TRUNCATED, "usable": False},
    )
    trace.record_failure(
        outcome="conversational_v1_owner_invalid",
        reason="the conversational owner did not answer usably after one repair",
        attempts=2,
        pinned_attempts=2,
        alternate_attempts=0,
        elapsed_ms=4_000,
        deadline_exceeded=False,
    )
    prepared = live_orchestration.PreparedTurn(
        loaded=loaded,
        decision=owner_contract.extract_owner_result(
            valid_payload(reply="", disposition="silence", hold="respect_silence"),
            source="conversational_owner_v1",
        ).decision,
        execution=live_orchestration.ApprovedExecution(),
        replies=[],
        provenance=SimpleNamespace(),
        writer_trace=trace,
        conversation_core="conversational_v1",
    )
    result = await live_orchestration.execute_auto_turn(prepared)

    assert result["outcome"] == live_orchestration.OUTCOME_OWNER_FAILED
    assert result["message_ids"] == []
    assert result["owner_failure_categories"] == [FAILURE_EMPTY_TRUNCATED]
    assert result["owner_attempts"][0]["finish_reason"] == "length"


# ---------------------------------------------------------------------------
# 5 & 6. transport diagnostics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_content_is_distinguishable_from_malformed_json():
    loaded = loaded_evidence()
    owner = scripted_owner(
        owner_response(
            "",
            finish_reason="length",
            content_is_null=True,
            reasoning_tokens=8_100,
            completion_tokens=8_192,
            latency_ms=97_800,
        ),
        owner_response("", finish_reason="length", content_is_null=True),
    )
    _decision, _replies, trace, _delta = (
        await live_orchestration.decide_conversational_v1(
            loaded, ConversationalWorkingState(), owner_complete=owner
        )
    )

    first = trace.owner_attempts[0]
    assert first["failure_category"] == FAILURE_EMPTY_TRUNCATED
    assert first["content_is_null"] is True
    assert first["content_chars"] == 0
    assert first["reasoning_tokens"] == 8_100
    assert first["finish_reason"] == "length"
    assert first["truncated"] is True
    assert first["latency_ms"] == 97_800
    assert first["response_id"] == "gen-test"
    assert first["max_tokens_requested"] == 8_192
    assert first["reasoning_requested"] == "on,max_tokens=1024"
    # The old message said only this; it must no longer be the whole story.
    assert "no JSON object" not in trace.failure_reason


def test_diagnostics_never_carry_conversation_text():
    diagnostics = ModelResponseDiagnostics(
        provider="openrouter",
        model="owner-model",
        content_chars=4_212,
        reasoning_chars=9_000,
        finish_reason="length",
        message_fields=("content", "reasoning"),
    )
    rendered = json.dumps(diagnostics.as_dict()) + diagnostics.describe()
    for forbidden in (REPLY, "unfinished story", "fan-1", "creator-1"):
        assert forbidden not in rendered


def test_an_openrouter_reasoning_response_is_read_without_raising():
    """Reasoning, usage details and finish reason are all optional shapes."""
    from ai import model_providers

    message = SimpleNamespace(
        content=None,
        reasoning="a long private trace",
        reasoning_details=[{"text": "more"}],
        refusal=None,
        tool_calls=None,
    )
    assert model_providers._reasoning_text(message) == "a long private trace"
    assert model_providers._populated_message_fields(message) == (
        "reasoning",
        "reasoning_details",
    )
    assert model_providers._text_of([{"text": "a"}, "b"]) == "ab"
    assert model_providers._response_error(
        SimpleNamespace(error={"message": "no allowed provider"}), None
    ) == "no allowed provider"


@pytest.mark.asyncio
async def test_a_transport_failure_is_categorised_rather_than_swallowed():
    loaded = loaded_evidence()

    class Timeout(Exception):
        pass

    Timeout.__name__ = "APITimeoutError"

    async def failing(_target, **_kwargs):
        raise Timeout("request timed out")

    _decision, replies, trace, _delta = (
        await live_orchestration.decide_conversational_v1(
            loaded, ConversationalWorkingState(), owner_complete=failing
        )
    )
    assert replies == []
    assert trace.owner_attempts[0]["failure_category"] == FAILURE_TIMEOUT
    assert trace.deadline_exceeded is True
    assert FAILURE_TIMEOUT in trace.failure_reason


def test_transport_errors_are_named_by_category():
    from ai.model_providers import classify_transport_error

    class APITimeoutError(Exception):
        pass

    class APIStatusError(Exception):
        pass

    assert classify_transport_error(APITimeoutError("x")) == FAILURE_TIMEOUT
    assert classify_transport_error(APIStatusError("x")) == FAILURE_PROVIDER_ERROR
    assert classify_transport_error(ValueError("x")) != FAILURE_TIMEOUT


# ---------------------------------------------------------------------------
# 7 & 8. what a repair may not do
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_repair_cannot_invent_an_authoritative_reference():
    loaded = loaded_evidence(
        snapshot(pending_payment={"reference": "pay-real"}),
    )
    loaded.pending_payment = {"reference": "pay-real"}
    owner = scripted_owner(
        owner_response("prose, no object"),
        owner_response(
            valid_payload(
                operation="check_payment_claim",
                operation_subject="their payment",
                operation_payment_reference="pay-invented",
            )
        ),
    )
    decision, replies, trace, _delta = await live_orchestration.decide_conversational_v1(
        loaded, ConversationalWorkingState(), owner_complete=owner
    )

    assert replies == [REPLY], "the repaired reply still goes out"
    assert decision.proposed_operation.kind is OperationKind.NONE
    assert decision.proposed_operation.payment_reference == ""
    assert "absent from the evidence" in trace.owner_attempts[1]["operation_discarded"]


@pytest.mark.asyncio
async def test_a_repair_may_restate_a_reference_the_evidence_contains():
    loaded = loaded_evidence(snapshot(pending_payment={"reference": "pay-real"}))
    loaded.pending_payment = {"reference": "pay-real"}
    owner = scripted_owner(
        owner_response("prose, no object"),
        owner_response(
            valid_payload(
                operation="check_payment_claim",
                operation_subject="their payment",
                operation_payment_reference="pay-real",
            )
        ),
    )
    decision, _replies, _trace, _delta = (
        await live_orchestration.decide_conversational_v1(
            loaded, ConversationalWorkingState(), owner_complete=owner
        )
    )
    assert decision.proposed_operation.kind is OperationKind.CHECK_PAYMENT_CLAIM
    assert decision.proposed_operation.payment_reference == "pay-real"


@pytest.mark.asyncio
async def test_a_repair_prepares_the_commercial_operation_exactly_once():
    """A repaired turn must not settle twice and execute twice."""
    loaded = loaded_evidence(snapshot(pending_payment={"reference": "pay-real"}))
    loaded.pending_payment = {"reference": "pay-real"}
    prepared_operations: list[str] = []

    real_prepare = live_orchestration._prepare_execution

    async def counting_prepare(decision, evidence, **kwargs):
        prepared_operations.append(decision.proposed_operation.kind.value)
        return await real_prepare(decision, evidence, **kwargs)

    owner = scripted_owner(
        owner_response("prose, no object"),
        owner_response(
            valid_payload(
                operation="check_payment_claim",
                operation_subject="their payment",
                operation_payment_reference="pay-real",
            )
        ),
    )
    decision, replies, _trace, _delta = (
        await live_orchestration.decide_conversational_v1(
            loaded, ConversationalWorkingState(), owner_complete=owner
        )
    )
    assert prepared_operations == [], "the owner boundary never executes anything"

    live_orchestration._prepare_execution = counting_prepare
    try:
        settlement = await settle(loaded, decision, replies)
    finally:
        live_orchestration._prepare_execution = real_prepare

    assert settlement.execution.operation == "check_payment_claim"
    assert prepared_operations == ["check_payment_claim"], (
        "an approved operation is prepared once, not once per owner call"
    )


# ---------------------------------------------------------------------------
# 9 & 10. the ordinary turn, and honest accounting
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_ordinary_turn_uses_exactly_one_owner_call():
    loaded = loaded_evidence()
    owner = scripted_owner(
        owner_response(
            valid_payload(state_delta={"current_focus": "the unfinished story"}),
            latency_ms=1_450,
        )
    )
    decision, replies, trace, raw_delta = (
        await live_orchestration.decide_conversational_v1(
            loaded, ConversationalWorkingState(), owner_complete=owner
        )
    )

    assert len(owner.calls) == 1
    assert replies == [REPLY]
    assert decision.source == "conversational_owner_v1"
    assert raw_delta == {"current_focus": "the unfinished story"}
    assert trace.repair_attempted is False
    assert trace.repaired is False
    assert trace.outcome == "conversational_v1_first_try_success"
    assert trace.attempts == 1
    assert trace.elapsed_ms == 1_450
    assert trace.as_metadata().get("repair_attempted") is None


@pytest.mark.asyncio
async def test_latency_and_attempts_accumulate_across_a_repair():
    loaded = loaded_evidence()
    owner = scripted_owner(
        owner_response("prose", latency_ms=41_000),
        owner_response(valid_payload(), latency_ms=1_900),
    )
    _decision, _replies, trace, _delta = (
        await live_orchestration.decide_conversational_v1(
            loaded, ConversationalWorkingState(), owner_complete=owner
        )
    )
    assert trace.attempts == 2
    assert trace.pinned_attempts == 2
    assert trace.elapsed_ms == 42_900, "a repair's cost is not hidden"
    metadata = trace.as_metadata()
    assert metadata["attempts"] == 2
    assert metadata["repair_attempted"] is True
    assert metadata["repaired"] is True
    assert [row["attempt"] for row in metadata["owner_attempts"]] == [
        "initial",
        "repair",
    ]


@pytest.mark.asyncio
async def test_a_failed_repair_reports_both_attempts_latency():
    loaded = loaded_evidence()
    owner = scripted_owner(
        owner_response("prose", latency_ms=5_000),
        owner_response("prose again", latency_ms=3_000),
    )
    _decision, _replies, trace, _delta = (
        await live_orchestration.decide_conversational_v1(
            loaded, ConversationalWorkingState(), owner_complete=owner
        )
    )
    assert trace.attempts == 2
    assert trace.elapsed_ms == 8_000


# ---------------------------------------------------------------------------
# Component reads that must never be fatal
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        valid_payload(response_intent="vibe_check"),
        valid_payload(hold="thinking_about_it"),
        valid_payload(active_needs="not a list"),
        valid_payload(must_address={"nope": 1}),
        valid_payload(hold_detail=17),
        json.dumps({"reply": REPLY}),
        "```json\n" + valid_payload() + "\n```",
        "Here you go:\n" + valid_payload(),
    ],
)
def test_malformed_optional_metadata_never_erases_the_reply(payload):
    result = owner_contract.extract_owner_result(
        payload, source="conversational_owner_v1"
    )
    assert result.usable is True
    assert result.reply == REPLY
    assert result.decision.disposition is ResponseDisposition.REPLY


def test_json_truncated_inside_the_state_delta_still_yields_the_reply():
    full = valid_payload(state_delta={"current_focus": "the unfinished story"})
    truncated = full[: full.index('"state_delta"') + 24]
    result = owner_contract.extract_owner_result(
        truncated, source="conversational_owner_v1"
    )
    assert result.usable is True
    assert result.reply == REPLY
    assert result.json_status == owner_contract.JSON_RECOVERED


def test_json_that_never_closes_still_yields_the_reply():
    result = owner_contract.extract_owner_result(
        '{"reply": "' + REPLY, source="conversational_owner_v1"
    )
    assert result.usable is True
    assert result.reply == REPLY
    assert result.json_status in {
        owner_contract.JSON_RECOVERED,
        owner_contract.JSON_REPLY_ONLY,
    }
    assert result.decision.proposed_operation.kind is OperationKind.NONE


def test_an_absent_reply_is_a_failure_rather_than_a_convenient_silence():
    """A response that simply stopped must not become a considered silence."""
    result = owner_contract.extract_owner_result(
        json.dumps({"active_needs": [], "operation": "none"}),
        source="conversational_owner_v1",
    )
    assert result.usable is False
    assert result.failure_category
    assert result.decision is None


def test_a_stated_silence_is_honoured():
    result = owner_contract.extract_owner_result(
        json.dumps(
            {"reply": "", "disposition": "silence", "hold": "respect_silence"}
        ),
        source="conversational_owner_v1",
    )
    assert result.usable is True
    assert result.reply == ""
    assert result.decision.disposition is ResponseDisposition.SILENCE
    assert result.decision.hold is HoldReason.RESPECT_SILENCE


def test_a_silent_turn_cannot_smuggle_an_operation():
    result = owner_contract.extract_owner_result(
        json.dumps(
            {
                "reply": "",
                "disposition": "silence",
                "hold": "respect_silence",
                "operation": "send_locked_paid_message",
                "operation_subject": "the set",
                "confidence": 0.9,
            }
        ),
        source="conversational_owner_v1",
    )
    assert result.decision.proposed_operation.kind is OperationKind.NONE
    assert result.operation_discarded


def test_a_handoff_never_becomes_an_ordinary_reply():
    result = owner_contract.extract_owner_result(
        json.dumps(
            {
                "reply": "let me get someone to look at this",
                "operation": "hand_off_to_human",
                "confidence": 0.5,
            }
        ),
        source="conversational_owner_v1",
    )
    assert result.decision.disposition is ResponseDisposition.HANDOFF
    assert result.decision.proposed_operation.kind is OperationKind.HAND_OFF_TO_HUMAN
