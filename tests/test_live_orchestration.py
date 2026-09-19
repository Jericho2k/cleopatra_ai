from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from models.commercial import CreatorPolicy, FanCommercialState, FanStatus, Offer
from models.conversation_continuity import ConversationEpisode
from models.conversation_decision import (
    ConversationDecision,
    OperationKind,
    ProposedOperation,
)
from models.live_orchestration import (
    ApprovedExecution,
    EvidenceFact,
    EvidenceSnapshot,
    TurnTrigger,
    ValidationResult,
)
from models.schemas import Fan, Message, Persona, SuggestionResponse
from services import conversation_core, live_orchestration, proactive, suggestions
from services.context_packet import ContextPacket
from services.decision_owners import build_semantic_prompt, parse_semantic_decision_result
from services.reply_provenance import PIPELINE_AUTO, ReplyProvenance


def run(coro):
    return asyncio.run(coro)


async def value(result):
    return result


def offer(*, offer_id: str = "offer-1", set_id: str = "set-1", cents: int = 2500):
    return Offer(
        offer_id=offer_id,
        set_id=set_id,
        label="approved photo set",
        price_cents=cents,
        media_count=2,
    )


def loaded(
    *,
    pending_offer: Offer | None = None,
    next_offer: Offer | None = None,
    pending_payment: dict | None = None,
    review: bool = False,
    offered_at: datetime | None = None,
):
    state = FanCommercialState(
        pending_offer=pending_offer,
        last_offer_at=offered_at,
        status=FanStatus.OFFER_PENDING if pending_offer else FanStatus.IDLE,
    )
    snapshot = EvidenceSnapshot(
        creator_id="creator-1",
        fan_id="fan-1",
        trigger=TurnTrigger(
            kind="fan_message", identity="message-1", latest_message="yes"
        ),
        state_revision="revision-1",
        pending_offer=(
            {"offer_id": pending_offer.offer_id, "set_id": pending_offer.set_id}
            if pending_offer
            else None
        ),
        spending_limits={"explicit_current_limit_cents": 5000},
    )
    return live_orchestration.LoadedEvidence(
        snapshot=snapshot,
        packet=ContextPacket(),
        history=[],
        fan=Fan(
            id="fan-1",
            display_name="Test Fan",
            platform_fan_id="test_fan_1",
            auto_mode=True,
            needs_human_review=review,
        ),
        persona=Persona(),
        commercial_state=state,
        policy=CreatorPolicy(),
        next_offer=next_offer,
        active_session=None,
        pending_payment=pending_payment,
        sent_ppv=[],
        within_daily_caps=True,
        stack=SimpleNamespace(profile_id="cleo_v3"),
    )


def live_payload(**overrides):
    payload = {
        "active_needs": ["ordinary conversation"],
        "supporting_messages": ["turn:latest"],
        "unresolved_references": [],
        "must_address": [],
        "response_intent": "ordinary_conversation",
        "disposition": "reply",
        "operation": "none",
        "operation_subject": "",
        "operation_because": "",
        "operation_offer_id": "",
        "operation_set_id": "",
        "operation_payment_reference": "",
        "operation_purchase_id": "",
        "hold": "none",
        "hold_detail": "",
        "confidence": 0.91,
    }
    payload.update(overrides)
    return json.dumps(payload)


def test_live_owner_requires_critical_fields_and_valid_enums():
    missing = json.loads(live_payload())
    missing.pop("disposition")
    assert not parse_semantic_decision_result(json.dumps(missing), strict_live=True).ok

    invalid = parse_semantic_decision_result(
        live_payload(operation="charge_card"), strict_live=True
    )
    assert not invalid.ok
    assert "not one this system knows" in invalid.reason


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        (
            {
                "disposition": "silence",
                "hold": "respect_silence",
                "operation": "present_offer",
            },
            "silent turn",
        ),
        (
            {"hold": "needs_human", "disposition": "reply"},
            "needs_human",
        ),
        (
            {"disposition": "silence", "hold": "none"},
            "must name why",
        ),
    ],
)
def test_live_owner_refuses_contradictory_decisions(changes, reason):
    parsed = parse_semantic_decision_result(live_payload(**changes), strict_live=True)
    assert not parsed.ok
    assert reason in parsed.reason


def test_exact_acceptance_is_distinct_from_presenting_an_offer():
    current = offer()
    evidence = loaded(pending_offer=current)
    decision = ConversationDecision(
        proposed_operation=ProposedOperation(
            kind=OperationKind.SEND_LOCKED_PAID_MESSAGE,
            subject="the current set",
            offer_id=current.offer_id,
            set_id=current.set_id,
        )
    )
    result = live_orchestration.validate_decision(decision, evidence)
    assert result.approved
    assert result.record_refs == {"offer_id": "offer-1", "set_id": "set-1"}

    ambiguous = ConversationDecision(
        proposed_operation=ProposedOperation(
            kind=OperationKind.SEND_LOCKED_PAID_MESSAGE,
            subject="that one",
            offer_id="offer-other",
            set_id="set-other",
        )
    )
    refused = live_orchestration.validate_decision(ambiguous, evidence)
    assert not refused.approved
    assert "exact pending offer" in refused.reasons[0]


def test_expired_offer_and_claimed_payment_cannot_authorize_delivery():
    current = offer()
    expired = loaded(
        pending_offer=current,
        offered_at=datetime.now(timezone.utc) - timedelta(days=2),
    )
    acceptance = ConversationDecision(
        proposed_operation=ProposedOperation(
            kind=OperationKind.SEND_LOCKED_PAID_MESSAGE,
            subject="the set",
            offer_id=current.offer_id,
            set_id=current.set_id,
        )
    )
    assert "expired" in " ".join(
        live_orchestration.validate_decision(acceptance, expired).reasons
    )

    claim = ConversationDecision(
        proposed_operation=ProposedOperation(
            kind=OperationKind.CHECK_PAYMENT_CLAIM,
            subject="claimed payment",
            payment_reference="pay-1",
        )
    )
    checked = live_orchestration.validate_decision(
        claim,
        loaded(pending_payment={"reference": "pay-1"}),
    )
    assert checked.approved
    assert checked.operation == "check_payment_claim"
    assert checked.operation != "send_locked_paid_message"


def test_review_hold_blocks_commercial_execution():
    candidate = offer()
    decision = ConversationDecision(
        proposed_operation=ProposedOperation(
            kind=OperationKind.PRESENT_OFFER,
            subject="approved set",
            offer_id=candidate.offer_id,
            set_id=candidate.set_id,
        )
    )
    result = live_orchestration.validate_decision(
        decision,
        loaded(next_offer=candidate, review=True),
    )
    assert not result.approved
    assert "frozen" in result.reasons[0]


def test_state_revision_detects_a_repeated_identical_fan_message():
    fan = Fan(id="fan-1", display_name="Test Fan")
    state = FanCommercialState()
    first = live_orchestration._state_material(
        fan=fan,
        commercial_state=state,
        active_session=None,
        pending_payment=None,
        latest_fan_message="yes",
        latest_fan_marker="2026-09-18T10:00:00+00:00",
    )
    repeated = live_orchestration._state_material(
        fan=fan,
        commercial_state=state,
        active_session=None,
        pending_payment=None,
        latest_fan_message="yes",
        latest_fan_marker="2026-09-18T10:01:00+00:00",
    )

    assert live_orchestration._revision(first) != live_orchestration._revision(repeated)


@pytest.mark.parametrize("episode_count", [0, 6])
def test_context_ceiling_keeps_latest_trigger_and_transaction_evidence(monkeypatch, episode_count):
    history = [
        Message(
            role="fan" if index % 2 == 0 else "creator",
            content=f"turn-{index} " + ("x" * 650),
        )
        for index in range(79)
    ]
    history.append(Message(role="fan", content="newest message"))
    candidate = offer()
    fan = Fan(
        id="fan-1",
        display_name="Long Conversation",
        platform_fan_id="test_long",
        auto_mode=True,
    )
    state = FanCommercialState()

    monkeypatch.setattr(
        live_orchestration, "get_conversation_history", lambda *_a: value(history)
    )
    monkeypatch.setattr(live_orchestration, "get_fan_by_id", lambda *_a: value(fan))
    monkeypatch.setattr(
        live_orchestration, "get_creator_persona", lambda *_a: value(Persona())
    )
    monkeypatch.setattr(
        live_orchestration,
        "get_creator_legend",
        lambda *_a: value({"background": "creator fact " + ("z" * 5000)}),
    )
    monkeypatch.setattr(
        live_orchestration,
        "get_fan_intelligence_context",
        lambda *_a: value(
            {
                "facts": [
                    {
                        "fact_key": f"fact-{index}",
                        "value": "y" * 600,
                        "source_message_id": f"message-{index}",
                    }
                    for index in range(40)
                ],
                "historical_backfill_complete": False,
            }
        ),
    )
    monkeypatch.setattr(
        live_orchestration, "get_fan_lifecycle_context", lambda *_a: value({})
    )
    monkeypatch.setattr(
        live_orchestration, "get_affordability_context", lambda *_a: value({})
    )
    monkeypatch.setattr(
        live_orchestration, "get_price_learning_context", lambda *_a: value({})
    )
    monkeypatch.setattr(
        live_orchestration,
        "get_sent_ppv",
        lambda *_a: value(
            [
                {
                    "reference": "purchase-1",
                    "set_id": "set-paid",
                    "price_cents": 2000,
                    "purchased": True,
                    "purchased_at": "2026-09-01T00:00:00+00:00",
                }
            ]
        ),
    )
    monkeypatch.setattr(live_orchestration, "get_fan_session", lambda *_a: value(None))
    monkeypatch.setattr(live_orchestration, "get_fan_state", lambda *_a: value(state))
    monkeypatch.setattr(
        live_orchestration, "get_creator_policy", lambda *_a: value(CreatorPolicy())
    )
    monkeypatch.setattr(live_orchestration, "open_threads_for", lambda *_a: value([]))
    episodes = [
        ConversationEpisode(
            id=f"episode-{index}", creator_id="creator-1", fan_id="fan-1",
            summary=f"Discussed the interview at company-{index}; waiting for the result.",
            first_message_at=datetime(2026, 9, 10 - index, tzinfo=timezone.utc),
            last_message_at=datetime(2026, 9, 10 - index, tzinfo=timezone.utc),
        )
        for index in range(episode_count)
    ]
    monkeypatch.setattr(
        live_orchestration, "recent_episodes_for", lambda *_a: value(episodes)
    )
    monkeypatch.setattr(
        live_orchestration, "_fan_pending_payment", lambda *_a: value(None)
    )
    monkeypatch.setattr(live_orchestration, "get_creator_caps", lambda *_a: value({}))
    monkeypatch.setattr(
        live_orchestration,
        "get_next_offer_with_inventory",
        lambda *_a, **_k: value((candidate, ("photo_set",))),
    )
    monkeypatch.setattr(
        live_orchestration,
        "resolve_ai_stack",
        lambda **_k: value(SimpleNamespace(profile_id="cleo_v3")),
    )

    result = run(
        live_orchestration.load_evidence(
            creator_id="creator-1",
            fan_id="fan-1",
            trigger_kind="fan_message",
            trigger_identity="message-80",
            latest_message="newest message",
        )
    )
    snapshot = result.snapshot
    assert snapshot.trigger.latest_message == "newest message"
    assert all(
        "newest message" not in bubble
        for turn in snapshot.recent_turns
        for bubble in turn["bubbles"]
    )
    assert snapshot.confirmed_purchases[0]["reference"] == "purchase-1"
    assert snapshot.approved_inventory[0]["offer_id"] == "offer-1"
    assert snapshot.memory_status["historical_backfill_complete"] is False
    assert snapshot.truncation
    assert len(snapshot.canonical_json()) <= live_orchestration.MAX_EVIDENCE_CHARS

    # Exercise the actual production prompt paths: a populated ContextPacket
    # alone did not put any episode into either model's input before this fix.
    _, owner_input = build_semantic_prompt(result.packet, {"evidence_snapshot": snapshot})
    owner_evidence = json.loads(owner_input.split("\n", 1)[1])
    writer_messages = live_orchestration.build_writer_prompt(
        result, ConversationDecision(), ApprovedExecution(), mode=live_orchestration.MODE_AUTO
    )
    writer_evidence = json.loads(writer_messages[1]["content"])["evidence"]
    assert writer_evidence == owner_evidence
    assert len(writer_evidence["conversation_episodes"]) == min(episode_count, 4)
    if episode_count:
        first_episode = writer_evidence["conversation_episodes"][0]
        assert first_episode["source_ref"] == "conversation_episodes:episode-0"
        assert first_episode["certainty"] == "inferred"
        assert "2026-09-10" in first_episode["value"]
        assert "company-0" in first_episode["value"]
        assert snapshot.truncation["conversation_episodes"] == 2


def test_episode_budget_drops_oldest_before_current_exchange(monkeypatch):
    original = loaded().snapshot
    current = ({"speaker": "fan", "bubbles": ["I changed jobs since then."]},)
    correction = ({"kind": "correction", "summary": "Now works at the library."},)
    newer = EvidenceFact("Recent interview " + "a" * 400, "conversation_episodes:new", "inferred")
    older = EvidenceFact("Old job " + "b" * 400, "conversation_episodes:old", "inferred")
    expected = replace(original, recent_turns=current, corrections=correction,
                       conversation_episodes=(newer,), truncation={"conversation_episodes": 3})
    monkeypatch.setattr(live_orchestration, "MAX_EVIDENCE_CHARS", len(expected.canonical_json()))
    snapshot = replace(expected, conversation_episodes=(newer, older),
                       truncation={"conversation_episodes": 2})

    result = live_orchestration._trim_snapshot(snapshot)

    assert result == expected
    assert result.trigger == original.trigger
    assert len(result.canonical_json()) <= live_orchestration.MAX_EVIDENCE_CHARS


def _semantic_resolution():
    return conversation_core.ConversationCoreResolution(
        conversation_core.CORE_SEMANTIC_V1,
        conversation_core.SOURCE_CREATOR,
    )


def _retired(*_args, **_kwargs):
    raise AssertionError("retired behavioral controller was called")


def test_assisted_entrypoint_bypasses_every_legacy_behavioral_controller(monkeypatch):
    monkeypatch.setattr(
        conversation_core,
        "resolve_conversation_core",
        lambda **_k: value(_semantic_resolution()),
    )
    monkeypatch.setattr(
        live_orchestration,
        "get_assisted_suggestions",
        lambda **_k: value(
            SuggestionResponse(
                suggestions=["ordinary reply"],
                conversation_core="semantic_v1",
            )
        ),
    )
    for name in (
        "analyze_situation",
        "orchestrate",
        "direct_conversation",
        "plan_next_action",
    ):
        monkeypatch.setattr(suggestions, name, _retired)

    result = run(
        suggestions.get_suggestions(
            fan_id="fan-1",
            creator_id="creator-1",
            fan_message="how was your day?",
            save_fan_message=False,
        )
    )
    assert result.conversation_core == "semantic_v1"
    assert result.suggestions == ["ordinary reply"]


def configure_auto_route(
    monkeypatch, *, outcome=None, failure: Exception | None = None
):
    fan = Fan(
        id="fan-1",
        display_name="Test Fan",
        platform_fan_id="test_auto",
        auto_mode=True,
    )
    monkeypatch.setattr(
        suggestions,
        "get_conversation_history",
        lambda *_a, **_k: value([Message(role="fan", content="hello")]),
    )
    monkeypatch.setattr(suggestions, "get_fan_by_id", lambda *_a: value(fan))
    monkeypatch.setattr(
        suggestions, "get_fan_intelligence_context", lambda *_a: value({})
    )
    monkeypatch.setattr(suggestions, "get_fan_lifecycle_context", lambda *_a: value({}))
    monkeypatch.setattr(suggestions, "get_affordability_context", lambda *_a: value({}))
    monkeypatch.setattr(
        suggestions, "get_price_learning_context", lambda *_a: value({})
    )
    monkeypatch.setattr(
        conversation_core,
        "resolve_conversation_core",
        lambda **_k: value(_semantic_resolution()),
    )

    async def semantic_turn(**_kwargs):
        if failure:
            raise failure
        return outcome or {"outcome": "replied", "message_ids": ["message-1"]}

    monkeypatch.setattr(live_orchestration, "run_auto_turn", semantic_turn)
    for name in (
        "analyze_situation",
        "orchestrate",
        "direct_conversation",
        "plan_next_action",
    ):
        monkeypatch.setattr(suggestions, name, _retired)
    suggestions._pending_auto_replies.clear()


def test_full_auto_entrypoint_bypasses_legacy_and_reports_semantic_outcome(monkeypatch):
    configure_auto_route(monkeypatch, outcome={"outcome": "no_send", "message_ids": []})
    sink = {}
    run(
        suggestions._debounced_auto_reply(
            "fan-1",
            "creator-1",
            skip_debounce=True,
            skip_availability=True,
            skip_human_delays=True,
            outcome_sink=sink,
        )
    )
    assert sink == {"outcome": "no_send"}


@pytest.mark.parametrize(
    "outcome,reason",
    [("no_send", None), ("human_review", "semantic_writer_contract_rejected: false_delivery_claim")],
)
def test_simulator_entrypoint_selects_the_real_semantic_auto_path(monkeypatch, outcome, reason):
    configure_auto_route(
        monkeypatch, outcome={"outcome": outcome, "message_ids": [], "reason": reason}
    )
    monkeypatch.setattr(
        suggestions, "save_message", lambda *_a, **_k: value("fan-message-1")
    )
    monkeypatch.setattr(suggestions, "mark_simulation_owned_message", lambda *_a: None)
    monkeypatch.setattr(
        suggestions, "_recent_creator_message_rows", lambda *_a: value([])
    )
    monkeypatch.setattr(
        suggestions,
        "resolve_ai_stack",
        lambda **_k: value(SimpleNamespace(profile_id="cleo_v3")),
    )
    monkeypatch.setattr(suggestions, "learn_from_fan_message", lambda **_k: value(None))

    result = run(
        suggestions.run_simulated_inbound(
            fan_id="fan-1",
            creator_id="creator-1",
            message="ordinary simulator turn",
            fast=True,
        )
    )

    assert result["simulation"] is True
    assert result["outcome"] == outcome
    assert result["reason"] == reason
    assert result["fan_message_id"] == "fan-message-1"


def test_semantic_provider_failure_never_falls_through_to_legacy(monkeypatch):
    configure_auto_route(
        monkeypatch,
        failure=live_orchestration.LiveOrchestrationError("owner unavailable"),
    )
    sink = {}
    with pytest.raises(live_orchestration.LiveOrchestrationError):
        run(
            suggestions._debounced_auto_reply(
                "fan-1",
                "creator-1",
                skip_debounce=True,
                skip_availability=True,
                skip_human_delays=True,
                outcome_sink=sink,
            )
        )
    assert sink == {"outcome": "owner_failed"}


def test_scheduled_entrypoint_uses_shared_core_and_not_legacy_writer(monkeypatch):
    fan = Fan(
        id="fan-1",
        display_name="Test Fan",
        platform_fan_id="test_proactive",
        auto_mode=True,
    )
    monkeypatch.setattr(proactive, "get_fan_by_id", lambda *_a: value(fan))
    monkeypatch.setattr(
        conversation_core,
        "resolve_conversation_core",
        lambda **_k: value(_semantic_resolution()),
    )
    monkeypatch.setattr(
        live_orchestration, "run_proactive_turn", lambda **_k: value(True)
    )
    monkeypatch.setattr(proactive, "generate_replies", _retired)

    assert run(
        proactive.send_proactive_message(
            "creator-1",
            "fan-1",
            "return event is due",
            action_id="action-1",
        )
    )


def test_locked_auto_execution_uses_controlled_delivery_adapter(monkeypatch):
    evidence = loaded(pending_offer=offer())
    evidence.commercial_state.status = FanStatus.OFFER_SELECTED
    execution = ApprovedExecution(
        operation="send_locked_paid_message",
        delivery={
            "media_ids": ["media-1", "media-2"],
            "price_cents": 2500,
            "set_id": "set-1",
            "step_index": 0,
        },
        validation=ValidationResult(
            approved=True,
            operation="send_locked_paid_message",
            state_revision="revision-1",
        ),
    )
    prepared = live_orchestration.PreparedTurn(
        loaded=evidence,
        decision=ConversationDecision(),
        execution=execution,
        replies=["attached in this locked message"],
        provenance=ReplyProvenance("creator-1", "fan-1", PIPELINE_AUTO),
        writer_trace=SimpleNamespace(),
    )
    calls = []
    monkeypatch.setattr(
        live_orchestration,
        "_expected_execution_revision",
        lambda *_a: value("revision-2"),
    )
    monkeypatch.setattr(
        live_orchestration, "_current_revision", lambda *_a: value("revision-2")
    )
    monkeypatch.setattr(
        live_orchestration, "_commit_locked_plan", lambda *_a: value("revision-2")
    )

    async def deliver(**kwargs):
        calls.append(kwargs)
        return {"message_id": "local-message", "platform_message_id": "local-test:1"}

    monkeypatch.setattr(live_orchestration, "send_locked_ppv", deliver)
    result = run(live_orchestration.execute_auto_turn(prepared))
    assert result["outcome"] == "replied"
    assert calls[0]["media_ids"] == ["media-1", "media-2"]
    assert calls[0]["price_cents"] == 2500
    assert calls[0]["source"] == "semantic_auto"


def test_core_resolution_defaults_to_legacy_and_test_fan_can_override(monkeypatch):
    monkeypatch.delenv(conversation_core.CORE_ENV_VAR, raising=False)
    monkeypatch.setattr(
        conversation_core, "creator_core_override", lambda *_a, **_k: value(None)
    )
    monkeypatch.setattr(
        conversation_core, "simulation_fan_core_override", lambda *_a, **_k: value(None)
    )
    default = run(
        conversation_core.resolve_conversation_core(
            creator_id="creator-1",
            fan_id="fan-1",
            platform_fan_id="real-fan",
        )
    )
    assert default.core_id == "legacy"
    assert default.source == "builtin"

    monkeypatch.setattr(
        conversation_core,
        "simulation_fan_core_override",
        lambda *_a, **_k: value("semantic_v1"),
    )
    selected = run(
        conversation_core.resolve_conversation_core(
            creator_id="creator-1",
            fan_id="fan-1",
            platform_fan_id="test_fan",
        )
    )
    assert selected.core_id == "semantic_v1"
    assert selected.source == "simulation_fan"


def test_real_fan_ignores_fan_override_and_invalid_environment_is_visible(monkeypatch):
    fan_reads = []

    async def fan_override(*_args, **_kwargs):
        fan_reads.append(True)
        return "semantic_v1"

    monkeypatch.setattr(conversation_core, "simulation_fan_core_override", fan_override)
    monkeypatch.setattr(
        conversation_core,
        "creator_core_override",
        lambda *_a, **_k: value("legacy"),
    )
    resolved = run(
        conversation_core.resolve_conversation_core(
            creator_id="creator-1",
            fan_id="fan-1",
            platform_fan_id="real-fan",
        )
    )
    assert resolved.core_id == "legacy"
    assert fan_reads == []

    monkeypatch.setattr(
        conversation_core, "creator_core_override", lambda *_a, **_k: value(None)
    )
    monkeypatch.setenv(conversation_core.CORE_ENV_VAR, "typo-core")
    with pytest.raises(conversation_core.InvalidConversationCoreConfiguration):
        run(
            conversation_core.resolve_conversation_core(
                creator_id="creator-1",
                platform_fan_id="real-fan",
            )
        )


# Production regressions: rejected model proposals are evidence for repair,
# never transaction permission and never an exception for the whole turn.


def owner_world(monkeypatch, evidence, outputs):
    calls = []
    responses = iter(outputs)
    evidence.stack.profile = SimpleNamespace(
        stage=lambda *_: SimpleNamespace(
            primary_target=lambda: None,
            fallback_target=lambda: None,
            prompt_version="v1",
        )
    )

    async def complete(_target, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(text=next(responses))

    monkeypatch.setattr(live_orchestration, "complete", complete)
    return calls


@pytest.mark.parametrize(
    "bad",
    [
        dict(
            operation="check_payment_claim",
            operation_subject="claimed payment",
            operation_payment_reference="imaginary-payment",
        ),
        dict(
            operation="present_offer",
            operation_subject="the $30 set",
            operation_offer_id="offer-1",
            operation_set_id="set-1",
        ),
        dict(disposition="handoff", operation="none", hold="needs_human"),
    ],
)
def test_rejected_owner_decisions_repair_using_same_snapshot(monkeypatch, bad):
    evidence = loaded(pending_offer=offer(cents=3000), next_offer=offer(cents=3000))
    before = evidence.snapshot.canonical_json()
    state_before = evidence.commercial_state.model_dump()
    calls = owner_world(monkeypatch, evidence, [live_payload(**bad), live_payload()])
    decision = run(live_orchestration.decide_turn(evidence))
    assert decision.proposed_operation.kind is OperationKind.NONE
    assert len(calls) == 2
    assert before in calls[0]["messages"][0]["content"]
    assert before in calls[1]["messages"][0]["content"]
    repair = calls[1]["messages"][0]["content"]
    assert (
        "validation_failures" in repair
        and "Authoritative state outranks fan wording" in repair
    )
    assert "check_payment_claim" not in live_orchestration.legal_operations(evidence)
    assert evidence.commercial_state.model_dump() == state_before
    assert evidence.active_session is None


def test_repeated_invalid_owner_decisions_handoff_without_execution(monkeypatch):
    from models.conversation_decision import ResponseDisposition

    evidence = loaded(pending_offer=offer())
    calls = owner_world(
        monkeypatch,
        evidence,
        [
            live_payload(
                operation="check_payment_claim",
                operation_subject="claimed payment",
                operation_payment_reference="fake",
            )
            for _ in range(3)
        ],
    )
    monkeypatch.setattr(
        live_orchestration, "load_evidence", lambda **_: value(evidence)
    )
    monkeypatch.setattr(live_orchestration, "_write_turn", _retired)
    monkeypatch.setattr(live_orchestration, "send_locked_ppv", _retired)
    frozen = []

    async def freeze(fan_id, reason):
        frozen.append((fan_id, reason))

    monkeypatch.setattr(live_orchestration, "freeze_fan_for_review", freeze)
    prepared = run(
        live_orchestration.prepare_turn(
            creator_id="creator-1",
            fan_id="fan-1",
            trigger_kind="fan_message",
            trigger_identity="message-1",
            latest_message="i dont see it",
        )
    )
    assert prepared.decision.disposition is ResponseDisposition.HANDOFF
    result = run(live_orchestration.execute_auto_turn(prepared))
    assert len(calls) == 3
    assert result["outcome"] == "human_review" and result["message_ids"] == []
    assert "no pending payment" in frozen[0][1]
    assert "repair_exhausted" in frozen[0][1]
    assert result["reason"] == frozen[0][1]
    assert evidence.commercial_state.accepted_offer_id is None


@pytest.mark.parametrize(
    "message", ["i dont see it", "cmon baby, send it to me", "I paid"]
)
def test_missing_delivery_never_manufactures_access_or_payment(monkeypatch, message):
    from dataclasses import replace

    evidence = loaded(pending_offer=offer(cents=3000))
    evidence.snapshot = replace(
        evidence.snapshot,
        trigger=replace(evidence.snapshot.trigger, latest_message=message),
    )
    calls = owner_world(
        monkeypatch,
        evidence,
        [
            live_payload(
                operation="repair_content_access",
                operation_subject="missing content",
                operation_purchase_id="fake",
            ),
            live_payload(
                operation="check_payment_claim",
                operation_subject="payment",
                operation_payment_reference="fake",
            ),
            live_payload(),
        ],
    )
    assert (
        run(live_orchestration.decide_turn(evidence)).proposed_operation.kind
        is OperationKind.NONE
    )
    assert len(calls) == 3
    assert evidence.snapshot.confirmed_purchases == ()
    assert evidence.pending_payment is None


def test_delivered_unpaid_content_can_check_payment_but_not_repair_purchase():
    evidence = loaded(pending_payment={"reference": "real-payment"})
    payment = ConversationDecision(
        proposed_operation=ProposedOperation(
            kind=OperationKind.CHECK_PAYMENT_CLAIM,
            subject="missing locked message",
            payment_reference="real-payment",
        )
    )
    assert live_orchestration.validate_decision(payment, evidence).approved
    repair = ConversationDecision(
        proposed_operation=ProposedOperation(
            kind=OperationKind.REPAIR_CONTENT_ACCESS,
            subject="missing content",
            purchase_id="real-payment",
        )
    )
    assert not live_orchestration.validate_decision(repair, evidence).approved
    from dataclasses import replace

    evidence.snapshot = replace(
        evidence.snapshot, confirmed_purchases=({"reference": "real-payment"},)
    )
    assert live_orchestration.validate_decision(repair, evidence).approved


@pytest.mark.parametrize(
    "text",
    [
        "just sent it your way",
        "check it",
        "it's there",
        "open it",
        "should be there now",
        "delivered it",
    ],
)
def test_text_offer_cannot_claim_delivery(text):
    execution = ApprovedExecution(
        operation="present_offer", offer={"price_cents": 3000}
    )
    assert "false_delivery_claim" in live_orchestration.writer_contract_reasons(
        [text], loaded(next_offer=offer(cents=3000)), execution, mode="auto"
    )


@pytest.mark.parametrize(
    "fan_text, caption, rejected",
    [
        ("i do baby", "$30 to unlock", True),
        ("i do baby", "30 dollars to unlock", True),
        ("i do baby", "knew you would 😏", False),
        ("i do baby", "you've got $300 reasons to smile", False),
        ("how much?", "$30", False),
        ("$30 is too much", "it's $30", False),
    ],
)
def test_locked_caption_price_contract(fan_text, caption, rejected):
    from dataclasses import replace

    evidence = loaded()
    evidence.snapshot = replace(
        evidence.snapshot,
        trigger=replace(evidence.snapshot.trigger, latest_message=fan_text),
    )
    execution = ApprovedExecution(
        operation="send_locked_paid_message",
        delivery={"media_ids": ["m1"], "price_cents": 3000},
    )
    reasons = live_orchestration.writer_contract_reasons(
        [caption], evidence, execution, mode="auto"
    )
    assert ("redundant_locked_price" in reasons) is rejected


def test_writer_voice_contract_and_current_life_grounding():
    execution = ApprovedExecution()
    evidence = loaded()
    prompt = live_orchestration.build_writer_prompt(
        evidence, ConversationDecision(), execution, mode="auto"
    )[0]["content"]
    for rule in [
        "does not require a question",
        "One natural thought",
        "Choose length",
        "Do not repeat a two-message template",
        "catalogue",
        "current-life facts",
        "emotional moment",
    ]:
        assert rule in prompt
    for legacy in [
        "Conversation Director",
        "Experience Director",
        "strategic_move",
        "session strategy",
        "escalation ladder",
    ]:
        assert legacy not in prompt
    assert (
        live_orchestration.writer_contract_reasons(
            ["you know I like that 😏"], evidence, execution, mode="auto"
        )
        == []
    )
    assert (
        "unsupported_current_life_claim"
        in live_orchestration.writer_contract_reasons(
            ["I'm cooking dinner right now"], evidence, execution, mode="auto"
        )
    )


def test_ready_offer_plans_exact_locked_delivery_without_pending_confirmation(
    monkeypatch,
):
    evidence = loaded(next_offer=offer(cents=3000))
    decision = ConversationDecision(
        proposed_operation=ProposedOperation(
            kind=OperationKind.SEND_LOCKED_PAID_MESSAGE,
            subject="the bikini content",
            offer_id="offer-1",
            set_id="set-1",
        )
    )
    planned = []

    async def plan(*args, **kwargs):
        planned.append(kwargs)
        return {
            "status": "ok",
            "session": {
                "plan": [
                    {
                        "set_id": "set-1",
                        "media_ids": ["approved-1", "approved-2"],
                        "price_cents": 3000,
                    }
                ]
            },
        }

    monkeypatch.setattr(live_orchestration, "plan_session_for_fan", plan)
    execution = run(
        live_orchestration._prepare_execution(
            decision, evidence, execute_operations=True
        )
    )
    assert execution.operation == "send_locked_paid_message"
    assert execution.delivery["media_ids"] == ["approved-1", "approved-2"]
    assert execution.delivery["price_cents"] == 3000
    assert planned[0] == {
        "accepted_set_id": "set-1",
        "accepted_price_cents": 3000,
        "persist": False,
    }
    assert evidence.commercial_state.accepted_offer_id is None
    evidence.within_daily_caps = False
    assert not live_orchestration.validate_decision(decision, evidence).approved
    evidence.within_daily_caps = True
    evidence.pending_payment = {"reference": "already-pending"}
    assert not live_orchestration.validate_decision(decision, evidence).approved


def test_failed_ppv_delivery_returns_review_without_false_creator_message(monkeypatch):
    evidence = loaded(pending_offer=offer())
    execution = ApprovedExecution(
        operation="send_locked_paid_message",
        delivery={"media_ids": ["media-1"], "price_cents": 2500, "set_id": "set-1"},
    )
    prepared = live_orchestration.PreparedTurn(
        evidence,
        ConversationDecision(),
        execution,
        ["just sent it your way"],
        ReplyProvenance("creator-1", "fan-1", PIPELINE_AUTO),
        SimpleNamespace(),
    )
    monkeypatch.setattr(
        live_orchestration, "_current_revision", lambda *_: value("revision-1")
    )
    monkeypatch.setattr(
        live_orchestration, "_commit_locked_plan", lambda *_: value("revision-1")
    )
    monkeypatch.setattr(live_orchestration, "save_message", _retired)

    async def refuse(**_):
        raise live_orchestration.PPVDeliveryError("send refused")

    monkeypatch.setattr(live_orchestration, "send_locked_ppv", refuse)
    frozen = []

    async def freeze(*args):
        frozen.append(args)

    monkeypatch.setattr(live_orchestration, "freeze_fan_for_review", freeze)
    result = run(live_orchestration.execute_auto_turn(prepared))
    assert result["outcome"] == "human_review" and result["message_ids"] == []
    assert "send refused" in frozen[0][1]


@pytest.mark.parametrize("initial_pending", [False, True])
@pytest.mark.parametrize("mirrored", [False, True])
def test_bikini_trajectory_persists_real_ppv_receipt_and_payment_state(
    monkeypatch, initial_pending, mirrored
):
    """Real executor, adapter, receipt and reconciliation; models/DB are stubs."""
    from dataclasses import replace
    from services import ppv_delivery, ppv_persistence
    from tests.test_full_auto_simulation import FakeDB

    media_ids = ["approved-1", "approved-2"]
    if mirrored:
        from core.simulation_catalog import simulation_media_id
        media_ids = [simulation_media_id("source-creator", mid) for mid in media_ids]

    class DB(FakeDB):
        def rpc(self, name, params):
            assert name == "attach_pending_ppv"
            self.tables["fans"][0]["pending_ppv_check"] = params["p_pending"]
            return SimpleNamespace(execute=lambda: SimpleNamespace(data="attached"))

    db = DB(
        {
            "messages": [],
            "fans": [
                {
                    "id": "fan-1",
                    "creator_id": "creator-1",
                    "platform_fan_id": "test_fan_1",
                    "pending_ppv_check": None,
                }
            ],
            "creators": [{"id": "creator-1", "apifansly_account_id": "account-1"}],
        }
    )
    evidence = loaded(
        next_offer=offer(cents=3000),
        pending_offer=offer(cents=3000) if initial_pending else None,
    )
    state = {"commercial": evidence.commercial_state, "session": None}
    claims = []

    async def save_state(_fan, _creator, commercial):
        state["commercial"] = commercial

    async def save_session(_fan, session):
        state["session"] = session

    async def noop(*_, **__):
        return None

    async def claim(**kwargs):
        claims.append(kwargs)

    for module in [live_orchestration, ppv_delivery, ppv_persistence]:
        monkeypatch.setattr(module, "get_supabase", lambda: db)
    monkeypatch.setattr("db.queries.get_supabase", lambda: db)
    monkeypatch.setattr("services.message_diagnostics.record_diagnostics", noop)
    for module in [live_orchestration, ppv_persistence]:
        monkeypatch.setattr(
            module, "get_fan_state", lambda *_: value(state["commercial"])
        )
        monkeypatch.setattr(module, "save_fan_state", save_state)
        monkeypatch.setattr(module, "save_fan_session", save_session)
    for module in [ppv_delivery, ppv_persistence]:
        monkeypatch.setattr(
            module, "get_fan_session", lambda *_: value(state["session"])
        )
        monkeypatch.setattr(
            module, "get_creator_policy", lambda *_: value(CreatorPolicy())
        )
    monkeypatch.setattr(ppv_persistence, "schedule_action", noop)
    monkeypatch.setattr(ppv_delivery, "claim_delivery", claim)
    monkeypatch.setattr(ppv_delivery, "transition_delivery", noop)
    monkeypatch.setattr(ppv_delivery, "send_apifansly_message", _retired)
    monkeypatch.setattr(
        live_orchestration, "_current_revision", lambda *_: value("revision-1")
    )
    monkeypatch.setattr(
        live_orchestration, "load_evidence", lambda **_: value(evidence)
    )
    calls = owner_world(
        monkeypatch,
        evidence,
        [
            live_payload(),
            live_payload(),
            live_payload(
                operation="send_locked_paid_message",
                response_intent="deliver_accepted_offer",
                operation_subject="the approved bikini content",
                operation_offer_id="offer-1",
                operation_set_id="set-1",
            ),
        ],
    )
    captions = iter(
        [
            "you like that bikini 😏",
            "without it is even better 😏|wanna find out?",
            "knew you would 😏",
        ]
    )

    async def write(*_, **__):
        return [next(captions)]

    monkeypatch.setattr(live_orchestration, "generate_replies", write)
    monkeypatch.setattr(
        live_orchestration,
        "plan_session_for_fan",
        lambda *_a, **_k: value(
            {
                "status": "ok",
                "session": {
                    "status": "active",
                    "current_index": 0,
                    "plan": [
                        {
                            "set_id": "set-1",
                            "media_ids": media_ids,
                            "price_cents": 3000,
                            "sent": False,
                            "purchased": False,
                        }
                    ],
                },
            }
        ),
    )
    for message in [
        "your ass looks soo good on that bikini photo, its so hot",
        "yes it does baby, makes me really excited of what it is without that bikini on",
        "i do baby",
    ]:
        evidence.snapshot = replace(
            evidence.snapshot,
            trigger=replace(evidence.snapshot.trigger, latest_message=message),
        )
        result = run(
            live_orchestration.run_auto_turn(
                creator_id="creator-1",
                fan_id="fan-1",
                latest_message=message,
                trigger_identity=message,
            )
        )
        assert result["outcome"] == "replied"
    assert len(calls) == 3 and len(claims) == 1
    row = db.tables["messages"][-1]
    ppv = row["media_context"]["ppv"]
    assert row["content"] == "knew you would 😏"
    assert ppv["media_ids"] == media_ids
    assert ppv["price_cents"] == 3000 and ppv["set_id"] == "set-1"
    assert (
        ppv["payment_reference"]
        == db.tables["fans"][0]["pending_ppv_check"]["reference"]
    )
    assert state["commercial"].status is FanStatus.PAYMENT_PENDING
    assert state["session"]["plan"][0]["purchased"] is False
    assert all(
        not r.get("media_context", {}).get("ppv") for r in db.tables["messages"][:-1]
    )
    assert all("sent it" not in r["content"] for r in db.tables["messages"])


def test_writer_rejection_suppresses_entire_turn_before_delivery(monkeypatch):
    evidence = loaded(next_offer=offer())
    owner_world(
        monkeypatch,
        evidence,
        [
            live_payload(
                operation="present_offer",
                operation_subject="the approved content",
                operation_offer_id="offer-1",
                operation_set_id="set-1",
            )
        ],
    )
    monkeypatch.setattr(
        live_orchestration, "load_evidence", lambda **_: value(evidence)
    )
    monkeypatch.setattr(
        live_orchestration,
        "generate_replies",
        lambda *_a, **_k: value(["knew you would|just sent it your way|$25 to unlock"]),
    )
    monkeypatch.setattr(live_orchestration, "_deliver_plain_parts", _retired)
    frozen = []

    async def freeze(*args):
        frozen.append(args)

    monkeypatch.setattr(live_orchestration, "freeze_fan_for_review", freeze)
    prepared = run(
        live_orchestration.prepare_turn(
            creator_id="creator-1",
            fan_id="fan-1",
            trigger_kind="fan_message",
            trigger_identity="m1",
            latest_message="yes",
        )
    )
    assert prepared.replies == []
    assert (
        run(live_orchestration.execute_auto_turn(prepared))["outcome"] == "human_review"
    )
    assert frozen
    assert evidence.commercial_state.pending_offer is None


@pytest.mark.parametrize(
    "cents, text", [(2500, "$25 to unlock"), (4250, "$42.50 to unlock")]
)
def test_price_guard_uses_approved_price_for_each_turn(cents, text):
    execution = ApprovedExecution(
        operation="send_locked_paid_message",
        delivery={"media_ids": ["m1"], "price_cents": cents},
    )
    assert "redundant_locked_price" in live_orchestration.writer_contract_reasons(
        [text], loaded(), execution, mode="auto"
    )


def test_validated_price_repetition_cannot_freeze_a_locked_delivery(monkeypatch):
    evidence = loaded(next_offer=offer(cents=3000))
    owner_world(monkeypatch, evidence, [])
    execution = ApprovedExecution(
        operation="send_locked_paid_message",
        delivery={"media_ids": ["approved-1"], "price_cents": 3000},
    )
    attempts = []

    async def writer(*args, **kwargs):
        attempts.append(kwargs["trace"])
        return ["The approved item is $30."]

    monkeypatch.setattr(live_orchestration, "generate_replies", writer)
    replies, trace = run(live_orchestration._write_turn(
        evidence, ConversationDecision(), execution, mode="auto"
    ))
    assert len(attempts) == 2
    assert replies == ["The approved item is $30."]
    assert trace is attempts[-1]
    assert not trace.failure_reason


@pytest.mark.parametrize("second", [[], ["dm me for the link"], ["$300 to unlock"]])
def test_expression_retry_keeps_prior_safe_caption_and_its_attribution(monkeypatch, second):
    evidence = loaded(next_offer=offer(cents=3000))
    owner_world(monkeypatch, evidence, [])
    attempts = []

    async def writer(*args, **kwargs):
        attempts.append(kwargs["trace"])
        return ["$30 to unlock"] if len(attempts) == 1 else second

    monkeypatch.setattr(live_orchestration, "generate_replies", writer)
    replies, trace = run(live_orchestration._write_turn(
        evidence, ConversationDecision(), ApprovedExecution(
            operation="send_locked_paid_message",
            delivery={"media_ids": ["approved-1"], "price_cents": 3000},
        ), mode="auto"
    ))
    assert replies == ["$30 to unlock"]
    assert trace is attempts[0]


@pytest.mark.parametrize("caption", ["$300 to unlock", "$30 plus $99", "Only 90 dollars"])
def test_wrong_price_is_a_hard_failure_even_with_correct_price_present(caption):
    reasons = live_orchestration.writer_contract_reasons(
        [caption], loaded(), ApprovedExecution(
            operation="send_locked_paid_message",
            delivery={"media_ids": ["approved-1"], "price_cents": 3000},
        ), mode="auto"
    )
    assert "unapproved_price_claim" in reasons


def test_price_counteroffer_can_be_discussed_without_repricing():
    from dataclasses import replace

    evidence = loaded()
    evidence.snapshot = replace(evidence.snapshot, trigger=replace(
        evidence.snapshot.trigger, latest_message="Can you do $20 instead?"
    ))
    execution = ApprovedExecution(
        operation="present_offer", offer={"price_cents": 3000}
    )
    assert live_orchestration.writer_contract_reasons(
        ["You asked for $20; this item is $30."], evidence, execution, mode="auto"
    ) == []
    assert execution.offer["price_cents"] == 3000


@pytest.mark.parametrize("operation", ["none", "present_offer", "send_locked_paid_message"])
def test_direct_message_redirect_is_invalid_in_every_semantic_operation(operation):
    assert "unsupported_delivery_route" in live_orchestration.writer_contract_reasons(
        ["dm me for the link"], loaded(), ApprovedExecution(operation=operation), mode="auto"
    )


def test_semantic_writer_platform_and_visual_boundaries():
    prompt = live_orchestration.build_writer_prompt(
        loaded(), ConversationDecision(), ApprovedExecution(), mode="auto"
    )[0]["content"]
    assert "there is no delivery link" in prompt
    assert "You cannot see the fan" in prompt
    assert live_orchestration.writer_contract_reasons(
        ["The link you shared was interesting."], loaded(), ApprovedExecution(), mode="auto"
    ) == []
    assert live_orchestration.writer_contract_reasons(
        ["Message me for advice any time."], loaded(), ApprovedExecution(), mode="auto"
    ) == []
    assert "unsupported_delivery_route" in live_orchestration.writer_contract_reasons(
        ["I'll send you the link"], loaded(), ApprovedExecution(operation="present_offer"), mode="auto"
    )


def test_planner_cannot_change_approved_price(monkeypatch):
    evidence = loaded(next_offer=offer(cents=3000))
    decision = ConversationDecision(
        proposed_operation=ProposedOperation(
            kind=OperationKind.SEND_LOCKED_PAID_MESSAGE,
            subject="the content",
            offer_id="offer-1",
            set_id="set-1",
        )
    )
    monkeypatch.setattr(
        live_orchestration,
        "plan_session_for_fan",
        lambda *_a, **_k: value(
            {
                "status": "ok",
                "session": {
                    "plan": [
                        {"set_id": "set-1", "media_ids": ["m1"], "price_cents": 2500}
                    ]
                },
            }
        ),
    )
    result = run(
        live_orchestration._prepare_execution(
            decision, evidence, execute_operations=True
        )
    )
    assert not result.validation.approved and result.delivery is None
    assert evidence.active_session is None


def test_production_pending_offer_send_request_repairs_into_exact_locked_delivery(
    monkeypatch,
):
    from dataclasses import replace

    evidence = loaded(pending_offer=offer(cents=3000))
    evidence.snapshot = replace(
        evidence.snapshot,
        trigger=replace(
            evidence.snapshot.trigger,
            latest_message="i dont see it, cmon baby send it to me",
        ),
    )
    before = evidence.commercial_state.model_dump()
    calls = owner_world(
        monkeypatch,
        evidence,
        [
            live_payload(
                operation="check_payment_claim",
                operation_subject="the $30 payment",
                operation_payment_reference="imaginary",
            ),
            live_payload(
                operation="send_locked_paid_message",
                operation_subject="the exact pending content",
                response_intent="deliver_accepted_offer",
                operation_offer_id="offer-1",
                operation_set_id="set-1",
            ),
        ],
    )
    decision = run(live_orchestration.decide_turn(evidence))
    assert decision.proposed_operation.kind is OperationKind.SEND_LOCKED_PAID_MESSAGE
    assert live_orchestration.validate_decision(decision, evidence).approved
    assert "there is no pending payment to check" in calls[1]["messages"][0]["content"]
    assert (
        "semantic owner attempted to state a price"
        in calls[1]["messages"][0]["content"]
    )
    assert evidence.commercial_state.model_dump() == before
    assert evidence.pending_payment is None
    evidence.active_session = {"status": "active", "plan": [{"sent": False}]}
    assert not live_orchestration.validate_decision(decision, evidence).approved


def test_writer_repairs_expression_without_changing_commercial_authority(monkeypatch):
    evidence = loaded(next_offer=offer())
    owner_world(monkeypatch, evidence, [])
    state_before = evidence.commercial_state.model_dump()
    replies = iter([["just sent it your way"], ["you've got me smiling 😏"]])
    calls = []

    async def generate(prompt, *_a, **_k):
        calls.append(json.loads(json.dumps(prompt)))
        return next(replies)

    monkeypatch.setattr(live_orchestration, "generate_replies", generate)
    execution = ApprovedExecution(
        operation="present_offer", offer={"price_cents": 2500}
    )
    result, trace = run(
        live_orchestration._write_turn(
            evidence, ConversationDecision(), execution, mode="auto"
        )
    )
    assert result == ["you've got me smiling 😏"] and len(calls) == 2
    assert "false_delivery_claim" in calls[1][0]["content"]
    assert calls[0][1] == calls[1][1]  # unchanged evidence and approved plan
    assert trace.failure_reason == ""
    assert evidence.commercial_state.model_dump() == state_before
    assert execution.operation == "present_offer" and execution.delivery is None


@pytest.mark.parametrize('text', ["its 30 if ur down", "it's 30", '30 to unlock'])
def test_bare_commercial_price_gets_a_style_rewrite(text):
    execution = ApprovedExecution(operation='present_offer', offer={'price_cents': 3000})
    assert 'unsolicited_offer_price' in live_orchestration.writer_contract_reasons(
        [text], loaded(), execution, mode='auto'
    )
    execution.offer['price_cents'] = 4000
    assert 'unapproved_price_claim' in live_orchestration.writer_contract_reasons(
        [text], loaded(), execution, mode='auto'
    )


def test_bare_price_detection_does_not_turn_ordinary_numbers_into_prices():
    assert live_orchestration._mentioned_prices('It is 30 degrees outside; I have 2 questions.') == set()


def test_review_resume_reuses_the_exact_unsent_plan_without_replanning(monkeypatch):
    evidence = loaded(pending_offer=offer(cents=3000))
    evidence.commercial_state.accepted_offer_id = 'offer-1'
    evidence.active_session = {
        'status': 'active', 'commercial_offer_id': 'offer-1', 'current_index': 0,
        'plan': [{'set_id': 'set-1', 'price_cents': 3000,
                  'media_ids': ['sim:source:one'], 'sent': False, 'purchased': False}],
    }
    decision = ConversationDecision(proposed_operation=ProposedOperation(
        kind=OperationKind.SEND_LOCKED_PAID_MESSAGE, offer_id='offer-1', set_id='set-1', subject='the approved item'
    ))
    monkeypatch.setattr(live_orchestration, 'plan_session_for_fan', _retired)
    execution = run(live_orchestration._prepare_execution(decision, evidence, execute_operations=True))
    assert execution.validation.approved
    assert execution.delivery['media_ids'] == ['sim:source:one']
    assert execution.delivery['price_cents'] == 3000
    evidence.fan.platform_fan_id = 'real-fan'
    assert not live_orchestration._resumable_locked_session(evidence, evidence.commercial_state.pending_offer)
    evidence.fan.platform_fan_id = 'test_fan_1'
    for field, bad in [('sent', True), ('purchased', True), ('set_id', 'another'), ('price_cents', 9000)]:
        step = evidence.active_session['plan'][0]
        before = step[field]
        step[field] = bad
        assert not live_orchestration._resumable_locked_session(evidence, evidence.commercial_state.pending_offer)
        step[field] = before
    evidence.active_session['awaiting_purchase_index'] = 0
    assert not live_orchestration.validate_decision(decision, evidence).approved
