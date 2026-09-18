from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from models.commercial import CreatorPolicy, FanCommercialState, FanStatus, Offer
from models.conversation_decision import (
    ConversationDecision,
    OperationKind,
    ProposedOperation,
)
from models.live_orchestration import (
    ApprovedExecution,
    EvidenceSnapshot,
    TurnTrigger,
    ValidationResult,
)
from models.schemas import Fan, Message, Persona, SuggestionResponse
from services import conversation_core, live_orchestration, proactive, suggestions
from services.context_packet import ContextPacket
from services.decision_owners import parse_semantic_decision_result
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


def test_context_ceiling_keeps_latest_trigger_and_transaction_evidence(monkeypatch):
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
    monkeypatch.setattr(
        live_orchestration, "recent_episodes_for", lambda *_a: value([])
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


def test_simulator_entrypoint_selects_the_real_semantic_auto_path(monkeypatch):
    configure_auto_route(monkeypatch, outcome={"outcome": "no_send", "message_ids": []})
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
    monkeypatch.setattr(
        suggestions, "learn_from_fan_message", lambda **_k: value(None)
    )

    result = run(
        suggestions.run_simulated_inbound(
            fan_id="fan-1",
            creator_id="creator-1",
            message="ordinary simulator turn",
            fast=True,
        )
    )

    assert result["simulation"] is True
    assert result["outcome"] == "no_send"
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
