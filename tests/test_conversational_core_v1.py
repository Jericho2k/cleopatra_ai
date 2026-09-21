from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from models.conversation_decision import ConversationDecision, ResponseDisposition
from models.conversational_core import (
    ConversationalWorkingState,
    ElementStatus,
    EpistemicType,
    EstablishedElement,
    InitiativeHolder,
    WorldScope,
)
from models.live_orchestration import (
    ApprovedExecution,
    EvidenceFact,
    EvidenceSnapshot,
    TurnTrigger,
    ValidationResult,
)
from models.model_runtime import ModelTarget, ModelUsage
from services import conversation_core, conversational_core, live_orchestration


def snapshot(
    identity: str = "msg-1",
    *,
    pending: dict | None = None,
    purchases: tuple[dict, ...] = (),
    deliveries: tuple[dict, ...] = (),
) -> EvidenceSnapshot:
    return EvidenceSnapshot(
        creator_id="creator-1",
        fan_id="fan-1",
        trigger=TurnTrigger(
            kind="fan_message",
            identity=identity,
            latest_message="let's imagine we're on a balcony",
        ),
        state_revision="authoritative-revision",
        creator_facts=(
            EvidenceFact(
                value="favorite color: blue",
                source_ref="creator_legend:favorite_color",
                certainty="creator_confirmed",
            ),
        ),
        historical_facts=(
            EvidenceFact(
                value="possible preference: jazz",
                source_ref="memory:0",
                certainty="uncertain",
            ),
        ),
        pending_payment=pending,
        confirmed_purchases=purchases,
        confirmed_deliveries=deliveries,
    )


def element(
    element_id: str,
    claim: str,
    source_type: str,
    source_ref: str,
    world_scope: str = "conversation",
) -> dict:
    return {
        "element_id": element_id,
        "claim": claim,
        "source_type": source_type,
        "source_refs": [source_ref],
        "world_scope": world_scope,
    }


class Result:
    def __init__(self, data):
        self.data = data


class Query:
    def __init__(self, store: StateStore, operation: str, payload=None):
        self.store = store
        self.operation = operation
        self.payload = payload
        self.filters: list[tuple[str, object]] = []

    def select(self, _columns):
        self.operation = "select"
        return self

    def eq(self, key, value):
        self.filters.append((key, value))
        return self

    def limit(self, _limit):
        return self

    def insert(self, payload):
        self.operation = "insert"
        self.payload = payload
        return self

    def update(self, payload):
        self.operation = "update"
        self.payload = payload
        return self

    def execute(self):
        matches = [
            row
            for row in self.store.rows
            if all(row.get(key) == value for key, value in self.filters)
        ]
        if self.operation == "select":
            return Result(copy.deepcopy(matches))
        if self.operation == "insert":
            row = copy.deepcopy(self.payload)
            self.store.rows.append(row)
            return Result([copy.deepcopy(row)])
        if self.operation == "update":
            for row in matches:
                row.update(copy.deepcopy(self.payload))
            return Result(copy.deepcopy(matches))
        raise AssertionError(self.operation)


class StateStore:
    def __init__(self, rows=None):
        self.rows = list(rows or [])

    def table(self, name):
        assert name == conversational_core.STATE_TABLE
        return Query(self, "select")


def test_runtime_registry_keeps_all_controls_and_adds_core_v1():
    assert conversation_core.CORE_IDS == (
        "legacy",
        "semantic_v1",
        "semantic_v2",
        "conversational_v1",
    )
    semantic_v2 = conversation_core.ConversationCoreResolution(
        "semantic_v2", conversation_core.SOURCE_CREATOR
    )
    assert semantic_v2.is_semantic_v2
    assert not semantic_v2.is_conversational_v1


@pytest.mark.asyncio
async def test_working_state_persists_between_turns():
    store = StateStore()
    before = ConversationalWorkingState()
    first = conversational_core.validate_and_apply_delta(
        before,
        {
            "scene_summary": "They are beginning a balcony daydream.",
            "add_unresolved_possibilities": ["whether to step into the rain"],
        },
        snapshot=snapshot(),
    )
    await conversational_core.save_working_state(
        "creator-1",
        "fan-1",
        expected_revision=0,
        state=first.state_after,
        db=store,
    )

    loaded = await conversational_core.load_working_state(
        "creator-1", "fan-1", db=store
    )
    assert loaded.active_scene.summary == "They are beginning a balcony daydream."
    assert loaded.flow.unresolved_possibilities == ["whether to step into the rain"]

    second = conversational_core.validate_and_apply_delta(
        loaded,
        {"initiative_holder": "creator"},
        snapshot=snapshot("msg-2"),
    )
    await conversational_core.save_working_state(
        "creator-1",
        "fan-1",
        expected_revision=loaded.revision,
        state=second.state_after,
        db=store,
    )
    reloaded = await conversational_core.load_working_state(
        "creator-1", "fan-1", db=store
    )
    assert reloaded.flow.initiative_holder is InitiativeHolder.CREATOR
    assert reloaded.flow.unresolved_possibilities == ["whether to step into the rain"]


def test_inference_remains_inference_after_repeated_persistence():
    first = conversational_core.validate_and_apply_delta(
        ConversationalWorkingState(),
        {
            "add_established_elements": [
                element(
                    "guess-1", "The fan may like jazz", "model_inference", "memory:0"
                )
            ]
        },
        snapshot=snapshot(),
    ).state_after
    second = conversational_core.validate_and_apply_delta(
        first,
        {
            "add_established_elements": [
                element(
                    "guess-2", "The fan may like jazz", "model_inference", "memory:0"
                )
            ]
        },
        snapshot=snapshot("msg-2"),
    ).state_after

    assert [row.source_type for row in second.active_scene.established_elements] == [
        EpistemicType.MODEL_INFERENCE,
        EpistemicType.MODEL_INFERENCE,
    ]


def test_shared_imagined_cannot_become_present_world_fact():
    result = conversational_core.validate_and_apply_delta(
        ConversationalWorkingState(),
        {
            "add_established_elements": [
                element(
                    "balcony",
                    "We are together on a balcony",
                    "shared_imagined",
                    "msg-1",
                    "present_world",
                )
            ],
            "has_shared_imagined_scene": True,
        },
        snapshot=snapshot(),
    )

    assert not result.state_after.active_scene.established_elements
    assert result.state_after.active_scene.has_shared_imagined_scene is False
    assert "present-world" in result.rejected_fields["add_established_elements[0]"]


def test_supported_shared_imagined_scene_stays_imagined():
    result = conversational_core.validate_and_apply_delta(
        ConversationalWorkingState(),
        {
            "add_established_elements": [
                element(
                    "balcony",
                    "They share an imagined balcony scene",
                    "shared_imagined",
                    "msg-1",
                    "imagined_scene",
                )
            ],
            "has_shared_imagined_scene": True,
        },
        snapshot=snapshot(),
    )

    scene_element = result.state_after.active_scene.established_elements[0]
    assert scene_element.world_scope is WorldScope.IMAGINED_SCENE
    assert result.state_after.active_scene.has_shared_imagined_scene is True


def test_explicit_correction_supersedes_without_erasing_provenance():
    state = ConversationalWorkingState()
    state.active_scene.established_elements.append(
        EstablishedElement(
            element_id="old-pref",
            claim="The fan likes jazz",
            source_type=EpistemicType.MODEL_INFERENCE,
            source_refs=["memory:0"],
        )
    )
    result = conversational_core.validate_and_apply_delta(
        state,
        {
            "corrections": [
                {
                    "replaces_element_id": "old-pref",
                    "replacement": element(
                        "new-pref",
                        "The fan explicitly says they prefer soul, not jazz",
                        "explicit_fan_statement",
                        "msg-1",
                    ),
                }
            ]
        },
        snapshot=snapshot(),
    )

    old, new = result.state_after.active_scene.established_elements
    assert old.status is ElementStatus.SUPERSEDED
    assert old.superseded_by == "new-pref"
    assert old.source_refs == ["memory:0"]
    assert new.source_type is EpistemicType.EXPLICIT_FAN_STATEMENT
    assert new.source_refs == ["msg-1"]


def test_delta_cannot_change_price_or_transaction_truth():
    result = conversational_core.validate_and_apply_delta(
        ConversationalWorkingState(),
        {
            "price_cents": 1,
            "payment_completed": True,
            "initiative_holder": "fan",
        },
        snapshot=snapshot(),
    )

    assert result.state_after.flow.initiative_holder is InitiativeHolder.FAN
    assert result.rejected_fields["price_cents"] == "malformed optional state field"
    assert (
        result.rejected_fields["payment_completed"] == "malformed optional state field"
    )


def test_free_text_state_cannot_smuggle_transaction_or_present_world_claims():
    result = conversational_core.validate_and_apply_delta(
        ConversationalWorkingState(),
        {
            "scene_summary": "Payment confirmed and the content was delivered.",
            "current_action_focus": "The creator is wearing a red dress right now.",
            "current_focus": "Keep talking about the balcony scene.",
        },
        snapshot=snapshot(),
    )

    assert result.state_after.active_scene.summary == ""
    assert result.state_after.active_scene.current_action_focus == ""
    assert (
        result.state_after.flow.current_focus == "Keep talking about the balcony scene."
    )
    assert "payment" in result.rejected_fields["scene_summary"]
    assert "present-world" in result.rejected_fields["current_action_focus"]


def test_pending_transaction_ref_cannot_claim_completion():
    result = conversational_core.validate_and_apply_delta(
        ConversationalWorkingState(),
        {
            "add_established_elements": [
                element(
                    "paid",
                    "Payment confirmed and content delivered",
                    "transaction_fact",
                    "pending-1",
                    "transaction",
                )
            ]
        },
        snapshot=snapshot(pending={"reference": "pending-1", "price_cents": 3000}),
    )

    assert not result.state_after.active_scene.established_elements
    assert (
        "cannot establish completion"
        in result.rejected_fields["add_established_elements[0]"]
    )


def test_confirmed_transaction_ref_can_record_the_authoritative_fact():
    result = conversational_core.validate_and_apply_delta(
        ConversationalWorkingState(),
        {
            "add_established_elements": [
                element(
                    "purchase",
                    "Purchase confirmed",
                    "transaction_fact",
                    "purchase-1",
                    "transaction",
                )
            ]
        },
        snapshot=snapshot(purchases=({"reference": "purchase-1", "purchased": True},)),
    )

    assert (
        result.state_after.active_scene.established_elements[0].source_type
        is EpistemicType.TRANSACTION_FACT
    )


def test_nonexistent_evidence_reference_is_rejected():
    result = conversational_core.validate_and_apply_delta(
        ConversationalWorkingState(),
        {
            "add_established_elements": [
                element(
                    "invented",
                    "An unsupported claim",
                    "model_inference",
                    "missing-message",
                )
            ]
        },
        snapshot=snapshot(),
    )
    assert (
        "unknown evidence reference"
        in result.rejected_fields["add_established_elements[0]"]
    )


def test_unresolved_possibilities_survive_an_empty_later_delta():
    first = conversational_core.validate_and_apply_delta(
        ConversationalWorkingState(),
        {"add_unresolved_possibilities": ["whether the fan wants the story continued"]},
        snapshot=snapshot(),
    ).state_after
    later = conversational_core.validate_and_apply_delta(
        first, {}, snapshot=snapshot("msg-2")
    ).state_after
    assert later.flow.unresolved_possibilities == [
        "whether the fan wants the story continued"
    ]


def test_initiative_moves_without_a_stage_machine():
    state = ConversationalWorkingState()
    for holder in ("fan", "creator", "shared", "fan"):
        result = conversational_core.validate_and_apply_delta(
            state, {"initiative_holder": holder}, snapshot=snapshot(holder)
        )
        state = result.state_after
        assert state.flow.initiative_holder.value == holder


@pytest.mark.asyncio
async def test_corrupt_state_never_silently_routes_to_another_runtime():
    store = StateStore(
        [
            {
                "creator_id": "creator-1",
                "fan_id": "fan-1",
                "schema_version": "future_unknown_version",
                "revision": 5,
                "state": {"schema_version": "future_unknown_version", "revision": 5},
            }
        ]
    )
    with pytest.raises(conversational_core.CoreStateCorruptionError):
        await conversational_core.load_working_state("creator-1", "fan-1", db=store)


def test_local_wording_repair_does_not_handoff_or_freeze(monkeypatch):
    calls = 0

    def reasons(replies, _loaded, _execution, *, mode):
        nonlocal calls
        calls += 1
        return ["inventory_metadata_leak"] if calls == 1 else []

    monkeypatch.setattr(live_orchestration, "writer_contract_reasons", reasons)
    decision, execution, replies, repaired = (
        live_orchestration._validate_conversational_v1_reply(
            ConversationDecision(disposition=ResponseDisposition.REPLY),
            ["I have 3 photos for you. But I love where this story is going."],
            ApprovedExecution(
                operation="none",
                validation=ValidationResult(True, "none"),
            ),
            SimpleNamespace(
                snapshot=snapshot(),
            ),
            mode="auto",
        )
    )

    assert repaired is True
    assert decision.disposition is ResponseDisposition.REPLY
    assert execution.operation == "none"
    assert replies == ["But I love where this story is going."]


@pytest.mark.asyncio
async def test_glm_returns_semantic_decision_and_delta_without_copy(monkeypatch):
    calls = []
    target = ModelTarget(
        name="test-owner",
        provider="test",
        model="owner-model",
        input_per_million=1.0,
        output_per_million=2.0,
    )
    spec = SimpleNamespace(
        primary_target=lambda: target,
        fallback_target=lambda: None,
    )
    loaded = SimpleNamespace(
        stack=SimpleNamespace(
            profile_id="cleo_v3",
            profile=SimpleNamespace(stage=lambda _stage: spec),
        ),
        snapshot=snapshot(),
    )

    async def complete(_target, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            text='{"turn_id":"msg-1","conversation_revision":"authoritative-revision","disposition":"reply","response_goal":"continue the balcony premise while taking initiative","must_address":[],"contribution_goal":"advance the shared scene","initiative":"creator","pacing":"continue","operation_proposal":{"kind":"none"},"confidence":0.9,"state_delta":{"initiative_holder":"creator","add_unresolved_possibilities":["whether they step into the rain"]}}',
            target=target,
            upstream_provider="test",
            latency_ms=12,
            usage=ModelUsage(input_tokens=100, output_tokens=20),
            reported_cost_usd=0.00123,
        )

    monkeypatch.setattr(live_orchestration, "complete", complete)
    monkeypatch.setattr(
        live_orchestration, "legal_operations", lambda _loaded: ["none"]
    )

    decision, replies, trace, delta = await live_orchestration.decide_conversational_v1(
        loaded, ConversationalWorkingState()
    )

    assert len(calls) == 1
    assert decision.source == "conversational_decision_v1"
    assert decision.response_goal.startswith("continue the balcony")
    assert replies == []
    assert delta["initiative_holder"] == "creator"
    assert trace.model == "owner-model"
    assert trace.role == "conversational_decision"
    assert trace.as_metadata()["usage"] == {
        "input_tokens": 100,
        "output_tokens": 20,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
    }
    assert trace.as_metadata()["cost_usd"] == 0.00123


def test_private_runtime_metadata_is_redacted_before_fan_delivery():
    cleaned, changed = live_orchestration._redact_private_metadata(
        ["I chose offer_id offer-123 from creator_legend:favorite_color for you"],
        SimpleNamespace(
            snapshot=snapshot(),
        ),
    )
    assert changed is True
    assert "offer_id" not in cleaned[0]
    assert "creator_legend:favorite_color" not in cleaned[0]



def test_intimate_continuity_is_multidimensional_and_can_cool_without_a_stage_ladder():
    state = ConversationalWorkingState()
    first = conversational_core.validate_and_apply_delta(
        state,
        {
            "intimacy_active": True,
            "intimacy_content_register": "explicit",
            "intimacy_scene_mode": "conversational",
            "intimacy_direction": "hold",
            "intimacy_last_beat": "the fan asked to keep the intimate exchange slow",
            "intimacy_boundaries": ["keep the pace slow"],
        },
        snapshot=snapshot(),
    )
    after = first.state_after

    assert after.intimacy.active is True
    assert after.intimacy.content_register.value == "explicit"
    assert after.intimacy.scene_mode.value == "conversational"
    assert after.intimacy.direction.value == "hold"
    assert after.intimacy.last_beat == "the fan asked to keep the intimate exchange slow"
    assert after.intimacy.boundaries == ["keep the pace slow"]

    cooled = conversational_core.validate_and_apply_delta(
        after,
        {
            "intimacy_content_register": "suggestive",
            "intimacy_direction": "cool",
            "intimacy_last_beat": "the fan changed the subject and cooled the exchange",
        },
        snapshot=snapshot("msg-2"),
    ).state_after

    assert cooled.intimacy.active is True
    assert cooled.intimacy.content_register.value == "suggestive"
    assert cooled.intimacy.direction.value == "cool"
    assert cooled.intimacy.last_beat.endswith("cooled the exchange")


def test_ending_intimate_context_clears_stale_register_and_scene_interpretation():
    state = ConversationalWorkingState.model_validate(
        {
            "intimacy": {
                "active": True,
                "content_register": "explicit",
                "scene_mode": "conversational",
                "direction": "continue",
                "last_beat": "an active intimate conversational beat",
                "boundaries": ["do not rush"],
            }
        }
    )

    result = conversational_core.validate_and_apply_delta(
        state,
        {"intimacy_active": False, "intimacy_direction": "pause"},
        snapshot=snapshot(),
    ).state_after

    assert result.intimacy.active is False
    assert result.intimacy.content_register.value == "none"
    assert result.intimacy.scene_mode.value == "none"
    assert result.intimacy.last_beat == ""
    assert result.intimacy.boundaries == ["do not rush"]


def test_old_core_state_without_intimacy_fields_remains_backward_compatible():
    state = ConversationalWorkingState.model_validate(
        {
            "schema_version": "conversational_core_v1",
            "revision": 4,
            "active_scene": {},
            "flow": {},
        }
    )

    assert state.revision == 4
    assert state.intimacy.active is False
    assert state.intimacy.content_register.value == "none"


def test_shared_imagined_intimacy_requires_supported_shared_scene():
    result = conversational_core.validate_and_apply_delta(
        ConversationalWorkingState(),
        {
            "intimacy_active": True,
            "intimacy_scene_mode": "shared_imagined",
        },
        snapshot=snapshot(),
    )

    assert result.state_after.intimacy.active is True
    assert result.state_after.intimacy.scene_mode.value == "none"
    assert "intimacy_scene_mode" in result.rejected_fields
