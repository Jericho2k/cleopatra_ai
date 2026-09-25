"""Behavioural acceptance tests for Conversational Core v2 (session-aware).

Every multi-turn test runs the real turn path (see tests/conversational_v2_harness.py):
real ``load_evidence``, real state persistence against a PostgREST-shaped fake,
real reconciliation and validation, real ``execute_auto_turn``. Only data
sources and the GLM/Kimi transports are scripted. Scenarios are neutral.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from models.commercial import Offer
from models.conversational_core import WorldScope
from models.conversational_session import (
    BeatSource,
    ContentLifecycle,
    ConversationalSessionState,
    SessionStatus,
)
from models.live_orchestration import EvidenceSnapshot, TurnTrigger
from models.schemas import Fan
from services import conversation_core, conversational_session, conversational_v2
from services import live_orchestration as lo
from services import suggestions
from services.session_immersion_eval import TurnRecord, score_session_trajectory
from tests.conversational_v2_harness import (
    CREATOR_ID,
    FAN_ID,
    Item,
    V2World,
    install,
    owner_json,
    writer_payload,
)

V1 = conversation_core.CORE_CONVERSATIONAL_V1
V2 = conversation_core.CORE_CONVERSATIONAL_V2


def catalog() -> dict[str, Item]:
    return {
        "A": Item("A", 4000, "content A: the first part of the date scenario"),
        "B": Item("B", 7500, "content B: a later part of the date scenario"),
        "C": Item("C", 3000, "content C: a short clip", asset_type="video"),
    }


@pytest.fixture
def world(monkeypatch):
    return install(monkeypatch, V2World(catalog=catalog()))


def start_session(message_ref: str, *, beats=None, handle_b: str = "") -> dict:
    delta = {
        "status": "active",
        "experience_premise": {
            "summary": "an interactive date scenario at an imagined rooftop cafe",
            "world_scope": "imagined_scene",
            "source_refs": [message_ref],
        },
        "interaction_goal": "build the date scenario together at an unhurried pace",
        "fan_participation": {"mode": "co_creating", "gist": "he set the scene"},
        "trajectory": {
            "reason": "",
            "beats": beats
            if beats is not None
            else [
                {"beat_id": "arrive", "kind": "conversation", "intent": "settle into the scenario"},
                {"beat_id": "his_turn", "kind": "participation", "intent": "let him choose what happens next"},
            ]
            + (
                [
                    {
                        "beat_id": "reveal_b",
                        "kind": "content",
                        "intent": "a visual moment once the scenario has built",
                        "candidate_handle": handle_b,
                        "media_role": "gives him the moment the scenario has been building to",
                    }
                ]
                if handle_b
                else []
            ),
        },
        "start_beat": {"beat_id": "arrive"},
        "tempo": "build",
    }
    return delta


def last_owner_session(world: V2World) -> dict:
    return world.owner_payloads[-1]["interaction_session"]


# ---------------------------------------------------------------------------
# 1. v1 remains unchanged
# ---------------------------------------------------------------------------

V1_OWNER_PAYLOAD_KEYS = {
    "turn_id",
    "conversation_revision",
    "evidence_snapshot",
    "evidence_catalog",
    "working_state",
    "working_state_fingerprint",
    "platform_context",
    "legal_operations",
    "scheduled_intent_affordance",
}
V1_WRITER_PAYLOAD_KEYS = {
    "raw_conversation",
    "creator_voice",
    "sourced_memory",
    "working_context",
    "semantic_decision",
    "platform_context",
    "deterministic_facts",
    "prepared_operation_facts",
    "execution_reality",
    "grounding",
    "voice_rhythm",
    "hermes_examples",
    "hermes_examples_notice",
    "mode",
}


@pytest.mark.asyncio
async def test_v1_turn_path_is_unchanged_and_never_touches_v2_state(monkeypatch):
    world = install(monkeypatch, V2World(core=V1, catalog=catalog()))
    result = await world.turn(
        "let's imagine a rooftop cafe",
        owner_json(state_delta={"initiative_holder": "creator"}),
        ["a rooftop cafe sounds lovely, you pick the table"],
    )

    assert result["outcome"] == "replied"
    assert world.owner_systems[-1] == lo.CONVERSATIONAL_V1_SYSTEM
    assert "SESSION-AWARE" not in lo.CONVERSATIONAL_V1_SYSTEM
    assert set(world.owner_payloads[-1]) == V1_OWNER_PAYLOAD_KEYS
    # v1 still sees exactly one next unlock, never the v2 candidate set.
    inventory = world.owner_payloads[-1]["evidence_snapshot"]["approved_inventory"]
    assert [row["candidate_handle"] for row in inventory] == ["offer_candidate_1"]
    writer = world.writer_prompts[-1]
    assert set(writer_payload(writer)) == V1_WRITER_PAYLOAD_KEYS
    assert "SESSION CONTEXT" not in writer[0]["content"]
    # v1 persisted its own working state; the v2 table is untouched.
    assert world.db.tables[conversational_session.SESSION_TABLE] == []
    assert len(world.db.tables["conversational_core_states"]) == 1


def test_v1_registry_identity_and_writer_prompt_default_are_unchanged():
    resolution = conversation_core.ConversationCoreResolution(V1, "creator")
    assert resolution.is_conversational_v1 and not resolution.is_conversational_v2
    v2 = conversation_core.ConversationCoreResolution(V2, "simulation_fan")
    assert v2.is_semantic and v2.is_conversational and v2.is_conversational_v2
    assert not v2.is_conversational_v1
    assert conversation_core.CORE_LABELS[V2] == "Conversational Core v2 — Session-aware"
    assert conversation_core.normalize_core_id("CONVERSATIONAL_V2") == V2


# ---------------------------------------------------------------------------
# 2 + 5. State survives turns; an active session can talk for many turns
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_session_state_survives_turns_and_carries_many_conversational_beats(world):
    first = world.fan_says("let's pretend we're on a date at a rooftop cafe")
    world.queue(
        owner_json(
            move="develop_premise",
            session_delta=start_session(first),
        ),
        ["okay, rooftop it is. i already picked the corner table"],
    )
    await lo.run_auto_turn(
        creator_id=CREATOR_ID,
        fan_id=FAN_ID,
        latest_message="let's pretend we're on a date at a rooftop cafe",
        trigger_identity=first,
        conversation_core=V2,
    )
    state = world.session_state()
    assert state.session.status is SessionStatus.ACTIVE
    assert state.session.experience_premise.world_scope is WorldScope.IMAGINED_SCENE
    assert state.session.current_beat.beat_id == "arrive"
    revision_after_first = state.revision

    moves = ["react", "linger", "invite_participation", "callback", "converse"]
    lines = [
        "i'd order something sweet",
        "tell me what you'd order",
        "you choose the next thing we do",
        "remember the corner table",
        "this is nice",
    ]
    for index, (move, line) in enumerate(zip(moves, lines)):
        delta = {}
        if index == 1:
            delta = {"complete_current_beat": {"outcome": "settled in"}, "start_beat": {"beat_id": "his_turn"}}
        result = await world.turn(
            line,
            owner_json(move=move, session_delta=delta),
            [f"creator beat {index}"],
        )
        assert result["outcome"] == "replied"
        view = last_owner_session(world)
        # Every later turn is decided FROM the persisted session, not from scratch.
        assert view["status"] == "active"
        assert view["experience_premise"]["summary"].startswith("an interactive date")

    state = world.session_state()
    assert state.revision > revision_after_first
    assert state.session.turns_in_session == 6
    assert [beat.beat_id for beat in state.session.completed_beats] == ["arrive"]
    assert state.session.current_beat.beat_id == "his_turn"
    assert state.session.next_experience_move.kind.value == "converse"
    # Five conversational turns inside an active session with no media at all.
    assert all(row.get("operation") == "none" for row in world.transcript if row["speaker"] == "creator")
    assert world.pending_offer is None and world.sent_ppv == []
    assert state.session.used_content == []


# ---------------------------------------------------------------------------
# 3. Tentative content is never offered / sent / purchased state
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_planned_content_is_not_offered_sent_or_purchased(world):
    first = world.fan_says("let's pretend we're on a date")
    world.queue(
        lambda payload: owner_json(
            session_delta={
                **start_session(first, handle_b=world.handle_for(payload, "B")),
                # A model trying to write application-owned facts:
                "used_content": [{"content_ref": "B", "lifecycle": "purchased"}],
            }
        ),
        ["sure, let's start slow"],
    )
    result = await lo.run_auto_turn(
        creator_id=CREATOR_ID,
        fan_id=FAN_ID,
        latest_message="let's pretend we're on a date",
        trigger_identity=first,
        conversation_core=V2,
    )
    state = world.session_state()
    planned = [beat for beat in state.session.tentative_trajectory if beat.content_ref]
    assert [beat.content_ref for beat in planned] == ["B"]
    # Planned is not presented, accepted, sent or purchased.
    assert state.session.used_content == []
    assert state.session.lifecycle_of("B") is None
    assert world.pending_offer is None
    assert world.sent_ppv == []
    assert world.pending_payment is None
    assert result["outcome"] == "replied"

    await world.turn("what should we do next", owner_json(), ["let's keep talking"])
    view = last_owner_session(world)
    b_candidate = next(
        row for row in view["content_candidates"] if row["description"].startswith("content B")
    )
    assert b_candidate["fact"] == "eligible_candidate_not_offered"
    assert b_candidate["planned_in_beat"] == "reveal_b"
    assert view["content_ledger"] == []
    # The writer is never told about planned future content.
    session_for_writer = writer_payload(world.writer_prompts[-1])["interaction_session"]
    assert "tentative_trajectory" not in session_for_writer
    assert "content_candidates" not in session_for_writer


def test_model_cannot_write_application_owned_content_facts():
    snapshot = EvidenceSnapshot(
        creator_id="c", fan_id="f", trigger=TurnTrigger("fan_message", "m1", "hi"), state_revision="r"
    )
    result = conversational_session.validate_session_delta(
        ConversationalSessionState(),
        {"used_content": [{"content_ref": "A", "lifecycle": "sent"}], "last_event": {"kind": "purchased"}},
        snapshot=snapshot,
    )
    assert "application-owned" in result.rejected_fields["used_content"]
    assert "application-owned" in result.rejected_fields["last_event"]
    assert result.state_after.session.used_content == []


# ---------------------------------------------------------------------------
# 4 + 6 + 12. Ledger authority, post-purchase conversation, no reuse
# ---------------------------------------------------------------------------


async def _present_send_and_buy_a(world: V2World) -> None:
    first = world.fan_says("let's pretend we're on a date")
    world.queue(
        lambda payload: owner_json(session_delta=start_session(first, handle_b=world.handle_for(payload, "B"))),
        ["okay, i'm in"],
    )
    await lo.run_auto_turn(
        creator_id=CREATOR_ID, fan_id=FAN_ID, latest_message="let's pretend we're on a date",
        trigger_identity=first, conversation_core=V2,
    )
    await world.turn(
        "can i buy content A?",
        lambda payload: owner_json(
            move="use_content",
            operation={
                "kind": "present_offer",
                "subject": "content A",
                "because": "he asked for it directly",
                "candidate_handle": world.handle_for(payload, "A"),
            },
        ),
        ["yes, it's the start of our date"],
    )
    assert world.pending_offer is not None and world.pending_offer.set_id == "A"
    await world.turn(
        "yes send it",
        owner_json(
            move="use_content",
            operation={
                "kind": "send_locked_paid_message",
                "subject": "content A",
                "because": "he accepted the pending offer",
                "candidate_handle": "pending_offer_1",
            },
        ),
        ["here it is"],
    )
    assert [row["set_id"] for row in world.sent_ppv] == ["A"]
    world.purchase("A")


@pytest.mark.asyncio
async def test_after_a_purchase_the_next_beat_stays_conversational(world):
    await _present_send_and_buy_a(world)

    # The owner is ALLOWED to stay in the moment: no operation, the turn replies.
    result = await world.turn("that was lovely", owner_json(move="react"), ["i'm glad. stay a minute"])
    assert result["outcome"] == "replied"
    view = last_owner_session(world)
    assert view["last_content_event"] == {"fact": "purchased", "happened_this_turn": True}
    state = world.session_state()
    assert state.session.lifecycle_of("A") is ContentLifecycle.PURCHASED
    assert state.session.last_event.kind == "purchased"
    assert world.pending_offer is None
    writer_session = writer_payload(world.writer_prompts[-1])["interaction_session"]
    assert "purchased" in writer_session["what_just_happened"]


@pytest.mark.asyncio
async def test_a_purchase_never_auto_authorises_the_next_paid_item(world):
    await _present_send_and_buy_a(world)
    # Owner tries to sell B right after the purchase, without the fan asking.
    result = await world.turn(
        "that was lovely",
        lambda payload: owner_json(
            move="use_content",
            operation={
                "kind": "present_offer",
                "subject": "content B",
                "because": "next in the plan",
                "candidate_handle": world.handle_for(payload, "B"),
            },
        ),
        ["i'm glad you liked it"],
    )
    assert result["outcome"] == "replied"
    assert world.pending_offer is None
    assert world.transcript[-1]["operation"] == "none"


@pytest.mark.asyncio
async def test_after_a_purchase_the_fan_can_still_ask_for_more(world):
    await _present_send_and_buy_a(world)
    await world.turn(
        "can i buy another one?",
        lambda payload: owner_json(
            move="use_content",
            operation={
                "kind": "present_offer",
                "subject": "content B",
                "because": "he asked for more",
                "candidate_handle": world.handle_for(payload, "B"),
            },
        ),
        ["there's a later part of our date"],
    )
    assert world.pending_offer is not None and world.pending_offer.set_id == "B"


@pytest.mark.asyncio
async def test_transactions_stay_application_authoritative(world):
    first = world.fan_says("let's pretend we're on a date")
    world.queue(owner_json(session_delta=start_session(first)), ["okay"])
    await lo.run_auto_turn(
        creator_id=CREATOR_ID, fan_id=FAN_ID, latest_message="let's pretend we're on a date",
        trigger_identity=first, conversation_core=V2,
    )
    # A fan's claim is not a receipt; model prose cannot manufacture one.
    claim = await world.turn(
        "i paid for content A already",
        owner_json(
            session_delta={
                "complete_current_beat": {"outcome": "he purchased content A"},
                "interaction_goal": "celebrate that the payment was confirmed",
                "add_constraints": [
                    {
                        "constraint_id": "paid",
                        "kind": "other",
                        "statement": "he has access",
                        "source_type": "transaction_fact",
                        "source_refs": ["invented-receipt"],
                    }
                ],
            }
        ),
        ["let me check on that"],
    )
    assert claim["outcome"] == "replied"
    state = world.session_state()
    assert state.session.used_content == []
    assert state.session.current_beat.beat_id == "arrive"  # completion refused
    assert state.session.interaction_goal.startswith("build the date")
    assert state.session.known_constraints == []
    packet = world.owner_payloads[-1]
    assert packet["interaction_session"]["content_ledger"] == []


def test_reconciliation_never_regresses_a_consumed_content_fact():
    purchased = EvidenceSnapshot(
        creator_id="c",
        fan_id="f",
        trigger=TurnTrigger("fan_message", "m1", "hi"),
        state_revision="r",
        confirmed_deliveries=({"set_id": "A", "platform_message_id": "p1", "delivered": True},),
        confirmed_purchases=({"set_id": "A", "reference": "buy-1", "price_cents": 4000, "purchased": True},),
    )
    state, changes = conversational_session.reconcile_with_authority(
        ConversationalSessionState(), snapshot=purchased
    )
    assert state.session.lifecycle_of("A") is ContentLifecycle.PURCHASED
    assert changes
    # A later, truncated ledger page no longer shows A. It stays consumed.
    truncated = EvidenceSnapshot(
        creator_id="c", fan_id="f", trigger=TurnTrigger("fan_message", "m2", "hi"), state_revision="r"
    )
    later, later_changes = conversational_session.reconcile_with_authority(state, snapshot=truncated)
    assert later.session.lifecycle_of("A") is ContentLifecycle.PURCHASED
    assert "A" in later.session.consumed_refs()
    assert later_changes == []


@pytest.mark.asyncio
async def test_delivered_content_can_never_be_treated_as_unused(world):
    await _present_send_and_buy_a(world)
    await world.turn("that was lovely", owner_json(move="react"), ["glad"])
    view = last_owner_session(world)
    descriptions = [row["description"] for row in view["content_candidates"]]
    assert not any(text.startswith("content A") for text in descriptions)
    assert {"content": "earlier_item_1", "fact": "purchased"} in view["content_ledger"]

    state = world.session_state()
    # Even a stale candidate list that still contained A cannot bind or sell it.
    stale = Offer(offer_id="offer:A", label="x", price_cents=4000, set_id="A")
    fake_loaded = SimpleNamespace(candidate_handles={"offer_candidate_1": stale})
    assert conversational_v2.exposed_candidates(fake_loaded, state) == {}
    result = conversational_session.validate_session_delta(
        state,
        {"trajectory": {"reason": "replay", "beats": [
            {"kind": "content", "candidate_handle": "offer_candidate_1", "media_role": "again"}
        ]}},
        snapshot=EvidenceSnapshot(
            creator_id="c", fan_id="f", trigger=TurnTrigger("fan_message", "m9", "x"), state_revision="r"
        ),
        candidates={"offer_candidate_1": stale},
    )
    assert "already sent or purchased" in result.rejected_fields["trajectory.beats[0]"]
    loaded = SimpleNamespace(
        commercial_state=SimpleNamespace(pending_offer=None),
        next_offer=stale,
        snapshot=EvidenceSnapshot(
            creator_id="c", fan_id="f", trigger=TurnTrigger("fan_message", "m9", "more please"),
            state_revision="r", commercial_opportunity={"fan_stated_buying_signal": True},
        ),
    )
    from models.conversation_decision import ConversationDecision, OperationKind, ProposedOperation

    refusals = conversational_v2.session_operation_refusals(
        loaded,
        ConversationDecision(
            proposed_operation=ProposedOperation(kind=OperationKind.PRESENT_OFFER, subject="A")
        ),
        state,
    )
    assert "that content was already sent or purchased" in refusals


# ---------------------------------------------------------------------------
# 7. A redirect replaces FUTURE beats and preserves completed history
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fan_redirect_replaces_future_trajectory_and_preserves_history(world):
    first = world.fan_says("let's do a shared roleplay scene on a train")
    world.queue(
        lambda payload: owner_json(
            session_delta=start_session(
                first,
                beats=[
                    {"beat_id": "board", "kind": "conversation", "intent": "board the imagined train"},
                    {
                        "beat_id": "show_a",
                        "kind": "content",
                        "intent": "a visual moment on the train",
                        "candidate_handle": world.handle_for(payload, "A"),
                        "media_role": "marks the train leaving the station",
                    },
                    {"beat_id": "wind_down", "kind": "close", "intent": "wind down"},
                ],
            )
            | {"start_beat": {"beat_id": "board"}}
        ),
        ["all aboard"],
    )
    await lo.run_auto_turn(
        creator_id=CREATOR_ID, fan_id=FAN_ID, latest_message="let's do a shared roleplay scene on a train",
        trigger_identity=first, conversation_core=V2,
    )
    await world.turn(
        "we found our seats",
        owner_json(session_delta={"complete_current_beat": {"outcome": "boarded together"}}),
        ["window seat is mine"],
    )
    before = world.session_state()
    completed_before = [beat.model_dump() for beat in before.session.completed_beats]
    assert [beat["beat_id"] for beat in completed_before] == ["board"]

    # A replan without a reason is refused: history of intent must be explained.
    await world.turn(
        "hmm",
        owner_json(session_delta={"trajectory": {"beats": [{"kind": "conversation", "intent": "x"}]}}),
        ["mm?"],
    )
    assert [b.beat_id for b in world.session_state().session.tentative_trajectory] == ["show_a", "wind_down"]

    await world.turn(
        "actually, change of plan: forget the train, i'd love a short clip instead",
        lambda payload: owner_json(
            move="alter_premise",
            session_delta={
                "fan_participation": {"mode": "redirecting", "gist": "he changed direction"},
                "content_direction": "a short clip",
                "trajectory": {
                    "reason": "fan redirected away from the train scene toward a short clip",
                    "beats": [
                        {"beat_id": "new_direction", "kind": "conversation", "intent": "acknowledge the change warmly"},
                        {
                            "beat_id": "clip",
                            "kind": "content",
                            "intent": "the clip he asked for, when it fits",
                            "candidate_handle": world.handle_for(payload, "C"),
                            "media_role": "answers the specific thing he asked for",
                        },
                    ],
                },
                "tempo": "redirect",
            },
        ),
        ["okay, new plan"],
    )
    after = world.session_state()
    assert [beat.model_dump() for beat in after.session.completed_beats] == completed_before
    assert [beat.beat_id for beat in after.session.tentative_trajectory] == ["new_direction", "clip"]
    assert after.session.tentative_trajectory[1].content_ref == "C"
    assert after.session.last_replan_reason.startswith("fan redirected")
    assert after.session.replans[-1].replaced_beat_ids == ["show_a", "wind_down"]
    assert after.session.content_direction == "a short clip"
    # Nothing was offered by replanning.
    assert world.pending_offer is None and world.sent_ppv == []


def test_replan_without_reason_is_refused_and_completed_beats_are_unreachable():
    snapshot = EvidenceSnapshot(
        creator_id="c", fan_id="f", trigger=TurnTrigger("fan_message", "m1", "go"), state_revision="r"
    )
    state = ConversationalSessionState()
    state.session.status = SessionStatus.ACTIVE
    first = conversational_session.validate_session_delta(
        state,
        {"trajectory": {"beats": [{"beat_id": "one", "kind": "conversation", "intent": "talk"}]}},
        snapshot=snapshot,
    ).state_after
    refused = conversational_session.validate_session_delta(
        first,
        {"trajectory": {"beats": [{"kind": "conversation", "intent": "other"}]}},
        snapshot=snapshot,
    )
    assert "replan reason" in refused.rejected_fields["trajectory"]
    assert [beat.beat_id for beat in refused.state_after.session.tentative_trajectory] == ["one"]



@pytest.mark.asyncio
async def test_creator_may_propose_a_session_the_fan_did_not_ask_for(world):
    """A skilled chatter can invite; the fan's answer decides; nothing is sold."""
    await world.turn("haha that story about your commute was great", owner_json(), ["right? chaos"])
    result = await world.turn(
        "honestly i'm free all evening",
        owner_json(
            move={"kind": "invite_participation", "intent": "lightly invite him into a longer shared date scenario"},
            session_delta={
                "status": "proposed",
                "interaction_goal": "see whether he wants a longer shared scenario tonight",
            },
        ),
        ["then how about a proper date night, you and me, starting now?"],
    )
    assert result["outcome"] == "replied"
    state = world.session_state()
    assert state.session.status is SessionStatus.PROPOSED
    assert state.session.session_id
    assert state.session.next_experience_move.kind.value == "invite_participation"
    # Proposing implies no content and no commercial state.
    assert state.session.tentative_trajectory == []
    assert world.pending_offer is None and world.sent_ppv == []
    assert world.transcript[-1]["operation"] == "none"

    # He declines: the proposal is abandoned and the conversation simply goes on.
    await world.turn(
        "maybe another night, i'm pretty tired",
        owner_json(move="react", session_delta={"status": "abandoned"}),
        ["fair, rest up"],
    )
    assert world.session_state().session.status is SessionStatus.ABANDONED


def test_owner_prompt_allows_creator_proposals_without_a_funnel_or_padding():
    system = conversational_v2.CONVERSATIONAL_V2_SYSTEM
    assert "You propose it" in system
    assert "never a routine step" in system
    assert "Intimacy, explicitness, elapsed turns or a past purchase are not by themselves a session opportunity" in system
    assert "merely because another candidate exists" in system
    assert "Never add turns just to create distance" in system
    assert "Several conversational turns between content events is normal" not in system

# ---------------------------------------------------------------------------
# 8 + 9. No funnel; discovery only when it materially matters
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_direct_concrete_request_is_handled_without_budget_discovery(world):
    result = await world.turn(
        "can i buy content A right now?",
        lambda payload: owner_json(
            move="use_content",
            operation={
                "kind": "present_offer",
                "subject": "content A",
                "because": "direct request",
                "candidate_handle": world.handle_for(payload, "A"),
            },
        ),
        ["of course"],
    )
    assert result["outcome"] == "replied"
    # No session, no constraint, no information need was required first.
    state = world.session_state()
    assert state.session.status is SessionStatus.INACTIVE
    assert state.session.known_constraints == []
    assert state.session.information_needs == []
    assert world.pending_offer is not None and world.pending_offer.set_id == "A"


@pytest.mark.asyncio
async def test_a_pending_offer_reads_as_presented_never_as_not_offered(world):
    await world.turn(
        "can i buy content A right now?",
        lambda payload: owner_json(
            operation={"kind": "present_offer", "subject": "content A", "because": "asked",
                       "candidate_handle": world.handle_for(payload, "A")},
        ),
        ["of course"],
    )
    await world.turn("hmm let me think", owner_json(move="linger"), ["take your time"])
    view = last_owner_session(world)
    a_row = next(row for row in view["content_candidates"] if row["description"].startswith("content A"))
    assert a_row["fact"] == "presented"
    assert {"content": a_row["candidate_handle"], "fact": "presented"} in view["content_ledger"]
    state = world.session_state()
    assert state.session.lifecycle_of("A") is ContentLifecycle.PRESENTED
    assert "A" not in state.session.consumed_refs()

@pytest.mark.asyncio
async def test_material_missing_information_becomes_a_discovery_goal_then_caps_spend(world):
    first = world.fan_says("i want a long evening with you, a whole date scenario")
    world.queue(
        owner_json(
            move="discover",
            goal="find out, conversationally, what he wants to spend tonight",
            session_delta={
                "status": "planning",
                "interaction_goal": "learn his current spending limit before planning a longer paid evening",
                "open_information_need": {
                    "topic": "spending_limit",
                    "why_material": "planning several paid moments depends on his limit",
                },
            },
        ),
        ["i love that. what kind of evening are you picturing?"],
    )
    await lo.run_auto_turn(
        creator_id=CREATOR_ID, fan_id=FAN_ID,
        latest_message="i want a long evening with you, a whole date scenario",
        trigger_identity=first, conversation_core=V2,
    )
    state = world.session_state()
    assert state.session.status is SessionStatus.PLANNING
    assert [need.topic.value for need in state.session.information_needs] == ["spending_limit"]
    assert state.session.next_experience_move.kind.value == "discover"

    limit_msg = world.fan_says("i can do $60 tonight")
    world.queue(
        owner_json(
            session_delta={
                "status": "active",
                "add_constraints": [
                    # Invented amount: refused, the message does not say it.
                    {"constraint_id": "wrong", "kind": "spending_limit", "statement": "100",
                     "amount_cents": 10000, "source_refs": [limit_msg]},
                ],
            }
        ),
        ["noted"],
    )
    await lo.run_auto_turn(
        creator_id=CREATOR_ID, fan_id=FAN_ID, latest_message="i can do $60 tonight",
        trigger_identity=limit_msg, conversation_core=V2,
    )
    assert world.session_state().session.spending_constraint() is None

    world.queue(
        owner_json(
            session_delta={
                "add_constraints": [
                    {"constraint_id": "limit60", "kind": "spending_limit", "statement": "up to 60 tonight",
                     "amount_cents": 6000, "source_refs": [limit_msg]},
                ],
            }
        ),
        ["perfect"],
    )
    world.fan_says("so what's first?")
    await lo.run_auto_turn(
        creator_id=CREATOR_ID, fan_id=FAN_ID, latest_message="so what's first?",
        trigger_identity=world.messages[-1].id, conversation_core=V2,
    )
    state = world.session_state()
    assert state.session.spending_constraint().amount_cents == 6000
    assert state.session.information_needs[0].status.value == "resolved"

    # Known now: asking again is refused, and B (above the limit) is not a candidate.
    await world.turn(
        "tell me more",
        owner_json(session_delta={"open_information_need": {"topic": "spending_limit", "why_material": "again"}}),
        ["mm"],
    )
    view = last_owner_session(world)
    assert view["spending_limit_known"] is True
    assert view["open_information_needs"] == []
    described = [row["description"] for row in view["content_candidates"]]
    assert not any(text.startswith("content B") for text in described)
    assert any(text.startswith("content A") for text in described)


def test_inferred_constraints_are_refused_and_application_limits_prevent_asking():
    snapshot = EvidenceSnapshot(
        creator_id="c",
        fan_id="f",
        trigger=TurnTrigger("fan_message", "m1", "i drive a nice car"),
        state_revision="r",
        spending_limits={"explicit_current_limit_cents": 5000},
    )
    result = conversational_session.validate_session_delta(
        ConversationalSessionState(),
        {
            "add_constraints": [
                {"constraint_id": "rich", "kind": "spending_limit", "statement": "can afford a lot",
                 "amount_cents": 50000, "source_type": "model_inference", "source_refs": ["m1"]},
            ],
            "open_information_need": {"topic": "spending_limit", "why_material": "plan"},
        },
        snapshot=snapshot,
    )
    # Inference is never a constraint source.
    assert "inference" in result.rejected_fields["add_constraints[0]"]
    assert "already holds" in result.rejected_fields["open_information_need"]


# ---------------------------------------------------------------------------
# 10. Switch the test fan v2 -> v1 -> v2 through the real simulator path
# ---------------------------------------------------------------------------


def _install_simulator(monkeypatch, world: V2World) -> None:
    async def save_message(fan_id, creator_id, role, content, **_k):
        return world.fan_says(content)

    async def creator_rows(_fan_id):
        return [
            {"id": message.id, "role": "creator", "content": message.content}
            for message in world.messages
            if message.role == "creator"
        ]

    fan = Fan(id=FAN_ID, display_name="Session Fan", creator_id=CREATOR_ID,
              platform_fan_id="test_session_fan", auto_mode=True)

    async def _v(x):
        return x

    def retired(*_a, **_k):
        raise AssertionError("a retired legacy controller ran on a selected core")

    monkeypatch.setattr(suggestions, "save_message", save_message)
    monkeypatch.setattr(suggestions, "mark_simulation_owned_message", lambda *_a: None)
    monkeypatch.setattr(suggestions, "_recent_creator_message_rows", creator_rows)
    monkeypatch.setattr(suggestions, "get_conversation_history", lambda *_a, **_k: _v(list(world.messages)))
    monkeypatch.setattr(suggestions, "get_fan_by_id", lambda *_a: _v(fan))
    monkeypatch.setattr(suggestions, "get_fan_intelligence_context", lambda *_a: _v({}))
    monkeypatch.setattr(suggestions, "get_fan_lifecycle_context", lambda *_a: _v({}))
    monkeypatch.setattr(suggestions, "get_affordability_context", lambda *_a: _v({}))
    monkeypatch.setattr(suggestions, "get_price_learning_context", lambda *_a: _v({}))
    monkeypatch.setattr(suggestions, "resolve_ai_stack", lambda **_k: _v(SimpleNamespace(profile_id="cleo_v3")))
    monkeypatch.setattr(suggestions, "learn_from_fan_message", lambda **_k: _v(None))
    monkeypatch.setattr(suggestions, "get_supabase", lambda: world.db)
    monkeypatch.setattr(conversation_core, "get_supabase", lambda: world.db)
    for name in ("analyze_situation", "orchestrate", "direct_conversation", "plan_next_action"):
        monkeypatch.setattr(suggestions, name, retired)
    suggestions._pending_auto_replies.clear()


@pytest.mark.asyncio
async def test_simulator_path_selects_v2_consumes_its_state_and_rolls_back_to_v1(monkeypatch, world):
    _install_simulator(monkeypatch, world)

    # Turn 1 — the test fan row says conversational_v2 (the real resolver reads it).
    world.queue(
        lambda payload: owner_json(
            move="develop_premise",
            session_delta=start_session(payload["turn_id"]),
        ),
        ["rooftop cafe, corner table, you're late"],
    )
    first = await suggestions.run_simulated_inbound(
        fan_id=FAN_ID, creator_id=CREATOR_ID, message="let's pretend we're on a date", fast=True
    )
    assert first["outcome"] == "replied"
    assert "SESSION-AWARE EXTENSION" in world.owner_systems[-1]
    assert world.session_state().session.status is SessionStatus.ACTIVE

    # Turn 2 — the simulator path loads the persisted session before deciding.
    world.queue(owner_json(move="linger"), ["take your time"])
    await suggestions.run_simulated_inbound(
        fan_id=FAN_ID, creator_id=CREATOR_ID, message="sorry, traffic", fast=True
    )
    assert last_owner_session(world)["experience_premise"]["world_scope"] == "imagined_scene"
    assert last_owner_session(world)["current_beat"]["beat_id"] == "arrive"
    v2_revision = world.session_state().revision

    # Rollback: one write, effective on the very next turn.
    await conversation_core.set_simulation_fan_core_override(FAN_ID, V1)
    world.queue(owner_json(), ["back on v1"])
    await suggestions.run_simulated_inbound(
        fan_id=FAN_ID, creator_id=CREATOR_ID, message="still there?", fast=True
    )
    assert world.owner_systems[-1] == lo.CONVERSATIONAL_V1_SYSTEM
    assert "interaction_session" not in world.owner_payloads[-1]
    assert world.session_state().revision == v2_revision  # v1 never touched it

    # And forward again: v2 resumes exactly where it stopped.
    await conversation_core.set_simulation_fan_core_override(FAN_ID, V2)
    world.queue(owner_json(move="callback"), ["the corner table is still ours"])
    await suggestions.run_simulated_inbound(
        fan_id=FAN_ID, creator_id=CREATOR_ID, message="okay i'm back", fast=True
    )
    assert "SESSION-AWARE EXTENSION" in world.owner_systems[-1]
    assert last_owner_session(world)["experience_premise"]["summary"].startswith("an interactive date")


# ---------------------------------------------------------------------------
# 11. Imagined continuity never authorises real-world claims
# ---------------------------------------------------------------------------


def test_shared_imagined_premise_needs_fan_participation_and_stays_imagined():
    snapshot = EvidenceSnapshot(
        creator_id="c",
        fan_id="f",
        trigger=TurnTrigger("fan_message", "fan-1", "let's pretend we're at a cafe"),
        state_revision="r",
        historical_facts=(),
    )
    catalog_ref = "memory:0"
    snapshot_with_memory = EvidenceSnapshot(
        **{**snapshot.__dict__, "historical_facts": (
            __import__("models.live_orchestration", fromlist=["EvidenceFact"]).EvidenceFact(
                value="likes cafes", source_ref=catalog_ref
            ),
        )}
    )
    inferred_only = conversational_session.validate_session_delta(
        ConversationalSessionState(),
        {"experience_premise": {"summary": "a cafe date", "world_scope": "imagined_scene",
                                "source_refs": [catalog_ref]}},
        snapshot=snapshot_with_memory,
    )
    assert "participation" in inferred_only.rejected_fields["experience_premise"]

    real_world = conversational_session.validate_session_delta(
        ConversationalSessionState(),
        {"experience_premise": {"summary": "she is at home in bed right now", "world_scope": "present_world",
                                "source_refs": ["fan-1"]}},
        snapshot=snapshot,
    )
    assert "never a real-world fact" in real_world.rejected_fields["experience_premise"]

    unscoped = conversational_session.validate_session_delta(
        ConversationalSessionState(),
        {"experience_premise": {"summary": "the creator is sitting at home right now",
                                "world_scope": "conversation", "source_refs": ["fan-1"]}},
        snapshot=snapshot,
    )
    assert "present-world" in unscoped.rejected_fields["experience_premise"]

    imagined = conversational_session.validate_session_delta(
        ConversationalSessionState(),
        {"experience_premise": {"summary": "she is sitting across from him at the imagined cafe",
                                "world_scope": "imagined_scene", "source_refs": ["fan-1"]}},
        snapshot=snapshot,
    )
    assert "experience_premise" in imagined.accepted_fields
    view = conversational_session.writer_session_view(
        imagined.state_after, snapshot=snapshot, operation="none"
    )
    assert view["premise_scope"].startswith("imagined")
    assert "never a real-world claim" in view["premise_scope"]


@pytest.mark.asyncio
async def test_active_imagined_premise_does_not_let_the_writer_claim_real_activity(world):
    first = world.fan_says("let's pretend we're on a date")
    # Kimi's first wording asserts real present activity; the contract refuses
    # it even though an imagined premise is active, and asks once more.
    world.queue(
        owner_json(session_delta=start_session(first)),
        ["okay, rooftop it is. i'm just taking a walk by the lake right now. the view is ours"],
    )
    world.writer_script.append(["okay, rooftop it is. imagine the view from our table"])
    await lo.run_auto_turn(
        creator_id=CREATOR_ID, fan_id=FAN_ID, latest_message="let's pretend we're on a date",
        trigger_identity=first, conversation_core=V2,
    )
    assert world.session_state().session.experience_premise.world_scope is WorldScope.IMAGINED_SCENE
    assert len(world.writer_prompts) == 2
    assert "rejected" in world.writer_prompts[-1][0]["content"]
    sent = [row["text"] for row in world.transcript if row["speaker"] == "creator"][-1]
    assert "taking a walk" not in sent
    assert "rooftop" in sent
    system = world.writer_prompts[-1][0]["content"]
    assert "SESSION CONTEXT" in system
    assert "never becomes a claim about the creator's real current activity" in system


# ---------------------------------------------------------------------------
# Long trajectory: immersion evaluation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_long_session_reads_as_one_interaction_with_media_removed(world):
    records: list[TurnRecord] = []

    def record(fan_text: str, *, fan_asked: bool = False) -> None:
        prompt = world.writer_prompts[-1]
        session = writer_payload(prompt)["interaction_session"]
        operation = world.transcript[-1].get("operation") or "none"
        if world.transcript[-1].get("media"):
            operation = next(
                row.get("operation") for row in reversed(world.transcript) if row["speaker"] == "creator"
            )
        creator_text = next(row["text"] for row in reversed(world.transcript) if row["speaker"] == "creator")
        state = world.session_state()
        planned_earlier = False
        if operation in {"present_offer", "send_locked_paid_message"}:
            ref = (world.pending_offer.set_id if world.pending_offer else world.sent_ppv[-1]["set_id"])
            planned_earlier = any(
                beat.content_ref == ref and beat.planned_turn_ref != world.messages[-2].id
                for beat in [*state.session.tentative_trajectory, *filter(None, [state.session.current_beat])]
            ) or any(beat.content_ref == ref for beat in state.session.completed_beats)
        records.append(
            TurnRecord(
                fan_text=fan_text,
                creator_text=creator_text,
                operation=operation,
                premise_given_to_writer=session["experience_premise"],
                next_move=session["next_experience_move"]["kind"],
                content_was_planned_earlier=planned_earlier,
                fan_asked=fan_asked,
                follows_content_event=session["what_just_happened"] != "nothing new in the ledger",
            )
        )

    first = world.fan_says("let's pretend we're on a date at a rooftop cafe")
    world.queue(
        lambda payload: owner_json(
            move="develop_premise",
            session_delta=start_session(
                first,
                beats=[
                    {"beat_id": "arrive", "kind": "conversation", "intent": "arrive and settle in"},
                    {"beat_id": "order", "kind": "participation", "intent": "he orders for both"},
                    {"beat_id": "view", "kind": "content", "intent": "the view as the evening turns",
                     "candidate_handle": world.handle_for(payload, "A"),
                     "media_role": "the moment the rooftop scene has been building toward"},
                    {"beat_id": "after", "kind": "conversation", "intent": "stay in what the view changed"},
                    {"beat_id": "walk", "kind": "content", "intent": "the walk home",
                     "candidate_handle": world.handle_for(payload, "C"),
                     "media_role": "closes the evening the way he pictured it"},
                ],
            ),
        ),
        ["rooftop cafe, corner table. you're a little late"],
    )
    await lo.run_auto_turn(creator_id=CREATOR_ID, fan_id=FAN_ID,
                           latest_message="let's pretend we're on a date at a rooftop cafe",
                           trigger_identity=first, conversation_core=V2)
    record("let's pretend we're on a date at a rooftop cafe")

    async def say(text, owner, writer, **kw):
        await world.turn(text, owner, writer)
        record(text, **kw)

    await say("sorry! traffic. what are we drinking?", owner_json(move="react"), ["i ordered for you. trust me"])
    await say("you ordered for me? bold", owner_json(
        move="invite_participation",
        session_delta={"complete_current_beat": {"outcome": "settled in"}, "start_beat": {"beat_id": "order"}},
    ), ["your turn then. dessert is yours to pick"])
    await say("the chocolate one, obviously", owner_json(move="callback"),
              ["called it. the corner table was a good choice too"])
    await say("the sun is going down", owner_json(
        move="use_content",
        session_delta={"complete_current_beat": {"outcome": "he chose dessert"}, "start_beat": {"beat_id": "view"}},
        operation=None,
    ) | {"operation_proposal": {"kind": "present_offer", "subject": "the view", "because": "the scene reached it",
                                "candidate_handle": "offer_candidate_1"}},
        ["the sky is doing the thing now. want to see the view from our table?"])
    await say("yes please", owner_json(
        move="use_content",
        operation={"kind": "send_locked_paid_message", "subject": "the view", "because": "he accepted",
                   "candidate_handle": "pending_offer_1"},
    ), ["here, from our corner"])
    world.purchase("A")
    await say("wow", owner_json(move="react", session_delta={"start_beat": {"beat_id": "after"}}),
              ["right? i didn't want you to miss it"])
    await say("this whole evening is great", owner_json(move="linger"), ["it is. stay a bit longer"])
    await say("okay, walk me home?", owner_json(
        move="transition",
        session_delta={"complete_current_beat": {"outcome": "lingered after the view"},
                       "start_beat": {"beat_id": "walk"}},
    ), ["come on then, the long way"])
    await say("can i see the walk home?", lambda payload: owner_json(
        move="use_content",
        operation={"kind": "present_offer", "subject": "the walk home", "because": "he asked",
                   "candidate_handle": world.handle_for(payload, "C")},
    ), ["the whole walk, just us"], fan_asked=True)
    await say("perfect", owner_json(move="close", session_delta={"tempo": "close"}),
              ["best date i've had in a while"])

    report = score_session_trajectory(records)
    assert report.passed, report.as_dict()
    assert report.media_events == 3  # offer A, send A, offer C
    assert report.post_event_sales == 0
    assert report.premise_continuity == 1.0
    # Reported for analysis, not enforced: pacing owns the gap.
    assert report.min_conversational_turns_between_media is not None
    # With media cards removed every fan line still has its creator line.
    assert len(report.dialogue_without_media) == 2 * len(records)

    state = world.session_state()
    assert [beat.beat_id for beat in state.session.completed_beats] == [
        "arrive", "order", "view", "after",
    ]
    assert state.session.completed_beats[2].source is BeatSource.APPLICATION


def test_short_gap_is_a_diagnostic_not_a_failure_when_the_fan_asks():
    records = [
        TurnRecord("tell me about the view", "it's gorgeous tonight", premise_given_to_writer="date"),
        TurnRecord("show me?", "here it is", operation="present_offer",
                   premise_given_to_writer="date", fan_asked=True),
        TurnRecord("yes", "sent", operation="send_locked_paid_message",
                   premise_given_to_writer="date", content_was_planned_earlier=True),
        TurnRecord("and the walk home too, now", "the walk, just us", operation="present_offer",
                   premise_given_to_writer="date", fan_asked=True),
    ]
    report = score_session_trajectory(records)
    assert report.passed, report.as_dict()
    assert report.min_conversational_turns_between_media == 0
    assert report.diagnostics  # still visible for analysis


def test_content_that_advances_only_because_a_candidate_exists_fails():
    records = [
        TurnRecord("hi", "hey you", premise_given_to_writer="date"),
        TurnRecord("nice evening", "it is", premise_given_to_writer="date"),
        TurnRecord("mm", "want this?", operation="present_offer", premise_given_to_writer="date"),
    ]
    report = score_session_trajectory(records)
    assert not report.passed
    assert report.unexplained_media_events == 1


def test_immersion_scorer_rejects_a_chain_of_ppvs():
    chain = [
        TurnRecord("hi", "want content A?", operation="present_offer", premise_given_to_writer="date"),
        TurnRecord("ok", "here", operation="send_locked_paid_message", premise_given_to_writer="date",
                   content_was_planned_earlier=True),
        TurnRecord("nice", "want content B?", operation="present_offer", follows_content_event=True),
        TurnRecord("ok", "here", operation="send_locked_paid_message", content_was_planned_earlier=True),
    ]
    report = score_session_trajectory(chain)
    assert not report.passed
    assert report.post_event_sales == 1
    assert report.unexplained_media_events >= 1
