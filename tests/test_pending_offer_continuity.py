"""Contracts for persisted pending offers and deterministic selection."""
import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai import prompt_builder  # noqa: E402
from models.commercial import (  # noqa: E402
    ActionType,
    CommercialEvent,
    CreatorPolicy,
    EventType,
    FanCommercialState,
    FanStatus,
    PackageOption,
)
from models.conversation_director import (  # noqa: E402
    ConversationPhase,
    DirectorAction,
    advance_conversation_director,
)
from models.session_strategy import (  # noqa: E402
    NextBestAction,
    SessionGoal,
    derive_session_strategy,
)
from services.commercial_events import (  # noqa: E402
    augment_pending_offer_events,
    is_pending_offer_detail_request,
    resolve_pending_offer_reference,
)
from services.commercial_policy import CommercialContext, decide_next_action  # noqa: E402


def offers() -> list[PackageOption]:
    return [
        PackageOption(
            package_id="pkg-quick",
            label="quick bedroom session",
            price_cents=3000,
            set_id="bedroom-1",
            set_ids=["bedroom-1"],
            experience="bedroom, lingerie",
            legal_description="bedroom, lingerie",
        ),
        PackageOption(
            package_id="pkg-full",
            label="full shower session",
            price_cents=5500,
            set_id="shower-1",
            set_ids=["shower-1", "shower-2"],
            experience="shower, wet, teasing",
            legal_description="shower, wet, teasing",
        ),
    ]


def selected(events: list[CommercialEvent]) -> CommercialEvent | None:
    return next((event for event in events if event.type == EventType.PACKAGE_SELECTED), None)


def test_detail_request_preserves_snapshot_and_overrides_accidental_selection():
    snapshot = offers()
    events = [
        CommercialEvent(
            type=EventType.PACKAGE_SELECTED,
            amount_cents=5500,
            metadata={"package_id": "pkg-full", "set_ids": ["shower-1", "shower-2"]},
        )
    ]

    augment_pending_offer_events(events, "what is the second one?", snapshot)

    assert selected(events) is None
    assert [event.type for event in events] == [EventType.OFFER_DETAILS_REQUESTED]


def test_ordinal_selection_maps_to_exact_second_snapshot_entry():
    snapshot = offers()
    package, ambiguous, reason = resolve_pending_offer_reference("the second one", snapshot)

    assert ambiguous is False
    assert reason == "ordinal_second"
    assert package == snapshot[1]

    events: list[CommercialEvent] = []
    augment_pending_offer_events(events, "the second one", snapshot)
    event = selected(events)
    assert event is not None
    assert event.metadata["package_id"] == "pkg-full"
    assert event.metadata["set_ids"] == ["shower-1", "shower-2"]
    assert event.amount_cents == 5500


def test_price_label_and_theme_references_resolve_only_inside_snapshot():
    snapshot = offers()

    assert resolve_pending_offer_reference("the cheaper one", snapshot)[0] == snapshot[0]
    assert resolve_pending_offer_reference("the full one", snapshot)[0] == snapshot[1]
    assert resolve_pending_offer_reference("that shower option", snapshot)[0] == snapshot[1]
    assert resolve_pending_offer_reference("I'll take the $55 one", snapshot)[0] == snapshot[1]


def test_generic_reference_is_ambiguous_and_never_guessed():
    snapshot = offers()
    package, ambiguous, reason = resolve_pending_offer_reference("that one", snapshot)

    assert package is None
    assert ambiguous is True
    assert reason == "generic_reference"

    events: list[CommercialEvent] = []
    augment_pending_offer_events(events, "that one", snapshot)
    assert selected(events) is None
    assert events[0].type == EventType.OFFER_SELECTION_AMBIGUOUS



def test_offer_detail_detection_does_not_hijack_unrelated_small_talk():
    assert is_pending_offer_detail_request("more about the second option") is True
    assert is_pending_offer_detail_request("tell me more about your day") is False
    assert is_pending_offer_detail_request("which one is your favorite movie?") is False

    package, ambiguous, reason = resolve_pending_offer_reference(
        "do you like shower scenes?",
        offers(),
    )
    assert package is None
    assert ambiguous is False
    assert reason == "no_selection_reference"

def test_policy_resumes_exact_pending_snapshot_for_detail_request():
    snapshot = offers()
    state = FanCommercialState(status=FanStatus.OFFER_PENDING, offered_packages=snapshot)
    detail = CommercialEvent(type=EventType.OFFER_DETAILS_REQUESTED, raw_expression="tell me more")

    decision = decide_next_action(
        CreatorPolicy(),
        state,
        [detail],
        CommercialContext(package_options=list(reversed(snapshot))),
    )

    assert decision.action == ActionType.RESUME_PREVIOUS_OFFER
    assert decision.new_status == FanStatus.OFFER_PENDING
    assert [option.package_id for option in decision.package_options] == ["pkg-quick", "pkg-full"]
    assert [option.price_cents for option in decision.package_options] == [3000, 5500]


def test_policy_asks_clarification_with_exact_pending_snapshot():
    snapshot = offers()
    state = FanCommercialState(status=FanStatus.OFFER_PENDING, offered_packages=snapshot)
    ambiguous = CommercialEvent(type=EventType.OFFER_SELECTION_AMBIGUOUS, raw_expression="that one")

    decision = decide_next_action(
        CreatorPolicy(),
        state,
        [ambiguous],
        CommercialContext(package_options=[]),
    )

    assert decision.action == ActionType.PRESENT_SESSION_OPTIONS
    assert decision.new_status == FanStatus.OFFER_PENDING
    assert [option.package_id for option in decision.package_options] == ["pkg-quick", "pkg-full"]
    assert "clarification" in decision.goal


def test_selection_is_not_a_purchase_event_and_keeps_exact_package():
    snapshot = offers()
    events: list[CommercialEvent] = []
    augment_pending_offer_events(events, "the second one", snapshot)

    assert EventType.PURCHASED not in {event.type for event in events}
    decision = decide_next_action(
        CreatorPolicy(),
        FanCommercialState(status=FanStatus.OFFER_PENDING, offered_packages=snapshot),
        events,
        CommercialContext(package_options=snapshot),
    )
    assert decision.action == ActionType.CREATE_PAID_SESSION
    assert decision.selected_package_set_ids == ["shower-1", "shower-2"]


def test_pending_offer_remains_authoritative_over_director_repetition_guard():
    director = advance_conversation_director(
        previous={
            "phase": "OFFER",
            "action": "PRESENT_APPROVED_OPTIONS",
            "turns_in_phase": 3,
            "same_action_streak": 4,
            "recent_actions": ["PRESENT_APPROVED_OPTIONS"] * 4,
        },
        situation={"purchase_signal": "none"},
        commercial_decision={"action": "RESUME_PREVIOUS_OFFER"},
        fan_turn_count=6,
        creator_turn_count=6,
    )

    assert director.phase == ConversationPhase.OFFER
    assert director.action == DirectorAction.PRESENT_APPROVED_OPTIONS


def test_session_strategy_treats_offer_resume_as_active_offer_not_reengagement():
    decision = {
        "action": "RESUME_PREVIOUS_OFFER",
        "package_options": [option.model_dump() for option in offers()],
    }
    strategy = derive_session_strategy(commercial_decision=decision)

    assert strategy.goal == SessionGoal.PRESENT_OFFER
    assert strategy.phase == "OFFER"
    assert strategy.next_action == NextBestAction.PRESENT_APPROVED_OPTIONS
    assert strategy.route_hint == "commercial_complex"
    assert strategy.approved_offer_ids == ["pkg-quick", "pkg-full"]
    assert "pending_offer_snapshot_continuity" in strategy.reason_codes


def test_selected_offer_strategy_does_not_claim_payment_confirmation():
    decision = {
        "action": "CREATE_PAID_SESSION",
        "session_budget_cents": 5500,
        "package_options": [offers()[1].model_dump()],
    }
    strategy = derive_session_strategy(commercial_decision=decision)

    assert strategy.phase == "OFFER_SELECTED"
    assert "not purchase" in " ".join(strategy.reason_codes).replace("_", " ")
    assert "already purchased" in strategy.writer_goal


def test_prompt_contract_numbers_and_preserves_persisted_options():
    source = inspect.getsource(prompt_builder.build_prompt)
    assert "EXACT PERSISTED OFFER SNAPSHOT, ORIGINAL ORDER" in source
    assert 'f"{index}) {label}: ${cents / 100:g}"' in source
    assert "Do not rebuild, replace, reorder, or" in source
    assert "This is a selection, not proof of payment" in source
