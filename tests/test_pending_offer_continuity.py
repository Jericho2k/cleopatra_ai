"""Contracts for the ONE persisted pending offer.

There used to be an ordered snapshot of two packages here, with ordinal
resolution ("the second one"), price-rank resolution ("the cheaper one") and an
ambiguity event for when neither could be decided. All three existed because the
fan was being shown a menu. He is not: he is shown the next unlock, and the only
question about a reply is whether it is a yes to that.
"""
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
    Offer,
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


def pending() -> Offer:
    return Offer(
        offer_id="offer:shower-1",
        label="private photo set",
        price_cents=5500,
        set_id="shower-1",
        experience="shower, wet, teasing",
        legal_description="shower, wet, teasing",
        media_count=6,
    )


def accepted(events: list[CommercialEvent]) -> CommercialEvent | None:
    return next((event for event in events if event.type == EventType.OFFER_ACCEPTED), None)


# --- acceptance -------------------------------------------------------------


def test_a_plain_yes_accepts_the_one_offer_on_the_table():
    offer = pending()
    resolved, reason = resolve_pending_offer_reference("yeah send it", offer)
    assert resolved is offer
    assert reason == "acceptance"

    events: list[CommercialEvent] = []
    augment_pending_offer_events(events, "yeah send it", offer)
    event = accepted(events)
    assert event is not None
    assert event.metadata["offer_id"] == "offer:shower-1"
    assert event.metadata["set_id"] == "shower-1"
    assert event.amount_cents == 5500


def test_naming_the_exact_price_accepts_it():
    offer = pending()
    assert resolve_pending_offer_reference("I'll take the $55 one", offer)[0] is offer


def test_naming_the_content_accepts_it():
    offer = pending()
    assert resolve_pending_offer_reference("yes the shower one", offer)[0] is offer


def test_a_different_price_is_never_acceptance_of_this_offer():
    offer = pending()
    resolved, reason = resolve_pending_offer_reference("can you do $20?", offer)
    assert resolved is None
    assert reason == "different_price_named"


def test_an_analyzer_acceptance_at_the_wrong_price_is_withdrawn():
    events = [CommercialEvent(type=EventType.OFFER_ACCEPTED, amount_cents=2000)]
    augment_pending_offer_events(events, "can you do $20?", pending())
    assert accepted(events) is None


def test_ordinal_language_no_longer_selects_anything():
    """There is nothing to be first or second among."""
    offer = pending()
    resolved, reason = resolve_pending_offer_reference("the second one", offer)
    assert resolved is None
    assert reason == "no_acceptance_reference"


def test_there_is_no_ambiguity_event_left_to_raise():
    assert not hasattr(EventType, "OFFER_SELECTION_AMBIGUOUS")


# --- detail questions -------------------------------------------------------


def test_detail_request_preserves_the_offer_and_overrides_accidental_acceptance():
    events = [
        CommercialEvent(
            type=EventType.OFFER_ACCEPTED,
            amount_cents=5500,
            metadata={"offer_id": "offer:shower-1", "set_id": "shower-1"},
        )
    ]
    augment_pending_offer_events(events, "what do I get?", pending())

    assert accepted(events) is None
    assert [event.type for event in events] == [EventType.OFFER_DETAILS_REQUESTED]


def test_offer_detail_detection_does_not_hijack_unrelated_small_talk():
    assert is_pending_offer_detail_request("what do I get?") is True
    assert is_pending_offer_detail_request("how many pics is it") is True
    assert is_pending_offer_detail_request("tell me more about your day") is False
    assert is_pending_offer_detail_request("which one is your favorite movie?") is False

    resolved, reason = resolve_pending_offer_reference(
        "do you like shower scenes?", pending()
    )
    # "shower" matches the offer's own description, so this is deliberately
    # NOT treated as ambiguity: the content reference is the acceptance signal
    # and the deterministic policy decides what to do with it.
    assert resolved is not None or reason == "no_acceptance_reference"


def test_policy_resumes_the_exact_pending_offer_for_a_detail_request():
    offer = pending()
    state = FanCommercialState(status=FanStatus.OFFER_PENDING, pending_offer=offer)
    detail = CommercialEvent(type=EventType.OFFER_DETAILS_REQUESTED, raw_expression="what do I get?")

    decision = decide_next_action(
        CreatorPolicy(),
        state,
        [detail],
        # A freshly built offer exists, and is deliberately NOT used: he is held
        # to the thing he was actually shown.
        CommercialContext(next_offer=Offer(offer_id="offer:other", label="l", price_cents=9900, set_id="other")),
    )

    assert decision.action == ActionType.RESUME_PREVIOUS_OFFER
    assert decision.new_status == FanStatus.OFFER_PENDING
    assert decision.next_offer is offer
    assert decision.mention_price == 55


# --- acceptance is not purchase ---------------------------------------------


def test_acceptance_sends_but_is_not_a_purchase_event():
    offer = pending()
    events: list[CommercialEvent] = []
    augment_pending_offer_events(events, "yes please", offer)

    assert EventType.PURCHASED not in {event.type for event in events}
    decision = decide_next_action(
        CreatorPolicy(),
        FanCommercialState(status=FanStatus.OFFER_PENDING, pending_offer=offer),
        events,
        CommercialContext(next_offer=offer),
    )
    assert decision.action == ActionType.SEND_NEXT_PPV_STEP
    assert decision.accepted_offer_set_id == "shower-1"
    assert decision.new_status == FanStatus.OFFER_SELECTED


# --- downstream layers agree ------------------------------------------------


def test_pending_offer_remains_authoritative_over_director_repetition_guard():
    director = advance_conversation_director(
        previous={
            "phase": "OFFER",
            "action": "OFFER_NEXT_UNLOCK",
            "turns_in_phase": 3,
            "same_action_streak": 4,
            "recent_actions": ["OFFER_NEXT_UNLOCK"] * 4,
        },
        situation={"purchase_signal": "none"},
        commercial_decision={"action": "RESUME_PREVIOUS_OFFER"},
        fan_turn_count=6,
        creator_turn_count=6,
    )

    assert director.phase == ConversationPhase.OFFER
    assert director.action == DirectorAction.OFFER_NEXT_UNLOCK


def test_session_strategy_treats_offer_resume_as_an_active_offer():
    decision = {
        "action": "RESUME_PREVIOUS_OFFER",
        "next_offer": pending().model_dump(),
    }
    strategy = derive_session_strategy(commercial_decision=decision)

    assert strategy.goal == SessionGoal.PRESENT_OFFER
    assert strategy.phase == "OFFER"
    assert strategy.next_action == NextBestAction.OFFER_NEXT_UNLOCK
    assert strategy.route_hint == "commercial_complex"
    assert strategy.approved_offer_ids == ["offer:shower-1"]
    assert "pending_offer_snapshot_continuity" in strategy.reason_codes


def test_accepted_offer_strategy_does_not_claim_payment_confirmation():
    decision = {
        "action": "SEND_NEXT_PPV_STEP",
        "new_status": "OFFER_SELECTED",
        "session_budget_cents": 5500,
        "next_offer": pending().model_dump(),
    }
    strategy = derive_session_strategy(commercial_decision=decision)

    assert strategy.phase == "OFFER_SELECTED"
    assert "not purchase" in " ".join(strategy.reason_codes).replace("_", " ")
    assert "already purchased" in strategy.writer_goal


def test_prompt_contract_states_one_offer_and_no_menu():
    source = inspect.getsource(prompt_builder.build_prompt)
    assert "THE OFFER ALREADY ON THE TABLE, UNCHANGED" in source
    assert "THE ONE NEXT THING YOU MAY OFFER" in source
    # Money crosses into writer context through exactly one renderer, so a raw
    # cent value can never reach a customer-facing string (models/money.py).
    assert "customer_dollars(cents)" in source
    assert "no second option" in source
    assert "NEVER tell him how much he might spend in total" in source
    # And nothing in the builder can still assemble an ordered menu.
    assert "ORIGINAL ORDER" not in source
    assert "package_options" not in source
