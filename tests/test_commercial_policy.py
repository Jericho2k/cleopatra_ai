"""Regression tests for deterministic commercial policy."""
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.commercial import (  # noqa: E402
    ActionType,
    CommercialEvent,
    CreatorPolicy,
    EventType,
    FanCommercialState,
    FanStatus,
    Offer,
    SextingMode,
)
from services.commercial_policy import CommercialContext, decide_next_action  # noqa: E402
from services.payday import resolve_payday  # noqa: E402


def ev(event_type, raw="", *, cents=None, metadata=None):
    return CommercialEvent(
        type=event_type,
        raw_expression=raw,
        amount_cents=cents,
        metadata=metadata or {},
    )


def offer(price=2800, set_id="set-28"):
    return Offer(
        offer_id=f"offer:{set_id}",
        label="lingerie set",
        price_cents=price,
        set_id=set_id,
    )


def test_acceptance_beats_a_payday_mention_in_the_same_message():
    accepted = ev(
        EventType.OFFER_ACCEPTED,
        "$28",
        cents=2800,
        metadata={"set_id": "set-28", "offer_id": "offer:set-28"},
    )
    decision = decide_next_action(
        CreatorPolicy(),
        FanCommercialState(status=FanStatus.OFFER_PENDING),
        [accepted, ev(EventType.BUDGET_LIMIT_STATED, cents=2800), ev(EventType.PAYDAY_MENTIONED, "Friday")],
        CommercialContext(next_offer=offer()),
    )
    # One meaningful yes is enough: the next move is the send, not a
    # confirmation of the yes he already gave.
    assert decision.action == ActionType.SEND_NEXT_PPV_STEP
    assert decision.must_not_send_media is False
    assert decision.must_not_ask_question is True
    assert decision.session_budget_cents == 2800
    assert decision.accepted_offer_set_id == "set-28"
    assert decision.schedule_payday_followup is False
    assert decision.new_status == FanStatus.OFFER_SELECTED


def test_cannot_afford_any_option_schedules_payday():
    decision = decide_next_action(
        CreatorPolicy(),
        FanCommercialState(),
        [ev(EventType.MONEY_UNAVAILABLE), ev(EventType.PAYDAY_MENTIONED, "Friday")],
        CommercialContext(next_offer=offer()),
    )
    assert decision.action == ActionType.PAUSE_UNTIL_PAYDAY
    assert decision.schedule_payday_followup is True
    assert decision.must_not_ask_question is True
    assert decision.conversation_continuation == "none"


def test_broke_fan_is_never_sold_to():
    decision = decide_next_action(
        CreatorPolicy(),
        FanCommercialState(status=FanStatus.PAUSED_UNTIL_PAYDAY),
        [ev(EventType.WANTS_MEDIA)],
        CommercialContext(next_offer=offer()),
    )
    assert decision.must_not_send_media is True
    assert decision.action in (ActionType.CONTINUE_NORMAL_CHAT, ActionType.CONTINUE_FREE_TEXT)


def test_money_available_lifts_pause():
    decision = decide_next_action(
        CreatorPolicy(),
        FanCommercialState(status=FanStatus.PAUSED_UNTIL_PAYDAY),
        [ev(EventType.MONEY_AVAILABLE)],
        CommercialContext(),
    )
    assert decision.action == ActionType.RESUME_PREVIOUS_OFFER
    assert decision.new_status == FanStatus.IDLE


def test_crisis_beats_offer_acceptance():
    decision = decide_next_action(
        CreatorPolicy(),
        FanCommercialState(),
        [ev(EventType.CRISIS), ev(EventType.OFFER_ACCEPTED, cents=2800)],
        CommercialContext(next_offer=offer()),
    )
    assert decision.action == ActionType.HAND_OFF_TO_HUMAN
    assert decision.must_not_send_media is True


def test_hybrid_teaser_gives_a_taste_to_a_fan_who_has_not_asked_outright():
    """WANTS_MEDIA alone is interest, not the explicit intent that skips ahead."""
    policy = CreatorPolicy(sexting_mode=SextingMode.HYBRID_TEASER, teaser_max_messages=4)
    option = offer()
    early = decide_next_action(
        policy,
        FanCommercialState(teaser_messages_used=1),
        [ev(EventType.WANTS_MEDIA)],
        CommercialContext(next_offer=option),
    )
    assert early.action == ActionType.START_FREE_TEASER
    exhausted = decide_next_action(
        policy,
        FanCommercialState(teaser_messages_used=4),
        [ev(EventType.WANTS_MEDIA)],
        CommercialContext(next_offer=option),
    )
    assert exhausted.action == ActionType.OFFER_NEXT_UNLOCK
    assert exhausted.next_offer is option


def test_explicit_intent_skips_the_teaser_entirely():
    """The high-intent fast path. No message count is consulted."""
    policy = CreatorPolicy(sexting_mode=SextingMode.HYBRID_TEASER, teaser_max_messages=4)
    decision = decide_next_action(
        policy,
        FanCommercialState(teaser_messages_used=0),
        [ev(EventType.WANTS_EXPLICIT), ev(EventType.WANTS_MEDIA)],
        CommercialContext(next_offer=offer()),
    )
    assert decision.action == ActionType.OFFER_NEXT_UNLOCK
    assert decision.mention_price == 28


def test_paid_only_never_gives_free_explicit():
    decision = decide_next_action(
        CreatorPolicy(sexting_mode=SextingMode.PAID_ONLY),
        FanCommercialState(),
        [ev(EventType.WANTS_EXPLICIT)],
        CommercialContext(),
    )
    assert decision.may_be_explicit is False
    assert decision.action != ActionType.START_FREE_TEASER


def test_free_text_mode_allows_text_while_paused():
    decision = decide_next_action(
        CreatorPolicy(sexting_mode=SextingMode.FREE_TEXT_ALLOWED),
        FanCommercialState(status=FanStatus.PAUSED_UNTIL_PAYDAY),
        [ev(EventType.WANTS_EXPLICIT)],
        CommercialContext(),
    )
    assert decision.action == ActionType.CONTINUE_FREE_TEXT
    assert decision.must_not_send_media is True



def test_an_acceptance_that_matches_no_approved_set_never_sends():
    decision = decide_next_action(
        CreatorPolicy(),
        FanCommercialState(status=FanStatus.OFFER_PENDING),
        [ev(EventType.OFFER_ACCEPTED, "$28", cents=2800)],
        CommercialContext(next_offer=offer(price=2500, set_id="set-25")),
    )
    assert decision.action == ActionType.OFFER_NEXT_UNLOCK
    assert decision.must_not_send_media is True
    assert decision.new_status == FanStatus.OFFER_PENDING


def test_one_offer_is_never_accompanied_by_a_second():
    """The whole point: a decision carries ONE offer, and there is no list."""
    decision = decide_next_action(
        CreatorPolicy(sexting_mode=SextingMode.PAID_ONLY),
        FanCommercialState(),
        [ev(EventType.WANTS_EXPLICIT), ev(EventType.WANTS_MEDIA)],
        CommercialContext(next_offer=offer()),
    )
    assert decision.action == ActionType.OFFER_NEXT_UNLOCK
    assert decision.next_offer is not None
    assert not hasattr(decision, "package_options")


def test_post_purchase_cooldown_stays_in_the_moment_instead_of_selling():
    decision = decide_next_action(
        CreatorPolicy(),
        FanCommercialState(status=FanStatus.PAID_SESSION_ACTIVE),
        [ev(EventType.WANTS_MEDIA)],
        CommercialContext(
            next_offer=offer(),
            session_exists=True,
            session_cooldown_active=True,
        ),
    )
    assert decision.action == ActionType.CONTINUE_NORMAL_CHAT
    assert decision.must_not_send_media is True
    assert decision.mention_price is None
    assert decision.next_offer is None


def test_timezone_aware_payday_resolver():
    now = datetime(2026, 7, 15, 12, 0, tzinfo=timezone.utc)
    target, confidence = resolve_payday(
        "Friday",
        now=now,
        send_hour=18,
        timezone_name="Europe/Berlin",
    )
    assert target is not None and confidence > 0.5
    assert target.weekday() == 4
    assert target.hour == 18
    assert str(target.tzinfo) == "Europe/Berlin"
