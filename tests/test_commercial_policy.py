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
    PackageOption,
    SextingMode,
)
from services.commercial_policy import CommercialContext, decide_next_action  # noqa: E402
from services.payday import resolve_payday  # noqa: E402


def ev(event_type, raw="", *, cents=None, position=None, metadata=None):
    return CommercialEvent(
        type=event_type,
        raw_expression=raw,
        amount_cents=cents,
        package_position=position,
        metadata=metadata or {},
    )


def package(price=2800, set_id="set-28"):
    return PackageOption(
        package_id=f"set:{set_id}",
        label="lingerie set",
        price_cents=price,
        set_id=set_id,
    )


def test_selected_cheaper_package_beats_payday_mention():
    selected = ev(
        EventType.PACKAGE_SELECTED,
        "$28",
        cents=2800,
        metadata={"set_id": "set-28", "package_id": "set:set-28"},
    )
    decision = decide_next_action(
        CreatorPolicy(),
        FanCommercialState(status=FanStatus.OFFER_PENDING),
        [selected, ev(EventType.BUDGET_LIMIT_STATED, cents=2800), ev(EventType.PAYDAY_MENTIONED, "Friday")],
        CommercialContext(package_options=[package()]),
    )
    assert decision.action == ActionType.CREATE_PAID_SESSION
    assert decision.session_budget_cents == 2800
    assert decision.schedule_payday_followup is False
    assert decision.new_status == FanStatus.PAID_SESSION_ACTIVE


def test_cannot_afford_any_option_schedules_payday():
    decision = decide_next_action(
        CreatorPolicy(),
        FanCommercialState(),
        [ev(EventType.MONEY_UNAVAILABLE), ev(EventType.PAYDAY_MENTIONED, "Friday")],
        CommercialContext(package_options=[package()]),
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
        CommercialContext(package_options=[package()]),
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


def test_crisis_beats_package_selection():
    decision = decide_next_action(
        CreatorPolicy(),
        FanCommercialState(),
        [ev(EventType.CRISIS), ev(EventType.PACKAGE_SELECTED, cents=2800)],
        CommercialContext(package_options=[package()]),
    )
    assert decision.action == ActionType.HAND_OFF_TO_HUMAN
    assert decision.must_not_send_media is True


def test_hybrid_teaser_gives_taste_then_offer():
    policy = CreatorPolicy(sexting_mode=SextingMode.HYBRID_TEASER, teaser_max_messages=4)
    option = package()
    early = decide_next_action(
        policy,
        FanCommercialState(teaser_messages_used=1),
        [ev(EventType.WANTS_EXPLICIT)],
        CommercialContext(package_options=[option]),
    )
    assert early.action == ActionType.START_FREE_TEASER
    exhausted = decide_next_action(
        policy,
        FanCommercialState(teaser_messages_used=4),
        [ev(EventType.WANTS_EXPLICIT)],
        CommercialContext(package_options=[option]),
    )
    assert exhausted.action == ActionType.END_TEASER_AND_OFFER


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



def test_unmatched_package_selection_does_not_start_session():
    decision = decide_next_action(
        CreatorPolicy(),
        FanCommercialState(status=FanStatus.OFFER_PENDING),
        [ev(EventType.PACKAGE_SELECTED, "$28", cents=2800)],
        CommercialContext(package_options=[package(price=2500, set_id="set-25")]),
    )
    assert decision.action == ActionType.PRESENT_SESSION_OPTIONS
    assert decision.must_not_send_media is True
    assert decision.new_status == FanStatus.OFFER_PENDING

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
