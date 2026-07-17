import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datetime import datetime, timedelta, timezone

from models.commercial import (
    ActionType,
    CreatorPolicy,
    EventType,
    CommercialEvent,
    FanCommercialState,
    FanStatus,
    SextingMode,
)
from services.commercial_policy import CommercialContext, decide_next_action


def test_active_session_waits_for_purchase():
    decision = decide_next_action(
        CreatorPolicy(),
        FanCommercialState(status=FanStatus.PAID_SESSION_ACTIVE),
        [CommercialEvent(type=EventType.WANTS_EXPLICIT)],
        CommercialContext(session_exists=True, session_has_pending_purchase=True, session_has_remaining_steps=True),
    )
    assert decision.action == ActionType.CONTINUE_NORMAL_CHAT
    assert decision.must_not_send_media is True
    assert decision.may_be_explicit is False


def test_active_session_sends_next_after_purchase_cooldown():
    decision = decide_next_action(
        CreatorPolicy(),
        FanCommercialState(status=FanStatus.PAID_SESSION_ACTIVE),
        [],
        CommercialContext(session_exists=True, session_has_remaining_steps=True),
    )
    assert decision.action == ActionType.SEND_NEXT_PPV_STEP
    assert decision.must_not_send_media is False


def test_free_text_cooldown_is_enforced():
    now = datetime.now(timezone.utc)
    state = FanCommercialState(
        status=FanStatus.IDLE,
        teaser_messages_used=20,
        free_session_ended_at=now - timedelta(hours=1),
    )
    decision = decide_next_action(
        CreatorPolicy(sexting_mode=SextingMode.FREE_TEXT_ALLOWED, free_session_cooldown_hours=24),
        state,
        [CommercialEvent(type=EventType.WANTS_EXPLICIT)],
        CommercialContext(now=now),
    )
    assert decision.action == ActionType.CONTINUE_NORMAL_CHAT
    assert decision.may_be_explicit is False


def test_money_available_resumes_paused_session():
    decision = decide_next_action(
        CreatorPolicy(),
        FanCommercialState(status=FanStatus.PAUSED_UNTIL_PAYDAY),
        [CommercialEvent(type=EventType.MONEY_AVAILABLE)],
        CommercialContext(paused_session_available=True),
    )
    assert decision.action == ActionType.RESUME_PREVIOUS_OFFER
    assert decision.new_status == FanStatus.OFFER_SELECTED
