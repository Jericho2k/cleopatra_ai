"""Tests for the commercial policy engine.

This is the whole point of making the decision a pure function: the business rules
that used to live in prompt paragraphs (and broke constantly) are now assertable.
Every bug we hit in live transcripts gets a test here.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.commercial import (  # noqa: E402
    ActionType, CommercialEvent, CreatorPolicy, EventType,
    FanCommercialState, FanStatus, SextingMode,
)
from services.commercial_policy import (  # noqa: E402
    CommercialContext, decide_next_action,
)
from services.payday import resolve_payday  # noqa: E402


def ev(t, raw=""):
    return CommercialEvent(type=t, raw_expression=raw)


def test_broke_fan_is_never_sold_to():
    """THE regression test. Live bug: fan said he was broke and got the same $28
    PPV three times."""
    d = decide_next_action(
        CreatorPolicy(),
        FanCommercialState(status=FanStatus.PAUSED_UNTIL_PAYDAY),
        [ev(EventType.WANTS_MEDIA)],   # he's asking for content while broke
        CommercialContext(),
    )
    assert d.must_not_send_media is True
    assert d.action in (ActionType.CONTINUE_NORMAL_CHAT, ActionType.CONTINUE_FREE_TEXT)
    assert d.mention_price is None


def test_payday_mention_schedules_followup():
    d = decide_next_action(
        CreatorPolicy(),
        FanCommercialState(),
        [ev(EventType.MONEY_UNAVAILABLE), ev(EventType.PAYDAY_MENTIONED, "Friday")],
        CommercialContext(),
    )
    assert d.action == ActionType.PAUSE_UNTIL_PAYDAY
    assert d.schedule_payday_followup is True
    assert d.new_status == FanStatus.PAUSED_UNTIL_PAYDAY
    assert d.must_not_send_media is True


def test_no_payday_means_no_schedule():
    d = decide_next_action(
        CreatorPolicy(),
        FanCommercialState(),
        [ev(EventType.MONEY_UNAVAILABLE)],
        CommercialContext(),
    )
    assert d.action == ActionType.PAUSE_NO_BUDGET
    assert d.schedule_payday_followup is False


def test_money_available_lifts_the_pause():
    d = decide_next_action(
        CreatorPolicy(),
        FanCommercialState(status=FanStatus.PAUSED_UNTIL_PAYDAY),
        [ev(EventType.MONEY_AVAILABLE)],
        CommercialContext(),
    )
    assert d.action == ActionType.RESUME_PREVIOUS_OFFER
    assert d.new_status == FanStatus.IDLE
    assert d.mention_previous_interest is True


def test_crisis_beats_everything():
    d = decide_next_action(
        CreatorPolicy(),
        FanCommercialState(status=FanStatus.PAID_SESSION_ACTIVE),
        [ev(EventType.CRISIS), ev(EventType.READY_TO_BUY)],
        CommercialContext(),
    )
    assert d.action == ActionType.HAND_OFF_TO_HUMAN
    assert d.must_not_send_media is True


def test_hybrid_teaser_gives_a_taste_then_offers():
    policy = CreatorPolicy(sexting_mode=SextingMode.HYBRID_TEASER, teaser_max_messages=4)
    # early: free teaser
    d = decide_next_action(
        policy, FanCommercialState(teaser_messages_used=1),
        [ev(EventType.WANTS_EXPLICIT)], CommercialContext(),
    )
    assert d.action == ActionType.START_FREE_TEASER
    assert d.may_be_explicit is True
    assert d.must_not_send_media is True
    # exhausted: transition to the offer
    d2 = decide_next_action(
        policy, FanCommercialState(teaser_messages_used=4),
        [ev(EventType.WANTS_EXPLICIT)], CommercialContext(),
    )
    assert d2.action == ActionType.END_TEASER_AND_OFFER


def test_paid_only_never_gives_free_explicit():
    d = decide_next_action(
        CreatorPolicy(sexting_mode=SextingMode.PAID_ONLY),
        FanCommercialState(),
        [ev(EventType.WANTS_EXPLICIT)],
        CommercialContext(),
    )
    assert d.may_be_explicit is False
    assert d.action != ActionType.START_FREE_TEASER


def test_free_text_mode_allows_explicit_even_when_broke():
    d = decide_next_action(
        CreatorPolicy(sexting_mode=SextingMode.FREE_TEXT_ALLOWED),
        FanCommercialState(status=FanStatus.PAUSED_UNTIL_PAYDAY),
        [ev(EventType.WANTS_EXPLICIT)],
        CommercialContext(),
    )
    assert d.action == ActionType.CONTINUE_FREE_TEXT
    assert d.may_be_explicit is True
    assert d.must_not_send_media is True   # media still paid


def test_qualification_does_not_fire_on_weak_signal():
    """Live bug: 'where are you from, James' fired mid-flirt out of nowhere."""
    d = decide_next_action(
        CreatorPolicy(sexting_mode=SextingMode.PAID_ONLY),
        FanCommercialState(),
        [],  # no real buying signal
        CommercialContext(),
    )
    assert d.action == ActionType.CONTINUE_NORMAL_CHAT


def test_no_approved_sets_means_no_sale():
    d = decide_next_action(
        CreatorPolicy(),
        FanCommercialState(),
        [ev(EventType.WANTS_MEDIA), ev(EventType.READY_TO_BUY)],
        CommercialContext(approved_sets_available=False),
    )
    assert d.action != ActionType.CREATE_PAID_SESSION


def test_payday_resolver():
    from datetime import datetime, timezone
    # Wednesday 2026-07-15
    now = datetime(2026, 7, 15, 12, 0, tzinfo=timezone.utc)
    dt, conf = resolve_payday("Friday", now=now, send_hour=18)
    assert dt is not None and conf > 0.5
    assert dt.weekday() == 4 and dt.hour == 18
    assert dt > now
    # unresolvable -> no guess
    dt2, conf2 = resolve_payday("when I feel like it", now=now)
    assert dt2 is None and conf2 == 0.0


if __name__ == "__main__":
    import traceback
    passed = failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS  {name}")
                passed += 1
            except Exception:
                print(f"  FAIL  {name}")
                traceback.print_exc()
                failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)