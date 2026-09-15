"""Sexual text is not a commercial authorization.

The regression this file exists for, stated plainly: a fan who arrives already
explicit, on a turn where there is correctly nothing to sell, used to be
answered non-explicitly — because ``CommercialDecision.may_be_explicit``
defaults to False and the prompt turned that into "Keep this response
non-explicit." The commercial engine was answering a question about words.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.commercial import CreatorPolicy, FanStatus, SextingMode  # noqa: E402
from services.text_intimacy import (  # noqa: E402
    TextIntimacy,
    decide_text_intimacy,
)

EXPLICIT_FAN = {"wants_explicit": "true", "conversation_energy": "rising"}
JUST_CHATTING = {"conversation_energy": "flat"}


def _decide(policy=None, **kwargs):
    return decide_text_intimacy(
        policy=policy or CreatorPolicy(sexting_mode=SextingMode.HYBRID_TEASER),
        **kwargs,
    )


# --- the decoupling itself ---------------------------------------------------


def test_an_explicit_fan_gets_an_explicit_register_with_nothing_being_sold():
    """The acceptance criterion, in one assertion.

    CONTINUE_NORMAL_CHAT means "no PPV this turn". It must not mean "speak
    non-explicitly".
    """
    decision = _decide(
        situation=EXPLICIT_FAN,
        commercial_decision={"action": "CONTINUE_NORMAL_CHAT", "may_be_explicit": False},
    )
    assert decision.level is TextIntimacy.EXPLICIT
    assert decision.may_be_explicit is True


def test_the_commercial_decisions_own_flag_is_not_consulted():
    """Same register either way: the flag is about media, not about words."""
    off = _decide(
        situation=EXPLICIT_FAN,
        commercial_decision={"action": "CONTINUE_NORMAL_CHAT", "may_be_explicit": False},
    )
    on = _decide(
        situation=EXPLICIT_FAN,
        commercial_decision={"action": "CONTINUE_NORMAL_CHAT", "may_be_explicit": True},
    )
    assert off.level is on.level


def test_a_fan_who_is_not_being_sexual_does_not_get_escalated_at():
    decision = _decide(
        situation=JUST_CHATTING,
        commercial_decision={"action": "CONTINUE_NORMAL_CHAT"},
    )
    assert decision.level is TextIntimacy.FLIRTY
    assert decision.may_be_explicit is False


# --- the agency's controls are preserved -------------------------------------


def test_paid_only_still_means_explicit_text_is_the_product():
    decision = _decide(
        policy=CreatorPolicy(sexting_mode=SextingMode.PAID_ONLY),
        situation=EXPLICIT_FAN,
        commercial_decision={"action": "CONTINUE_NORMAL_CHAT"},
    )
    assert decision.level is TextIntimacy.FLIRTY
    assert "paid_only_mode_caps_free_text" in decision.reason_codes


def test_paid_only_does_allow_the_register_inside_something_he_paid_for():
    """He bought the scene. Talking about it chastely is the regression."""
    decision = _decide(
        policy=CreatorPolicy(sexting_mode=SextingMode.PAID_ONLY),
        situation=EXPLICIT_FAN,
        commercial_decision={"action": "CONTINUE_NORMAL_CHAT"},
        scene={"beat": "PLAY", "intimacy_level": 4},
    )
    assert decision.level is TextIntimacy.EXPLICIT
    assert decision.consumes_free_allowance is False, "he already paid for this"


def test_free_explicit_text_spends_the_configured_allowance():
    """The safeguard against decoupling becoming unlimited free service."""
    decision = _decide(
        policy=CreatorPolicy(sexting_mode=SextingMode.HYBRID_TEASER, teaser_max_messages=4),
        situation=EXPLICIT_FAN,
        commercial_decision={"action": "CONTINUE_NORMAL_CHAT"},
        teaser_messages_used=0,
    )
    assert decision.may_be_explicit is True
    assert decision.consumes_free_allowance is True


def test_an_exhausted_allowance_ends_the_free_explicit_register():
    decision = _decide(
        policy=CreatorPolicy(sexting_mode=SextingMode.HYBRID_TEASER, teaser_max_messages=4),
        situation=EXPLICIT_FAN,
        commercial_decision={"action": "CONTINUE_NORMAL_CHAT"},
        teaser_messages_used=4,
    )
    assert decision.level is TextIntimacy.FLIRTY
    assert decision.consumes_free_allowance is False
    assert "free_allowance_exhausted" in decision.reason_codes


def test_a_cooling_down_allowance_does_not_reopen_by_itself():
    decision = _decide(
        policy=CreatorPolicy(sexting_mode=SextingMode.FREE_TEXT_ALLOWED),
        situation=EXPLICIT_FAN,
        commercial_decision={"action": "CONTINUE_NORMAL_CHAT"},
        free_mode_on_cooldown=True,
    )
    assert decision.level is TextIntimacy.FLIRTY


def test_a_paused_fan_is_not_handed_the_paid_experience_for_free():
    decision = _decide(
        policy=CreatorPolicy(sexting_mode=SextingMode.HYBRID_TEASER),
        situation=EXPLICIT_FAN,
        commercial_decision={"action": "CONTINUE_NORMAL_CHAT"},
        fan_status=FanStatus.PAUSED_NO_BUDGET,
    )
    assert decision.level is TextIntimacy.FLIRTY
    assert "paused_no_free_paid_experience" in decision.reason_codes


# --- safety is still absolute ------------------------------------------------


def test_a_crisis_signal_stops_everything_including_flirting():
    decision = _decide(
        situation={**EXPLICIT_FAN, "crisis_signal": "self_harm"},
        commercial_decision={"action": "CONTINUE_NORMAL_CHAT"},
    )
    assert decision.level is TextIntimacy.NONE
    assert decision.reason_codes == ["crisis"]


def test_a_frozen_account_stops_everything():
    decision = _decide(situation=EXPLICIT_FAN, frozen_for_review=True)
    assert decision.level is TextIntimacy.NONE


def test_a_handoff_stops_everything():
    decision = _decide(
        situation=EXPLICIT_FAN,
        commercial_decision={"action": "HAND_OFF_TO_HUMAN"},
    )
    assert decision.level is TextIntimacy.NONE
