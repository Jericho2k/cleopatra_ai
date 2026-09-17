"""After a purchase, say something that fits — or say nothing.

WHAT THIS REPLACES
------------------
Seven sentences in services/suggestions.py:

    "let me know what you think 🙈", "dying to know your reaction 😏", ...

picked with random.choice at SCHEDULE time and frozen into the queued action.
Every customer, after every purchase, got one of seven, chosen before anything
was known about how the next thirty seconds would go.

Freezing it also bypassed the writer: services/proactive.py generates from a
goal and the conversation unless `_delivery.text` is already set. So the one
proactive message that follows money changing hands was the only one that
never read the conversation it was about.

THE ONE THAT MATTERS MOST
-------------------------
POST_PURCHASE_REACTION was explicitly exempt from the worker's auto-mode gate.
A customer whose creator had switched automation OFF still received an
automated message about their purchase. Auto off means an operator has the
conversation, and money having just changed hands is the worst possible moment
to message over the top of them.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from models.conversation_continuity import (
    EvidenceType,
    OpenThread,
    ThreadKind,
    ThreadParty,
)
from services.post_purchase import (
    REACTION_GOAL,
    asked_for_no_follow_up,
    decide,
    he_replied_since,
)

BOUGHT_AT = datetime(2026, 9, 17, 12, 0, tzinfo=timezone.utc)


def _msg(role: str, *, minutes: float, content: str = "hey"):
    return {
        "role": role,
        "content": content,
        "sent_at": (BOUGHT_AT + timedelta(minutes=minutes)).isoformat(),
    }


def _silence_thread(summary: str) -> OpenThread:
    return OpenThread(
        creator_id="creator-1",
        fan_id="fan-1",
        kind=ThreadKind.DEFERRED_TOPIC,
        raised_by=ThreadParty.FAN,
        summary=summary,
        evidence_type=EvidenceType.STATED,
    )


# ===========================================================================
# 1. Operator takeover — the check that was missing entirely
# ===========================================================================


def test_auto_mode_off_means_silence():
    """A customer whose creator turned automation off still got messaged.

    POST_PURCHASE_REACTION was exempted from the worker's auto-mode gate by
    name. Auto off means an operator has this conversation.
    """
    result = decide(history=[], purchased_at=BOUGHT_AT, auto_mode=False)

    assert result.silent
    assert "auto mode is off" in result.reason


def test_a_conversation_on_hold_means_silence():
    result = decide(
        history=[], purchased_at=BOUGHT_AT, auto_mode=True, frozen_for_review=True
    )

    assert result.silent
    assert "on hold" in result.reason


def test_operator_takeover_outranks_everything_else():
    """Reported as the takeover, not as one of the softer reasons."""
    result = decide(
        history=[_msg("fan", minutes=1)],
        open_threads=[_silence_thread("he asked for no follow up")],
        purchased_at=BOUGHT_AT,
        auto_mode=False,
        frozen_for_review=True,
    )

    assert "on hold" in result.reason


# ===========================================================================
# 2. He already replied
# ===========================================================================


def test_a_customer_who_already_replied_is_not_nudged():
    """"Don't leave me hanging" sent to somebody who did not leave anyone
    hanging reads as not having been listened to."""
    result = decide(history=[_msg("fan", minutes=1)], purchased_at=BOUGHT_AT)

    assert result.silent
    assert "already replied" in result.reason


def test_a_customer_who_said_nothing_gets_the_nudge():
    """The case the nudge exists for."""
    result = decide(
        history=[_msg("fan", minutes=-30), _msg("creator", minutes=-1)],
        purchased_at=BOUGHT_AT,
    )

    assert result.send
    assert "has not said anything since" in result.reason


def test_the_creators_own_message_is_not_him_replying():
    result = decide(history=[_msg("creator", minutes=1)], purchased_at=BOUGHT_AT)

    assert result.send


def test_a_message_from_before_the_purchase_is_not_a_reply_to_it():
    assert he_replied_since([_msg("fan", minutes=-5)], BOUGHT_AT) is False
    assert he_replied_since([_msg("fan", minutes=5)], BOUGHT_AT) is True


def test_an_undated_message_is_not_counted_as_a_reply():
    """Guessing it came after would suppress a nudge on no evidence.

    The quieter failure, and still a failure.
    """
    assert he_replied_since([{"role": "fan", "content": "hey"}], BOUGHT_AT) is False


def test_no_purchase_time_means_no_reply_can_be_established():
    assert he_replied_since([_msg("fan", minutes=5)], None) is False


# ===========================================================================
# 3. He asked not to be followed up
# ===========================================================================


@pytest.mark.parametrize(
    "summary",
    [
        "he asked for no follow up tonight",
        "do not message him until the weekend",
        "he asked us to stop messaging for a while",
    ],
)
def test_a_request_not_to_be_followed_up_is_respected(summary):
    result = decide(
        history=[], open_threads=[_silence_thread(summary)], purchased_at=BOUGHT_AT
    )

    assert result.silent
    assert "not to be followed up" in result.reason


def test_an_ordinary_deferred_topic_is_not_a_request_for_silence():
    """Putting off a subject is not asking to be left alone."""
    result = decide(
        history=[],
        open_threads=[_silence_thread("his sister's wedding, until it is closer")],
        purchased_at=BOUGHT_AT,
    )

    assert result.send


def test_a_question_is_never_read_as_a_request_for_silence():
    question = OpenThread(
        creator_id="creator-1",
        fan_id="fan-1",
        kind=ThreadKind.QUESTION,
        raised_by=ThreadParty.FAN,
        summary="do not message him — whether she gets to chicago",
        evidence_type=EvidenceType.STATED,
    )

    assert asked_for_no_follow_up([question]) is False


# ===========================================================================
# 4. What is sent when something is sent
# ===========================================================================


def test_the_goal_is_a_goal_and_not_a_line():
    """services/proactive.py builds the message from this plus the actual
    conversation — the path every other proactive action already took, and the
    one a frozen string skipped."""
    assert "short message" in REACTION_GOAL
    assert "in your own voice" in REACTION_GOAL


def test_the_goal_forbids_the_things_a_post_purchase_message_must_not_do():
    lowered = REACTION_GOAL.lower()

    assert "do not state a price" in lowered
    assert "do not offer anything else" in lowered
    assert "do not send media" in lowered
    # And the specific failure the frozen lines had: asking him something he
    # has already answered.
    assert "already answered" in lowered


def test_the_same_situation_always_produces_the_same_decision():
    """What replacing random.choice actually bought.

    Asserted behaviourally rather than by scanning the source for the word
    "random" — a first draft did that and failed on the module's own docstring
    explaining that randomness was removed. That is the fourth substring
    assertion over prose to misfire on this branch, after the health-check
    timestamp, the context digest and the _delivery comment. The behaviour is
    what matters and the behaviour is testable.
    """
    import services.post_purchase as module

    assert not hasattr(module, "random"), "the module imports no randomness"

    situations = [
        dict(history=[], purchased_at=BOUGHT_AT),
        dict(history=[_msg("fan", minutes=1)], purchased_at=BOUGHT_AT),
        dict(history=[], purchased_at=BOUGHT_AT, auto_mode=False),
    ]
    for situation in situations:
        answers = {decide(**situation).reason for _ in range(20)}
        assert len(answers) == 1, answers
