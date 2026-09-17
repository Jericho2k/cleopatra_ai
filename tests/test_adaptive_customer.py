"""A customer that reacts, so earlier choices have later consequences.

eval/trajectories.json used to say this about itself:

    The customer here is SCRIPTED. §5 asks for adaptive trajectories too —
    'adaptive trajectories expose the consequences of earlier choices' — and a
    script cannot do that, because it says the same thing whatever the system
    replied.

That is the gap. A scripted customer asks a question and then says its next
scripted line whether the creator answered it, ignored it, or sold something
instead — so the behaviour §5 most wants to measure, what a dropped obligation
does to a conversation over time, is invisible to it.

The tests that matter here are the ones showing DIFFERENT creator behaviour
producing DIFFERENT conversations. Everything else is bookkeeping.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from services.adaptive_customer import RULES, AdaptiveCustomer, build
from services.trajectory_eval import load_trajectories

ROOT = Path(__file__).resolve().parents[1]
TRAJECTORIES = ROOT / "eval" / "trajectories.json"


def _converse(customer: AdaptiveCustomer, creator, turns: int) -> list[str]:
    """Run ``turns`` exchanges, with ``creator`` deciding each reply."""
    replies: list[str] = []
    said: list[str] = []
    for index in range(turns):
        said.append(customer.next_message(replies))
        replies.append(creator(index, said[-1]))
    return said


def _always_sells(_index: int, _message: str) -> str:
    return "you should unlock my new set, only $15"


def _answers_everything(_index: int, message: str) -> str:
    return f"oh really? tell me more about {message}"


def _repeats_itself(_index: int, _message: str) -> str:
    return "that sounds like it was a really long week for you honestly"


# ===========================================================================
# 1. The consequence of an earlier choice
# ===========================================================================


def test_ignoring_a_question_gets_it_asked_again():
    """A dropped obligation becomes a thing the customer does, not just a finding."""
    customer = AdaptiveCustomer(seed=1)

    _converse(customer, _always_sells, 8)

    assert "reask_an_unanswered_question" in customer.fired


def test_answering_a_question_stops_it_being_re_asked():
    """The other half. A customer that re-asks regardless is a script again."""
    customer = AdaptiveCustomer(seed=1, decline_at_turn=0)

    _converse(customer, _answers_everything, 8)

    assert "reask_an_unanswered_question" not in customer.fired


def test_selling_after_a_no_gets_pushed_back_on_every_time():
    customer = AdaptiveCustomer(seed=1, decline_at_turn=3)

    _converse(customer, _always_sells, 12)

    assert "decline_to_spend" in customer.fired
    assert customer.fired.count("push_back_after_a_no") >= 3
    # Counted, so "pressed after a no" is a number a report can carry.
    assert customer.state.pushed_after_no >= 3


def test_not_selling_after_a_no_is_never_pushed_back_on():
    customer = AdaptiveCustomer(seed=1, decline_at_turn=3)

    _converse(customer, _answers_everything, 12)

    assert "push_back_after_a_no" not in customer.fired
    assert customer.state.pushed_after_no == 0


def test_repeating_a_reply_is_noticed():
    customer = AdaptiveCustomer(seed=1, decline_at_turn=0)

    _converse(customer, _repeats_itself, 6)

    assert "call_out_a_repeated_reply" in customer.fired


def test_a_creator_stuck_in_a_loop_does_not_capture_the_customer():
    """The first version of this rule fired twelve turns running.

    It starved every other rule — including the two that matter most — so a
    creator repeating itself made the customer blind to everything else.
    """
    customer = AdaptiveCustomer(seed=1, decline_at_turn=0)

    _converse(customer, _repeats_itself, 12)

    assert "call_out_a_repeated_reply" in customer.fired
    assert customer.fired.count("call_out_a_repeated_reply") < 12
    # It never calls out the same repeat twice in a row.
    pairs = zip(customer.fired, customer.fired[1:])
    assert not any(a == b == "call_out_a_repeated_reply" for a, b in pairs)


def test_two_creators_produce_two_different_conversations():
    """The property the whole module exists for, stated directly."""
    selling = _converse(AdaptiveCustomer(seed=99), _always_sells, 12)
    answering = _converse(AdaptiveCustomer(seed=99), _answers_everything, 12)

    assert selling != answering


def test_the_customer_can_always_say_no():
    """A regression on a real bug.

    The decline used to be checked AFTER the rules, so any rule matching on
    that turn skipped it permanently: said_no_to_spending stayed False forever
    and the push-back rule was dead code. A customer that can never say no
    cannot test what happens when a no is ignored.
    """
    customer = AdaptiveCustomer(seed=5, decline_at_turn=2)

    _converse(customer, _repeats_itself, 10)

    assert customer.state.said_no_to_spending is True
    assert "decline_to_spend" in customer.fired


# ===========================================================================
# 2. Reproducible
# ===========================================================================


def test_the_same_seed_replays_the_same_conversation():
    """What makes a failure investigable and two candidates comparable."""
    first = _converse(AdaptiveCustomer(seed=4242), _always_sells, 25)
    second = _converse(AdaptiveCustomer(seed=4242), _always_sells, 25)

    assert first == second


def test_a_different_seed_is_a_different_conversation():
    first = _converse(AdaptiveCustomer(seed=1), _always_sells, 25)
    second = _converse(AdaptiveCustomer(seed=2), _always_sells, 25)

    assert first != second


def test_nothing_is_drawn_from_a_shared_random_source():
    """Interleaving two customers must not change either one.

    A module-level `random` would make every conversation depend on every other
    conversation in the same process, which is reproducibility in name only.
    """
    alone = _converse(AdaptiveCustomer(seed=77), _always_sells, 10)

    a = AdaptiveCustomer(seed=77)
    b = AdaptiveCustomer(seed=1234)
    interleaved: list[str] = []
    replies_a: list[str] = []
    replies_b: list[str] = []
    for index in range(10):
        interleaved.append(a.next_message(replies_a))
        replies_a.append(_always_sells(index, ""))
        b.next_message(replies_b)
        replies_b.append(_always_sells(index, ""))

    assert interleaved == alone


def test_every_rule_carries_what_it_is_for():
    """A transcript has to be readable against the behaviour it provoked."""
    for rule in RULES:
        assert rule.name and rule.tests


# ===========================================================================
# 3. The trajectories built from it
# ===========================================================================


def test_an_adaptive_trajectory_generates_the_turns_it_declares():
    trajectories = load_trajectories([
        {
            "name": "generated",
            "covers": "a long one",
            "adaptive": {"seed": 11, "turns": 40},
            "requires": {"turns": 40},
        }
    ])

    assert len(trajectories[0].disturbances) == 40


def test_the_generated_turns_share_one_customer():
    """A fresh customer per turn would be a script with extra steps."""
    trajectories = load_trajectories([
        {"name": "g", "adaptive": {"seed": 11, "turns": 6, "decline_at_turn": 2}}
    ])
    disturbances = trajectories[0].disturbances

    replies: list[str] = []
    said = []
    for disturbance in disturbances:
        said.append(disturbance.next_message(replies))
        replies.append("you should unlock my new set, only $15")

    # State carried across turns: it declined, then objected to being sold to.
    assert any("cant" in line or "broke" in line or "tight" in line for line in said)
    assert any("told you" in line or "sales pitch" in line or "just said" in line
               for line in said)


def test_an_adaptive_trajectory_can_place_gaps_between_turns():
    trajectories = load_trajectories([
        {
            "name": "g",
            "adaptive": {"seed": 11, "turns": 5, "gaps": {"2": 1.0, "4": 7.0}},
        }
    ])
    days = [d.days_since_previous for d in trajectories[0].disturbances]

    assert days == [0.0, 0.0, 1.0, 0.0, 7.0]


def test_build_reads_the_configuration_block():
    customer = build({"seed": 9, "decline_at_turn": 2, "low_engagement_rate": 0.0})

    assert customer.seed == 9
    assert customer.decline_at_turn == 2
    assert customer.low_engagement_rate == 0.0


# ===========================================================================
# 4. The shipped file
# ===========================================================================


def test_the_long_conversation_rows_are_actually_long():
    """The claim and the thing, finally the same length.

    The scripted row claiming "40-80 turns of ordinary conversation" was seven
    turns long. It no longer makes that claim, and a row that does is now
    genuinely that long.
    """
    trajectories = load_trajectories(
        json.loads(TRAJECTORIES.read_text())["trajectories"]
    )
    by_name = {t.name: t for t in trajectories}

    long_form = by_name["ordinary conversation, adaptive"]
    assert long_form.requires_turns == 40
    assert 40 <= len(long_form.disturbances) <= 80

    deferred = by_name["a question deferred across a long conversation"]
    assert deferred.requires_turns == 30
    assert len(deferred.disturbances) >= 30


def test_the_short_scripted_rows_no_longer_claim_a_length_they_lack():
    """Checked by what the harness enforces, not by the prose.

    The prose still says "40-80" — as a cross-reference to the row that now
    covers it, which is worth keeping. What matters is that this row no longer
    REQUIRES that length, so the coverage check reports no gap for it: the
    claim and the fixture agree again.
    """
    from services.trajectory_eval import TrajectoryReport, TurnRecord, coverage_gaps

    trajectories = load_trajectories(
        json.loads(TRAJECTORIES.read_text())["trajectories"]
    )
    by_name = {t.name: t for t in trajectories}

    for name in (
        "ordinary conversation with no purchase goal",
        "a question deferred behind two other topics",
    ):
        row = by_name[name]
        assert row.requires_turns == 0, name
        ran = TrajectoryReport(
            trajectory=name,
            turns=[
                TurnRecord(index=index, customer_message="x")
                for index in range(len(row.disturbances))
            ],
        )
        assert coverage_gaps(row, ran) == [], name


@pytest.mark.parametrize(
    "name",
    [
        "ordinary conversation, adaptive",
        "a question deferred across a long conversation",
        "he returns after a day and a week, adaptive",
    ],
)
def test_every_shipped_adaptive_row_is_reproducible(name):
    def _load():
        trajectories = load_trajectories(
            json.loads(TRAJECTORIES.read_text())["trajectories"]
        )
        row = next(t for t in trajectories if t.name == name)
        replies: list[str] = []
        said = []
        for disturbance in row.disturbances:
            said.append(disturbance.next_message(replies))
            replies.append("mm, anyway. want to unlock something?")
        return said

    assert _load() == _load()
