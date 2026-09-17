"""A customer that reacts to what was actually said to it.

WHY A SCRIPT IS NOT ENOUGH
--------------------------
``eval/trajectories.json`` says so itself:

    The customer here is SCRIPTED. §5 asks for adaptive trajectories too —
    'adaptive trajectories expose the consequences of earlier choices' — and a
    script cannot do that, because it says the same thing whatever the system
    replied.

That is the whole problem. A scripted customer asks "do you ever get to
chicago", and on the next turn says the next scripted line whether the creator
answered, ignored it, or sold something instead. The one behaviour §5 most
wants to measure — what happens to a conversation over time when an obligation
is dropped — is invisible to it, because the customer never notices.

This customer notices. It carries state, it re-asks questions nobody answered,
it pushes back when it is sold to after saying no, and it calls out a reply it
has already seen. Those reactions are what turn a detector's finding into a
consequence.

IT IS RULES, AND IT SAYS SO
---------------------------
No model is involved. The brief allows "reviewable rules or model
configuration", and rules are the option that can be read, argued with and
diffed — and the option that costs nothing and needs no provider access.

The cost is honesty about what this is: a rule-driven customer is not a
realistic one. It has a small vocabulary, it reacts to shallow features of the
creator's text, and a system could in principle be tuned to satisfy it without
being any good. It is a way to make earlier choices have later consequences,
which is exactly what §5 asks adaptive trajectories for, and it is not a
substitute for expert human review — §5 says that too, and it is the reason
this module produces no score.

REPRODUCIBLE
------------
Everything random comes from ``random.Random(seed)`` and nothing else — no
module-level ``random``, no clock, no iteration over a set. The same seed
replays the same conversation exactly, which is what makes a failure
investigable and two candidate systems comparable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from random import Random
from typing import Any, Callable

#: Topics the customer can raise, each with the word a detector matches on.
#:
#: Small and dull on purpose. The point is the SHAPE of the conversation —
#: a topic raised, dropped, returned to — and a larger vocabulary would make
#: transcripts harder to read without making the shape any truer.
TOPICS: tuple[tuple[str, str], ...] = (
    ("chicago", "do you ever get to chicago"),
    ("trip", "how was your trip"),
    ("dog", "the dog got into the bins again"),
    ("project", "finally finished that project i mentioned"),
    ("show", "did you watch that show everyone goes on about"),
    ("weekend", "what are you doing this weekend"),
    ("work", "work has been relentless this week"),
    ("sleep", "i barely slept last night"),
)

_QUESTION = re.compile(r"\?")
_SALES = re.compile(
    r"\b(buy|unlock|tip|ppv|\$\d|price|special|offer|treat|spoil)\b", re.I
)


@dataclass
class CustomerState:
    """What this customer remembers, and what a rule may look at."""

    #: Topics raised whose keyword no creator reply has mentioned yet.
    open_questions: list[str] = field(default_factory=list)
    #: Topics already raised, so one is not raised twice by accident.
    raised: list[str] = field(default_factory=list)
    #: True once the customer has declined to spend.
    said_no_to_spending: bool = False
    #: How many times a sale was pushed after that.
    pushed_after_no: int = 0
    #: Replies already seen, for noticing a repeat.
    seen_replies: list[str] = field(default_factory=list)
    #: Turns taken.
    turn: int = 0
    #: Which rule produced the previous turn, so a rule can decline to repeat
    #: itself without every rule having to track that separately.
    last_rule: str = ""


def _mentions(reply: str, keyword: str) -> bool:
    return keyword.lower() in reply.lower()


def _asks_something(reply: str) -> bool:
    return bool(_QUESTION.search(reply))


def _pushes_a_sale(reply: str) -> bool:
    return bool(_SALES.search(reply))


# ---------------------------------------------------------------------------
# The rules
# ---------------------------------------------------------------------------
#
# Each is a named pair: when it applies, and what the customer then says.
# Ordered — the first match wins — so the ordering IS the customer's priority,
# and it is readable as one list rather than buried in branches.


@dataclass(frozen=True)
class Rule:
    name: str
    #: What this rule exists to expose, recorded on the turn it fires.
    tests: str
    when: Callable[[CustomerState, str], bool]
    say: Callable[[CustomerState, Random], str]


def _reask(state: CustomerState, rng: Random) -> str:
    keyword = state.open_questions[0]
    return rng.choice(
        [
            f"you never said about {keyword}",
            f"still curious about {keyword} though",
            f"anyway, {keyword}?",
        ]
    )


def _push_back(state: CustomerState, rng: Random) -> str:
    return rng.choice(
        [
            "i literally just said i cant right now",
            "not buying anything, i told you",
            "can we just talk without the sales pitch",
        ]
    )


def _decline(state: CustomerState, rng: Random) -> str:
    return rng.choice(
        [
            "id love to but im broke until friday",
            "cant spend anything this week honestly",
            "not right now, money is tight",
        ]
    )


def _call_out_repeat(state: CustomerState, rng: Random) -> str:
    return rng.choice(
        ["you said that already", "you literally just sent that", "deja vu"]
    )


def _answer(state: CustomerState, rng: Random) -> str:
    return rng.choice(
        ["yeah pretty much", "honestly not really", "sort of, hard to say"]
    )


def _new_topic(state: CustomerState, rng: Random) -> str:
    available = [pair for pair in TOPICS if pair[0] not in state.raised]
    if not available:
        return rng.choice(["anyway", "mm", "yeah", "fair enough"])
    keyword, line = available[0]
    state.raised.append(keyword)
    state.open_questions.append(keyword)
    return line


def _low_energy(state: CustomerState, rng: Random) -> str:
    return rng.choice(["mm", "yeah", "k", "same honestly", "tired"])


#: First match wins, so this ordering is the customer's priority: a dropped
#: obligation outranks being sold to, which outranks small talk.
RULES: tuple[Rule, ...] = (
    Rule(
        name="push_back_after_a_no",
        tests="pressing a sale after an explicit decline",
        when=lambda state, reply: state.said_no_to_spending and _pushes_a_sale(reply),
        say=_push_back,
    ),
    Rule(
        name="call_out_a_repeated_reply",
        tests="sending the same line twice",
        # Once, not every turn. A creator stuck in a loop would otherwise
        # capture this customer completely — the first version of this rule
        # fired twelve turns running and starved every other rule, including
        # the two that matter most, which is its own kind of blindness.
        when=lambda state, reply: (
            bool(reply)
            and reply in state.seen_replies
            and state.last_rule != "call_out_a_repeated_reply"
        ),
        say=_call_out_repeat,
    ),
    Rule(
        name="reask_an_unanswered_question",
        tests="an obligation dropped and never picked back up",
        # Not immediately: a customer who re-asks on the very next turn is not
        # testing memory, it is testing the last reply. Three turns of silence
        # is what makes it a continuity question.
        when=lambda state, reply: bool(state.open_questions) and state.turn >= 3,
        say=_reask,
    ),
    Rule(
        name="answer_a_direct_question",
        tests="ordinary back and forth",
        when=lambda state, reply: _asks_something(reply),
        say=_answer,
    ),
)


@dataclass
class AdaptiveCustomer:
    """A reproducible customer that reacts to the creator's replies.

    ``next_message`` matches ``Disturbance.responds_to``, which is the seam the
    harness already had and nothing supplied.
    """

    seed: int = 0
    #: Turns before the customer declines to spend, so "pressing after a no"
    #: has a no to press. 0 means it never declines.
    decline_at_turn: int = 4
    #: How often it gives a low-engagement reply instead of raising something.
    low_engagement_rate: float = 0.25

    state: CustomerState = field(default_factory=CustomerState)
    #: Which rule produced each turn, so a transcript can be read back against
    #: the behaviour it was meant to provoke.
    fired: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._rng = Random(self.seed)

    def _close_answered_questions(self, reply: str) -> None:
        if not reply:
            return
        self.state.open_questions = [
            keyword
            for keyword in self.state.open_questions
            if not _mentions(reply, keyword)
        ]

    def next_message(self, creator_replies: list[str]) -> str:
        """The customer's next message, given everything said to it so far."""
        last = str(creator_replies[-1]) if creator_replies else ""
        self._close_answered_questions(last)

        if self.state.said_no_to_spending and _pushes_a_sale(last):
            self.state.pushed_after_no += 1

        # Declining comes BEFORE the rules, because it is a state transition on
        # a turn number rather than a reaction to the last reply — and because
        # putting it after them meant any matching rule on that turn skipped it
        # permanently, leaving said_no_to_spending False forever and the
        # push-back rule dead. A customer that can never say no cannot test
        # what happens when a no is ignored.
        if (
            self.decline_at_turn
            and self.state.turn == self.decline_at_turn
            and not self.state.said_no_to_spending
        ):
            self.state.said_no_to_spending = True
            return self._emit("decline_to_spend", _decline(self.state, self._rng), last)

        for rule in RULES:
            if rule.when(self.state, last):
                return self._emit(rule.name, rule.say(self.state, self._rng), last)

        if self._rng.random() < self.low_engagement_rate:
            return self._emit("low_engagement", _low_energy(self.state, self._rng), last)

        return self._emit("raise_a_topic", _new_topic(self.state, self._rng), last)

    def _emit(self, rule: str, message: str, last_reply: str) -> str:
        self.fired.append(rule)
        self._advance(last_reply)
        self.state.last_rule = rule
        return message

    def _advance(self, last_reply: str) -> None:
        if last_reply:
            self.state.seen_replies.append(last_reply)
        self.state.turn += 1


def build(config: dict[str, Any]) -> AdaptiveCustomer:
    """One customer from a trajectory's ``adaptive`` block."""
    return AdaptiveCustomer(
        seed=int(config.get("seed") or 0),
        decline_at_turn=int(config.get("decline_at_turn", 4) or 0),
        low_engagement_rate=float(config.get("low_engagement_rate", 0.25)),
    )
