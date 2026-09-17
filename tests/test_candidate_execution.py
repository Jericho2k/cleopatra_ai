"""Two candidates, one executor, compared on what they would actually do.

THE GAP
-------
services/decision_replay.py compared ConversationDecision objects, and the
review's objection was that this is not a comparison of conversational cores:

    replay does not compare two complete new conversational cores through a
    shared writer/executor.

Two candidates agreeing that a turn should `offer_content` tells you almost
nothing. What reaches the customer is a sentence, and what happens to their
account is an operation that ran or was refused — neither of which a decision
comparison can see.

The other half of the gap was that there was only one candidate to compare:

    The one-call reply-plus-intent candidate is absent.

Nothing here sends anything. "Executed" means the deterministic constraints
permitted the operation. A candidate that would have sent is compared against
one that would not have, which is the whole question, and neither sends.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from models.conversation_decision import (
    ConversationDecision,
    HoldReason,
    OperationKind,
    ProposedOperation,
)
from services.candidate_execution import (
    compare_candidates,
    compare_executions,
    execute_candidate,
)
from services.context_packet import build_context_packet
from services.decision_owners import (
    CandidateAnswer,
    DecideThenWrite,
    ReplyPlusIntentOwner,
    parse_reply_plus_intent,
)


def run(coro):
    return asyncio.run(coro)


PACKET = build_context_packet(history=[{"role": "fan", "content": "what have you got"}])


def _answer(**overrides) -> CandidateAnswer:
    decision = overrides.pop(
        "decision",
        ConversationDecision(source="candidate", confidence=0.8),
    )
    return CandidateAnswer(decision=decision, **overrides)


def _decision(**overrides) -> ConversationDecision:
    values = {"source": "candidate", "confidence": 0.8}
    values.update(overrides)
    return ConversationDecision(**values)


def _offer(subject: str = "the hotel set") -> ConversationDecision:
    return _decision(
        proposed_operation=ProposedOperation(
            kind=OperationKind.OFFER_CONTENT, subject=subject, because="he asked"
        )
    )


AUTHORIZED = frozenset({OperationKind.OFFER_CONTENT})
KNOWN = frozenset({"the hotel set"})


# ===========================================================================
# 1. The one-call candidate, which did not exist
# ===========================================================================


def _one_call(**payload) -> str:
    body = {
        "reply": "i do, want to see?",
        "operation": "none",
        "hold": "none",
        "confidence": 0.8,
    }
    body.update(payload)
    return json.dumps(body)


def test_a_one_call_answer_carries_both_halves():
    answer, refusal = parse_reply_plus_intent(_one_call(), source="one_call")

    assert refusal == ""
    assert answer.reply == "i do, want to see?"
    assert answer.wrote_its_own_reply is True
    assert answer.decision.confidence == 0.8


def test_the_decision_half_goes_through_the_same_strict_parser():
    """A candidate with a laxer parser would win by being marked wrong less
    often, which would make the comparison measure the parsers."""
    answer, refusal = parse_reply_plus_intent(
        _one_call(operation="charge_his_card"), source="one_call"
    )

    assert answer is None
    assert "charge_his_card" in refusal


@pytest.mark.parametrize(
    "payload,expected",
    [
        ({"confidence": 5}, "outside"),
        ({"hold": "typo"}, "not one this system knows"),
    ],
)
def test_the_one_call_candidate_inherits_every_refusal(payload, expected):
    _, refusal = parse_reply_plus_intent(_one_call(**payload), source="one_call")

    assert expected in refusal


def test_a_missing_reply_is_refused():
    body = json.loads(_one_call())
    body.pop("reply")

    answer, refusal = parse_reply_plus_intent(json.dumps(body), source="one_call")

    assert answer is None
    assert "reply" in refusal


def test_saying_nothing_is_allowed_when_a_hold_says_why():
    answer, refusal = parse_reply_plus_intent(
        _one_call(reply="", hold="respect_silence", hold_detail="he asked for quiet"),
        source="one_call",
    )

    assert refusal == ""
    assert answer.reply == ""
    assert "quiet" in answer.reason


def test_an_empty_reply_with_no_hold_is_a_failure_wearing_a_success():
    answer, refusal = parse_reply_plus_intent(_one_call(reply=""), source="one_call")

    assert answer is None
    assert "needs a hold" in refusal


def test_the_owner_reports_a_refusal_as_insufficient_evidence():
    async def answers_badly(**_kwargs):
        return SimpleNamespace(text='{"reply": "hi", "operation": "typo", '
                                    '"hold": "none", "confidence": 0.5}')

    answer = run(ReplyPlusIntentOwner(answers_badly).answer(PACKET, {}))

    assert answer.decision.hold is HoldReason.INSUFFICIENT_EVIDENCE
    assert answer.decision.confidence == 0.0
    assert "typo" in answer.decision.hold_detail


def test_a_provider_that_cannot_be_reached_is_not_a_decision():
    async def unreachable(**_kwargs):
        raise RuntimeError("503")

    answer = run(ReplyPlusIntentOwner(unreachable).answer(PACKET, {}))

    assert answer.decision.hold is HoldReason.INSUFFICIENT_EVIDENCE
    assert answer.decision.confidence == 0.0


def test_both_candidates_are_given_the_same_evidence():
    """§5: "Replay gives candidates the same evidence" — and evidence
    assembled twice is two pieces of evidence, however similar."""
    from services.decision_owners import build_semantic_prompt

    _, shared_user = build_semantic_prompt(PACKET, {"latest_message": "hey"})
    _, one_call_user = ReplyPlusIntentOwner(None)._build(
        PACKET, {"latest_message": "hey"}
    )

    assert one_call_user == shared_user


# ===========================================================================
# 2. The shared executor
# ===========================================================================


def test_a_candidate_that_wrote_its_own_reply_is_not_rewritten():
    async def write(*_args, **_kwargs):
        return "the writer wrote this"

    executed = run(
        execute_candidate(
            _answer(reply="the candidate wrote this", wrote_its_own_reply=True),
            packet=PACKET,
            state={},
            write=write,
        )
    )

    assert executed.reply == "the candidate wrote this"
    assert executed.wrote_its_own_reply is True


def test_a_candidate_that_did_not_write_gets_the_shared_writer():
    async def write(*_args, **_kwargs):
        return "the writer wrote this"

    executed = run(
        execute_candidate(
            _answer(reply="", wrote_its_own_reply=False),
            packet=PACKET,
            state={},
            write=write,
        )
    )

    assert executed.reply == "the writer wrote this"
    assert executed.wrote_its_own_reply is False


def test_an_authorized_operation_is_executed():
    executed = run(
        execute_candidate(
            _answer(decision=_offer(), reply="here you go", wrote_its_own_reply=True),
            packet=PACKET,
            state={},
            authorized_operations=AUTHORIZED,
            known_subjects=KNOWN,
        )
    )

    assert executed.executed is True
    assert executed.suppressed_because == ()


def test_an_unauthorized_operation_is_suppressed_with_its_reason():
    executed = run(
        execute_candidate(
            _answer(decision=_offer(), reply="here you go", wrote_its_own_reply=True),
            packet=PACKET,
            state={},
            authorized_operations=frozenset(),
            known_subjects=KNOWN,
        )
    )

    assert executed.executed is False
    assert executed.suppressed
    assert "not authorized" in executed.suppressed_because[0]


def test_an_invented_subject_is_suppressed():
    executed = run(
        execute_candidate(
            _answer(
                decision=_offer("a set that does not exist"),
                reply="here you go",
                wrote_its_own_reply=True,
            ),
            packet=PACKET,
            state={},
            authorized_operations=AUTHORIZED,
            known_subjects=KNOWN,
        )
    )

    assert executed.executed is False
    assert "does not exist" in executed.suppressed_because[0]


def test_the_constraints_apply_identically_to_both_candidates():
    """A candidate that could skip them would win by being allowed to do what
    the other was not."""
    for wrote_own in (True, False):

        async def write(*_args, **_kwargs):
            return "written"

        executed = run(
            execute_candidate(
                _answer(
                    decision=_offer("an invented set"),
                    reply="here you go" if wrote_own else "",
                    wrote_its_own_reply=wrote_own,
                ),
                packet=PACKET,
                state={},
                write=write,
                authorized_operations=AUTHORIZED,
                known_subjects=KNOWN,
            )
        )

        assert executed.executed is False, wrote_own


def test_a_hold_does_not_call_the_writer():
    called = []

    async def write(*_args, **_kwargs):
        called.append(True)
        return "written"

    executed = run(
        execute_candidate(
            _answer(decision=_decision(hold=HoldReason.RESPECT_SILENCE)),
            packet=PACKET,
            state={},
            write=write,
        )
    )

    assert called == []
    assert executed.said_nothing


def test_a_failed_writer_is_not_mistaken_for_restraint():
    """Collapsing them would make a broken writer look like judgment."""

    async def write(*_args, **_kwargs):
        raise RuntimeError("provider down")

    executed = run(
        execute_candidate(
            _answer(reply="", wrote_its_own_reply=False),
            packet=PACKET,
            state={},
            write=write,
        )
    )

    assert executed.said_nothing
    assert "writer failed" in executed.silent_because
    assert "RuntimeError" in executed.suppressed_because[0]


# ===========================================================================
# 3. Disagreements, by consequence
# ===========================================================================


def _executed(**overrides):
    return run(
        execute_candidate(
            _answer(
                decision=overrides.pop("decision", _decision()),
                reply=overrides.pop("reply", "hello"),
                wrote_its_own_reply=True,
            ),
            packet=PACKET,
            state={},
            **overrides,
        )
    )


def test_a_difference_in_what_happens_to_the_account_is_reported_first():
    left = _executed(
        decision=_offer(), authorized_operations=AUTHORIZED, known_subjects=KNOWN
    )
    right = _executed(decision=_decision())

    found = compare_executions(left, right)

    assert found.operation
    assert found.lines()[0].startswith("operation:")


def test_the_same_operation_refused_for_one_candidate_is_a_finding():
    """One was permitted and one refused on the same evidence."""
    permitted = _executed(
        decision=_offer(), authorized_operations=AUTHORIZED, known_subjects=KNOWN
    )
    refused = _executed(
        decision=_offer("an invented set"),
        authorized_operations=AUTHORIZED,
        known_subjects=KNOWN,
    )

    found = compare_executions(permitted, refused)

    # Same KIND of operation, so this is an execution difference rather than an
    # operation one.
    assert found.operation == ""
    assert "refused" in found.execution


def test_one_candidate_staying_quiet_is_reported():
    spoke = _executed(reply="hello")
    quiet = _executed(reply="", decision=_decision(hold=HoldReason.RESPECT_SILENCE))

    found = compare_executions(spoke, quiet)

    assert "said nothing" in found.silence


def test_two_different_replies_are_reported_as_wording_and_not_as_more():
    """A phrasing difference is interesting and is not the same kind of fact as
    an operation difference."""
    found = compare_executions(_executed(reply="hey there"), _executed(reply="hello"))

    assert found.wording
    assert found.operation == ""
    assert found.execution == ""


def test_identical_behaviour_is_no_disagreement():
    found = compare_executions(_executed(reply="hello"), _executed(reply="hello"))

    assert not found.any
    assert found.lines() == []


def test_whitespace_is_not_a_disagreement():
    found = compare_executions(_executed(reply="hello  there"), _executed(reply="Hello there"))

    assert found.wording == ""


# ===========================================================================
# 4. The whole comparison
# ===========================================================================


class _StubCandidate:
    def __init__(self, name, answer):
        self.name = name
        self._answer = answer

    async def answer(self, _packet, _state):
        return self._answer


class _Turn:
    name = "turn-1"
    state: dict = {}
    authorized_operations = AUTHORIZED
    known_subjects = KNOWN

    def packet(self):
        return PACKET


def test_the_report_says_what_each_candidate_would_have_done():
    report = run(
        compare_candidates(
            [_Turn()],
            [
                _StubCandidate(
                    "one_call",
                    _answer(
                        decision=_offer(),
                        reply="here you go",
                        wrote_its_own_reply=True,
                    ),
                ),
                _StubCandidate(
                    "two_call",
                    _answer(
                        decision=_decision(hold=HoldReason.WAITING_ON_CUSTOMER),
                        reply="",
                    ),
                ),
            ],
        )
    )

    summary = report.summary()

    assert summary["turns"] == 1
    assert summary["disagreed"] == 1
    assert summary["operations_executed"]["one_call"] == 1
    assert summary["operations_executed"]["two_call"] == 0
    assert summary["turns_silent"]["two_call"] == 1


def test_the_report_has_no_score_and_no_winner():
    """§5 rules one out, and the brief adds that the more elaborate candidate
    must not be selected by assumption."""
    report = run(
        compare_candidates([_Turn()], [_StubCandidate("a", _answer(reply="hi",
                                                                   wrote_its_own_reply=True))])
    )

    rendered = json.dumps(report.summary()) + report.render()

    for word in ("score", "winner", "better", "recommend"):
        assert word not in rendered.lower()


def test_the_adapter_leaves_the_writing_to_the_executor():
    """The architectural difference is WHEN the reply is written, so anything
    the adapter did beyond deferring it would be a third difference."""

    class _Owner:
        name = "two_call"

        async def decide(self, _packet, _state):
            return _decision()

    answer = run(DecideThenWrite(_Owner()).answer(PACKET, {}))

    assert answer.reply == ""
    assert answer.wrote_its_own_reply is False
