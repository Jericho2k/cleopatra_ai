"""One executor, so two candidates are compared on what they actually do.

WHAT WAS MISSING
----------------
``services/decision_replay.py`` compared ``ConversationDecision`` objects. The
review's objection was that this is not a comparison of conversational cores:

    replay does not compare two complete new conversational cores through a
    shared writer/executor.

Two candidates agreeing that a turn should ``offer_content`` tells you almost
nothing. What reaches the customer is a sentence, and what happens to their
account is an operation that either ran or was refused — and those are the two
things a decision comparison cannot see.

WHAT THIS ADDS
--------------
Both candidates go through the same three steps, in the same order:

1. the reply is obtained — written by the candidate itself (candidate 1) or by
   the shared writer from the candidate's decision (candidate 2);
2. ``deterministic_violations`` runs, unchanged, on the decision;
3. the operation is executed or suppressed on that result alone.

Step 2 is the important one and it is deliberately not negotiable per
candidate. §4 says those checks always run "whatever produced the decision",
and a candidate that could skip them would win by being allowed to do things
the other was not.

NOTHING HERE SENDS ANYTHING
---------------------------
"Operation permitted in dry-run" means the deterministic constraints would
have permitted the operation. It does NOT mean an external operation ran. This
is an offline comparison: no platform call, no database write, no money. A
candidate that would have sent is compared against a candidate that would not
have, which is the whole question, and neither of them sends.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Sequence

from models.conversation_decision import (
    ConversationDecision,
    OperationKind,
    deterministic_violations,
)

#: Writes a reply from a decision and its evidence. The seam that lets
#: candidate 2 use the real writer, a stub, or a recorded response without the
#: choice leaking into the comparison.
Writer = Callable[[ConversationDecision, Any, dict[str, Any]], Awaitable[str]]


@dataclass(frozen=True)
class ExecutedTurn:
    """What one candidate would actually have done on one turn."""

    candidate: str
    decision: ConversationDecision
    #: The message that would have reached the customer. Empty when the
    #: candidate chose to say nothing, which is a legitimate answer.
    reply: str = ""
    #: The operation, and whether the deterministic constraints let it run.
    operation: OperationKind = OperationKind.NONE
    executed: bool = False
    #: Why it did not run. Non-empty only when the operation was suppressed.
    suppressed_because: tuple[str, ...] = ()
    #: True when the candidate wrote its own reply in the deciding call.
    wrote_its_own_reply: bool = False
    #: Why there is no reply, when there is none.
    silent_because: str = ""

    @property
    def said_nothing(self) -> bool:
        return not self.reply.strip()

    @property
    def suppressed(self) -> bool:
        return bool(self.suppressed_because)

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate,
            "reply": self.reply,
            "said_nothing": self.said_nothing,
            "silent_because": self.silent_because,
            "operation": self.operation.value,
            # The unambiguous public name. ``executed`` remains below for
            # backwards-compatible readers of the original replay format; it
            # has never represented a platform or database side effect.
            "operation_permitted_in_dry_run": self.executed,
            "executed": self.executed,
            "suppressed_because": list(self.suppressed_because),
            "wrote_its_own_reply": self.wrote_its_own_reply,
            "decision": self.decision.as_dict(),
        }


@dataclass
class TurnDisagreement:
    """Where two candidates would have done different things.

    Ordered by consequence: what happened to the customer's account first,
    whether anything was said second, and the wording last. A difference in
    phrasing between two replies that both offered the same thing is
    interesting; a difference in whether an offer was made is not the same
    kind of fact and must not be listed beside it as though it were.
    """

    left: str
    right: str
    operation: str = ""
    execution: str = ""
    silence: str = ""
    wording: str = ""

    @property
    def any(self) -> bool:
        return bool(self.operation or self.execution or self.silence or self.wording)

    def lines(self) -> list[str]:
        out = []
        for label in ("operation", "execution", "silence", "wording"):
            value = getattr(self, label)
            if value:
                out.append(f"{label}: {value}")
        return out


async def execute_candidate(
    answer: Any,
    *,
    packet: Any,
    state: dict[str, Any],
    write: Writer | None = None,
    authorized_operations: frozenset[OperationKind] = frozenset(),
    known_subjects: frozenset[str] = frozenset(),
) -> ExecutedTurn:
    """Run one candidate's answer through the shared executor.

    ``write`` is used only when the candidate did not write its own reply. That
    asymmetry IS the architectural difference being measured, so it lives here
    where it is visible rather than inside either candidate.
    """
    decision = answer.decision
    reply = str(getattr(answer, "reply", "") or "")
    wrote_own = bool(getattr(answer, "wrote_its_own_reply", False))
    silent_because = str(getattr(answer, "reason", "") or "")

    if not wrote_own and write is not None and not decision.is_hold:
        try:
            reply = str(await write(decision, packet, state) or "")
        except Exception as exc:
            # A writer that failed is not a candidate that chose silence, and
            # collapsing them would make a broken writer look like restraint.
            return ExecutedTurn(
                candidate=decision.source,
                decision=decision,
                reply="",
                operation=decision.proposed_operation.kind,
                executed=False,
                suppressed_because=(f"the writer failed: {type(exc).__name__}",),
                silent_because=f"writer failed: {type(exc).__name__}",
            )

    if not reply.strip() and not silent_because:
        silent_because = decision.hold_detail or decision.hold.value

    # The always-on half, identical for every candidate. A candidate that could
    # skip this would win by being allowed to do what the other was not.
    problems = deterministic_violations(
        decision,
        authorized_operations=authorized_operations,
        known_subjects=known_subjects,
    )
    operation = decision.proposed_operation.kind
    wants_operation = decision.proposed_operation.is_external
    executed = bool(wants_operation and not problems)

    return ExecutedTurn(
        candidate=decision.source,
        decision=decision,
        reply=reply,
        operation=operation,
        executed=executed,
        suppressed_because=tuple(problems) if wants_operation and problems else (),
        wrote_its_own_reply=wrote_own,
        silent_because=silent_because if not reply.strip() else "",
    )


def _normalise(text: str) -> str:
    return " ".join(str(text or "").split()).strip().lower()


def compare_executions(left: ExecutedTurn, right: ExecutedTurn) -> TurnDisagreement:
    """What two candidates would have done differently, by consequence."""
    disagreement = TurnDisagreement(left=left.candidate, right=right.candidate)

    if left.operation is not right.operation:
        disagreement.operation = (
            f"{left.candidate} would {left.operation.value}, "
            f"{right.candidate} would {right.operation.value}"
        )
    elif left.executed != right.executed:
        # Same operation, different outcome: one was permitted and one was
        # refused on the same evidence, which is a finding about the decision
        # rather than about the constraint.
        permitted, refused = (left, right) if left.executed else (right, left)
        disagreement.execution = (
            f"{permitted.candidate} would have run {permitted.operation.value}; "
            f"{refused.candidate} was refused: {'; '.join(refused.suppressed_because)}"
        )

    if left.said_nothing != right.said_nothing:
        spoke, quiet = (left, right) if right.said_nothing else (right, left)
        disagreement.silence = (
            f"{quiet.candidate} said nothing ({quiet.silent_because or 'no reason given'}); "
            f"{spoke.candidate} replied"
        )
    elif not left.said_nothing and _normalise(left.reply) != _normalise(right.reply):
        disagreement.wording = "the two replies differ"

    return disagreement


@dataclass
class ExecutionReport:
    """A whole comparison of executed turns. Evidence, not a verdict.

    No score, and no winner. §5 is explicit that a scripted customer and a
    self-grading model cannot substitute for expert review, and the brief adds
    that the more elaborate candidate must not be selected by assumption. What
    comes out is what each candidate would have said and done, and where they
    differed.
    """

    turns: list[dict[str, Any]] = field(default_factory=list)
    candidates: tuple[str, ...] = ()

    @property
    def disagreed(self) -> int:
        return sum(1 for turn in self.turns if turn["disagreements"])

    def summary(self) -> dict[str, Any]:
        executed: dict[str, int] = {name: 0 for name in self.candidates}
        suppressed: dict[str, int] = {name: 0 for name in self.candidates}
        silent: dict[str, int] = {name: 0 for name in self.candidates}
        for turn in self.turns:
            for name, record in turn["executions"].items():
                executed[name] = executed.get(name, 0) + int(record["executed"])
                suppressed[name] = suppressed.get(name, 0) + int(
                    bool(record["suppressed_because"])
                )
                silent[name] = silent.get(name, 0) + int(record["said_nothing"])
        return {
            "candidates": list(self.candidates),
            "turns": len(self.turns),
            "disagreed": self.disagreed,
            "operations_executed": executed,
            "operations_permitted_in_dry_run": executed,
            "operations_suppressed": suppressed,
            "turns_silent": silent,
        }

    def render(self) -> str:
        summary = self.summary()
        lines = [
            f"{' vs '.join(summary['candidates'])} over {summary['turns']} turns",
            f"  disagreed on: {summary['disagreed']}",
            "  operations permitted in dry-run:  "
            f"{summary['operations_permitted_in_dry_run']}",
            f"  operations suppressed: {summary['operations_suppressed']}",
            f"  turns with no reply:   {summary['turns_silent']}",
        ]
        for turn in self.turns:
            if not turn["disagreements"]:
                continue
            lines.append(f"  {turn['turn']}:")
            lines.extend(f"    {line}" for line in turn["disagreements"])
        return "\n".join(lines)


async def compare_candidates(
    turns: Sequence[Any],
    candidates: Sequence[Any],
    *,
    write: Writer | None = None,
) -> ExecutionReport:
    """Run every candidate over every turn, through one executor.

    Sequential per turn for the reason ``compare_owners`` gives: one candidate
    is a model call and interleaving would make its latency depend on the
    other's.
    """
    report = ExecutionReport(candidates=tuple(c.name for c in candidates))

    for turn in turns:
        packet = turn.packet()
        state = dict(turn.state)
        executions: dict[str, ExecutedTurn] = {}

        for candidate in candidates:
            answer = await candidate.answer(packet, state)
            executions[candidate.name] = await execute_candidate(
                answer,
                packet=packet,
                state=state,
                write=write,
                authorized_operations=turn.authorized_operations,
                known_subjects=turn.known_subjects,
            )

        names = list(executions)
        disagreements: list[str] = []
        for index, left in enumerate(names):
            for right in names[index + 1 :]:
                found = compare_executions(executions[left], executions[right])
                disagreements.extend(found.lines())

        report.turns.append(
            {
                "turn": turn.name,
                "executions": {
                    name: run.as_dict() for name, run in executions.items()
                },
                "disagreements": disagreements,
            }
        )

    return report
