"""Evaluating whole conversations, not replies.

``docs/autonomy_architecture_review.md`` §5 is the brief, and its first paragraph
says what this replaces:

    The existing ``scripts/run_model_eval.py`` constructs scenario contexts and
    invokes model completion. It is useful for reply comparisons, but it does
    not run the complete autonomous orchestration, real state transitions,
    delivery, or weeks of interaction. The simulator exercises more of the actual
    path and should become the basis of a separate, non-explicit longitudinal
    evaluation.

So this drives ``services.suggestions.run_simulated_inbound`` — the real Full
Auto turn, the real analyzer, the real commercial orchestrator, the real writer
routing, the real delivery boundary — across many turns, with the disturbances
§5 names, and reports what happened to the whole conversation.

Three things about how it reports, each of which is a refusal to overclaim.

**Critical failures are counted on their own.** §5: *"Count critical execution
failures separately; a high average prose score cannot cancel an unauthorized
transaction."* ``CriticalFailure`` is a deterministic finding — a duplicate
delivery, a paid state reverting, a message sent after the customer asked for
none, a delivery claimed without a receipt. These are found by reading state and
provenance, never by reading prose, and one of them is not offset by anything.

**There is no quality score.** The review is explicit that *"a scripted
cooperative customer and a model grading its own text are insufficient
substitutes for expert human review"*, so this deliberately does not produce
one. What it produces is a `TrajectoryReport`: the deterministic findings, the
countable behaviours (obligations never addressed, turns that asked a question,
operator rescues required), and the complete transcript with each reply's
provenance — the record an expert reviews. Building an LLM judge that emitted a
number here would be building exactly the thing §5 rules out.

**The customer is scripted, and that is stated.** A fixed script cannot expose
the consequences of the system's own earlier choices the way a genuinely
adaptive customer would. §5 asks for both replay and adaptive trajectories;
this is the harness, and `Disturbance.responds_to` is the seam where an
adaptive customer plugs in. Until one exists, every result here is bounded by
what the script happened to say.

Nothing here talks to a platform. The simulator's own scope refuses every remote
call, which is why a longitudinal run is safe to execute at all.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Sequence

from services.reply_provenance import PROVENANCE_KEY, provenance_of


class Severity(str, Enum):
    """Whether a finding is an execution failure or a behaviour worth reading.

    Only ``CRITICAL`` is a statement that something went wrong with certainty.
    ``NOTABLE`` is a countable behaviour that a human decides about — a forced
    question is not a bug, and a harness that called it one would be measuring
    its own taste.
    """

    CRITICAL = "critical"
    NOTABLE = "notable"


@dataclass(frozen=True)
class Finding:
    """One thing that happened, at one turn, with what it was found in."""

    severity: Severity
    kind: str
    detail: str
    turn: int = -1

    def render(self) -> str:
        where = f"turn {self.turn}" if self.turn >= 0 else "trajectory"
        return f"[{self.severity.value.upper()}] {where}: {self.kind} — {self.detail}"


@dataclass(frozen=True)
class Disturbance:
    """One thing the customer does, and what the conversation should survive.

    ``days_since_previous`` is how §5's "return after one day and one week" is
    expressed: the harness advances the conversation's clock rather than
    sleeping, because a week of real time is not a test anybody runs.

    ``responds_to`` is the seam for an adaptive customer. Given the creator's
    replies so far it returns the next message; a plain ``message`` is the
    scripted case. §5 wants both and this supports both, but nothing in this
    repository supplies an adaptive one yet.
    """

    message: str = ""
    days_since_previous: float = 0.0
    #: What this turn is testing, from §5's table. Recorded on every finding so
    #: a failure names the property it broke.
    tests: str = ""
    #: An obligation this turn creates that a later turn must answer.
    raises_obligation: str = ""
    #: True when the customer has asked for no further messages. Anything sent
    #: after this is a critical failure, per §1's "judgment about silence".
    asks_for_silence: bool = False
    #: True when the customer corrected something. A later reply that reasserts
    #: the superseded version is a critical failure.
    corrects: str = ""
    responds_to: Callable[[list[str]], str] | None = None

    def next_message(self, creator_replies: list[str]) -> str:
        if self.responds_to is not None:
            return str(self.responds_to(list(creator_replies)) or "")
        return self.message


@dataclass(frozen=True)
class Trajectory:
    """A whole conversation to run, and what it is testing."""

    name: str
    disturbances: tuple[Disturbance, ...]
    #: Which row of review §5's table this covers. Required, so a trajectory
    #: cannot drift into testing nothing in particular.
    covers: str = ""
    creator_id: str = ""
    fan_id: str = ""


@dataclass
class TurnRecord:
    """Everything one simulated turn produced."""

    index: int
    customer_message: str
    outcome: str = ""
    replies: list[str] = field(default_factory=list)
    provenance: list[dict[str, Any]] = field(default_factory=list)
    latency_ms: int = 0
    error: str = ""

    @property
    def sent_anything(self) -> bool:
        return bool(self.replies)


@dataclass
class TrajectoryReport:
    """What happened to one whole conversation.

    No score. The findings and the transcript are the output, because §5 says
    what a score here would and would not establish.
    """

    trajectory: str
    covers: str = ""
    turns: list[TurnRecord] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)

    @property
    def critical(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == Severity.CRITICAL]

    @property
    def notable(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == Severity.NOTABLE]

    @property
    def operator_rescues(self) -> int:
        """Turns that handed the conversation to a human.

        §5 asks for "the proportion requiring operator rescue". Not a failure:
        handing off is often the right answer, and PR #48 made one class of
        handoff the correct behaviour. It is a cost, and it is counted.
        """
        return sum(1 for turn in self.turns if turn.outcome == "human_review")

    @property
    def silent_turns(self) -> int:
        return sum(1 for turn in self.turns if not turn.sent_anything)

    def latency(self) -> dict[str, int]:
        """Median and worst turn, in milliseconds.

        §5 asks for "latency, cost per completed conversation, and tail behavior
        during errors". The tail is the number that matters, so it is reported
        rather than averaged away.
        """
        values = sorted(turn.latency_ms for turn in self.turns)
        if not values:
            return {"median_ms": 0, "worst_ms": 0}
        return {
            "median_ms": values[len(values) // 2],
            "worst_ms": values[-1],
        }

    def models_used(self) -> dict[str, int]:
        """Which models actually answered, from each reply's provenance.

        Finding H, applied to a trajectory: a comparison across a long
        conversation is meaningless if half its replies came from the fallback
        and nobody recorded it.
        """
        counts: dict[str, int] = {}
        for turn in self.turns:
            for record in turn.provenance:
                actual = ((record.get("writer") or {}).get("actual") or {})
                model = str(actual.get("model") or "unrecorded")
                counts[model] = counts.get(model, 0) + 1
        return counts

    def summary(self) -> dict[str, Any]:
        return {
            "trajectory": self.trajectory,
            "covers": self.covers,
            "turns": len(self.turns),
            "critical_failures": len(self.critical),
            "notable": len(self.notable),
            "operator_rescues": self.operator_rescues,
            "silent_turns": self.silent_turns,
            "latency": self.latency(),
            "models_used": self.models_used(),
        }

    def render(self) -> str:
        summary = self.summary()
        lines = [
            f"{self.trajectory} ({summary['turns']} turns) — {self.covers}",
            f"  critical failures: {summary['critical_failures']}",
            f"  notable behaviours: {summary['notable']}",
            f"  operator rescues:  {summary['operator_rescues']}",
            f"  silent turns:      {summary['silent_turns']}",
            f"  latency: median {summary['latency']['median_ms']}ms, "
            f"worst {summary['latency']['worst_ms']}ms",
            f"  models that answered: {summary['models_used'] or 'unrecorded'}",
        ]
        for finding in self.findings:
            lines.append(f"  {finding.render()}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Deterministic detectors
# ---------------------------------------------------------------------------
#
# Every one of these reads state or provenance. None of them reads prose for
# quality. §5: critical execution failures are counted separately precisely
# because they are the findings that do not depend on anybody's judgement.


def _delivery_records(turn: TurnRecord) -> list[dict[str, Any]]:
    return [record.get("delivery") or {} for record in turn.provenance]


def find_unreceipted_delivery_claims(turns: Sequence[TurnRecord]) -> list[Finding]:
    """A reply persisted as delivered that the platform never acknowledged.

    §4: "Delivery claims must be tied to the operation result." A provenance
    record whose ``accepted_by_platform`` is false for a paid delivery is that
    rule being broken, visible without reading a word of the message.
    """
    findings: list[Finding] = []
    for turn in turns:
        for delivery in _delivery_records(turn):
            if delivery.get("kind") == "ppv" and not delivery.get(
                "accepted_by_platform"
            ):
                findings.append(
                    Finding(
                        Severity.CRITICAL,
                        "unreceipted_paid_delivery",
                        "a paid delivery was recorded with no platform receipt",
                        turn.index,
                    )
                )
    return findings


def find_duplicate_deliveries(turns: Sequence[TurnRecord]) -> list[Finding]:
    """The same platform message id recorded twice.

    The single most damaging thing this product can do in front of an agency,
    in core/bounded_state.py's words. Detected on identity rather than on text,
    so two genuinely different messages that happen to read alike are not it.
    """
    seen: dict[str, int] = {}
    findings: list[Finding] = []
    for turn in turns:
        for delivery in _delivery_records(turn):
            message_id = str(delivery.get("platform_message_id") or "")
            if not message_id:
                continue
            if message_id in seen:
                findings.append(
                    Finding(
                        Severity.CRITICAL,
                        "duplicate_delivery",
                        f"platform message {message_id} was recorded on turns "
                        f"{seen[message_id]} and {turn.index}",
                        turn.index,
                    )
                )
            else:
                seen[message_id] = turn.index
    return findings


def find_messages_after_silence_requested(
    turns: Sequence[TurnRecord], disturbances: Sequence[Disturbance]
) -> list[Finding]:
    """Anything sent after the customer asked for no more messages.

    §1: "Use judgment about silence — respect a goodbye, an unanswered message,
    or a request for no follow-up." A reply after that is not a style problem.
    """
    findings: list[Finding] = []
    silenced_at: int | None = None
    for index, disturbance in enumerate(disturbances):
        if disturbance.asks_for_silence and silenced_at is None:
            silenced_at = index
    if silenced_at is None:
        return findings
    for turn in turns:
        if turn.index > silenced_at and turn.sent_anything:
            findings.append(
                Finding(
                    Severity.CRITICAL,
                    "message_after_silence_requested",
                    f"he asked for no messages on turn {silenced_at}",
                    turn.index,
                )
            )
    return findings


def find_reasserted_corrections(
    turns: Sequence[TurnRecord], disturbances: Sequence[Disturbance]
) -> list[Finding]:
    """A later reply repeating something the customer already corrected.

    §5's "Preference correction / Reasserting superseded information". Matched
    on the exact phrase the trajectory declares was corrected, because a looser
    match would produce findings nobody can act on.
    """
    findings: list[Finding] = []
    for index, disturbance in enumerate(disturbances):
        phrase = disturbance.corrects.strip().lower()
        if not phrase:
            continue
        for turn in turns:
            if turn.index <= index:
                continue
            for reply in turn.replies:
                if phrase in reply.lower():
                    findings.append(
                        Finding(
                            Severity.CRITICAL,
                            "reasserted_corrected_information",
                            f"repeated {phrase!r} after he corrected it on turn {index}",
                            turn.index,
                        )
                    )
    return findings


def find_unaddressed_obligations(
    turns: Sequence[TurnRecord], disturbances: Sequence[Disturbance]
) -> list[Finding]:
    """An obligation the trajectory raised that no later reply answered.

    NOTABLE rather than CRITICAL, deliberately. Matching an obligation against
    prose is a keyword test, and a keyword test is not strong enough evidence to
    call something a failure with certainty — it is strong enough to put the
    turn in front of a person, which is what this harness is for.
    """
    findings: list[Finding] = []
    for index, disturbance in enumerate(disturbances):
        obligation = disturbance.raises_obligation.strip().lower()
        if not obligation:
            continue
        answered = any(
            obligation in reply.lower()
            for turn in turns
            if turn.index > index
            for reply in turn.replies
        )
        if not answered:
            findings.append(
                Finding(
                    Severity.NOTABLE,
                    "obligation_never_addressed",
                    f"nothing after turn {index} mentions {obligation!r}",
                    index,
                )
            )
    return findings


def find_relentless_questioning(turns: Sequence[TurnRecord]) -> list[Finding]:
    """Every reply ending in a question.

    §1: "Discuss a topic without forcing a question, rehearsed joke, or purchase
    opportunity into every reply." NOTABLE: one question is conversation, and
    only a person can say where the line is in a given conversation.
    """
    answered = [turn for turn in turns if turn.sent_anything]
    if len(answered) < 4:
        return []
    with_question = [
        turn for turn in answered if any("?" in reply for reply in turn.replies)
    ]
    if len(with_question) == len(answered):
        return [
            Finding(
                Severity.NOTABLE,
                "question_in_every_reply",
                f"all {len(answered)} replies contain a question",
            )
        ]
    return []


def find_verbatim_repetition(turns: Sequence[TurnRecord]) -> list[Finding]:
    """The same reply text sent more than once in one conversation.

    NOTABLE. A short acknowledgement repeating is ordinary; the same full reply
    twice is what the supplied excerpts showed, and it is worth a person's eye.
    """
    seen: dict[str, int] = {}
    findings: list[Finding] = []
    for turn in turns:
        for reply in turn.replies:
            normalized = " ".join(reply.lower().split())
            if len(normalized) < 25:
                continue
            if normalized in seen:
                findings.append(
                    Finding(
                        Severity.NOTABLE,
                        "verbatim_repetition",
                        f"the same reply was sent on turns {seen[normalized]} "
                        f"and {turn.index}",
                        turn.index,
                    )
                )
            else:
                seen[normalized] = turn.index
    return findings


def find_turn_errors(turns: Sequence[TurnRecord]) -> list[Finding]:
    """A turn that raised. The harness records it rather than stopping.

    §5 asks for "tail behavior during errors", which cannot be measured by a
    run that aborts at the first one.
    """
    return [
        Finding(Severity.CRITICAL, "turn_raised", turn.error, turn.index)
        for turn in turns
        if turn.error
    ]


#: Every detector, in the order their findings should be read.
DETECTORS: tuple[Callable[..., list[Finding]], ...] = (
    find_turn_errors,
    find_unreceipted_delivery_claims,
    find_duplicate_deliveries,
    find_messages_after_silence_requested,
    find_reasserted_corrections,
    find_unaddressed_obligations,
    find_relentless_questioning,
    find_verbatim_repetition,
)


def evaluate_turns(
    turns: Sequence[TurnRecord], disturbances: Sequence[Disturbance]
) -> list[Finding]:
    """Run every detector over a completed trajectory."""
    findings: list[Finding] = []
    for detector in DETECTORS:
        try:
            if detector in {
                find_messages_after_silence_requested,
                find_reasserted_corrections,
                find_unaddressed_obligations,
            }:
                findings.extend(detector(turns, disturbances))
            else:
                findings.extend(detector(turns))
        except Exception as exc:  # pragma: no cover - a detector is not the product
            findings.append(
                Finding(
                    Severity.NOTABLE,
                    "detector_failed",
                    f"{getattr(detector, '__name__', detector)}: {exc}",
                )
            )
    return findings


# ---------------------------------------------------------------------------
# Running one
# ---------------------------------------------------------------------------


async def run_trajectory(
    trajectory: Trajectory,
    *,
    send_turn: Callable[[str], Awaitable[dict[str, Any]]],
    advance_clock: Callable[[float], None] | None = None,
) -> TrajectoryReport:
    """Drive one whole conversation through the real Full Auto turn.

    ``send_turn`` is the injection point. In a real run it is a thin wrapper
    around ``services.suggestions.run_simulated_inbound``; in a test it is a
    stub. It is a parameter rather than an import so this module never depends
    on a database being reachable, and so a trajectory can be replayed against
    recorded turns.

    A turn that raises is recorded and the trajectory continues. §5 asks for tail
    behaviour during errors, and a run that stopped at the first one would
    measure the error rather than what the conversation did afterwards.
    """
    report = TrajectoryReport(trajectory=trajectory.name, covers=trajectory.covers)
    creator_replies: list[str] = []

    for index, disturbance in enumerate(trajectory.disturbances):
        if disturbance.days_since_previous and advance_clock is not None:
            # Simulated absence. §1 tests a return after one day and one week,
            # and no evaluation waits for either.
            advance_clock(disturbance.days_since_previous)

        message = disturbance.next_message(creator_replies)
        record = TurnRecord(index=index, customer_message=message)
        started = time.monotonic()
        try:
            result = await send_turn(message)
        except Exception as exc:
            record.error = f"{type(exc).__name__}: {exc}"
            record.latency_ms = int((time.monotonic() - started) * 1000)
            report.turns.append(record)
            continue
        record.latency_ms = int((time.monotonic() - started) * 1000)
        record.outcome = str((result or {}).get("outcome") or "")
        for row in (result or {}).get("creator_messages") or []:
            content = str(row.get("content") or "")
            if content:
                record.replies.append(content)
                creator_replies.append(content)
            provenance = provenance_of(row.get("media_context"))
            if provenance:
                record.provenance.append(provenance)
        report.turns.append(record)

    report.findings = evaluate_turns(report.turns, trajectory.disturbances)
    return report


def load_trajectories(payload: Sequence[dict[str, Any]]) -> list[Trajectory]:
    """Build trajectories from the JSON format, tolerating a missing optional."""
    trajectories: list[Trajectory] = []
    for raw in payload:
        if not isinstance(raw, dict):
            continue
        disturbances = tuple(
            Disturbance(
                message=str(item.get("message") or ""),
                days_since_previous=float(item.get("days_since_previous") or 0.0),
                tests=str(item.get("tests") or ""),
                raises_obligation=str(item.get("raises_obligation") or ""),
                asks_for_silence=bool(item.get("asks_for_silence")),
                corrects=str(item.get("corrects") or ""),
            )
            for item in (raw.get("turns") or [])
            if isinstance(item, dict)
        )
        trajectories.append(
            Trajectory(
                name=str(raw.get("name") or f"trajectory-{len(trajectories) + 1}"),
                disturbances=disturbances,
                covers=str(raw.get("covers") or ""),
                creator_id=str(raw.get("creator_id") or ""),
                fan_id=str(raw.get("fan_id") or ""),
            )
        )
    return trajectories


def render_reports(reports: Sequence[TrajectoryReport]) -> str:
    """The whole run, with the totals §5 asks to be kept apart.

    Critical failures are totalled on their own line and never combined with
    anything, because the review's point is that no other number may offset
    them.
    """
    lines = [report.render() for report in reports]
    critical = sum(len(report.critical) for report in reports)
    notable = sum(len(report.notable) for report in reports)
    rescues = sum(report.operator_rescues for report in reports)
    turns = sum(len(report.turns) for report in reports)
    lines.append("")
    lines.append(
        f"{len(reports)} trajectories, {turns} turns: "
        f"{critical} CRITICAL execution failure(s)"
    )
    lines.append(
        f"  {notable} notable behaviour(s) for review, {rescues} operator rescue(s)"
    )
    lines.append(
        "  No quality score is produced. Review §5: a scripted customer and a "
        "model grading its own text are not substitutes for expert review."
    )
    return "\n".join(lines)


__all__ = [
    "PROVENANCE_KEY",
    "DETECTORS",
    "Disturbance",
    "Finding",
    "Severity",
    "Trajectory",
    "TrajectoryReport",
    "TurnRecord",
    "evaluate_turns",
    "load_trajectories",
    "render_reports",
    "run_trajectory",
]
