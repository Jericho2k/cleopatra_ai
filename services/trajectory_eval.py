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

from services.message_diagnostics import read_diagnostics
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
    #: True when the customer has asked for no further messages.
    #:
    #: What that forbids is UNPROMPTED contact — a proactive message, a queued
    #: follow-up, a scheduled nudge. Replying to a message he himself sent is
    #: not breaking a request for quiet; it is answering him. The detector used
    #: to flag every turn after the request, which made "he asked for quiet,
    #: came back a week later and asked a question, and got an answer" read as
    #: a critical failure.
    asks_for_silence: bool = False
    #: How long the request covers, in simulated days. 0 means "until lifted".
    #:
    #: §1 calls for judgment about silence, and judgment includes knowing when
    #: it has run out. A preference with no scope is a preference nothing can
    #: ever satisfy.
    silence_expires_after_days: float = 0.0
    #: True when this turn is the customer withdrawing an earlier request for
    #: quiet — "actually, message me whenever". A preference the customer set
    #: is a preference the customer can change.
    lifts_silence: bool = False
    #: What the customer corrected: the claim that is no longer true.
    corrects: str = ""
    #: What is true instead, when the trajectory says. A reply carrying BOTH
    #: the old phrase and the new one is discussing the correction, which is
    #: the opposite of reasserting it.
    corrected_to: str = ""
    #: Authoritative markers — media ids, set ids, references — that a delivery
    #: after this correction must not carry.
    #:
    #: This is the difference between a suspicion and a finding. A reply that
    #: MENTIONS a corrected topic is prose, and prose needs a person; a
    #: DELIVERY whose recorded contents contradict the correction is an
    #: executed action with a record behind it.
    forbids_delivery_of: tuple[str, ...] = ()
    responds_to: Callable[[list[str]], str] | None = None

    def next_message(self, creator_replies: list[str]) -> str:
        if self.responds_to is not None:
            return str(self.responds_to(list(creator_replies)) or "")
        return self.message


@dataclass(frozen=True)
class CoverageGap:
    """Something a trajectory claims to cover that the run did not establish.

    Deliberately not a ``Finding``. A finding is about the system's behaviour;
    this is about the evaluation's reach. "The conversation forced a commercial
    pivot" and "this run was seven turns long and the claim says forty" are
    different kinds of statement, and merging them would let a clean findings
    list read as a covered claim.

    This exists because the labels were not true. ``--describe`` printed
    "40-80 turns of ordinary conversation" above a seven-turn script, and
    "a question deferred across 30+ turns" above a five-turn one. Nothing
    compared the two, so the claim was the only thing a reader ever saw.
    """

    claim: str
    required: str
    actual: str

    def render(self) -> str:
        return f"[NOT COVERED] {self.claim}: needs {self.required}, run had {self.actual}"


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

    # --- what the claim above actually needs, to be checked against the run --
    #
    # Declared per trajectory rather than parsed out of `covers`, because prose
    # is not a specification and a regex over it would be a second thing that
    # can disagree with reality.

    #: Turns the claim needs. A "40-80 turn" row needs 40.
    requires_turns: int = 0
    #: Simulated days the claim needs to have elapsed.
    requires_elapsed_days: float = 0.0
    #: Whether the claim needs queued work to have actually become due and run.
    requires_due_worker: bool = False
    #: Whether the claim needs a purchase the ledger records, rather than a
    #: customer message asserting one. A complaint is not proof of payment, and
    #: a fixture that only says "i paid" tests the complaint handling and not
    #: the thing the row claims.
    requires_authoritative_purchase: bool = False

    #: Authoritative state to establish before the first turn — currently
    #: purchases the ledger should show. Applied by
    #: services/trajectory_fixtures.py, which refuses to write it against
    #: anything but a simulator test fan.
    seed: dict[str, Any] = field(default_factory=dict)


class TurnOutcome(str, Enum):
    """What actually happened on one turn, told apart from what it looks like.

    A turn that sent nothing is not one thing. Full Auto deciding not to write
    is the product working; the writer failing, the analyzer degrading, an
    inventory guard refusing and a handoff to a human are four different
    events; and a turn nothing recorded anything about is not a silence at all,
    it is an absence of evidence.

    Collapsing those into one "silent turns" count — which is what this harness
    did — means a deployment whose writer is failing on every turn reports the
    same number as one exercising judgment, and the count that is supposed to
    detect the first is the one hiding it.

    ``UNKNOWN`` is never a pass. A harness that cannot say what happened has
    not established the thing it exists to establish, so it says so.
    """

    #: Sent, and a delivery receipt proves it arrived.
    REPLIED = "replied"
    #: Sent, and nothing proves it arrived. Not a failure and not a success.
    DELIVERY_UNKNOWN = "delivery_unknown"
    #: Deliberately said nothing. The product exercising judgment.
    CHOSE_SILENCE = "chose_silence"
    #: Handed to a human. A cost, never a failure.
    HANDED_OFF = "handed_off"
    #: The analyzer could not read the situation.
    ANALYSIS_FAILED = "analysis_failed"
    #: The writer produced nothing usable, or its plan could not be recovered.
    WRITER_FAILED = "writer_failed"
    #: A commercial guard refused to send what was planned.
    INVENTORY_BLOCKED = "inventory_blocked"
    #: The turn raised. Infrastructure, not behaviour.
    INFRASTRUCTURE_ERROR = "infrastructure_error"
    #: Nothing recorded why. Missing evidence, reported as missing.
    UNKNOWN = "unknown"


#: The pipeline's own outcome strings (services/suggestions.py), mapped to what
#: they mean for an evaluation. Read from the vocabulary the product already
#: writes rather than re-derived, so a new outcome shows up as UNKNOWN here
#: instead of being quietly folded into silence.
_OUTCOME_MEANING: dict[str, "TurnOutcome"] = {
    "replied": TurnOutcome.REPLIED,
    "no_send": TurnOutcome.CHOSE_SILENCE,
    "human_review": TurnOutcome.HANDED_OFF,
    "analyzer_degraded": TurnOutcome.ANALYSIS_FAILED,
    "writer_failed": TurnOutcome.WRITER_FAILED,
    "plan_unrecoverable": TurnOutcome.WRITER_FAILED,
    "inventory_unsafe": TurnOutcome.INVENTORY_BLOCKED,
}

#: Outcomes that are not the system working as intended. Counted separately,
#: and never cancelled by anything a rubric says about the prose: §5 is
#: explicit that a quality score must not absolve an execution failure.
FAILED_OUTCOMES: frozenset["TurnOutcome"] = frozenset(
    {
        TurnOutcome.ANALYSIS_FAILED,
        TurnOutcome.WRITER_FAILED,
        TurnOutcome.INFRASTRUCTURE_ERROR,
        TurnOutcome.UNKNOWN,
    }
)


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

    @property
    def delivery_confirmed(self) -> bool:
        """Whether every reply this turn sent has a platform receipt.

        No provenance at all is not confirmation. It is the absence of the
        record that would confirm it, and the two must not read the same.
        """
        records = [record.get("delivery") or {} for record in self.provenance]
        if not records:
            return False
        return all(bool(record.get("accepted_by_platform")) for record in records)

    def classify(self) -> TurnOutcome:
        """What happened here, on the evidence this turn actually carries."""
        if self.error:
            return TurnOutcome.INFRASTRUCTURE_ERROR
        if self.sent_anything:
            return (
                TurnOutcome.REPLIED
                if self.delivery_confirmed
                else TurnOutcome.DELIVERY_UNKNOWN
            )
        recorded = str(self.outcome or "").strip()
        if not recorded:
            return TurnOutcome.UNKNOWN
        return _OUTCOME_MEANING.get(recorded, TurnOutcome.UNKNOWN)


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
    #: What this run did NOT establish, against what the trajectory claims.
    coverage_gaps: list[CoverageGap] = field(default_factory=list)
    #: Simulated days the run actually advanced. Zero when no clock was
    #: injected, which is not the same as a conversation that happened in one
    #: sitting — hence `clock_injected` below.
    elapsed_days: float = 0.0
    #: Whether a clock was supplied at all. Without one, a trajectory declaring
    #: "he comes back a week later" ran its turns back to back, and every
    #: expiry, schedule and continuity window saw one continuous session.
    clock_injected: bool = False
    #: How many turns ran queued work instead of delivering a customer message.
    due_worker_runs: int = 0
    #: Authoritative state established before the first turn. Reported, so a
    #: reader can see what the conversation started from rather than inferring
    #: it from the transcript.
    seeded_purchases: list[dict[str, Any]] = field(default_factory=list)

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

    def outcomes(self) -> dict[str, int]:
        """How many turns ended each way, by evidence rather than by appearance.

        This is what ``silent_turns`` cannot tell you. Two runs with the same
        silent count can be a system exercising judgment and a system whose
        writer is failing, and only this distinguishes them.
        """
        counts: dict[str, int] = {}
        for turn in self.turns:
            key = turn.classify().value
            counts[key] = counts.get(key, 0) + 1
        return counts

    @property
    def failed_turns(self) -> int:
        """Turns that did not work, including the ones nothing explains."""
        return sum(1 for turn in self.turns if turn.classify() in FAILED_OUTCOMES)

    @property
    def unexplained_turns(self) -> int:
        """Turns with no evidence of what happened. Never a pass."""
        return sum(
            1 for turn in self.turns if turn.classify() is TurnOutcome.UNKNOWN
        )

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
            # What the silent count cannot say, and the reason it is not the
            # headline number: a failure and a judgment call look identical to
            # it and are not the same event.
            "outcomes": self.outcomes(),
            "failed_turns": self.failed_turns,
            "unexplained_turns": self.unexplained_turns,
            "latency": self.latency(),
            "models_used": self.models_used(),
            # What this run does not establish, stated in the summary rather
            # than left for a reader to notice from the transcript's length.
            "coverage_gaps": [
                {"claim": gap.claim, "required": gap.required, "actual": gap.actual}
                for gap in self.coverage_gaps
            ],
            "fully_covered": not self.coverage_gaps,
            "elapsed_days": self.elapsed_days,
            "clock_injected": self.clock_injected,
            "due_worker_runs": self.due_worker_runs,
            "seeded_purchases": [
                {
                    "reference": row.get("reference"),
                    "price_cents": row.get("price_cents"),
                    "media_ids": row.get("media_ids"),
                }
                for row in self.seeded_purchases
            ],
        }

    def render(self) -> str:
        summary = self.summary()
        lines = [
            f"{self.trajectory} ({summary['turns']} turns) — {self.covers}",
            f"  critical failures: {summary['critical_failures']}",
            f"  notable behaviours: {summary['notable']}",
            f"  operator rescues:  {summary['operator_rescues']}",
            f"  silent turns:      {summary['silent_turns']}",
            f"  failed turns:      {summary['failed_turns']}"
            + (
                f" ({summary['unexplained_turns']} with no recorded reason)"
                if summary["unexplained_turns"]
                else ""
            ),
            f"  turn outcomes:     {summary['outcomes']}",
            f"  latency: median {summary['latency']['median_ms']}ms, "
            f"worst {summary['latency']['worst_ms']}ms",
            f"  models that answered: {summary['models_used'] or 'unrecorded'}",
        ]
        for finding in self.findings:
            lines.append(f"  {finding.render()}")
        # Last, and never omitted when present: a clean findings list above an
        # uncovered claim is the exact misreading this section exists to stop.
        for gap in self.coverage_gaps:
            lines.append(f"  {gap.render()}")
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


def _silence_window(
    disturbances: Sequence[Disturbance],
) -> tuple[int, int | None, float] | None:
    """When a request for quiet starts, when it ends, and how long it runs.

    Returns ``(requested_at, lifted_at, expires_after_days)`` or None. A
    preference the customer set is a preference the customer can change, so a
    later turn marked ``lifts_silence`` closes the window; a
    ``silence_expires_after_days`` closes it by elapsed simulated time.
    """
    requested_at: int | None = None
    expires_after = 0.0
    for index, disturbance in enumerate(disturbances):
        if disturbance.asks_for_silence and requested_at is None:
            requested_at = index
            expires_after = float(disturbance.silence_expires_after_days or 0.0)
            continue
        if requested_at is not None and disturbance.lifts_silence:
            return requested_at, index, expires_after
    if requested_at is None:
        return None
    return requested_at, None, expires_after


def _days_between(
    disturbances: Sequence[Disturbance], start: int, end: int
) -> float:
    """Simulated days elapsed between two turns, from the trajectory's clock."""
    return sum(
        float(disturbances[i].days_since_previous or 0.0)
        for i in range(start + 1, min(end + 1, len(disturbances)))
    )


def find_messages_after_silence_requested(
    turns: Sequence[TurnRecord], disturbances: Sequence[Disturbance]
) -> list[Finding]:
    """UNPROMPTED contact after the customer asked for no more messages.

    §1: "Use judgment about silence — respect a goodbye, an unanswered message,
    or a request for no follow-up."

    Unprompted is the whole content of the rule, and this detector used to
    ignore it: it flagged every turn after the request, including turns the
    customer started. "He asked for quiet, came back a week later and asked a
    question, and got an answer" was reported as a critical failure — which
    trains a reader to discount the detector, and would make honouring a
    request indistinguishable from breaking it.

    A turn is unprompted when it carries no customer message: a queued
    follow-up, a scheduled nudge, a proactive turn. Those are the ones the
    request forbids, and those are what this reports.

    The window closes when the customer lifts it or when the scope they gave it
    runs out. A preference with no expiry and no way to withdraw it is not a
    preference; it is a permanent state the customer cannot get out of.
    """
    findings: list[Finding] = []
    window = _silence_window(disturbances)
    if window is None:
        return findings
    requested_at, lifted_at, expires_after = window

    for turn in turns:
        if turn.index <= requested_at or not turn.sent_anything:
            continue
        if lifted_at is not None and turn.index >= lifted_at:
            continue
        if expires_after > 0.0:
            elapsed = _days_between(disturbances, requested_at, turn.index)
            if elapsed >= expires_after:
                continue
        if str(turn.customer_message or "").strip():
            # He wrote first. Answering him is not following up at him.
            continue
        findings.append(
            Finding(
                Severity.CRITICAL,
                "unprompted_message_after_silence_requested",
                f"he asked for no messages on turn {requested_at} and this turn "
                "sent without him writing first",
                turn.index,
            )
        )
    return findings


#: Words that, next to a corrected phrase, mean the reply is HONOURING the
#: correction rather than repeating what was corrected — "you don't like
#: outdoor shots", "no more outdoor stuff", "indoor instead of outdoor".
#:
#: Deliberately small. This is not an attempt to understand the sentence; it is
#: an attempt to stop the most common false positive from being reported with
#: certainty, and everything it does not catch still gets reported — as a
#: suspicion, for a person to read.
_HONOURING_MARKERS = (
    "not ",
    "no more ",
    "n't ",
    "never ",
    "dislike",
    "don't like",
    "do not like",
    "hate",
    "avoid",
    "instead of",
    "rather than",
    "away from",
    "other than",
    "except",
    "without",
    "remember you",
    "you said",
    "you told me",
)


def _reads_as_honouring(reply: str, phrase: str, corrected_to: str) -> bool:
    """Whether this reply is acknowledging the correction, not repeating it."""
    lowered = reply.lower()
    if corrected_to and corrected_to.lower() in lowered:
        # It names the correction as well as what was corrected, which is what
        # discussing a change of mind looks like.
        return True
    position = lowered.find(phrase)
    if position < 0:
        return False
    # Only the run-up to the phrase matters: "you don't like outdoor" honours
    # it, "here's an outdoor set, you don't like the indoor ones" does not.
    preceding = lowered[max(0, position - 40):position]
    return any(marker in preceding for marker in _HONOURING_MARKERS)


def find_reasserted_corrections(
    turns: Sequence[TurnRecord], disturbances: Sequence[Disturbance]
) -> list[Finding]:
    """A later turn acting on something the customer already corrected.

    §5's "Preference correction / Reasserting superseded information", split
    into the two things it actually is:

    **A delivery that contradicts the correction** is CRITICAL. The evidence is
    a delivery record — an executed action with contents recorded on it — so
    the finding does not rest on anyone's reading of a sentence.

    **A reply that mentions the corrected phrase** is NOTABLE, and named as a
    suspicion. It used to be CRITICAL on a bare substring match, which made
    "i remember you dislike outdoor photos, so here's an indoor set" — a reply
    that honours the correction perfectly — a critical failure for containing
    the word "outdoor". A keyword is not proof that an obsolete fact was
    reasserted, and reporting it as proof is how a detector stops being read.
    """
    findings: list[Finding] = []
    for index, disturbance in enumerate(disturbances):
        phrase = disturbance.corrects.strip().lower()
        forbidden = {str(marker) for marker in disturbance.forbids_delivery_of}

        for turn in turns:
            if turn.index <= index:
                continue

            # Authoritative first: what was actually delivered.
            for record in _delivery_records(turn):
                reference = str(record.get("reference") or "")
                if reference and reference in forbidden:
                    findings.append(
                        Finding(
                            Severity.CRITICAL,
                            "delivery_contradicts_correction",
                            f"delivered {reference!r} after he ruled it out on "
                            f"turn {index}",
                            turn.index,
                        )
                    )

            if not phrase:
                continue
            for reply in turn.replies:
                if phrase not in reply.lower():
                    continue
                if _reads_as_honouring(reply, phrase, disturbance.corrected_to):
                    continue
                findings.append(
                    Finding(
                        Severity.NOTABLE,
                        "correction_possibly_reasserted",
                        f"mentions {phrase!r} after he corrected it on turn "
                        f"{index}; read the reply to see which",
                        turn.index,
                    )
                )
    return findings


def find_unaddressed_obligations(
    turns: Sequence[TurnRecord], disturbances: Sequence[Disturbance]
) -> list[Finding]:
    """An obligation the trajectory raised that no reply answered.

    NOTABLE rather than CRITICAL, deliberately. Matching an obligation against
    prose is a keyword test, and a keyword test is not strong enough evidence to
    call something a failure with certainty — it is strong enough to put the
    turn in front of a person, which is what this harness is for.

    A reply on the turn that raises the obligation counts. It used to require a
    STRICTLY later turn, so "how was your trip?" answered in the same breath
    with "the trip was amazing, so much sun" was reported as never addressed —
    the single most natural place for an answer to be was the one place that
    could not contain one.
    """
    findings: list[Finding] = []
    for index, disturbance in enumerate(disturbances):
        obligation = disturbance.raises_obligation.strip().lower()
        if not obligation:
            continue
        answered = any(
            obligation in reply.lower()
            for turn in turns
            if turn.index >= index
            for reply in turn.replies
        )
        if not answered:
            findings.append(
                Finding(
                    Severity.NOTABLE,
                    "obligation_never_addressed",
                    f"nothing from turn {index} onwards mentions {obligation!r}",
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


def find_unexplained_turns(turns: Sequence[TurnRecord]) -> list[Finding]:
    """A turn that sent nothing and recorded no reason for it.

    CRITICAL, and the reasoning is the point of this whole harness: an
    evaluation exists to establish what a system did. A turn it cannot describe
    has not been evaluated, and counting it as a quiet turn would report the
    absence of evidence as evidence of good behaviour.

    So this is not "the system misbehaved" — it is "this run does not
    establish that it behaved", which is the finding that must not be
    silently absorbed into a silence count.
    """
    return [
        Finding(
            Severity.CRITICAL,
            "turn_outcome_unknown",
            "sent nothing and recorded no outcome; whether this was judgment "
            "or a failure is not established by this run",
            turn.index,
        )
        for turn in turns
        if turn.classify() is TurnOutcome.UNKNOWN
    ]


def find_unverified_deliveries(turns: Sequence[TurnRecord]) -> list[Finding]:
    """A turn that replied with nothing recording that the reply arrived.

    NOTABLE, not CRITICAL: the reply exists, so something happened. What is
    missing is the proof that it reached the customer, and a run assembling
    evidence about delivery has to say when it has none.

    Distinct from ``find_unreceipted_delivery_claims``, which is about a record
    that exists and says the platform did not accept. This is about there being
    no record at all.
    """
    findings: list[Finding] = []
    for turn in turns:
        if turn.classify() is not TurnOutcome.DELIVERY_UNKNOWN:
            continue
        if turn.provenance:
            # There is a record and it is unreceipted; that is the other
            # detector's finding, reported there with its own evidence.
            continue
        findings.append(
            Finding(
                Severity.NOTABLE,
                "delivery_unverified",
                "replied, but nothing recorded whether the platform took it",
                turn.index,
            )
        )
    return findings


#: Every detector, in the order their findings should be read.
DETECTORS: tuple[Callable[..., list[Finding]], ...] = (
    find_turn_errors,
    find_unexplained_turns,
    find_unverified_deliveries,
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
    report.clock_injected = advance_clock is not None
    creator_replies: list[str] = []

    for index, disturbance in enumerate(trajectory.disturbances):
        if disturbance.days_since_previous:
            if advance_clock is not None:
                # Simulated absence. §1 tests a return after one day and one
                # week, and no evaluation waits for either.
                try:
                    advance_clock(disturbance.days_since_previous)
                except Exception as exc:
                    # A clock that refuses (core.clock.ClockNotMovable in a
                    # process that may not simulate time) must not abort the
                    # conversation. The turns still run and the report says
                    # the time did not pass, which is the honest outcome and
                    # the one the coverage check reads.
                    report.clock_injected = False
                    report.findings.append(
                        Finding(
                            Severity.NOTABLE,
                            "clock_did_not_advance",
                            f"{type(exc).__name__}: {exc}",
                            index,
                        )
                    )
                else:
                    report.elapsed_days += float(disturbance.days_since_previous)
            # Without a clock, the turn still runs — and the report says the
            # week did not pass, rather than the trajectory's label implying
            # it did. `days_since_previous` was previously a no-op whenever
            # the caller supplied nothing, silently, and the CLI supplied
            # nothing.

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
        if (result or {}).get("due_worker_ran"):
            report.due_worker_runs += 1
        rows = (result or {}).get("creator_messages") or []
        # Provenance moved off the message row into the owner-only
        # message_diagnostics table (db/owner_only_diagnostics_v1.sql), because
        # media_context is a column agency browsers select directly. An
        # evaluation runs as the owner, so it reads the table — but only for
        # the rows that need it.
        #
        # A row written before the migration, or by a harness that builds the
        # result itself, still carries the record inline. Asking the database
        # about those spends a round trip per turn to be told what the row
        # already said, which over a 40-turn trajectory is 40 of them.
        inline = {
            str(row.get("id")): provenance_of(row.get("media_context"))
            for row in rows
        }
        unresolved = [
            str(row.get("id"))
            for row in rows
            if row.get("id") and not inline.get(str(row.get("id")))
        ]
        traces = await read_diagnostics(unresolved) if unresolved else {}
        for row in rows:
            content = str(row.get("content") or "")
            if content:
                record.replies.append(content)
                creator_replies.append(content)
            row_id = str(row.get("id"))
            trace = traces.get(row_id) or {}
            # Inline first, then the table. Mixed history is read completely
            # rather than half-attributed.
            provenance = inline.get(row_id) or (trace.get("record") or {}).get(
                PROVENANCE_KEY
            ) or {}
            if provenance:
                record.provenance.append(provenance)
        report.turns.append(record)

    # Extended, not replaced: a clock refusal recorded during the loop is a
    # finding about the run and must survive the detectors being run.
    report.findings.extend(evaluate_turns(report.turns, trajectory.disturbances))
    report.coverage_gaps = coverage_gaps(trajectory, report)
    return report


def coverage_gaps(
    trajectory: Trajectory, report: "TrajectoryReport"
) -> list[CoverageGap]:
    """What this run did not establish about what the trajectory claims.

    Compared against the RUN, not against the fixture. A fixture can declare
    forty turns and still produce seven if turns failed, and the claim is about
    what was exercised rather than what was written down.

    This is the direct answer to the labels being wrong. "40-80 turns of
    ordinary conversation" sat above a seven-turn script and "a question
    deferred across 30+ turns" above a five-turn one, and nothing anywhere
    compared the sentence to the thing. Reporting the gap does not make the
    fixture longer; it stops the report claiming it is.
    """
    gaps: list[CoverageGap] = []
    ran = len(report.turns)

    if trajectory.requires_turns and ran < trajectory.requires_turns:
        gaps.append(
            CoverageGap(
                claim="conversation length",
                required=f"{trajectory.requires_turns} turns",
                actual=f"{ran}",
            )
        )

    if trajectory.requires_elapsed_days:
        if not report.clock_injected:
            gaps.append(
                CoverageGap(
                    claim="elapsed time",
                    required=(
                        f"{trajectory.requires_elapsed_days:g} simulated days, "
                        "through expiry, scheduling and continuity"
                    ),
                    actual="no clock was injected; the turns ran back to back",
                )
            )
        elif report.elapsed_days < trajectory.requires_elapsed_days:
            gaps.append(
                CoverageGap(
                    claim="elapsed time",
                    required=f"{trajectory.requires_elapsed_days:g} simulated days",
                    actual=f"{report.elapsed_days:g}",
                )
            )

    if trajectory.requires_due_worker and report.due_worker_runs == 0:
        gaps.append(
            CoverageGap(
                claim="queued work becoming due",
                required="a due-worker cycle to actually run",
                actual="no cycle ran, so a queued follow-up could not fire",
            )
        )

    if trajectory.requires_authoritative_purchase:
        # Either the run seeded a purchase before it started, or a delivery it
        # made recorded a price. Both are the ledger saying money moved; a
        # customer message saying so is not.
        recorded = bool(report.seeded_purchases) or any(
            record.get("delivery", {}).get("price_cents")
            for turn in report.turns
            for record in turn.provenance
        )
        if not recorded:
            gaps.append(
                CoverageGap(
                    claim="a purchase the ledger records",
                    required="a seeded delivery the ledger shows as paid",
                    actual=(
                        "only the customer's own claim of payment, which is "
                        "what this row exists to distinguish from a purchase"
                    ),
                )
            )

    return gaps


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
                silence_expires_after_days=float(
                    item.get("silence_expires_after_days") or 0.0
                ),
                lifts_silence=bool(item.get("lifts_silence")),
                corrects=str(item.get("corrects") or ""),
                corrected_to=str(item.get("corrected_to") or ""),
                forbids_delivery_of=tuple(
                    str(value) for value in (item.get("forbids_delivery_of") or [])
                ),
            )
            for item in (raw.get("turns") or [])
            if isinstance(item, dict)
        )
        requires = raw.get("requires") or {}

        adaptive = raw.get("adaptive") or {}
        if adaptive:
            # A generated conversation, not a scripted one. One customer object
            # across every turn, so it carries what it has been told and can
            # react to it — a fresh one per turn would be a script again.
            from services.adaptive_customer import build as build_customer

            customer = build_customer(adaptive)
            length = max(1, int(adaptive.get("turns") or 40))
            gaps = {
                int(index): float(days)
                for index, days in (adaptive.get("gaps") or {}).items()
            }
            disturbances = tuple(
                Disturbance(
                    responds_to=customer.next_message,
                    days_since_previous=gaps.get(index, 0.0),
                    tests="adaptive",
                )
                for index in range(length)
            )
        trajectories.append(
            Trajectory(
                name=str(raw.get("name") or f"trajectory-{len(trajectories) + 1}"),
                disturbances=disturbances,
                covers=str(raw.get("covers") or ""),
                creator_id=str(raw.get("creator_id") or ""),
                fan_id=str(raw.get("fan_id") or ""),
                requires_turns=int(requires.get("turns") or 0),
                requires_elapsed_days=float(requires.get("elapsed_days") or 0.0),
                requires_due_worker=bool(requires.get("due_worker")),
                requires_authoritative_purchase=bool(
                    requires.get("authoritative_purchase")
                ),
                seed=dict(raw.get("seed") or {}),
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
