"""One statement of what a turn should do, whoever decided it.

``docs/autonomy_architecture_review.md`` finding F, confirmed:

    The current path combines a stage classifier, situation analysis, commercial
    policy, conversation director, session strategy, experience director,
    expression guidance, and writer instructions. [...] The problem is not the
    number of files. It is that several representations can prescribe what a
    single conversation should do next.

Four of them prescribe a next move in their own vocabulary, today:
``CommercialDecision.action`` (``ActionType``), ``ConversationDirectorState``
(``DirectorAction`` plus a phase), ``SessionStrategy`` (``NextBestAction``, a
goal and a ``writer_goal`` sentence), and the analyzer's ``strategic_move``.
Each is reasonable alone. Together nothing in the system states what the turn is
actually doing, so nothing can compare two ways of deciding it.

``ConversationDecision`` is that statement. §4 specifies its contents exactly:

    The decision object should identify the customer's active needs, the
    messages supporting that interpretation, unresolved references, which
    questions the reply must address, any proposed operation, and why waiting or
    handing off is needed. It should not prescribe a mandatory emotional ladder
    or a fixed sentence shape.

The second sentence is a constraint on the type, not a note. There is no field
here for a phase, a tone ladder, a bubble count or a sentence template — and
``forbidden_fields`` exists so that a future owner adding one fails a test
rather than quietly reintroducing what this replaces.

**This does not add an authority.** The review warns that "another planner added
on top would introduce another authority unless ownership is explicitly
simplified", and that removal must be tested under replay rather than assumed.
So the first thing built on this interface is a *projection* of the controllers
that already exist (``services/decision_owners.current_stack_decision``), which
changes no behaviour and makes today's decision inspectable. A single semantic
owner is a second implementation of the same interface, compared against the
first offline with the executor held fixed. Which one ships is a question for
evidence, and this type is what makes the comparison possible.

**Neither owner may authorize anything.** ``ProposedOperation`` is a proposal in
its name and in its semantics: §4 is explicit that "Neither model can authorize
a charge or declare a tool succeeded", so an operation carries what is being
asked for and nothing that could be mistaken for permission or for a result.
``deterministic_violations`` is the check that holds it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

#: Field names a decision may never carry. Finding F's failure mode is a new
#: representation acquiring the prescriptive vocabulary of the old ones, and the
#: only defence that survives a refactor is a test.
FORBIDDEN_FIELDS = (
    "phase",
    "tone",
    "emotional_ladder",
    "sentence_shape",
    "max_messages",
    "target_bubbles",
    "writer_goal",
    "script",
)


class OperationKind(str, Enum):
    """What a decision may ask the executor to consider doing.

    Deliberately short. These are operations with external effects, which is the
    only category where "the model proposed it" and "the system did it" must be
    kept apart. Ordinary conversation proposes nothing.
    """

    NONE = "none"
    OFFER_CONTENT = "offer_content"
    DELIVER_PAID_CONTENT = "deliver_paid_content"
    REPAIR_CONTENT_ACCESS = "repair_content_access"
    HAND_OFF_TO_HUMAN = "hand_off_to_human"


class HoldReason(str, Enum):
    """Why this turn does not simply reply."""

    NONE = "none"
    WAITING_ON_CUSTOMER = "waiting_on_customer"
    WAITING_ON_PAYMENT = "waiting_on_payment"
    NEEDS_HUMAN = "needs_human"
    RESPECT_SILENCE = "respect_silence"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


@dataclass(frozen=True)
class ProposedOperation:
    """Something with an external effect that this turn is asking for.

    A proposal, never an authorization. It names the kind and the exact subject,
    and carries no price, no media id and no success flag — deterministic code
    resolves those from inventory and the ledger, which is where §4 puts money,
    entitlements and idempotency. A decision that could carry a price would be a
    decision that could invent one.
    """

    kind: OperationKind = OperationKind.NONE
    #: What the operation is about, in the conversation's own words — "the set
    #: he asked about", not an identifier. Resolving it is the executor's job.
    subject: str = ""
    #: Why this turn is proposing it.
    because: str = ""

    @property
    def is_external(self) -> bool:
        return self.kind != OperationKind.NONE


@dataclass(frozen=True)
class ConversationDecision:
    """What this turn is doing, and the evidence it rests on.

    Every field is either a reading of the conversation or a reference back to
    what supports that reading. Nothing here says how to write.
    """

    #: What the customer actually wants right now. More than one is normal: §1
    #: asks for "mixed-intent turns with competing priorities", and a decision
    #: that can only hold one need is how the second gets dropped.
    active_needs: tuple[str, ...] = ()

    #: Fingerprints of the messages that support the reading above. Not the
    #: text — services/reply_provenance.py's discipline — so a decision can be
    #: audited against the conversation without copying it.
    supporting_messages: tuple[str, ...] = ()

    #: Things referred to that this turn cannot resolve: "the one from before",
    #: "that other set". §1 lists "ambiguous pronouns" and "wrong referent" as
    #: things a test must catch, and they are only catchable if a decision can
    #: say it did not know.
    unresolved_references: tuple[str, ...] = ()

    #: Questions this reply has to answer. The obligations from
    #: services/conversation_continuity.py land here.
    must_address: tuple[str, ...] = ()

    #: The one external thing being asked for, if any.
    proposed_operation: ProposedOperation = field(default_factory=ProposedOperation)

    #: Why this turn waits or hands over instead of answering.
    hold: HoldReason = HoldReason.NONE
    hold_detail: str = ""

    #: Which owner produced this. The whole point of the replay comparison.
    source: str = ""
    #: How confident the owner is in its reading. A low number is not a reason
    #: to act differently by itself; it is what makes disagreement between two
    #: owners interpretable.
    confidence: float = 1.0

    @property
    def is_hold(self) -> bool:
        return self.hold != HoldReason.NONE

    def disagreement_with(self, other: "ConversationDecision") -> list[str]:
        """What two owners decided differently about the same evidence.

        The unit of the replay comparison. Ordered so the most consequential
        difference reads first: what the turn DOES diverging matters more than
        how the two described the customer's mood.
        """
        differences: list[str] = []
        if self.proposed_operation.kind != other.proposed_operation.kind:
            differences.append(
                f"operation: {self.proposed_operation.kind.value} vs "
                f"{other.proposed_operation.kind.value}"
            )
        if self.hold != other.hold:
            differences.append(f"hold: {self.hold.value} vs {other.hold.value}")
        missed = set(other.must_address) - set(self.must_address)
        if missed:
            differences.append(
                f"{self.source or 'a'} would not address: {', '.join(sorted(missed))}"
            )
        extra = set(self.must_address) - set(other.must_address)
        if extra:
            differences.append(
                f"{other.source or 'b'} would not address: {', '.join(sorted(extra))}"
            )
        if set(self.active_needs) != set(other.active_needs):
            differences.append(
                f"needs: {sorted(self.active_needs)} vs {sorted(other.active_needs)}"
            )
        return differences

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "active_needs": list(self.active_needs),
            "supporting_messages": list(self.supporting_messages),
            "unresolved_references": list(self.unresolved_references),
            "must_address": list(self.must_address),
            "operation": self.proposed_operation.kind.value,
            "operation_subject": self.proposed_operation.subject,
            "operation_because": self.proposed_operation.because,
            "hold": self.hold.value,
            "hold_detail": self.hold_detail,
            "confidence": self.confidence,
        }


def deterministic_violations(
    decision: ConversationDecision,
    *,
    authorized_operations: frozenset[OperationKind] = frozenset(),
    known_subjects: frozenset[str] = frozenset(),
) -> list[str]:
    """The checks §4 says must always run, whatever produced the decision.

        Always run deterministic checks for unauthorized operations, stale
        state, invented identifiers/prices, duplicate sends, and unsupported
        success claims.

    This is the always-on half. It takes what deterministic code has already
    established — which operations this turn is permitted, and which subjects
    actually exist — and reports what the decision claims beyond it. It does not
    judge prose, and it has no notion of a good reply: everything here is a
    statement about permission or about evidence.

    A violation is a refusal, not a warning. A caller that gets a non-empty list
    must not execute the operation.
    """
    problems: list[str] = []
    operation = decision.proposed_operation

    if operation.is_external and operation.kind not in authorized_operations:
        problems.append(
            f"proposed {operation.kind.value} which this turn is not authorized to do"
        )

    if operation.is_external and not operation.subject:
        problems.append(
            f"proposed {operation.kind.value} without saying what it is about"
        )

    if (
        operation.is_external
        and known_subjects
        and operation.subject
        and operation.subject not in known_subjects
    ):
        problems.append(
            f"proposed {operation.kind.value} about {operation.subject!r}, "
            "which is not something this turn knows to exist"
        )

    # An invented price. A decision may say what it wants to happen; it may
    # never say what it costs, because it has no way to know and every way to
    # guess. Checked on the text because that is where one would appear.
    for text in (operation.subject, operation.because):
        if "$" in str(text):
            problems.append(
                f"named a price in {text!r}; price comes from inventory, not from a decision"
            )

    # An unsupported claim about something already having happened. §4: a
    # delivery claim must be tied to the operation result.
    claims = ("already sent", "has been sent", "i sent", "payment went through",
              "it worked", "delivered it")
    lowered = f"{operation.because} {decision.hold_detail}".lower()
    for claim in claims:
        if claim in lowered:
            problems.append(
                f"claimed {claim!r} happened; only a receipt can establish that"
            )

    if decision.is_hold and not decision.hold_detail:
        problems.append(f"held as {decision.hold.value} without saying why")

    if not 0.0 <= decision.confidence <= 1.0:
        problems.append(f"confidence {decision.confidence} is not a probability")

    return problems
