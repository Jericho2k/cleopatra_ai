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

**This does not add an authority.** On the selected replacement path, the single
semantic owner replaces the legacy behavioural controllers and its proposal is
checked by a deterministic validator before execution. A projection of the old
controllers (``services.decision_owners.current_stack_decision``) remains only
as the frozen comparison baseline; it is not called by the selected path.

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
    PRESENT_OFFER = "present_offer"
    SEND_LOCKED_PAID_MESSAGE = "send_locked_paid_message"
    CHECK_PAYMENT_CLAIM = "check_payment_claim"
    OFFER_CONTENT = "offer_content"
    DELIVER_PAID_CONTENT = "deliver_paid_content"
    REPAIR_CONTENT_ACCESS = "repair_content_access"
    HAND_OFF_TO_HUMAN = "hand_off_to_human"


class ResponseDisposition(str, Enum):
    """Whether this turn writes, intentionally stays quiet, or hands off."""

    REPLY = "reply"
    SILENCE = "silence"
    HANDOFF = "handoff"


class ResponseIntent(str, Enum):
    """Semantic purpose of the reply, without prescribing its wording."""

    ORDINARY_CONVERSATION = "ordinary_conversation"
    ANSWER_AND_CONTINUE = "answer_and_continue"
    CLARIFY_REFERENCE = "clarify_reference"
    PRESENT_OFFER = "present_offer"
    DELIVER_ACCEPTED_OFFER = "deliver_accepted_offer"
    ACKNOWLEDGE_PAYMENT_CHECK = "acknowledge_payment_check"
    SUPPORT_HANDOFF = "support_handoff"
    RESPECT_SILENCE = "respect_silence"


class HoldReason(str, Enum):
    """Why this turn does not simply reply."""

    NONE = "none"
    WAITING_ON_CUSTOMER = "waiting_on_customer"
    WAITING_ON_PAYMENT = "waiting_on_payment"
    NEEDS_HUMAN = "needs_human"
    RESPECT_SILENCE = "respect_silence"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


@dataclass(frozen=True)
class IntimacyContext:
    """Descriptive intimate/adult context for the current turn.

    This is intentionally not a stage machine. Every dimension may move in any
    direction on the next turn and none of them authorize escalation, media,
    price, or a sale.
    """

    active: bool = False
    content_register: str = "none"
    scene_mode: str = "none"
    direction: str = "continue"
    last_beat: str = ""
    boundaries: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "active": self.active,
            "content_register": self.content_register,
            "scene_mode": self.scene_mode,
            "direction": self.direction,
            "last_beat": self.last_beat,
            "boundaries": list(self.boundaries),
        }


#: What a scheduled conversational intention may be about. A short vocabulary
#: on purpose: it names the KIND of future obligation, never a stage in a
#: funnel, and application code reads it only to choose a deterministic timing
#: policy.
INTENT_KINDS = (
    "short_continuation",
    "scene_resume",
    "check_back",
    "commercial_callback",
    "payday_followup",
)

#: How a scheduled intention reacts to the fan speaking first.
CANCEL_ON_ACTIVITY = "cancel_on_activity"
REVALIDATE_ON_ACTIVITY = "revalidate_on_activity"
INTENT_ACTIVITY_POLICIES = (CANCEL_ON_ACTIVITY, REVALIDATE_ON_ACTIVITY)


@dataclass(frozen=True)
class ScheduledIntent:
    """A future conversational obligation, stated semantically and never written.

    "Wait right there" is a promise. Cleopatra could keep several specific ones
    — payday, post-session, abandoned offer — but had no way to make a general
    one, so the promise was simply dropped.

    What this object deliberately does NOT carry is the message. Freezing Kimi's
    wording hours in advance would send copy written against a conversation that
    has since moved, and it would put fan-facing prose back inside the decision
    role. It carries the GOAL; the words are written when it comes due, by the
    same GLM -> Kimi path any other turn uses.

    Timing is a REQUEST, not authority. Application code owns the normalized
    ``execute_at``: a relative delay is clamped to deterministic bounds, and a
    named reference such as payday resolves against evidence the application
    already parsed, never against a time a model invented.
    """

    kind: str = ""
    #: What the future turn should accomplish, semantically. Not a sentence to
    #: say, and never a price.
    goal: str = ""
    #: "relative" (a delay this conversation implies) or "reference" (an
    #: application-owned evidenced time such as payday).
    timing_kind: str = ""
    relative_seconds: int = 0
    reference: str = ""
    #: Message/event ids that make this obligation real.
    source_ids: tuple[str, ...] = ()
    activity_policy: str = CANCEL_ON_ACTIVITY

    @property
    def requested(self) -> bool:
        return bool(self.kind and self.goal and self.timing_kind)

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "goal": self.goal,
            "timing_kind": self.timing_kind,
            "relative_seconds": int(self.relative_seconds),
            "reference": self.reference,
            "source_ids": list(self.source_ids),
            "activity_policy": self.activity_policy,
        }


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
    #: Opaque, turn-local handle selected from application-supplied candidates.
    #: Only application code resolves it to inventory identity and price.
    candidate_handle: str = ""
    # Exact opaque references copied from the evidence snapshot.  The semantic
    # owner may choose among these values but may not create one.  Prices and
    # media identifiers remain absent: the executor resolves both from the
    # authoritative rows at execution time.
    offer_id: str = ""
    set_id: str = ""
    payment_reference: str = ""
    purchase_id: str = ""

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

    #: Semantic guidance for the writer. These fields describe what the turn
    #: must accomplish, never how a sentence should be phrased.
    response_goal: str = ""
    contribution_goal: str = ""
    relevant_thread_ids: tuple[str, ...] = ()
    initiative: str = "shared"
    pacing: str = "continue"

    #: Adult/intimate continuity is carried as independent descriptive
    #: dimensions, never as an escalation stage or scalar engagement score.
    intimacy_context: IntimacyContext = field(default_factory=IntimacyContext)

    #: Application-owned evidence categories the decision layer could not
    #: resolve from the supplied snapshot. The orchestrator may satisfy these
    #: once in the same fan turn and ask for a final decision.
    evidence_requests: tuple[str, ...] = ()

    #: Candidate facts proposed for the provenance-aware memory layer. They are
    #: still only proposals until deterministic validation accepts them.
    memory_candidates: tuple[dict[str, Any], ...] = ()

    #: The one external thing being asked for, if any.
    proposed_operation: ProposedOperation = field(default_factory=ProposedOperation)

    #: An optional future conversational obligation this turn wants to create.
    #: A request: application code normalizes its timing, validates it against
    #: evidence, and owns whether it is persisted at all.
    scheduled_intent: ScheduledIntent = field(default_factory=ScheduledIntent)

    #: What the turn means to do and whether it should produce customer text.
    #: Defaults preserve the offline comparison contract that predates the live
    #: migration; the selected live core requires both fields explicitly.
    response_intent: ResponseIntent = ResponseIntent.ORDINARY_CONVERSATION
    disposition: ResponseDisposition = ResponseDisposition.REPLY

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
        if self.disposition != other.disposition:
            differences.append(
                f"disposition: {self.disposition.value} vs {other.disposition.value}"
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
            "response_goal": self.response_goal,
            "contribution_goal": self.contribution_goal,
            "relevant_thread_ids": list(self.relevant_thread_ids),
            "initiative": self.initiative,
            "pacing": self.pacing,
            "intimacy_context": self.intimacy_context.as_dict(),
            "evidence_requests": list(self.evidence_requests),
            "memory_candidates": [dict(row) for row in self.memory_candidates],
            "response_intent": self.response_intent.value,
            "disposition": self.disposition.value,
            "operation": self.proposed_operation.kind.value,
            "operation_subject": self.proposed_operation.subject,
            "operation_because": self.proposed_operation.because,
            "operation_candidate_handle": self.proposed_operation.candidate_handle,
            "operation_offer_id": self.proposed_operation.offer_id,
            "operation_set_id": self.proposed_operation.set_id,
            "operation_payment_reference": self.proposed_operation.payment_reference,
            "operation_purchase_id": self.proposed_operation.purchase_id,
            "hold": self.hold.value,
            "hold_detail": self.hold_detail,
            "confidence": self.confidence,
            "scheduled_intent": self.scheduled_intent.as_dict(),
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
