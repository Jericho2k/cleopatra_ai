"""Two ways of deciding what a turn does, behind one interface.

``docs/autonomy_architecture_review.md`` §6.4:

    **Compare replacement conversational cores offline.** Keep the executor
    fixed; compare the current controller stack with a single semantic decision
    owner. Change model routing separately so results remain attributable.

and §4, on which candidates are worth comparing:

    Two viable candidates should be compared: one model call returning an
    ordinary reply and typed intent, versus a separate semantic planner followed
    by a writer. The second costs more and adds another failure point. Select it
    only if complete conversation evaluation establishes a benefit.

Both owners here implement ``DecisionOwner`` and neither is wired into the live
path. That is the point of the sprint: the review warns that adding a planner on
top would introduce another authority, and says the effect of removing
duplicated guidance must be *tested under replay*, not assumed. So:

``current_stack_decision``
    A pure projection of what the existing controllers already decided. No model
    call, no new judgement, nothing invented. Running it changes nothing; what
    it produces is the first statement this system has ever had of what a turn
    is actually doing, in one object, which is what makes the second candidate
    comparable at all.

``SemanticDecisionOwner``
    One model call that reads the same context packet and returns the same typed
    object. Offline only. It is the candidate, not a replacement — and it is
    built so that comparing it to the projection compares two ways of deciding
    and nothing else: same evidence, same executor, same routing.

Neither may authorize anything. Whatever either returns goes through
``models.conversation_decision.deterministic_violations`` before an executor
would look at it, and an operation it is not permitted is refused there rather
than negotiated here.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Protocol

from models.conversation_decision import (
    ConversationDecision,
    HoldReason,
    OperationKind,
    ProposedOperation,
)
from services.context_packet import ContextPacket

SOURCE_CURRENT_STACK = "current_stack"
SOURCE_SEMANTIC = "semantic_owner"


class DecisionOwner(Protocol):
    """Anything that can say what a turn should do, given the same evidence."""

    name: str

    async def decide(
        self, packet: ContextPacket, state: dict[str, Any]
    ) -> ConversationDecision:
        ...


# ---------------------------------------------------------------------------
# Candidate 1 — what the current controllers already decided
# ---------------------------------------------------------------------------

#: How each commercial action reads as an operation. Anything absent proposes
#: nothing: most of what this system does in a conversation is talk, and only
#: the moves with an external effect belong in an operation field.
_COMMERCIAL_OPERATIONS: dict[str, OperationKind] = {
    "OFFER_NEXT_UNLOCK": OperationKind.OFFER_CONTENT,
    "RESUME_PREVIOUS_OFFER": OperationKind.OFFER_CONTENT,
    "PAYDAY_REENGAGEMENT": OperationKind.OFFER_CONTENT,
    "SEND_NEXT_PPV_STEP": OperationKind.DELIVER_PAID_CONTENT,
}

#: Which commercial actions are the system declining to move, and why.
_COMMERCIAL_HOLDS: dict[str, tuple[HoldReason, str]] = {
    "PAUSE_NO_BUDGET": (
        HoldReason.WAITING_ON_CUSTOMER,
        "he has said he cannot spend right now",
    ),
    "PAUSE_UNTIL_PAYDAY": (
        HoldReason.WAITING_ON_CUSTOMER,
        "he has said he will have money later",
    ),
}

#: What the analyzer's purchase signal says he is here for. The analyzer's
#: vocabulary, translated once, rather than re-derived by each reader.
_PURCHASE_NEEDS: dict[str, str] = {
    "ready_to_buy": "he wants to buy something now",
    "money_available": "he has money and is deciding",
    "interested": "he is interested but not committed",
    "no_money": "he cannot spend right now",
    "declined": "he has said no to what was offered",
}


def _text(value: Any) -> str:
    return "" if value is None else str(getattr(value, "value", value)).strip()


def current_stack_decision(
    packet: ContextPacket,
    state: dict[str, Any],
    *,
    source: str = SOURCE_CURRENT_STACK,
) -> ConversationDecision:
    """State what the existing controllers decided, in one object.

    A projection, not a re-decision. Every value here is read from something
    another part of the system already computed — the analyzer's situation, the
    commercial policy's action, the open threads the continuity layer is
    carrying. Nothing is inferred and no model is called, so this is safe to run
    anywhere and its output is exactly today's behaviour, described.

    What it deliberately drops is as informative as what it keeps. The director's
    phase, the session strategy's ``writer_goal`` sentence and the analyzer's
    ``strategic_move`` are three more prescriptions of the next move, and §4
    says a decision must not carry an emotional ladder or a sentence shape. They
    are not represented, so the replay comparison can show whether the reply
    actually needed them — which is the removal test the review asks for and
    warns against skipping.
    """
    situation = dict(state.get("situation") or {})
    commercial = dict(state.get("commercial_decision") or {})

    needs: list[str] = []
    purchase_signal = _text(situation.get("purchase_signal")).lower()
    if purchase_signal and purchase_signal != "none":
        described = _PURCHASE_NEEDS.get(purchase_signal)
        if described:
            needs.append(described)

    if _text(situation.get("resend_requested")).lower() == "true":
        # Ahead of everything else, deliberately. §4: "A request to fix access
        # outranks a new commercial suggestion."
        needs.insert(0, "he cannot access content he paid for")

    crisis = _text(situation.get("crisis_signal")).lower()
    if crisis and crisis != "none":
        needs.insert(0, "he needs support, not a conversation about content")

    # must_address stays EMPTY, and that is the finding rather than a gap in
    # this function.
    #
    # The open threads are in the packet, so they reach the prompt. But no
    # controller in the current stack decides which of them this reply has to
    # answer: the commercial policy decides a business move, the director
    # decides a conversational move, the session strategy decides a goal, and
    # none of them takes an obligation as input. The reply either happens to
    # pick one up from the transcript or it does not.
    #
    # Filling this in from packet.open_threads would make the projection look
    # like it decides something it does not, and would make the replay's
    # missed-obligation count vacuous for this candidate — it could never miss
    # one. Leaving it empty is what lets the comparison measure the difference
    # honestly.
    must_address: tuple[str, ...] = ()
    if not needs and packet.open_threads:
        needs.append("he is waiting on something from an earlier message")
    if not needs:
        needs.append("ordinary conversation")

    action = _text(commercial.get("action")).upper()
    operation = ProposedOperation()
    if action in _COMMERCIAL_OPERATIONS:
        offer = commercial.get("next_offer") or {}
        subject = _text(
            (offer.get("label") if isinstance(offer, dict) else "")
            or commercial.get("accepted_offer_set_id")
            or "the next unlock"
        )
        operation = ProposedOperation(
            kind=_COMMERCIAL_OPERATIONS[action],
            subject=subject,
            because=f"the commercial policy decided {action}",
        )

    hold, hold_detail = HoldReason.NONE, ""
    if crisis and crisis != "none":
        hold, hold_detail = HoldReason.NEEDS_HUMAN, "a crisis signal freezes this chat"
    elif _text(situation.get("resend_requested")).lower() == "true":
        hold, hold_detail = (
            HoldReason.NEEDS_HUMAN,
            "an access complaint is resolved by an operator, not by a reply",
        )
    elif state.get("frozen_for_review"):
        hold, hold_detail = HoldReason.NEEDS_HUMAN, "this conversation is on hold"
    elif action in _COMMERCIAL_HOLDS:
        hold, hold_detail = _COMMERCIAL_HOLDS[action]

    if state.get("analysis_degraded"):
        # REL-001 in this vocabulary: a fabricated analysis is not evidence, and
        # the projection must not present a confident reading built on one.
        hold, hold_detail = (
            HoldReason.INSUFFICIENT_EVIDENCE,
            "the situation analysis was fabricated after the analyzer failed",
        )

    if hold != HoldReason.NONE and operation.is_external:
        # A turn that is holding is not also proposing a sale. One decision
        # states one thing.
        #
        # The commercial layer computed that offer independently of the
        # complaint, the crisis or the freeze, and in the live path an ordering
        # guard returns before it is ever reached
        # (services/suggestions.py, PR #48). That guard is real and this is not
        # a claim that it fails — but the offer existing at all, in a decision
        # whose own reason is "hand this to a human", is finding F exactly: two
        # representations of the next move, and only the order in which they
        # happen to run keeping them apart. Recorded in the reason so the
        # replay can show it rather than hiding it behind a silent drop.
        hold_detail = (
            f"{hold_detail}; the commercial layer separately wanted "
            f"{operation.kind.value} about {operation.subject!r}"
        )
        operation = ProposedOperation()

    return ConversationDecision(
        active_needs=tuple(needs),
        supporting_messages=tuple(state.get("supporting_messages") or ()),
        unresolved_references=tuple(state.get("unresolved_references") or ()),
        must_address=must_address,
        proposed_operation=operation,
        hold=hold,
        hold_detail=hold_detail,
        source=source,
        confidence=0.0 if state.get("analysis_degraded") else 1.0,
    )


class CurrentStackOwner:
    """The projection, as a ``DecisionOwner``."""

    name = SOURCE_CURRENT_STACK

    async def decide(
        self, packet: ContextPacket, state: dict[str, Any]
    ) -> ConversationDecision:
        return current_stack_decision(packet, state)


# ---------------------------------------------------------------------------
# Candidate 2 — one semantic owner
# ---------------------------------------------------------------------------

SEMANTIC_SYSTEM = """You read one conversation between a creator and a customer on a paid content platform, and you state what the creator's next message needs to do. You do not write the message.

Answer with a JSON object and nothing else:

{
  "active_needs": ["what he actually wants right now, in plain words"],
  "unresolved_references": ["anything he referred to that the conversation does not make clear"],
  "must_address": ["questions or obligations this reply has to answer"],
  "operation": "none" | "offer_content" | "deliver_paid_content" | "repair_content_access" | "hand_off_to_human",
  "operation_subject": "what that operation is about, in his words, or \\"\\"",
  "operation_because": "why, or \\"\\"",
  "hold": "none" | "waiting_on_customer" | "waiting_on_payment" | "needs_human" | "respect_silence" | "insufficient_evidence",
  "hold_detail": "why this turn waits or hands over, or \\"\\"",
  "confidence": 0.0 to 1.0
}

Rules:
- Report what is true of this conversation. Do not decide tone, length, or what the message should say.
- A request to fix access to something already paid for outranks any suggestion to sell.
- Never state a price. Never say something was sent, delivered or paid unless the conversation shows a confirmation.
- An operation is a request for someone else to check and carry out, never permission.
- If the evidence does not support a reading, say so with "insufficient_evidence" rather than guessing.
- More than one active need is normal. Do not collapse a mixed message into one."""


#: The fields the contract in SEMANTIC_SYSTEM asks for and a decision cannot be
#: read without. `active_needs` and the other lists may legitimately be empty,
#: so their absence is tolerated; these three ARE the decision — what the turn
#: proposes, whether it waits, and how far to trust either.
REQUIRED_DECISION_FIELDS: tuple[str, ...] = ("operation", "hold", "confidence")


@dataclass(frozen=True)
class DecisionParse:
    """A read of the semantic owner's answer, or a named refusal.

    The reason is the point. ``parse_semantic_decision`` returned a bare None,
    so every different way of failing — a truncated response, an invented
    enum, a missing field — became the same "did not answer in the required
    shape" line, and a comparison could not say WHICH candidate failed how.
    """

    decision: "ConversationDecision | None" = None
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.decision is not None


def _refuse(reason: str) -> DecisionParse:
    return DecisionParse(decision=None, reason=reason)


def parse_semantic_decision_result(
    text: str, *, source: str = SOURCE_SEMANTIC
) -> DecisionParse:
    """Read the semantic owner's answer strictly, or say why it was refused.

    STRICT, AND IT WAS NOT
    ----------------------
    The docstring of the version this replaces said "Returns None for anything
    unparseable. There is no partial credit and no repair pass." The code did
    the opposite for the cases that matter most: an unrecognised enum was
    silently coerced to NONE and a missing ``confidence`` defaulted to 1.0, so

        {"hold": "typo", "operation": "typo"}

    produced a decision that proposed nothing, held nothing, and claimed FULL
    confidence in that reading. Reproduced against backend 4a1683a. That is a
    fabricated confident decision assembled from a response the parser could
    not understand — REL-001's shape exactly, in the component whose entire
    purpose is to be trusted enough to compare.

    So every one of these is a refusal, and each says which it was:

    * the response is not a JSON object;
    * a required field is missing — a response that does not say what it
      proposes or whether it holds has not answered;
    * an enum is not one of the values the contract lists;
    * a field is the wrong type;
    * confidence is not a finite number, or is outside 0..1.

    Out-of-range confidence is refused rather than clamped, which is the one
    choice here worth arguing about. Clamping 1.5 to 1.0 RAISES a malformed
    claim to maximum confidence — the failure mode this function exists to
    prevent, applied by the repair.
    """
    raw = str(text or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        return _refuse("the response contained no JSON object")
    try:
        payload = json.loads(raw[start : end + 1])
    except (ValueError, TypeError):
        return _refuse("the response was not valid JSON")
    if not isinstance(payload, dict):
        return _refuse("the response was not a JSON object")

    missing = [key for key in REQUIRED_DECISION_FIELDS if key not in payload]
    if missing:
        return _refuse(f"the response left out {', '.join(missing)}")

    def _list(key: str) -> tuple[str, ...] | None:
        value = payload.get(key, [])
        if value is None:
            return ()
        if not isinstance(value, list):
            return None
        return tuple(str(item).strip() for item in value if str(item).strip())

    lists: dict[str, tuple[str, ...]] = {}
    for key in (
        "active_needs",
        "unresolved_references",
        "must_address",
        "supporting_messages",
    ):
        parsed = _list(key)
        if parsed is None:
            return _refuse(f"{key} was not a list")
        lists[key] = parsed

    for key in ("operation", "hold"):
        if not isinstance(payload.get(key), str):
            return _refuse(f"{key} was not a string")
    try:
        kind = OperationKind(payload["operation"].strip() or "none")
    except ValueError:
        return _refuse(f"operation {payload['operation']!r} is not one this system knows")
    try:
        hold = HoldReason(payload["hold"].strip() or "none")
    except ValueError:
        return _refuse(f"hold {payload['hold']!r} is not one this system knows")

    confidence_raw = payload["confidence"]
    if isinstance(confidence_raw, bool) or not isinstance(confidence_raw, (int, float)):
        # `bool` first: it is a subclass of int, and True would otherwise read
        # as a confidence of 1.0 — the exact fabrication being prevented.
        return _refuse("confidence was not a number")
    confidence = float(confidence_raw)
    if not math.isfinite(confidence):
        return _refuse("confidence was not a finite number")
    if not 0.0 <= confidence <= 1.0:
        return _refuse(f"confidence {confidence} is outside 0..1")

    for key in ("operation_subject", "operation_because", "hold_detail"):
        if key in payload and payload[key] is not None and not isinstance(payload[key], str):
            return _refuse(f"{key} was not a string")

    return DecisionParse(
        decision=ConversationDecision(
            active_needs=lists["active_needs"],
            supporting_messages=lists["supporting_messages"],
            unresolved_references=lists["unresolved_references"],
            must_address=lists["must_address"],
            proposed_operation=ProposedOperation(
                kind=kind,
                subject=str(payload.get("operation_subject") or "").strip(),
                because=str(payload.get("operation_because") or "").strip(),
            ),
            hold=hold,
            hold_detail=str(payload.get("hold_detail") or "").strip(),
            source=source,
            confidence=confidence,
        )
    )


def parse_semantic_decision(
    text: str, *, source: str = SOURCE_SEMANTIC
) -> ConversationDecision | None:
    """The decision, or None. ``parse_semantic_decision_result`` says why."""
    return parse_semantic_decision_result(text, source=source).decision


def build_semantic_prompt(packet: ContextPacket, state: dict[str, Any]) -> tuple[str, str]:
    """The (system, user) pair for the semantic owner.

    Built from the SAME context packet the current stack reads, which is what
    makes the comparison a comparison. §5: "Replay gives candidates the same
    evidence."
    """
    parts = [f"Conversation so far:\n{packet.render_transcript()}"]
    continuity = packet.render_continuity()
    if continuity:
        parts.append(continuity)
    operational = state.get("operational_facts")
    if operational:
        # Facts the conversation cannot establish: what is actually pending,
        # what was actually paid. Supplied rather than inferred, because §4 puts
        # them under database authority and nowhere else.
        parts.append(f"OPERATIONAL FACTS (authoritative):\n{operational}")
    latest = str(state.get("latest_message") or "")
    if latest:
        parts.append(f'Latest message from him: "{latest}"')
    return SEMANTIC_SYSTEM, "\n\n".join(parts)


class SemanticDecisionOwner:
    """One model call that reads the packet and states what the turn needs to do.

    Offline only. It exists to be compared, and it is constructed with an
    explicit ``complete`` callable so a replay can run it against a stub, a
    recorded response, or a real provider without any of those choices leaking
    into the comparison.
    """

    name = SOURCE_SEMANTIC

    def __init__(self, complete, *, target=None) -> None:
        self._complete = complete
        self._target = target

    async def decide(
        self, packet: ContextPacket, state: dict[str, Any]
    ) -> ConversationDecision:
        system, user = build_semantic_prompt(packet, state)
        try:
            result = await self._complete(
                self._target,
                system=system,
                messages=[{"role": "user", "content": user}],
                max_tokens=600,
            )
        except Exception as exc:
            return ConversationDecision(
                active_needs=(),
                hold=HoldReason.INSUFFICIENT_EVIDENCE,
                hold_detail=f"the semantic owner could not be reached: {exc}",
                source=self.name,
                confidence=0.0,
            )
        parsed = parse_semantic_decision_result(
            getattr(result, "text", "") or "", source=self.name
        )
        if not parsed.ok:
            # Naming the failure rather than flattening every one of them into
            # "did not answer in the required shape": a comparison that cannot
            # say WHICH candidate failed HOW is not much of a comparison.
            return ConversationDecision(
                active_needs=(),
                hold=HoldReason.INSUFFICIENT_EVIDENCE,
                hold_detail=f"the semantic owner did not answer usably: {parsed.reason}",
                source=self.name,
                confidence=0.0,
            )
        return parsed.decision
