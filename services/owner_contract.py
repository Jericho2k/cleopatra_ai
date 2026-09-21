"""Read a conversational owner's answer one component at a time.

WHY THIS REPLACES THE ONE-SHOT PARSE
------------------------------------
Conversational Core v1 asked the owner for a single strict JSON object carrying
the fan-facing reply, the typed intent, a proposed commercial operation, four
lists of interpretive metadata and a working-state delta. It then read that
object with ``parse_reply_plus_intent``, whose contract is all-or-nothing by
design — it exists to make an *offline comparison* of two candidates fair, and a
candidate that got partial credit would win by being marked wrong less often.

Applying that parser to a live turn made every recoverable mistake fatal. A
confidence of ``1.5``, an unknown ``response_intent``, a ``state_delta``
truncated by the token budget, an operation naming an offer that no longer
exists: each one produced ``decision = None``, which produced ``owner_failed``,
which sent the fan nothing. The reply was sitting in the response the whole
time.

The rule here is the inverse:

    A recoverable format, state or operation mistake must not kill an otherwise
    valid conversation.

So the three components fail independently.

``reply``
    The only required output. It survives malformed optional metadata, an
    invalid operation, a rejected delta, and JSON truncated part-way through the
    object — including, as a last resort, JSON that never closed at all.

``operation``
    Optional. Anything the contract cannot read becomes ``none`` and is recorded
    as discarded. Deterministic validation in ``services.live_orchestration``
    remains the only authority that can approve one; this layer only decides
    whether there is a coherent proposal worth handing it.

``state_delta``
    Optional and opaque here. It is handed to
    ``services.conversational_core.validate_and_apply_delta`` exactly as
    received, because that validator already rejects fields individually. A
    delta that is not even an object is dropped, and the reply is untouched.

NOTHING HERE AUTHORIZES ANYTHING
--------------------------------
Every recovery in this module makes the owner's answer *smaller*: a field is
dropped, an operation is downgraded to ``none``, a confidence is lowered to
zero. No recovery invents a reference, raises a confidence, or promotes a
proposal. ``evidenced_refs`` exists so a repair attempt cannot introduce an
operation reference the original evidence never contained.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any

from models.conversation_decision import (
    ConversationDecision,
    HoldReason,
    OperationKind,
    ProposedOperation,
    ResponseDisposition,
    ResponseIntent,
)
from models.model_runtime import (
    FAILURE_EMPTY_REPLY,
    FAILURE_INVALID_JSON,
    FAILURE_MISSING_REPLY,
    FAILURE_NO_JSON,
    FAILURE_NOT_AN_OBJECT,
)

#: How the owner's JSON object was obtained.
JSON_OK = "ok"
#: Parsed only after unterminated strings and brackets were closed locally.
JSON_RECOVERED = "recovered"
#: No object could be parsed; only the reply string itself was recovered.
JSON_REPLY_ONLY = "reply_only"
#: Nothing usable at all.
JSON_ABSENT = "absent"

#: The smallest answer this runtime will act on. Used verbatim in the repair
#: instruction so the model is asked for exactly what the extractor requires.
MINIMUM_OWNER_CONTRACT = (
    '{"reply": "the customer-facing message", '
    '"response_intent": "ordinary_conversation", '
    '"operation": "none"}'
)

_MAX_REPAIR_TRIMS = 400
_MAX_REPLY_CHARS = 4_000


@dataclass(frozen=True)
class OwnerResult:
    """What could be read from one owner response, and what could not.

    ``decision`` is ``None`` only when there is no usable turn at all — no
    reply, and no explicit hold saying why there is none. Everything else is a
    successful read with recorded degradations.
    """

    decision: ConversationDecision | None = None
    reply: str = ""
    state_delta: Any = None
    json_status: str = JSON_ABSENT
    failure_category: str = ""
    failure_detail: str = ""
    #: ``field: why`` for every part that was dropped or downgraded.
    degradations: dict[str, str] = field(default_factory=dict)
    #: Why a proposed operation was refused before deterministic validation.
    operation_discarded: str = ""

    @property
    def usable(self) -> bool:
        return self.decision is not None

    @property
    def degraded(self) -> bool:
        return bool(self.degradations) or bool(self.operation_discarded)

    def describe(self) -> str:
        parts = [f"json={self.json_status}", f"reply_chars={len(self.reply)}"]
        if self.failure_category:
            parts.append(f"failure={self.failure_category}")
        if self.operation_discarded:
            parts.append(f"operation_discarded={self.operation_discarded}")
        if self.degradations:
            parts.append("degraded=" + ",".join(sorted(self.degradations)))
        return " ".join(parts)


# ---------------------------------------------------------------------------
# JSON recovery
# ---------------------------------------------------------------------------


def _strip_fences(text: str) -> str:
    raw = str(text or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
    return raw.strip()


def _close_open_json(fragment: str) -> str | None:
    """Close an object truncated mid-flight, or return None if it cannot be.

    A reasoning model that runs out of completion budget stops in the middle of
    a string, a key or a nested object. The prefix before that point is still
    the model's real answer — most importantly the reply, which the contract
    puts first — so it is worth closing rather than discarding.
    """

    stack: list[str] = []
    in_string = False
    escaped = False
    for char in fragment:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append(char)
        elif char in "}]":
            if not stack:
                return None
            opener = stack.pop()
            if (opener, char) not in {("{", "}"), ("[", "]")}:
                return None
    if not stack and not in_string:
        return fragment
    closed = fragment
    if escaped:
        # A dangling backslash would escape the quote that closes the string.
        closed = closed[:-1]
    if in_string:
        closed += '"'
    for opener in reversed(stack):
        closed += "}" if opener == "{" else "]"
    return closed


def _trim_to_previous_boundary(fragment: str) -> str | None:
    """Drop the last incomplete member of a truncated object or array."""

    in_string = False
    escaped = False
    depth = 0
    last_boundary = -1
    for index, char in enumerate(fragment):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            depth += 1
        elif char in "}]":
            depth = max(depth - 1, 0)
        elif char == "," and depth >= 1:
            last_boundary = index
    if last_boundary < 0:
        return None
    return fragment[:last_boundary]


def _load_json_object(text: str) -> tuple[dict[str, Any] | None, str]:
    """Read the owner's object, repairing truncation but never inventing data."""

    raw = _strip_fences(text)
    start = raw.find("{")
    if start < 0:
        return None, JSON_ABSENT

    end = raw.rfind("}")
    if end > start:
        try:
            value = json.loads(raw[start : end + 1])
        except (TypeError, ValueError):
            value = None
        if isinstance(value, dict):
            return value, JSON_OK
        if value is not None:
            return None, JSON_ABSENT

    fragment = raw[start:]
    for _ in range(_MAX_REPAIR_TRIMS):
        closed = _close_open_json(fragment)
        if closed is not None:
            try:
                value = json.loads(closed)
            except (TypeError, ValueError):
                value = None
            if isinstance(value, dict):
                return value, JSON_RECOVERED
        trimmed = _trim_to_previous_boundary(fragment)
        if trimmed is None or trimmed == fragment:
            break
        fragment = trimmed
    return None, JSON_ABSENT


def _json_string_at(text: str, index: int) -> str | None:
    """Read the JSON string literal starting at ``index``, even if unterminated."""

    if index >= len(text) or text[index] != '"':
        return None
    out: list[str] = []
    escaped = False
    for char in text[index + 1 :]:
        if escaped:
            out.append({"n": "\n", "t": "\t", "r": "\r"}.get(char, char))
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            break
        out.append(char)
    return "".join(out)


_REPLY_KEY = re.compile(r'"reply"\s*:\s*')


def recover_reply_only(text: str) -> str:
    """Pull just the reply out of a response no object could be read from.

    The last line of defence. The contract puts ``reply`` first, so a response
    cut off anywhere after it still contains the whole fan-facing message even
    when nothing else survived.
    """

    raw = _strip_fences(text)
    match = _REPLY_KEY.search(raw)
    if not match:
        return ""
    value = _json_string_at(raw, match.end())
    if value is None:
        return ""
    return " ".join(value.split())[:_MAX_REPLY_CHARS]


# ---------------------------------------------------------------------------
# Component reads
# ---------------------------------------------------------------------------


def _string_list(
    payload: dict[str, Any],
    key: str,
    degradations: dict[str, str],
) -> tuple[str, ...]:
    value = payload.get(key)
    if value in (None, ""):
        return ()
    if not isinstance(value, list):
        degradations[key] = "not a list; dropped"
        return ()
    return tuple(str(item).strip() for item in value if str(item).strip())


def _enum(
    payload: dict[str, Any],
    key: str,
    enum_type: Any,
    fallback: Any,
    degradations: dict[str, str],
) -> Any:
    value = payload.get(key)
    if value in (None, ""):
        # An omitted optional enum is not a defect. Recording one would make
        # every ordinary turn look degraded and would hide the real rate.
        return fallback
    if not isinstance(value, str):
        degradations[key] = f"not a string; assumed {fallback.value}"
        return fallback
    try:
        return enum_type(value.strip())
    except ValueError:
        degradations[key] = f"unknown value; assumed {fallback.value}"
        return fallback


def _ref(payload: dict[str, Any], key: str, degradations: dict[str, str]) -> str:
    value = payload.get(key)
    if value in (None, ""):
        return ""
    if not isinstance(value, str):
        degradations[key] = "not a string; dropped"
        return ""
    return value.strip()


def _confidence(
    payload: dict[str, Any],
    degradations: dict[str, str],
) -> tuple[float, bool]:
    """Read confidence, or refuse to guess one.

    The old parser defaulted a missing confidence to 1.0, which turned "the
    model did not say how sure it was" into "the model was certain". Here an
    unreadable confidence becomes 0.0 AND disqualifies any operation: an
    external effect must never rest on a number this layer made up.
    """

    if "confidence" not in payload:
        degradations["confidence"] = "missing; treated as 0.0"
        return 0.0, False
    value = payload["confidence"]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        degradations["confidence"] = "not a number; treated as 0.0"
        return 0.0, False
    number = float(value)
    if not math.isfinite(number):
        degradations["confidence"] = "not finite; treated as 0.0"
        return 0.0, False
    if not 0.0 <= number <= 1.0:
        # Clamping 1.5 to 1.0 would RAISE a malformed claim to maximum
        # confidence, so it is refused rather than repaired.
        degradations["confidence"] = f"{number} outside 0..1; treated as 0.0"
        return 0.0, False
    return number, True


def _reply_text(payload: dict[str, Any], degradations: dict[str, str]) -> str:
    value = payload.get("reply")
    if value is None:
        return ""
    if isinstance(value, list):
        # Some providers answer json_object mode with an array of bubbles.
        degradations["reply"] = "list of parts joined into one reply"
        value = " | ".join(str(part).strip() for part in value if str(part).strip())
    if not isinstance(value, str):
        degradations["reply"] = "not a string; dropped"
        return ""
    return value.strip()[:_MAX_REPLY_CHARS]


def _coherent_disposition(
    *,
    reply: str,
    disposition: ResponseDisposition,
    hold: HoldReason,
    kind: OperationKind,
    degradations: dict[str, str],
) -> tuple[ResponseDisposition, HoldReason, OperationKind, str]:
    """Reconcile the fields that describe what this turn does.

    Contradictions are resolved by taking the *safer* half and keeping the
    reply. Every resolution narrows what the turn may do; none widens it.
    """

    discarded = ""

    if kind is OperationKind.HAND_OFF_TO_HUMAN and (
        disposition is not ResponseDisposition.HANDOFF
    ):
        # Asking for a human is never overridden into an ordinary reply.
        degradations["disposition"] = "handoff operation forces a handoff"
        disposition = ResponseDisposition.HANDOFF
    if hold is HoldReason.NEEDS_HUMAN and disposition is not ResponseDisposition.HANDOFF:
        degradations["disposition"] = "needs_human forces a handoff"
        disposition = ResponseDisposition.HANDOFF

    if disposition is ResponseDisposition.HANDOFF:
        if kind not in {OperationKind.HAND_OFF_TO_HUMAN, OperationKind.REPAIR_CONTENT_ACCESS}:
            if kind is not OperationKind.NONE:
                discarded = "a handoff turn may not also run a commercial operation"
            kind = OperationKind.HAND_OFF_TO_HUMAN
        if hold is HoldReason.NONE:
            hold = HoldReason.NEEDS_HUMAN
            degradations["hold"] = "handoff without a hold; recorded as needs_human"
        return disposition, hold, kind, discarded

    if disposition is ResponseDisposition.SILENCE:
        if kind is not OperationKind.NONE:
            discarded = "a silent turn cannot also propose an operation"
            kind = OperationKind.NONE
        if hold is HoldReason.NONE:
            hold = HoldReason.RESPECT_SILENCE
            degradations["hold"] = "silence without a hold; recorded as respect_silence"
        return disposition, hold, kind, discarded

    # An ordinary reply that also claims to be holding for silence or a human is
    # contradictory. The reply is real and was written to be sent, so the hold
    # is dropped — and with it any operation, because a decision this confused
    # has not earned an external effect.
    if hold in {HoldReason.RESPECT_SILENCE, HoldReason.NEEDS_HUMAN}:
        degradations["hold"] = f"{hold.value} contradicts a reply; hold dropped"
        hold = HoldReason.NONE
        if kind is not OperationKind.NONE:
            discarded = "operation dropped with a contradictory hold"
            kind = OperationKind.NONE
    if hold is HoldReason.INSUFFICIENT_EVIDENCE and reply:
        # ``validate_decision`` refuses any operation under this hold, and a
        # turn that produced a reply is not evidence-starved in the sense the
        # hold means. Keep the reply, drop the operation.
        if kind is not OperationKind.NONE:
            discarded = "insufficient_evidence forbids an operation"
            kind = OperationKind.NONE
        degradations["hold"] = "insufficient_evidence alongside a reply; hold dropped"
        hold = HoldReason.NONE
    return disposition, hold, kind, discarded


def _operation_refs(
    payload: dict[str, Any],
    degradations: dict[str, str],
) -> dict[str, str]:
    return {
        "offer_id": _ref(payload, "operation_offer_id", degradations),
        "set_id": _ref(payload, "operation_set_id", degradations),
        "payment_reference": _ref(
            payload, "operation_payment_reference", degradations
        ),
        "purchase_id": _ref(payload, "operation_purchase_id", degradations),
    }


def extract_owner_result(
    text: str,
    *,
    source: str,
    evidenced_refs: frozenset[str] | None = None,
) -> OwnerResult:
    """Read whatever valid owner result exists in ``text``.

    ``evidenced_refs``, when given, is the exact set of references the owner was
    shown. Any operation citing something outside it is discarded before it can
    reach deterministic validation. It is passed on a repair attempt, where a
    model re-emitting a structured object must not be able to introduce an
    identifier that was never in the evidence.
    """

    degradations: dict[str, str] = {}
    payload, json_status = _load_json_object(text)

    if payload is None:
        salvaged = recover_reply_only(text)
        if salvaged:
            degradations["state_delta"] = "unreadable response; delta discarded"
            degradations["operation"] = "unreadable response; operation discarded"
            return OwnerResult(
                decision=ConversationDecision(
                    proposed_operation=ProposedOperation(),
                    response_intent=ResponseIntent.ORDINARY_CONVERSATION,
                    disposition=ResponseDisposition.REPLY,
                    hold=HoldReason.NONE,
                    source=source,
                    confidence=0.0,
                ),
                reply=salvaged,
                state_delta=None,
                json_status=JSON_REPLY_ONLY,
                degradations=degradations,
                operation_discarded="no readable JSON object around the reply",
            )
        has_brace = "{" in _strip_fences(text)
        return OwnerResult(
            json_status=JSON_ABSENT,
            failure_category=FAILURE_INVALID_JSON if has_brace else FAILURE_NO_JSON,
            failure_detail=(
                "the response contained a JSON fragment that could not be closed"
                if has_brace
                else "the response contained no JSON object"
            ),
        )

    if not isinstance(payload, dict):  # pragma: no cover - _load_json_object guards
        return OwnerResult(
            json_status=JSON_ABSENT,
            failure_category=FAILURE_NOT_AN_OBJECT,
            failure_detail="the response was not a JSON object",
        )

    reply = _reply_text(payload, degradations)
    # The fallback is REPLY even with no reply text, deliberately. Treating a
    # response that simply stopped as a considered silence is how a broken turn
    # used to become a clean-looking ``no_send``; staying quiet has to be
    # something the owner SAID, not something absence is read as.
    disposition = _enum(
        payload,
        "disposition",
        ResponseDisposition,
        ResponseDisposition.REPLY,
        degradations,
    )
    hold = _enum(payload, "hold", HoldReason, HoldReason.NONE, degradations)
    response_intent = _enum(
        payload,
        "response_intent",
        ResponseIntent,
        ResponseIntent.ORDINARY_CONVERSATION,
        degradations,
    )
    kind = _enum(payload, "operation", OperationKind, OperationKind.NONE, degradations)
    confidence, confidence_read = _confidence(payload, degradations)

    operation_discarded = ""
    if not confidence_read and kind is not OperationKind.NONE:
        operation_discarded = "confidence could not be read"
        kind = OperationKind.NONE

    refs = _operation_refs(payload, degradations)
    if evidenced_refs is not None and kind is not OperationKind.NONE:
        unevidenced = sorted(
            value for value in refs.values() if value and value not in evidenced_refs
        )
        if unevidenced:
            operation_discarded = (
                "repair cited references absent from the evidence: "
                + ", ".join(unevidenced[:3])
            )
            kind = OperationKind.NONE
            refs = dict.fromkeys(refs, "")

    disposition, hold, kind, incoherent = _coherent_disposition(
        reply=reply,
        disposition=disposition,
        hold=hold,
        kind=kind,
        degradations=degradations,
    )
    operation_discarded = operation_discarded or incoherent
    if kind is OperationKind.NONE:
        refs = dict.fromkeys(refs, "")

    if not reply and disposition is ResponseDisposition.REPLY:
        return OwnerResult(
            state_delta=payload.get("state_delta"),
            json_status=json_status,
            failure_category=(
                FAILURE_EMPTY_REPLY if "reply" in payload else FAILURE_MISSING_REPLY
            ),
            failure_detail=(
                "the response carried an empty reply and no hold explaining it"
                if "reply" in payload
                else "the response left out reply"
            ),
            degradations=degradations,
            operation_discarded=operation_discarded,
        )

    # The delta is passed through exactly as received. Judging it here would
    # put two validators in front of the working state; the one in
    # ``services.conversational_core`` already rejects fields individually and
    # records every refusal, and it is the only one that can see the evidence.
    raw_delta = payload.get("state_delta")

    decision = ConversationDecision(
        active_needs=_string_list(payload, "active_needs", degradations),
        supporting_messages=_string_list(payload, "supporting_messages", degradations),
        unresolved_references=_string_list(
            payload, "unresolved_references", degradations
        ),
        must_address=_string_list(payload, "must_address", degradations),
        proposed_operation=ProposedOperation(
            kind=kind,
            subject=_ref(payload, "operation_subject", degradations),
            because=_ref(payload, "operation_because", degradations),
            **refs,
        ),
        response_intent=response_intent,
        disposition=disposition,
        hold=hold,
        hold_detail=_ref(payload, "hold_detail", degradations),
        source=source,
        confidence=confidence,
    )
    return OwnerResult(
        decision=decision,
        reply=reply,
        state_delta=raw_delta,
        json_status=json_status,
        degradations=degradations,
        operation_discarded=operation_discarded,
    )
