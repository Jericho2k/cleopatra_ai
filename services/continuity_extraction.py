"""Turning what the analyzer noticed into records the conversation carries.

THE GAP THIS CLOSES
-------------------
``db/conversation_continuity_v1.sql`` created the tables and
``services/conversation_continuity.py`` the lifecycle, and almost nothing wrote
to them. The only live ``record_open_thread`` call outside the service was the
content-access complaint in ``services/suggestions.py``, and nothing anywhere
called ``record_episode``. So an ordinary unanswered question, a promise, a
deferred topic and a correction never acquired the durable lifecycle the whole
mechanism exists to give them — the review asked for "the state of the
interaction" and the state of the interaction was one row type deep.

WHY EXTRACTION IS SEMANTIC AND VALIDATION IS NOT
------------------------------------------------
Deciding that "yeah but what about the thing you said last week" is an
unanswered question is interpretation, and the brief is explicit that
interpretation belongs in semantic extraction rather than in a longer keyword
list. The analyzer is already a model call with a structured contract, so it is
where that happens.

Deciding whether the result may be WRITTEN is not interpretation, and must not
depend on the model having followed its instructions. Everything below is
deterministic, runs on every proposal, and is the reason a hallucinating or
prompt-injected analyzer cannot put arbitrary records into a customer's
conversation.

THE MONEY RULE
--------------
The prompt says never to put money in these lists. That is a request, and a
request is not an enforcement mechanism: the analyzer reads customer-supplied
text, and customer-supplied text is exactly what would try to make it say "he
paid for the premium set". ``ppv_deliveries`` is the only authority on whether
money moved, and a thread claiming otherwise would be a fabricated financial
record sitting next to real ones.

So monetary proposals are dropped here, in code, whatever the analyzer says.
Dropped and counted, never silently: a rejection rate that climbs is how
somebody notices the analyzer has started doing something it should not.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from models.conversation_continuity import (
    EvidenceType,
    OpenThread,
    ThreadKind,
    ThreadParty,
)

#: The analyzer field each kind of unfinished business arrives in, and who the
#: obligation falls on. A question HE asked is owed by the creator; a promise
#: the creator made is owed by the creator; a correction is his to make and
#: ours to respect.
EXTRACTION_FIELDS: tuple[tuple[str, ThreadKind, ThreadParty], ...] = (
    ("open_questions_raised", ThreadKind.QUESTION, ThreadParty.FAN),
    ("commitments_made", ThreadKind.PROMISE, ThreadParty.CREATOR),
    ("topics_deferred", ThreadKind.DEFERRED_TOPIC, ThreadParty.FAN),
    ("corrections_stated", ThreadKind.CORRECTION, ThreadParty.FAN),
)

#: At most this many new records from one turn, per kind.
#:
#: A cap rather than a trust boundary: one exchange genuinely raising six
#: distinct unanswered questions is vanishingly rare, and an analyzer emitting
#: twelve is malfunctioning. Without it, one bad response writes a conversation
#: nobody can read afterwards.
MAX_PER_KIND = 3

#: Shorter than the column allows. These are meant to be read in a context
#: packet alongside a transcript, and a paragraph-length "summary" is a second
#: transcript.
MAX_SUMMARY_CHARS = 160

#: Anything financial. Deliberately broad, and deliberately applied to the
#: WHOLE proposal rather than to a word position: a false rejection costs a
#: conversational record, and a false acceptance is a fabricated claim about
#: money sitting beside the ledger that decides it.
_MONEY = re.compile(
    r"""
    \$\d                                  # a price
    | \b\d+\s?(?:usd|dollars?|bucks|quid) # an amount
    | \b(?:paid|pay|paying|payment|purchase[ds]?|buy|bought|buying
        | refund(?:ed)?|charge[ds]?|charging|billed|invoice
        | unlock(?:ed|s)?|ppv|tip(?:ped|s)?|subscription|renew(?:al|ed)?
        | owe[sd]?|owing|credit|price[ds]?|cost(?:s|ed)?|spend|spent
      )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)


@dataclass
class ExtractionResult:
    """What one turn proposed, and what survived validation."""

    threads: list[OpenThread] = field(default_factory=list)
    #: Opaque stored-thread references the analyzer offered as resolved.  Free
    #: text is deliberately insufficient: a summary is not an identity and an
    #: uncertain match must never close an unrelated obligation.
    resolved: list["ResolutionProposal"] = field(default_factory=list)
    #: Corrections that explicitly name the stored record they replace.  A
    #: correction without a validated reference is still recorded as visible
    #: new evidence, but it cannot suppress an old record automatically.
    supersessions: list["SupersessionProposal"] = field(default_factory=list)
    #: Why proposals were refused, by reason. Reported rather than swallowed.
    rejected: dict[str, int] = field(default_factory=dict)

    def _reject(self, reason: str) -> None:
        self.rejected[reason] = self.rejected.get(reason, 0) + 1

    @property
    def rejected_total(self) -> int:
        return sum(self.rejected.values())

    def as_metadata(self) -> dict[str, Any]:
        """A compact record for a reply's provenance.

        Counts, never the proposals themselves. What matters afterwards is that
        extraction ran, how much it produced and how much was refused — a
        rejection rate climbing is the signal that something upstream changed.
        """
        return {
            "threads_recorded": len(self.threads),
            "threads_resolved_proposed": len(self.resolved),
            "threads_superseded_proposed": len(self.supersessions),
            "rejected": dict(self.rejected),
        }


@dataclass(frozen=True)
class ResolutionProposal:
    thread_id: str
    evidence: str


@dataclass(frozen=True)
class SupersessionProposal:
    thread: OpenThread
    replaces_thread_id: str
    evidence: str


def mentions_money(text: object) -> bool:
    """Whether a proposal is making a claim about money.

    Used to refuse it. ``ppv_deliveries`` is the only authority on whether money
    moved, and a thread saying otherwise is a fabricated financial record.
    """
    return bool(_MONEY.search(str(text or "")))


def _clean(value: object) -> str:
    return " ".join(str(value or "").split())


def _proposals(situation: Any, key: str) -> list[str]:
    """One analyzer list, tolerating everything a model might return instead.

    A string where a list belongs, a list with a dict in it, None — all of it
    is an analyzer being imprecise rather than an emergency, and none of it may
    raise inside a reply pipeline.
    """
    if not isinstance(situation, dict):
        return []
    raw = situation.get(key)
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        return []
    return [_clean(item) for item in raw if isinstance(item, (str, int, float))]


_THREAD_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


def _objects(situation: Any, key: str) -> list[dict[str, Any]]:
    if not isinstance(situation, dict):
        return []
    raw = situation.get(key)
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        return []
    return [item for item in raw if isinstance(item, dict)]


def _thread_id(value: object) -> str:
    candidate = _clean(value)
    return candidate if _THREAD_ID.fullmatch(candidate) else ""


def extract_threads(
    situation: Any,
    *,
    creator_id: str,
    fan_id: str,
    source_turn_id: str = "",
    source_message_fingerprint: str = "",
    evidence_type: EvidenceType = EvidenceType.STATED,
    confidence: float = 1.0,
) -> ExtractionResult:
    """Validate what the analyzer proposed into records worth storing.

    Total: it never raises. Extraction feeding a reply must not be able to stop
    one, and a failure here costs a memory, while a failure in the reply costs a
    conversation.
    """
    result = ExtractionResult()
    if not isinstance(situation, dict):
        return result

    # A degraded analysis extracted nothing it can vouch for. Recording its
    # proposals anyway would be writing durable memory from a reading the
    # analyzer itself reports as unreliable.
    if str(situation.get("analysis_degraded") or "").strip().lower() == "true":
        result._reject("analysis_degraded")
        return result

    for key, kind, raised_by in EXTRACTION_FIELDS:
        kept = 0
        proposals: list[tuple[str, str, str]] = [
            (summary, "", "") for summary in _proposals(situation, key)
        ]
        if key == "corrections_stated":
            proposals.extend(
                (
                    _clean(item.get("summary") or item.get("corrected_to")),
                    _thread_id(item.get("supersedes_thread_id")),
                    _clean(item.get("evidence")),
                )
                for item in _objects(situation, key)
            )
        for summary, replaces_thread_id, evidence in proposals:
            if not summary:
                result._reject("empty")
                continue
            if len(summary) > MAX_SUMMARY_CHARS:
                result._reject("too_long")
                continue
            if mentions_money(summary):
                # The rule the prompt asks for and this enforces. The analyzer
                # reads customer-supplied text, and customer-supplied text is
                # exactly what would try to make it assert a payment.
                result._reject("monetary")
                continue
            if kept >= MAX_PER_KIND:
                result._reject("over_limit")
                continue
            kept += 1
            thread = OpenThread(
                creator_id=str(creator_id),
                fan_id=str(fan_id),
                kind=kind,
                raised_by=raised_by,
                summary=summary,
                evidence_type=evidence_type,
                confidence=max(0.0, min(1.0, float(confidence))),
                source_turn_id=str(source_turn_id or ""),
                source_message_fingerprint=str(source_message_fingerprint or ""),
            )
            if replaces_thread_id and evidence:
                result.supersessions.append(
                    SupersessionProposal(
                        thread=thread,
                        replaces_thread_id=replaces_thread_id,
                        evidence=evidence[:MAX_SUMMARY_CHARS],
                    )
                )
            else:
                # Missing or malformed references never suppress anything. The
                # new correction remains visible so an operator can reconcile
                # it instead of the system silently choosing between records.
                result.threads.append(thread)
                if replaces_thread_id and not evidence:
                    result._reject("missing_supersession_evidence")

    for proposal in _objects(situation, "threads_resolved"):
        thread_id = _thread_id(proposal.get("thread_id"))
        evidence = _clean(proposal.get("evidence"))
        if not thread_id:
            result._reject("invalid_thread_reference")
            continue
        if not evidence or len(evidence) > MAX_SUMMARY_CHARS:
            result._reject("empty" if not evidence else "too_long")
            continue
        if mentions_money(evidence):
            # Resolving a monetary obligation from prose is the same mistake in
            # the other direction: "he says he got it" is not delivery.
            result._reject("monetary_resolution")
            continue
        result.resolved.append(ResolutionProposal(thread_id, evidence))

    # Older/unstructured analyzer output is explicitly refused.  It can be
    # logged and measured, but never fuzzy-matched against stored summaries.
    for _summary in _proposals(situation, "threads_resolved"):
        result._reject("unscoped_resolution")

    return result
