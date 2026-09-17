"""Typed records for what a conversation is still carrying.

``docs/autonomy_architecture_review.md`` §3E: the existing memory is real, and
the review says so — fan-fact evidence validation, historical compaction, saved
creator facts, persistent scene state. What none of it holds is the state of the
interaction:

    why a topic matters, which question remains unanswered, whether a
    misunderstanding was repaired, and whether a prior invitation to continue is
    still current.

An ``OpenThread`` is one piece of that: something unfinished, with the condition
that would finish it. A ``ConversationEpisode`` is a completed stretch, summarised
with the range of messages it covers.

Two rules from §4 are enforced in the types rather than left to callers:

*Evidence types do not merge.* ``EvidenceType`` separates what someone said from
what was read out of what they said, from what an external system confirmed.
"The customer said payment succeeded" and "the platform confirmed order X" are
different facts and must stay different.

*A correction supersedes; it does not coexist.* ``ThreadStatus.SUPERSEDED`` plus
``superseded_by`` is the only way a thread stops being current because a newer
one replaced it. Nothing is deleted, so the history of what changed survives —
which is what lets a later reply avoid reasserting something already corrected.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class ThreadKind(str, Enum):
    """What kind of unfinished business this is.

    Exactly the five the review names. A wider vocabulary is how fan_facts
    became a CRM schema; these describe a conversation, not a customer record.
    """

    QUESTION = "question"
    PROMISE = "promise"
    DEFERRED_TOPIC = "deferred_topic"
    COMPLAINT = "complaint"
    CORRECTION = "correction"


class ThreadParty(str, Enum):
    """Who is waiting on whom.

    A question the customer asked and one the creator asked are different
    obligations. Answering the wrong one is one of the failures review §1 asks a
    test to catch ("Forgotten obligation or wrong referent").
    """

    FAN = "fan"
    CREATOR = "creator"


class ThreadStatus(str, Enum):
    OPEN = "open"
    FULFILLED = "fulfilled"
    CANCELLED = "cancelled"
    SUPERSEDED = "superseded"
    EXPIRED = "expired"


class ResolvedBy(str, Enum):
    FAN_MESSAGE = "fan_message"
    CREATOR_REPLY = "creator_reply"
    OPERATOR = "operator"
    EXPIRY = "expiry"
    SUPERSESSION = "supersession"


class EvidenceType(str, Enum):
    """Where a record came from, and therefore how much it may be relied on."""

    #: Someone said it, in the conversation.
    STATED = "stated"
    #: Read out of the conversation rather than said outright.
    INFERRED = "inferred"
    #: An external system asserted it. The only kind that may stand for an
    #: operational fact — and even then, money stays with ppv_deliveries.
    PLATFORM_CONFIRMED = "platform_confirmed"
    #: A human recorded it.
    OPERATOR = "operator"


class EpisodeEnding(str, Enum):
    """How a stretch of conversation ended, in the conversation's own terms.

    Not a commercial outcome. Review §1 asks for judgment about silence — a
    goodbye respected, a quiet period noticed — and that needs the ending
    recorded as what it was rather than as whether a sale happened.
    """

    RESOLVED = "resolved"
    WENT_QUIET = "went_quiet"
    SAID_GOODBYE = "said_goodbye"
    INTERRUPTED = "interrupted"
    UNKNOWN = "unknown"


#: The longest a summary may be. Threads are rendered into a prompt together, so
#: one verbose thread would spend the budget the others need.
SUMMARY_MAX_CHARS = 400
EPISODE_SUMMARY_MAX_CHARS = 800


class OpenThread(BaseModel):
    """One unfinished thing, and what would finish it."""

    creator_id: str
    fan_id: str
    kind: ThreadKind
    raised_by: ThreadParty
    summary: str = Field(min_length=1, max_length=SUMMARY_MAX_CHARS)
    resolution_condition: str = Field(default="", max_length=SUMMARY_MAX_CHARS)
    status: ThreadStatus = ThreadStatus.OPEN
    evidence_type: EvidenceType = EvidenceType.STATED
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    #: Provenance back to the turn that raised it. A fingerprint, not the text —
    #: the same discipline as services/reply_provenance.py, for the same reason.
    source_message_fingerprint: str = ""
    source_turn_id: str = ""
    evidence_text: str = Field(default="", max_length=500)

    id: str = ""
    first_seen_at: datetime | None = None
    last_seen_at: datetime | None = None
    expires_at: datetime | None = None
    resolved_at: datetime | None = None
    resolved_by: ResolvedBy | None = None
    resolution_note: str = Field(default="", max_length=SUMMARY_MAX_CHARS)
    superseded_by: str = ""

    @field_validator("summary", "resolution_condition", "resolution_note")
    @classmethod
    def _tidy(cls, value: str) -> str:
        return " ".join(str(value or "").split())

    @property
    def is_open(self) -> bool:
        return self.status == ThreadStatus.OPEN

    @property
    def is_obligation_on_us(self) -> bool:
        """Whether the next move is the creator's.

        A question the fan asked, a promise the creator made, and a complaint
        the fan raised are all waiting on us. A question the creator asked is
        waiting on him, and pressing it again is nagging rather than continuity.
        """
        if self.kind == ThreadKind.QUESTION:
            return self.raised_by == ThreadParty.FAN
        if self.kind == ThreadKind.PROMISE:
            return self.raised_by == ThreadParty.CREATOR
        if self.kind == ThreadKind.COMPLAINT:
            return self.raised_by == ThreadParty.FAN
        return False

    def render(self) -> str:
        """One line for a prompt. No ids, no timestamps, no internal vocabulary."""
        who = "he" if self.raised_by == ThreadParty.FAN else "you"
        lead = {
            ThreadKind.QUESTION: f"{who} asked",
            ThreadKind.PROMISE: f"{who} said",
            ThreadKind.DEFERRED_TOPIC: f"{who} put off",
            ThreadKind.COMPLAINT: f"{who} raised",
            ThreadKind.CORRECTION: f"{who} corrected",
        }[self.kind]
        line = f"{lead}: {self.summary}"
        if self.resolution_condition:
            line = f"{line} (closes when: {self.resolution_condition})"
        return line

    def to_row(self, *, dedupe_key: str) -> dict[str, Any]:
        """The insert payload. Only the columns a new thread sets."""
        row: dict[str, Any] = {
            "creator_id": self.creator_id,
            "fan_id": self.fan_id,
            "kind": self.kind.value,
            "raised_by": self.raised_by.value,
            "summary": self.summary,
            "status": self.status.value,
            "evidence_type": self.evidence_type.value,
            "confidence": float(self.confidence),
            "dedupe_key": dedupe_key,
        }
        if self.resolution_condition:
            row["resolution_condition"] = self.resolution_condition
        if self.source_message_fingerprint:
            row["source_message_fingerprint"] = self.source_message_fingerprint
        if self.source_turn_id:
            row["source_turn_id"] = self.source_turn_id
        if self.evidence_text:
            row["evidence_text"] = self.evidence_text
        if self.expires_at is not None:
            row["expires_at"] = self.expires_at.isoformat()
        return row

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "OpenThread":
        """Read a row back, tolerating a value this build does not know.

        A row written by a newer deployment must not raise here. An unreadable
        kind or status degrades to something harmless rather than taking down a
        turn: continuity is an input to a reply, never a precondition for one.
        """

        def _enum(enum_cls, value, default):
            try:
                return enum_cls(str(value))
            except ValueError:
                return default

        return cls(
            id=str(row.get("id") or ""),
            creator_id=str(row.get("creator_id") or ""),
            fan_id=str(row.get("fan_id") or ""),
            kind=_enum(ThreadKind, row.get("kind"), ThreadKind.DEFERRED_TOPIC),
            raised_by=_enum(ThreadParty, row.get("raised_by"), ThreadParty.FAN),
            summary=str(row.get("summary") or "")[:SUMMARY_MAX_CHARS] or "unspecified",
            resolution_condition=str(row.get("resolution_condition") or "")[
                :SUMMARY_MAX_CHARS
            ],
            status=_enum(ThreadStatus, row.get("status"), ThreadStatus.OPEN),
            evidence_type=_enum(
                EvidenceType, row.get("evidence_type"), EvidenceType.INFERRED
            ),
            confidence=float(row.get("confidence") or 1.0),
            source_message_fingerprint=str(row.get("source_message_fingerprint") or ""),
            source_turn_id=str(row.get("source_turn_id") or ""),
            evidence_text=str(row.get("evidence_text") or "")[:500],
            first_seen_at=_parse_time(row.get("first_seen_at")),
            last_seen_at=_parse_time(row.get("last_seen_at")),
            expires_at=_parse_time(row.get("expires_at")),
            resolved_at=_parse_time(row.get("resolved_at")),
            resolved_by=(
                _enum(ResolvedBy, row.get("resolved_by"), None)
                if row.get("resolved_by")
                else None
            ),
            resolution_note=str(row.get("resolution_note") or "")[:SUMMARY_MAX_CHARS],
            superseded_by=str(row.get("superseded_by") or ""),
        )


class ConversationEpisode(BaseModel):
    """What a completed stretch of conversation was about, and how it ended.

    Carries no amount, no price and no purchase flag, by design. The review is
    unambiguous that an episode is "never proof of payment", and a summary that
    could be read as a receipt is how a model comes to believe in a purchase
    that did not happen.
    """

    creator_id: str
    fan_id: str
    summary: str = Field(min_length=1, max_length=EPISODE_SUMMARY_MAX_CHARS)
    ended_with: EpisodeEnding = EpisodeEnding.UNKNOWN
    first_message_at: datetime
    last_message_at: datetime
    message_count: int = Field(default=0, ge=0)
    evidence_type: EvidenceType = EvidenceType.INFERRED
    id: str = ""

    @field_validator("summary")
    @classmethod
    def _tidy(cls, value: str) -> str:
        return " ".join(str(value or "").split())

    def render(self) -> str:
        """One line for a prompt, with when rather than how much."""
        when = self.last_message_at.date().isoformat() if self.last_message_at else ""
        ending = {
            EpisodeEnding.RESOLVED: "sorted out",
            EpisodeEnding.WENT_QUIET: "he went quiet",
            EpisodeEnding.SAID_GOODBYE: "he said goodbye",
            EpisodeEnding.INTERRUPTED: "it was cut short",
            EpisodeEnding.UNKNOWN: "",
        }[self.ended_with]
        line = f"{when}: {self.summary}" if when else self.summary
        return f"{line} ({ending})" if ending else line

    def to_row(self, *, dedupe_key: str) -> dict[str, Any]:
        return {
            "creator_id": self.creator_id,
            "fan_id": self.fan_id,
            "summary": self.summary,
            "ended_with": self.ended_with.value,
            "first_message_at": self.first_message_at.isoformat(),
            "last_message_at": self.last_message_at.isoformat(),
            "message_count": int(self.message_count),
            "evidence_type": self.evidence_type.value,
            "dedupe_key": dedupe_key,
        }

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "ConversationEpisode":
        first = _parse_time(row.get("first_message_at"))
        last = _parse_time(row.get("last_message_at"))

        def _enum(enum_cls, value, default):
            try:
                return enum_cls(str(value))
            except ValueError:
                return default

        return cls(
            id=str(row.get("id") or ""),
            creator_id=str(row.get("creator_id") or ""),
            fan_id=str(row.get("fan_id") or ""),
            summary=str(row.get("summary") or "")[:EPISODE_SUMMARY_MAX_CHARS]
            or "unspecified",
            ended_with=_enum(
                EpisodeEnding, row.get("ended_with"), EpisodeEnding.UNKNOWN
            ),
            first_message_at=first or last,
            last_message_at=last or first,
            message_count=int(row.get("message_count") or 0),
            evidence_type=_enum(
                EvidenceType, row.get("evidence_type"), EvidenceType.INFERRED
            ),
        )


def _parse_time(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
