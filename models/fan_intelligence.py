"""Typed passive fan-intelligence models.

The extractor proposes observations. Deterministic validation and merge logic decide
what becomes durable fan knowledge.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class FactCategory(str, Enum):
    IDENTITY = "identity"
    AVAILABILITY = "availability"
    PREFERENCE = "preference"
    BOUNDARY = "boundary"
    COMMERCIAL = "commercial"
    BEHAVIOR = "behavior"


class FactCertainty(str, Enum):
    EXPLICIT = "explicit"
    STRONG_INFERENCE = "strong_inference"


class FactStatus(str, Enum):
    INFERRED = "inferred"
    EXPLICIT = "explicit"
    CONFIRMED = "confirmed"
    CONTRADICTED = "contradicted"


class ProposedObservation(BaseModel):
    category: FactCategory
    fact_key: str
    value: Any
    certainty: FactCertainty
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: str

    @field_validator("fact_key", "evidence")
    @classmethod
    def strip_text(cls, value: str) -> str:
        return value.strip()


class ExtractionEnvelope(BaseModel):
    observations: list[ProposedObservation] = Field(default_factory=list)


class ProposedHistoricalObservation(ProposedObservation):
    """One fact a historical chunk proposes, with the message it came from.

    The extra field is the whole point. Live extraction has exactly one
    candidate source — the latest fan message — so the evidence quote can be
    checked against it implicitly. A historical chunk shows the model up to
    forty messages from both speakers, so a proposal that does not name WHICH
    message it came from cannot be verified at all, and one that names a
    creator message is a fact about the creator being laundered into the fan's
    record. Both are rejected deterministically; see
    services/fan_history_memory.validate_historical_observation.
    """

    source_message_id: str

    @field_validator("source_message_id")
    @classmethod
    def strip_source(cls, value: str) -> str:
        return value.strip()


class HistoricalExtractionEnvelope(BaseModel):
    """One chunk's proposals plus the compact continuity it observed."""

    observations: list[ProposedHistoricalObservation] = Field(default_factory=list)
    ongoing_topics: list[str] = Field(default_factory=list)
    commercial_context: list[str] = Field(default_factory=list)
    relationship_summary: str = ""


class ValidatedObservation(BaseModel):
    category: FactCategory
    fact_key: str
    value_json: Any
    normalized_value: str
    certainty: FactCertainty
    confidence: float
    evidence_text: str
    source_type: str = "fan_message"


# Where a piece of evidence came from. Historical evidence is real evidence —
# it is quoted from a message the fan actually sent — but it is OLDER than
# anything the live path has seen, which is why plan_fact_merge treats it as
# unable to overturn a current explicit or confirmed fact.
SOURCE_TYPE_FAN_MESSAGE = "fan_message"
SOURCE_TYPE_HISTORICAL_MESSAGE = "historical_message"


class MergeAction(str, Enum):
    CREATE = "create"
    REINFORCE = "reinforce"
    REPLACE_INFERRED = "replace_inferred"
    ADD_MULTI_VALUE = "add_multi_value"
    CONFLICT = "conflict"
    IGNORE = "ignore"


class MergePlan(BaseModel):
    action: MergeAction
    matched_fact_id: str | None = None
    conflicting_fact_ids: list[str] = Field(default_factory=list)
    reason: str = ""
