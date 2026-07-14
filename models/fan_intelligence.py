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


class ValidatedObservation(BaseModel):
    category: FactCategory
    fact_key: str
    value_json: Any
    normalized_value: str
    certainty: FactCertainty
    confidence: float
    evidence_text: str
    source_type: str = "fan_message"


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
