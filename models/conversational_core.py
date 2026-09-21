"""Versioned, evidence-grounded working state for ``conversational_v1``.

This is an interpretation of the current interaction, never a replacement for
raw messages, creator configuration, or transaction ledgers.  Every established
element therefore carries an immutable epistemic source class and references to
the evidence that supported it.  Superseded elements remain in the state so a
correction changes what is current without erasing what was previously believed.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

CORE_STATE_SCHEMA_VERSION = "conversational_core_v1"


class EpistemicType(str, Enum):
    EXPLICIT_FAN_STATEMENT = "explicit_fan_statement"
    CREATOR_CONFIG = "creator_config"
    MODEL_INFERENCE = "model_inference"
    SCENE_ASSUMPTION = "scene_assumption"
    SHARED_IMAGINED = "shared_imagined"
    TRANSACTION_FACT = "transaction_fact"


class WorldScope(str, Enum):
    CONVERSATION = "conversation"
    IMAGINED_SCENE = "imagined_scene"
    PRESENT_WORLD = "present_world"
    TRANSACTION = "transaction"


class ElementStatus(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"


class InitiativeHolder(str, Enum):
    FAN = "fan"
    CREATOR = "creator"
    SHARED = "shared"


class Pacing(str, Enum):
    BUILD = "build"
    HOLD = "hold"
    CONTINUE = "continue"
    COOL = "cool"
    REDIRECT = "redirect"
    PAUSE = "pause"
    RESUME = "resume"


class IntimacyRegister(str, Enum):
    """Descriptive content register, never a progression ladder."""

    NONE = "none"
    FLIRTY = "flirty"
    SUGGESTIVE = "suggestive"
    EXPLICIT = "explicit"


class IntimacySceneMode(str, Enum):
    """How intimate content is situated in the conversation."""

    NONE = "none"
    CONVERSATIONAL = "conversational"
    SHARED_IMAGINED = "shared_imagined"


class IntimacyContext(BaseModel):
    """Independent intimate-continuity dimensions.

    These fields describe the current interaction so a later turn can preserve
    or cool it.  They never authorize escalation, media, price, or a sale.
    """

    model_config = ConfigDict(extra="forbid")

    active: bool = False
    content_register: IntimacyRegister = IntimacyRegister.NONE
    scene_mode: IntimacySceneMode = IntimacySceneMode.NONE
    direction: Pacing = Pacing.CONTINUE
    last_beat: str = Field(default="", max_length=400)
    boundaries: list[str] = Field(default_factory=list, max_length=8)


class EstablishedElement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    element_id: str = Field(min_length=1, max_length=100)
    claim: str = Field(min_length=1, max_length=500)
    source_type: EpistemicType
    source_refs: list[str] = Field(default_factory=list, max_length=8)
    world_scope: WorldScope = WorldScope.CONVERSATION
    status: ElementStatus = ElementStatus.ACTIVE
    superseded_by: str = Field(default="", max_length=100)
    introduced_turn_ref: str = Field(default="", max_length=200)

    @field_validator("claim")
    @classmethod
    def _tidy_claim(cls, value: str) -> str:
        return " ".join(value.split())

    @field_validator("source_refs")
    @classmethod
    def _tidy_refs(cls, values: list[str]) -> list[str]:
        return list(
            dict.fromkeys(str(value).strip() for value in values if str(value).strip())
        )[:8]


class ActiveScene(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # This prose is explicitly an interpretation. It is not an evidence item
    # and may never be cited as a source_ref by a later turn.
    summary: str = Field(default="", max_length=1000)
    summary_source_type: EpistemicType = EpistemicType.MODEL_INFERENCE
    established_elements: list[EstablishedElement] = Field(
        default_factory=list, max_length=40
    )
    current_action_focus: str = Field(default="", max_length=300)
    unresolved_possibilities: list[str] = Field(default_factory=list, max_length=12)
    current_direction: str = Field(default="", max_length=300)
    pacing: Pacing = Pacing.CONTINUE
    has_shared_imagined_scene: bool = False

    @field_validator("summary_source_type")
    @classmethod
    def _summary_is_always_interpretation(cls, value: EpistemicType) -> EpistemicType:
        if value is not EpistemicType.MODEL_INFERENCE:
            raise ValueError("a scene summary is always model_inference")
        return value


class ConversationFlow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    initiative_holder: InitiativeHolder = InitiativeHolder.SHARED
    current_focus: str = Field(default="", max_length=300)
    unresolved_possibilities: list[str] = Field(default_factory=list, max_length=12)
    active_thread_ids: list[str] = Field(default_factory=list, max_length=12)
    participation_gist: str = Field(default="", max_length=300)


class ConversationalWorkingState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = CORE_STATE_SCHEMA_VERSION
    revision: int = Field(default=0, ge=0)
    active_scene: ActiveScene = Field(default_factory=ActiveScene)
    flow: ConversationFlow = Field(default_factory=ConversationFlow)
    intimacy: IntimacyContext = Field(default_factory=IntimacyContext)

    @field_validator("schema_version")
    @classmethod
    def _known_version(cls, value: str) -> str:
        if value != CORE_STATE_SCHEMA_VERSION:
            raise ValueError(f"unsupported working-state version {value!r}")
        return value

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class ProposedElement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    element_id: str = Field(min_length=1, max_length=100)
    claim: str = Field(min_length=1, max_length=500)
    source_type: EpistemicType
    source_refs: list[str] = Field(default_factory=list, max_length=8)
    world_scope: WorldScope = WorldScope.CONVERSATION


class ProposedCorrection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    replaces_element_id: str = Field(min_length=1, max_length=100)
    replacement: ProposedElement


class WorkingStateDelta(BaseModel):
    """Optional owner-authored changes; absence means preserve current state."""

    model_config = ConfigDict(extra="forbid")

    scene_summary: str | None = Field(default=None, max_length=1000)
    current_action_focus: str | None = Field(default=None, max_length=300)
    current_direction: str | None = Field(default=None, max_length=300)
    pacing: Pacing | None = None
    has_shared_imagined_scene: bool | None = None
    initiative_holder: InitiativeHolder | None = None
    current_focus: str | None = Field(default=None, max_length=300)
    participation_gist: str | None = Field(default=None, max_length=300)
    add_unresolved_possibilities: list[str] = Field(default_factory=list, max_length=12)
    resolve_unresolved_possibilities: list[str] = Field(
        default_factory=list, max_length=12
    )
    active_thread_ids: list[str] | None = Field(default=None, max_length=12)

    # Adult/intimate continuity is deliberately multidimensional.  These
    # descriptors may move in any direction on any turn; they are not stages
    # and must never be interpreted as "advance to the next level".
    intimacy_active: bool | None = None
    intimacy_content_register: IntimacyRegister | None = None
    intimacy_scene_mode: IntimacySceneMode | None = None
    intimacy_direction: Pacing | None = None
    intimacy_last_beat: str | None = Field(default=None, max_length=400)
    intimacy_boundaries: list[str] | None = Field(default=None, max_length=8)

    add_established_elements: list[ProposedElement] = Field(
        default_factory=list, max_length=12
    )
    corrections: list[ProposedCorrection] = Field(default_factory=list, max_length=8)


class StateDeltaValidation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    proposed: dict[str, Any] = Field(default_factory=dict)
    accepted_fields: list[str] = Field(default_factory=list)
    rejected_fields: dict[str, str] = Field(default_factory=dict)
    state_after: ConversationalWorkingState
