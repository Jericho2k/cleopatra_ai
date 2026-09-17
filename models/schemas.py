"""All Pydantic models for Cleopatra. Nothing else in this file."""

from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, field_validator


class StageType(str, Enum):
    COLD_OPEN = "COLD_OPEN"
    WARMING_UP = "WARMING_UP"
    FLIRTING = "FLIRTING"
    PRE_UPSELL = "PRE_UPSELL"
    UPSELL_ACTIVE = "UPSELL_ACTIVE"
    OBJECTION = "OBJECTION"
    RETENTION = "RETENTION"
    HIGH_VALUE = "HIGH_VALUE"


class Fan(BaseModel):
    id: str
    display_name: str
    auto_mode: bool | None = None  # None = inherit creator setting
    platform_fan_id: str | None = None
    fansly_group_id: str | None = None
    total_spent: int = 0
    spend_tier: str = "cold"  # whale | active | casual | cold
    needs_human_review: bool = False
    sale_paused_at: str | None = None
    last_active: datetime | None = None
    preferences: list[str] = []
    notes: str = ""
    member_note: str = ""
    model_note: str = ""
    ai_summary: dict | None = None
    pre_session_qual: dict | None = None


class Creator(BaseModel):
    id: str
    name: str


class Persona(BaseModel):
    avg_message_length: str = "short"  # short | medium | long
    sends_multiple_messages: bool = False
    emoji_usage: str = "moderate"  # none | rare | moderate | heavy
    signature_emojis: list[str] = []
    vocabulary: list[str] = []
    capitalization: str = "mixed"  # none | lowercase | mixed | normal
    punctuation_style: str = ""
    flirt_style: str = ""
    upsell_style: str = ""
    example_greetings: list[str] = []
    example_flirts: list[str] = []
    dont_list: list[str] = []
    character: str = ""
    communication_style: str = ""
    example_phrases: str = ""
    hard_limits: str = ""
    emoji_style: str = ""
    voice_calibration_enabled: bool = False
    voice_calibration_samples: list[str] = Field(default_factory=list)
    voice_calibration_message_ids: list[str] = Field(default_factory=list)

    @field_validator("voice_calibration_samples")
    @classmethod
    def normalize_voice_calibration_samples(cls, values: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for value in values or []:
            sample = " ".join(str(value or "").split()).strip()
            key = sample.casefold()
            if not sample or key in seen:
                continue
            if len(sample) > 500:
                sample = sample[:500].rstrip()
            normalized.append(sample)
            seen.add(key)
            if len(normalized) == 30:
                break
        return normalized

    @field_validator("voice_calibration_message_ids")
    @classmethod
    def normalize_voice_calibration_message_ids(cls, values: list[str]) -> list[str]:
        return list(dict.fromkeys(str(value).strip() for value in (values or []) if str(value).strip()))[:30]


class Message(BaseModel):
    role: str  # "fan" | "creator"
    content: str
    sent_at: datetime | None = None
    media_context: dict | None = None


class ExchangeExample(BaseModel):
    """One RAG example: a past fan message and creator reply."""

    fan_message: str
    creator_reply: str


class ConversationContext(BaseModel):
    """Full context for building the suggestion prompt."""

    fan_message: str
    conversation_history: list[Message]
    fan_profile: Fan
    creator_persona: Persona
    similar_exchanges: list[ExchangeExample]
    conversation_stage: StageType
    creator_name: str = "a creator"
    situation: dict | None = None
    ppv_offers: list[dict] = []
    sent_ppv: list[dict] = []
    active_session: dict | None = None
    creator_legend: dict = {}
    commercial_decision: dict | None = None
    fan_intelligence: dict = Field(default_factory=dict)
    buyer_lifecycle: dict = Field(default_factory=dict)
    affordability: dict = Field(default_factory=dict)
    price_learning: dict = Field(default_factory=dict)
    session_strategy: dict = Field(default_factory=dict)
    conversation_director: dict = Field(default_factory=dict)
    # Deterministic bubble-count target for this turn (services/message_shape.py).
    message_shape: dict = Field(default_factory=dict)
    # The authoritative statement of what media actually exists for this turn
    # (services/inventory_authority.py). The writer is told; it never infers.
    media_inventory: dict = Field(default_factory=dict)
    # Set when the backend will attach a locked PPV to this turn's message
    # (services/ppv_turn.py). Writer-safe: asset type, how many pieces, the
    # price and the approved description — never media ids, and never a tag to
    # serialise. Empty means nothing is being attached.
    ppv_delivery: dict = Field(default_factory=dict)
    # The persistent conversational scene (services/experience_director.py),
    # already narrowed to its writer-safe projection: no set id, no counters
    # and no money. It decides WHEN something may happen; commercial_decision
    # still decides whether it may happen at all.
    scene: dict = Field(default_factory=dict)
    # How sexual this reply may be (services/text_intimacy.py). Separate from
    # the commercial decision on purpose — "nothing to sell this turn" is not a
    # reason to stop talking to him like an adult.
    text_intimacy: dict = Field(default_factory=dict)

    # Which AI Stack Profile is answering this turn, and which writer voice it
    # selects (ai/stack_profiles.py, ai/writer_style.py). Resolved once per turn
    # and carried here so the writer, the telemetry and the persisted message
    # metadata cannot disagree about which brain produced the reply.
    ai_stack_profile: str = "cleo_legacy_v1"
    writer_prompt_version: str = "writer_v1"

    # What this conversation is still carrying, already rendered to one line
    # each (services/conversation_continuity.py). Given their own allowance in
    # the context packet so recent small talk cannot evict an unanswered
    # question — docs/autonomy_architecture_review.md §4. Empty is the
    # pre-Sprint-2 behaviour exactly.
    open_threads: list[str] = Field(default_factory=list)
    # What earlier stretches of this conversation were about. Never proof of
    # payment; ppv_deliveries is the authority on money.
    conversation_episodes: list[str] = Field(default_factory=list)


class SuggestionRequest(BaseModel):
    """Request body for the suggestion API."""

    fan_id: str
    creator_id: str
    message: str


class SuggestionResponse(BaseModel):
    """Response: one to three reply options.

    COST-001 — this used to require exactly three. Combined with the writer's
    "must produce three survivors" rule it meant a turn that yielded one good
    reply raised a ValidationError and returned a 500, so the operator saw an
    error rather than the usable suggestion. One is a usable answer; the
    dashboard already renders a variable-length list.

    Zero is still rejected. Full Auto must fail closed rather than send filler,
    and an empty list is a failure, not a response.
    """

    suggestions: list[str]
    stage: StageType = StageType.WARMING_UP
    # REL-001 — true when the situation analysis behind these suggestions was
    # fabricated because the analyzer failed. Assisted still returns copy (an
    # operator reads it before anything is sent), but it must not be presented
    # as a normally analysed suggestion.
    analysis_degraded: bool = False
    analysis_degraded_reason: str = ""
    # An opaque handle to this turn's provenance record, held in memory by the
    # backend until the operator sends one of these candidates
    # (services/reply_provenance.py). The dashboard returns it on POST /reply so
    # the sent message can be attributed to the turn, the context and the model
    # attempt that actually produced it. Empty when provenance is unavailable;
    # the dashboard treats it as opaque and never displays it.
    suggestion_token: str = ""

    @field_validator("suggestions")
    @classmethod
    def between_one_and_three(cls, v: list[str]) -> list[str]:
        if not 1 <= len(v) <= 3:
            raise ValueError("suggestions must contain between 1 and 3 items")
        return v
