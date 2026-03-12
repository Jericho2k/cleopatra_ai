"""All Pydantic models for Cleopatra. Nothing else in this file."""

from datetime import datetime
from enum import Enum
from pydantic import BaseModel, field_validator


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
    total_spent: int = 0
    spend_tier: str = "cold"  # whale | active | casual | cold
    last_active: datetime | None = None
    preferences: list[str] = []
    notes: str = ""


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


class Message(BaseModel):
    role: str  # "fan" | "creator"
    content: str
    sent_at: datetime | None = None


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


class SuggestionRequest(BaseModel):
    """Request body for the suggestion API."""

    fan_id: str
    creator_id: str
    message: str


class SuggestionResponse(BaseModel):
    """Response: exactly 3 reply options."""

    suggestions: list[str]

    @field_validator("suggestions")
    @classmethod
    def exactly_three(cls, v: list[str]) -> list[str]:
        if len(v) != 3:
            raise ValueError("suggestions must contain exactly 3 items")
        return v
