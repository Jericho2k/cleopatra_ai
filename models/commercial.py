"""Typed commercial layer.

The dialogue model decides what the fan MEANT. This layer decides what the
business DOES. Keeping these separate is the point: business rules must not
depend on an LLM correctly following a paragraph of prompt instructions.
"""
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class SextingMode(str, Enum):
    PAID_ONLY = "PAID_ONLY"
    HYBRID_TEASER = "HYBRID_TEASER"
    FREE_TEXT_ALLOWED = "FREE_TEXT_ALLOWED"


class FanStatus(str, Enum):
    IDLE = "IDLE"
    FREE_TEASER = "FREE_TEASER"
    FREE_TEXT_SESSION = "FREE_TEXT_SESSION"
    OFFER_PENDING = "OFFER_PENDING"
    PAID_SESSION_ACTIVE = "PAID_SESSION_ACTIVE"
    PAUSED_NO_BUDGET = "PAUSED_NO_BUDGET"
    PAUSED_UNTIL_PAYDAY = "PAUSED_UNTIL_PAYDAY"
    HUMAN_REVIEW = "HUMAN_REVIEW"


class EventType(str, Enum):
    """Typed events the analyzer extracts. These are observations, not decisions."""
    WANTS_EXPLICIT = "WANTS_EXPLICIT"          # asked to sext / strongly escalating
    WANTS_MEDIA = "WANTS_MEDIA"                # asked to see content
    MONEY_UNAVAILABLE = "MONEY_UNAVAILABLE"    # can't afford right now
    MONEY_AVAILABLE = "MONEY_AVAILABLE"        # got paid / can pay now
    PAYDAY_MENTIONED = "PAYDAY_MENTIONED"      # stated when money arrives
    BUDGET_STATED = "BUDGET_STATED"            # "I have $50 tonight"
    READY_TO_BUY = "READY_TO_BUY"
    PURCHASED = "PURCHASED"
    CRISIS = "CRISIS"


class CommercialEvent(BaseModel):
    type: EventType
    raw_expression: str = ""       # e.g. "Friday", "$50"
    confidence: float = 1.0


class CreatorPolicy(BaseModel):
    """Per-creator commercial policy. The agency's dials."""
    sexting_mode: SextingMode = SextingMode.HYBRID_TEASER
    teaser_max_messages: int = 4
    free_text_max_messages: int = 20
    free_session_cooldown_hours: int = 24
    media_always_paid: bool = True
    payday_reengagement_enabled: bool = True
    payday_send_hour_local: int = 18
    timezone: str = "UTC"
    offer_two_packages: bool = True


class FanCommercialState(BaseModel):
    """Durable per-fan commercial state. Source of truth — NOT ai_summary."""
    status: FanStatus = FanStatus.IDLE
    desired_experience: str | None = None
    preferences_snapshot: dict = Field(default_factory=dict)

    # CONFIRMED only. We never store or optimize against an inferred spend ceiling.
    confirmed_budget_cents: int | None = None
    budget_source: str | None = None   # fan_explicit | package_selected

    payday_raw: str | None = None
    payday_at: datetime | None = None
    payday_confidence: float | None = None

    last_declined_price_cents: int | None = None
    teaser_messages_used: int = 0
    free_session_started_at: datetime | None = None


class ActionType(str, Enum):
    """What the policy engine decides. The generator only expresses these."""
    CONTINUE_NORMAL_CHAT = "CONTINUE_NORMAL_CHAT"
    CONTINUE_FREE_TEXT = "CONTINUE_FREE_TEXT"
    START_FREE_TEASER = "START_FREE_TEASER"
    END_TEASER_AND_OFFER = "END_TEASER_AND_OFFER"
    ASK_ONE_QUALIFYING_QUESTION = "ASK_ONE_QUALIFYING_QUESTION"
    PRESENT_SESSION_OPTIONS = "PRESENT_SESSION_OPTIONS"
    CREATE_PAID_SESSION = "CREATE_PAID_SESSION"
    SEND_NEXT_PPV_STEP = "SEND_NEXT_PPV_STEP"
    PAUSE_NO_BUDGET = "PAUSE_NO_BUDGET"
    PAUSE_UNTIL_PAYDAY = "PAUSE_UNTIL_PAYDAY"
    RESUME_PREVIOUS_OFFER = "RESUME_PREVIOUS_OFFER"
    PAYDAY_REENGAGEMENT = "PAYDAY_REENGAGEMENT"
    HAND_OFF_TO_HUMAN = "HAND_OFF_TO_HUMAN"


class CommercialDecision(BaseModel):
    """The decided action, handed to the generator. Claude's ONLY job is to
    express this naturally in the creator's voice — it does not get to choose
    whether to sell, pause, tease, or schedule."""
    action: ActionType
    goal: str = ""                      # plain-English intent for the writer
    must_not_send_media: bool = True
    may_be_explicit: bool = False
    mention_price: int | None = None
    package_options: list[int] = Field(default_factory=list)  # e.g. [25, 60]
    mention_previous_interest: bool = False
    tone: str = ""
    new_status: FanStatus | None = None
    schedule_payday_followup: bool = False
    reason: str = ""                    # for logs/debugging, never shown to fans