"""Typed commercial layer.

The dialogue model decides what the fan MEANT. This layer decides what the
business DOES. Keeping these separate is the point: business rules must not
depend on an LLM correctly following a paragraph of prompt instructions.
"""
from datetime import datetime
from enum import Enum
from typing import Literal

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
    OFFER_SELECTED = "OFFER_SELECTED"
    PAYMENT_PENDING = "PAYMENT_PENDING"
    PAID_SESSION_ACTIVE = "PAID_SESSION_ACTIVE"
    PAUSED_NO_BUDGET = "PAUSED_NO_BUDGET"
    PAUSED_UNTIL_PAYDAY = "PAUSED_UNTIL_PAYDAY"
    HUMAN_REVIEW = "HUMAN_REVIEW"


class EventType(str, Enum):
    """Typed observations extracted from the fan's message.

    These intentionally separate package acceptance, present affordability and a
    future payday. A fan can accept a cheaper package *and* mention payday in the
    same message; that must not collapse into a generic decline.
    """

    WANTS_EXPLICIT = "WANTS_EXPLICIT"
    WANTS_MEDIA = "WANTS_MEDIA"
    MONEY_UNAVAILABLE = "MONEY_UNAVAILABLE"  # cannot buy any available option now
    MONEY_AVAILABLE = "MONEY_AVAILABLE"
    PAYDAY_MENTIONED = "PAYDAY_MENTIONED"
    BUDGET_STATED = "BUDGET_STATED"  # voluntarily states an amount available now
    BUDGET_LIMIT_STATED = "BUDGET_LIMIT_STATED"  # accepts/limits current spend to X
    COUNTEROFFER_STATED = "COUNTEROFFER_STATED"  # explicit negotiated amount, not an offered package
    PACKAGE_SELECTED = "PACKAGE_SELECTED"
    OFFER_DETAILS_REQUESTED = "OFFER_DETAILS_REQUESTED"
    OFFER_SELECTION_AMBIGUOUS = "OFFER_SELECTION_AMBIGUOUS"
    OFFER_DECLINED = "OFFER_DECLINED"
    DEFERRED_PURCHASE = "DEFERRED_PURCHASE"
    READY_TO_BUY = "READY_TO_BUY"
    PURCHASED = "PURCHASED"
    CRISIS = "CRISIS"


class CommercialEvent(BaseModel):
    type: EventType
    raw_expression: str = ""
    confidence: float = 1.0
    amount_cents: int | None = None
    package_position: Literal["first", "second"] | None = None
    metadata: dict = Field(default_factory=dict)


class PackageOption(BaseModel):
    """A real package backed by one or more approved vault sets.

    ``set_id`` is retained for backward compatibility and points to the first
    step. ``set_ids`` is authoritative for multi-step sessions.
    """

    package_id: str
    label: str
    price_cents: int
    set_id: str | None = None
    set_ids: list[str] = Field(default_factory=list)
    experience: str | None = None
    legal_description: str | None = None


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
    quick_package_target_cents: int = 2500
    full_package_target_cents: int = 6000
    session_min_steps: int = 2
    session_max_steps: int = 4
    post_purchase_cooldown_messages: int = 2
    require_purchase_before_next_step: bool = True


class FanCommercialState(BaseModel):
    """Durable per-fan commercial state. Source of truth — NOT ai_summary."""

    status: FanStatus = FanStatus.IDLE
    desired_experience: str | None = None
    preferences_snapshot: dict = Field(default_factory=dict)

    # CONFIRMED only. We never store or optimize against an inferred spend ceiling.
    confirmed_budget_cents: int | None = None
    budget_source: str | None = None  # fan_explicit | package_selected

    # Exact ordered offer snapshot. While OFFER_PENDING this list is immutable
    # except when a brand-new approved offer is intentionally presented.
    offered_packages: list[PackageOption] = Field(default_factory=list)
    selected_package_id: str | None = None
    selected_package_set_id: str | None = None
    selected_package_set_ids: list[str] = Field(default_factory=list)
    selected_package_label: str | None = None
    selected_package_price_cents: int | None = None
    last_offer_at: datetime | None = None

    payday_raw: str | None = None
    payday_at: datetime | None = None
    payday_confidence: float | None = None

    last_declined_price_cents: int | None = None
    teaser_messages_used: int = 0
    free_session_started_at: datetime | None = None
    free_session_ended_at: datetime | None = None
    last_session_completed_at: datetime | None = None
    last_session_revenue_cents: int = 0


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
    """The decided action handed to the writer.

    The response-shape fields are deterministic constraints. They prevent a
    business decision such as PAUSE_UNTIL_PAYDAY from being followed by an
    awkward stock question merely because the generic voice prompt says to keep
    every conversation moving.
    """

    action: ActionType
    goal: str = ""
    must_not_send_media: bool = True
    may_be_explicit: bool = False
    mention_price: int | None = None
    package_options: list[PackageOption] = Field(default_factory=list)
    mention_previous_interest: bool = False
    tone: str = ""
    new_status: FanStatus | None = None
    schedule_payday_followup: bool = False
    session_budget_cents: int | None = None
    selected_package_set_id: str | None = None
    selected_package_set_ids: list[str] = Field(default_factory=list)

    must_not_ask_question: bool = False
    max_messages: int | None = None
    conversation_continuation: Literal["required", "optional", "none"] = "optional"

    reason: str = ""
