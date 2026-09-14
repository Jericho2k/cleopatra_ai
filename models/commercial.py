"""Typed commercial layer.

The dialogue model decides what the fan MEANT. This layer decides what the
business DOES. Keeping these separate is the point: business rules must not
depend on an LLM correctly following a paragraph of prompt instructions.
"""
from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator


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

    These intentionally separate offer acceptance, present affordability and a
    future payday. A fan can accept the offer *and* mention payday in the same
    message; that must not collapse into a generic decline.
    """

    WANTS_EXPLICIT = "WANTS_EXPLICIT"
    WANTS_MEDIA = "WANTS_MEDIA"
    MONEY_UNAVAILABLE = "MONEY_UNAVAILABLE"  # cannot buy what is on the table now
    MONEY_AVAILABLE = "MONEY_AVAILABLE"
    PAYDAY_MENTIONED = "PAYDAY_MENTIONED"
    BUDGET_STATED = "BUDGET_STATED"  # voluntarily states an amount available now
    BUDGET_LIMIT_STATED = "BUDGET_LIMIT_STATED"  # accepts/limits current spend to X
    COUNTEROFFER_STATED = "COUNTEROFFER_STATED"  # explicit negotiated amount below the offer
    OFFER_ACCEPTED = "OFFER_ACCEPTED"
    OFFER_DETAILS_REQUESTED = "OFFER_DETAILS_REQUESTED"
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
    metadata: dict = Field(default_factory=dict)


class Offer(BaseModel):
    """ONE next paid unlock, backed by exactly one approved vault set.

    There is deliberately no plural here and no notion of position. The fan is
    never shown a menu, never told how many further steps might follow, and
    never quoted a session total: he sees the next thing and its price. What
    might come after it is internal choreography (services/media_packages.py
    plans it from the same approved rows) and is not part of this object,
    precisely so it cannot leak into the prompt.
    """

    offer_id: str
    label: str
    price_cents: int
    set_id: str
    experience: str | None = None
    legal_description: str | None = None

    media_count: int = 0
    asset_type: str = "photo_set"

    # Provenance of the price. Approved content bounds first, fan probe second.
    content_floor_cents: int | None = None
    content_ceiling_cents: int | None = None
    price_reason_codes: list[str] = Field(default_factory=list)

    @property
    def set_ids(self) -> list[str]:
        """The delivery path plans in step lists; one offer is one step."""
        return [self.set_id]

    @property
    def asset_types(self) -> list[str]:
        return [self.asset_type]

    @property
    def includes_video(self) -> bool:
        return self.asset_type == "video"


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
    # How much approved content the NEXT unlock should be sized around. It is a
    # content-size hint, not a price: the price comes from the approved range of
    # the set that ends up chosen and from where this fan should be probed
    # inside it. There is exactly one of these because there is exactly one next
    # offer; the pair of "quick" and "full" budgets it replaces existed only to
    # build the two-branch menu.
    next_offer_target_cents: int = 2500
    post_purchase_cooldown_messages: int = 2
    require_purchase_before_next_step: bool = True
    require_operator_ppv_approval: bool = False
    ppv_recheck_minutes: int = Field(default=20, ge=5, le=1_440)
    ppv_payment_window_hours: int = Field(default=2, ge=1, le=168)
    abandoned_ppv_followup_enabled: bool = True
    abandoned_ppv_followup_delay_hours: int = Field(default=18, ge=1, le=720)
    pending_offer_expiry_hours: int = Field(default=24, ge=1, le=168)
    abandoned_offer_followup_enabled: bool = True
    abandoned_offer_followup_delay_hours: int = Field(default=18, ge=1, le=720)
    post_session_followup_enabled: bool = True
    post_session_followup_delay_hours: int = Field(default=18, ge=1, le=720)
    followup_recent_activity_suppression_hours: int = Field(default=6, ge=0, le=168)
    inactivity_reengagement_enabled: bool = False
    inactivity_reengagement_delay_hours: int = Field(default=48, ge=6, le=720)
    inactivity_reengagement_cooldown_hours: int = Field(default=168, ge=24, le=2_160)
    inactivity_reengagement_max_per_30_days: int = Field(default=2, ge=1, le=10)

    @model_validator(mode="after")
    def enforce_purchase_gating_invariants(self) -> "CreatorPolicy":
        """Media access and multi-step progression are never optional in v1.

        Older rows and clients may still submit these legacy fields as false.
        Normalizing them here prevents a stale dashboard from weakening the
        purchase-gated delivery contract.
        """
        self.media_always_paid = True
        self.require_purchase_before_next_step = True
        return self


class FanCommercialState(BaseModel):
    """Durable per-fan commercial state. Source of truth — NOT ai_summary."""

    status: FanStatus = FanStatus.IDLE
    desired_experience: str | None = None
    preferences_snapshot: dict = Field(default_factory=dict)

    # CONFIRMED only. We never store or optimize against an inferred spend ceiling.
    confirmed_budget_cents: int | None = None
    budget_source: str | None = None  # fan_explicit | offer_accepted

    # The exact offer currently on the table, or None. Singular: the fan is
    # shown one next unlock, so there is one snapshot to hold him to, and no
    # ordinal for him to pick from. While OFFER_PENDING it is immutable except
    # when a brand-new approved offer is intentionally presented.
    pending_offer: Offer | None = None
    accepted_offer_id: str | None = None
    accepted_offer_set_id: str | None = None
    accepted_offer_label: str | None = None
    accepted_offer_price_cents: int | None = None
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
    last_session_offer_id: str | None = None
    last_session_set_ids: list[str] = Field(default_factory=list)
    last_session_experience: str | None = None
    last_abandoned_ppv_at: datetime | None = None
    last_abandoned_media_id: str | None = None
    next_followup_at: datetime | None = None
    next_followup_type: str | None = None
    next_followup_payload: dict = Field(default_factory=dict)
    next_followup_dedupe_key: str | None = None
    last_followup_at: datetime | None = None
    last_inactivity_reengagement_at: datetime | None = None
    inactivity_reengagement_window_started_at: datetime | None = None
    inactivity_reengagement_count: int = 0


class ActionType(str, Enum):
    """What the policy engine decides. The generator only expresses these.

    There is one commercial forward move — ``OFFER_NEXT_UNLOCK`` — and one
    delivery move — ``SEND_NEXT_PPV_STEP``. The three actions they replace
    (``PRESENT_SESSION_OPTIONS``, ``END_TEASER_AND_OFFER`` and
    ``CREATE_PAID_SESSION``) all existed to run the same menu: present two
    branches, wait for a choice, confirm the choice, then ask again before
    sending. Acceptance now moves straight to delivery.
    """

    CONTINUE_NORMAL_CHAT = "CONTINUE_NORMAL_CHAT"
    CONTINUE_FREE_TEXT = "CONTINUE_FREE_TEXT"
    START_FREE_TEASER = "START_FREE_TEASER"
    DISCOVER_DESIRED_EXPERIENCE = "DISCOVER_DESIRED_EXPERIENCE"
    OFFER_NEXT_UNLOCK = "OFFER_NEXT_UNLOCK"
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
    # The ONE next unlock this decision may talk about, or None. Never a list:
    # the fan is not choosing between branches, and a decision that carried two
    # was the thing that produced "quick $60 or full $140?".
    next_offer: Offer | None = None
    mention_previous_interest: bool = False
    tone: str = ""
    new_status: FanStatus | None = None
    schedule_payday_followup: bool = False
    session_budget_cents: int | None = None
    accepted_offer_set_id: str | None = None

    must_not_ask_question: bool = False
    max_messages: int | None = None
    conversation_continuation: Literal["required", "optional", "none"] = "optional"

    # Authoritative media capabilities for this turn. The writer is TOLD what
    # exists; it never infers it from a tag, a title or the fan's request.
    # ``authorized`` is what this decision itself authorises, ``vault`` what
    # exists in approved unsent inventory at all.
    authorized_asset_types: list[str] = Field(default_factory=list)
    vault_asset_types: list[str] = Field(default_factory=list)

    # Set when the fan explicitly asked for a media type that is not available.
    # A deterministic pivot, never a stall and never a fabricated promise.
    unavailable_asset_type_requested: str | None = None

    # Set when the exact offer the fan accepted can no longer be delivered and
    # a replacement is being presented instead of silently substituted.
    replacement_for_unavailable: bool = False

    reason: str = ""
