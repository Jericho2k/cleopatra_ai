"""Typed buyer-lifecycle models and deterministic stage derivation.

The lifecycle engine never asks an LLM to decide whether somebody is a buyer,
repeat buyer, or VIP. Those labels come from authoritative purchase history,
commercial state, and explicit purchase-intent signals.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class BuyerLifecycleStage(str, Enum):
    PROSPECT = "PROSPECT"
    FIRST_PURCHASE_PROSPECT = "FIRST_PURCHASE_PROSPECT"
    FIRST_TIME_BUYER = "FIRST_TIME_BUYER"
    REPEAT_BUYER = "REPEAT_BUYER"
    VIP = "VIP"


class LifecyclePolicy(BaseModel):
    vip_spend_cents: int = Field(default=50_000, ge=0)
    vip_purchase_count: int = Field(default=5, ge=1)
    repeat_buyer_purchase_count: int = Field(default=2, ge=2)
    first_purchase_intent_ttl_hours: int = Field(default=72, ge=1, le=720)


class LifecycleInputs(BaseModel):
    purchase_count: int = Field(default=0, ge=0)
    purchase_revenue_cents: int = Field(default=0, ge=0)
    fan_total_spent_cents: int = Field(default=0, ge=0)
    first_purchase_at: datetime | None = None
    last_purchase_at: datetime | None = None
    purchase_intent_signal: bool = False
    existing_intent_expires_at: datetime | None = None
    active_paid_session: bool = False
    sales_paused: bool = False
    needs_human_review: bool = False
    now: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DerivedLifecycle(BaseModel):
    stage: BuyerLifecycleStage
    purchase_count: int
    purchase_revenue_cents: int
    total_spent_cents: int
    first_purchase_at: datetime | None = None
    last_purchase_at: datetime | None = None
    intent_expires_at: datetime | None = None
    flags: dict[str, bool] = Field(default_factory=dict)
    reason_codes: list[str] = Field(default_factory=list)

    def to_context(self) -> dict[str, Any]:
        return {
            "stage": self.stage.value,
            "purchase_count": self.purchase_count,
            "purchase_revenue_cents": self.purchase_revenue_cents,
            "total_spent_cents": self.total_spent_cents,
            "first_purchase_at": (
                self.first_purchase_at.isoformat() if self.first_purchase_at else None
            ),
            "last_purchase_at": (
                self.last_purchase_at.isoformat() if self.last_purchase_at else None
            ),
            "intent_expires_at": (
                self.intent_expires_at.isoformat() if self.intent_expires_at else None
            ),
            "flags": self.flags,
            "reason_codes": self.reason_codes,
        }


def _as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def derive_lifecycle(
    inputs: LifecycleInputs,
    policy: LifecyclePolicy,
) -> DerivedLifecycle:
    """Derive one buyer stage from authoritative, deterministic inputs."""

    now = _as_utc(inputs.now) or datetime.now(timezone.utc)
    purchase_count = max(0, int(inputs.purchase_count))
    purchase_revenue_cents = max(0, int(inputs.purchase_revenue_cents))
    total_spent_cents = max(
        purchase_revenue_cents,
        max(0, int(inputs.fan_total_spent_cents)),
    )

    intent_expires_at = _as_utc(inputs.existing_intent_expires_at)
    if purchase_count > 0:
        intent_expires_at = None
    elif inputs.purchase_intent_signal:
        intent_expires_at = now + timedelta(
            hours=policy.first_purchase_intent_ttl_hours
        )
    elif intent_expires_at is not None and intent_expires_at <= now:
        intent_expires_at = None

    reason_codes: list[str] = []

    vip_by_spend = total_spent_cents >= policy.vip_spend_cents
    vip_by_frequency = purchase_count >= policy.vip_purchase_count
    if vip_by_spend or vip_by_frequency:
        stage = BuyerLifecycleStage.VIP
        if vip_by_spend:
            reason_codes.append("vip_spend_threshold")
        if vip_by_frequency:
            reason_codes.append("vip_purchase_frequency")
    elif purchase_count >= policy.repeat_buyer_purchase_count:
        stage = BuyerLifecycleStage.REPEAT_BUYER
        reason_codes.append("repeat_purchase_count")
    elif purchase_count == 1:
        stage = BuyerLifecycleStage.FIRST_TIME_BUYER
        reason_codes.append("first_confirmed_purchase")
    elif intent_expires_at is not None:
        stage = BuyerLifecycleStage.FIRST_PURCHASE_PROSPECT
        reason_codes.append("active_first_purchase_intent")
    else:
        stage = BuyerLifecycleStage.PROSPECT
        reason_codes.append("no_confirmed_purchase")

    flags = {
        "active_paid_session": bool(inputs.active_paid_session),
        "sales_paused": bool(inputs.sales_paused),
        "needs_human_review": bool(inputs.needs_human_review),
    }

    return DerivedLifecycle(
        stage=stage,
        purchase_count=purchase_count,
        purchase_revenue_cents=purchase_revenue_cents,
        total_spent_cents=total_spent_cents,
        first_purchase_at=_as_utc(inputs.first_purchase_at),
        last_purchase_at=_as_utc(inputs.last_purchase_at),
        intent_expires_at=intent_expires_at,
        flags=flags,
        reason_codes=reason_codes,
    )
