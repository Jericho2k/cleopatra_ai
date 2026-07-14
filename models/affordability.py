"""Typed commercial-affordability ledger and deterministic state transitions.

This module intentionally keeps these concepts separate:
- what a fan says is available now;
- a current-session ceiling;
- an accepted offer that is not yet purchased;
- confirmed purchase evidence;
- temporary inability to spend;
- future liquidity such as payday.

Nothing in this model represents estimated wealth or a permanent spending ceiling.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class AffordabilityStatus(str, Enum):
    UNKNOWN = "UNKNOWN"
    AVAILABLE_NOW = "AVAILABLE_NOW"
    LIMITED_NOW = "LIMITED_NOW"
    TEMPORARILY_UNAVAILABLE = "TEMPORARILY_UNAVAILABLE"


class AffordabilityEventType(str, Enum):
    CURRENT_AMOUNT_STATED = "CURRENT_AMOUNT_STATED"
    CURRENT_LIMIT_STATED = "CURRENT_LIMIT_STATED"
    OFFER_SELECTED = "OFFER_SELECTED"
    COUNTEROFFER_STATED = "COUNTEROFFER_STATED"
    OFFER_DECLINED = "OFFER_DECLINED"
    MONEY_UNAVAILABLE = "MONEY_UNAVAILABLE"
    MONEY_AVAILABLE = "MONEY_AVAILABLE"
    PAYDAY_MENTIONED = "PAYDAY_MENTIONED"
    PURCHASE_CONFIRMED = "PURCHASE_CONFIRMED"


class AffordabilityAuthority(str, Enum):
    CHAT_EXPLICIT = "CHAT_EXPLICIT"
    PAYMENT_CONFIRMED = "PAYMENT_CONFIRMED"
    SYSTEM_DERIVED = "SYSTEM_DERIVED"


class AffordabilityEvent(BaseModel):
    event_type: AffordabilityEventType
    authority: AffordabilityAuthority = AffordabilityAuthority.CHAT_EXPLICIT
    amount_cents: int | None = Field(default=None, ge=0)
    raw_expression: str = ""
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime | None = None
    source_message_id: str | None = None
    source_ref: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AffordabilityState(BaseModel):
    status: AffordabilityStatus = AffordabilityStatus.UNKNOWN

    current_available_cents: int | None = Field(default=None, ge=0)
    current_limit_cents: int | None = Field(default=None, ge=0)
    current_signal_expires_at: datetime | None = None

    temporary_constraint: bool = False
    constraint_until: datetime | None = None

    payday_raw: str | None = None
    payday_at: datetime | None = None
    payday_confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    latest_offer_selected_cents: int | None = Field(default=None, ge=0)
    latest_counteroffer_cents: int | None = Field(default=None, ge=0)
    latest_rejected_price_cents: int | None = Field(default=None, ge=0)

    last_confirmed_purchase_cents: int | None = Field(default=None, ge=0)
    highest_confirmed_purchase_cents: int | None = Field(default=None, ge=0)
    confirmed_purchase_count: int = Field(default=0, ge=0)
    confirmed_purchase_total_cents: int = Field(default=0, ge=0)
    last_confirmed_purchase_at: datetime | None = None

    reason_codes: list[str] = Field(default_factory=list)
    state_version: int = 1
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def normalized(self, *, now: datetime | None = None) -> "AffordabilityState":
        current = _as_utc(now) or datetime.now(timezone.utc)
        state = self.model_copy(deep=True)

        if state.current_signal_expires_at:
            expires = _as_utc(state.current_signal_expires_at)
            if expires and expires <= current:
                state.current_available_cents = None
                state.current_limit_cents = None
                state.latest_offer_selected_cents = None
                state.current_signal_expires_at = None

        if state.temporary_constraint:
            until = _as_utc(state.constraint_until)
            if until and until <= current:
                state.temporary_constraint = False
                state.constraint_until = None

        state.status = _derive_status(state)
        state.reason_codes = _derive_reasons(state)
        return state

    def to_context(self, *, now: datetime | None = None) -> dict[str, Any]:
        state = self.normalized(now=now)
        return {
            "status": state.status.value,
            "current_available_cents": state.current_available_cents,
            "current_limit_cents": state.current_limit_cents,
            "current_signal_expires_at": _iso(state.current_signal_expires_at),
            "temporary_constraint": state.temporary_constraint,
            "constraint_until": _iso(state.constraint_until),
            "payday_raw": state.payday_raw,
            "payday_at": _iso(state.payday_at),
            "payday_confidence": state.payday_confidence,
            "latest_offer_selected_cents": state.latest_offer_selected_cents,
            "latest_counteroffer_cents": state.latest_counteroffer_cents,
            "latest_rejected_price_cents": state.latest_rejected_price_cents,
            "last_confirmed_purchase_cents": state.last_confirmed_purchase_cents,
            "highest_confirmed_purchase_cents": state.highest_confirmed_purchase_cents,
            "confirmed_purchase_count": state.confirmed_purchase_count,
            "confirmed_purchase_total_cents": state.confirmed_purchase_total_cents,
            "last_confirmed_purchase_at": _iso(state.last_confirmed_purchase_at),
            "reason_codes": state.reason_codes,
            "updated_at": _iso(state.updated_at),
        }


def apply_affordability_event(
    state: AffordabilityState,
    event: AffordabilityEvent,
    *,
    current_signal_ttl_hours: int = 24,
    constraint_ttl_hours: int = 72,
) -> AffordabilityState:
    """Apply one immutable ledger event to the current snapshot.

    The transition is conservative: purchases never become a permanent budget,
    an accepted offer is not a confirmed purchase, and a payday does not imply
    the fan is unable to pay now.
    """

    now = _as_utc(event.occurred_at) or datetime.now(timezone.utc)
    output = state.normalized(now=now)
    output = output.model_copy(deep=True)

    if event.event_type == AffordabilityEventType.CURRENT_AMOUNT_STATED:
        output.current_available_cents = event.amount_cents
        output.current_signal_expires_at = event.expires_at or (
            now + timedelta(hours=max(1, current_signal_ttl_hours))
        )
        output.temporary_constraint = False
        output.constraint_until = None

    elif event.event_type == AffordabilityEventType.CURRENT_LIMIT_STATED:
        output.current_limit_cents = event.amount_cents
        output.current_signal_expires_at = event.expires_at or (
            now + timedelta(hours=max(1, current_signal_ttl_hours))
        )
        output.temporary_constraint = False
        output.constraint_until = None

    elif event.event_type == AffordabilityEventType.OFFER_SELECTED:
        output.latest_offer_selected_cents = event.amount_cents
        output.current_signal_expires_at = event.expires_at or (
            now + timedelta(hours=max(1, current_signal_ttl_hours))
        )
        # Selection proves the selected option is presently viable. It does not
        # prove broader liquidity or create a permanent ceiling.
        output.temporary_constraint = False
        output.constraint_until = None

    elif event.event_type == AffordabilityEventType.COUNTEROFFER_STATED:
        output.latest_counteroffer_cents = event.amount_cents
        output.current_limit_cents = event.amount_cents
        output.current_signal_expires_at = event.expires_at or (
            now + timedelta(hours=max(1, current_signal_ttl_hours))
        )
        output.temporary_constraint = False
        output.constraint_until = None

    elif event.event_type == AffordabilityEventType.OFFER_DECLINED:
        if event.amount_cents is not None:
            output.latest_rejected_price_cents = event.amount_cents
        # A plain no is not proof of poverty and does not create a pause.

    elif event.event_type == AffordabilityEventType.MONEY_UNAVAILABLE:
        output.temporary_constraint = True
        output.constraint_until = event.expires_at or (
            now + timedelta(hours=max(1, constraint_ttl_hours))
        )

    elif event.event_type == AffordabilityEventType.MONEY_AVAILABLE:
        output.temporary_constraint = False
        output.constraint_until = None

    elif event.event_type == AffordabilityEventType.PAYDAY_MENTIONED:
        output.payday_raw = event.raw_expression or output.payday_raw
        output.payday_at = _as_utc(event.metadata.get("payday_at")) or output.payday_at
        output.payday_confidence = event.confidence
        # A payday mention alone does not mean the fan cannot buy now.
        if output.temporary_constraint and output.payday_at:
            output.constraint_until = output.payday_at

    elif event.event_type == AffordabilityEventType.PURCHASE_CONFIRMED:
        cents = int(event.amount_cents or 0)
        output.last_confirmed_purchase_cents = cents
        output.highest_confirmed_purchase_cents = max(
            int(output.highest_confirmed_purchase_cents or 0), cents
        )
        output.confirmed_purchase_count += 1
        output.confirmed_purchase_total_cents += cents
        output.last_confirmed_purchase_at = now
        output.temporary_constraint = False
        output.constraint_until = None

    output.updated_at = now
    output.status = _derive_status(output)
    output.reason_codes = _derive_reasons(output)
    return output


def state_from_row(row: dict[str, Any] | None) -> AffordabilityState:
    if not row:
        return AffordabilityState()
    payload = dict(row)
    for key in (
        "fan_id",
        "creator_id",
        "created_at",
    ):
        payload.pop(key, None)
    return AffordabilityState.model_validate(payload)


def _derive_status(state: AffordabilityState) -> AffordabilityStatus:
    if state.temporary_constraint:
        return AffordabilityStatus.TEMPORARILY_UNAVAILABLE
    if state.current_available_cents is not None or state.current_limit_cents is not None:
        return AffordabilityStatus.LIMITED_NOW
    if state.latest_offer_selected_cents is not None:
        return AffordabilityStatus.AVAILABLE_NOW
    return AffordabilityStatus.UNKNOWN


def _derive_reasons(state: AffordabilityState) -> list[str]:
    reasons: list[str] = []
    if state.temporary_constraint:
        reasons.append("temporary_cash_constraint")
    if state.current_available_cents is not None:
        reasons.append("explicit_current_amount")
    if state.current_limit_cents is not None:
        reasons.append("explicit_current_limit")
    if state.latest_offer_selected_cents is not None:
        reasons.append("offer_selected_pending_purchase")
    if state.payday_raw:
        reasons.append("future_liquidity_mentioned")
    if state.confirmed_purchase_count:
        reasons.append("confirmed_purchase_history")
    return reasons


def _as_utc(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _iso(value: datetime | None) -> str | None:
    parsed = _as_utc(value)
    return parsed.isoformat() if parsed else None
