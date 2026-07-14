"""Natural budget discovery and evidence-backed affordability state.

The service consumes deterministic commercial events. It never asks an LLM to
estimate wealth and never turns a purchase into a permanent budget ceiling.
"""

from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from typing import Iterable

from db.affordability_queries import (
    get_affordability_state,
    insert_affordability_event,
    save_affordability_state,
)
from models.affordability import (
    AffordabilityAuthority,
    AffordabilityEvent,
    AffordabilityEventType,
    AffordabilityState,
    apply_affordability_event,
)
from models.commercial import CommercialEvent, EventType
from services.commercial_events import extract_events
from services.payday import resolve_payday


def affordability_enabled() -> bool:
    return os.getenv("AFFORDABILITY_ENABLED", "").lower() in {"1", "true", "yes"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)) or default)
    except (TypeError, ValueError):
        return default


async def get_affordability_context(fan_id: str) -> dict:
    if not affordability_enabled():
        return {}
    state = await get_affordability_state(fan_id)
    return state.to_context()


async def refresh_affordability_from_situation(
    *,
    creator_id: str,
    fan_id: str,
    situation: dict,
    source_ref: str | None = None,
    source_message_id: str | None = None,
    offered_price_cents: int | None = None,
    timezone_name: str = "UTC",
    payday_send_hour: int = 18,
) -> dict:
    if not affordability_enabled():
        return {}
    events = extract_events(situation)
    safe_source_ref = _safe_source_ref(source_ref)
    context = await refresh_affordability_from_events(
        creator_id=creator_id,
        fan_id=fan_id,
        events=events,
        source_ref=safe_source_ref,
        source_message_id=source_message_id,
        offered_price_cents=offered_price_cents,
        timezone_name=timezone_name,
        payday_send_hour=payday_send_hour,
    )
    situation["affordability"] = context
    return context


async def refresh_affordability_from_events(
    *,
    creator_id: str,
    fan_id: str,
    events: Iterable[CommercialEvent],
    source_ref: str | None = None,
    source_message_id: str | None = None,
    offered_price_cents: int | None = None,
    timezone_name: str = "UTC",
    payday_send_hour: int = 18,
) -> dict:
    if not affordability_enabled():
        return {}

    state = await get_affordability_state(fan_id)
    changed = False
    now = datetime.now(timezone.utc)

    for commercial_event in events:
        ledger_event = _to_affordability_event(
            commercial_event,
            occurred_at=now,
            source_ref=source_ref,
            source_message_id=source_message_id,
            offered_price_cents=offered_price_cents,
            timezone_name=timezone_name,
            payday_send_hour=payday_send_hour,
        )
        if ledger_event is None:
            continue
        dedupe_key = _dedupe_key(fan_id, ledger_event)
        inserted = await insert_affordability_event(
            creator_id=creator_id,
            fan_id=fan_id,
            event=ledger_event,
            dedupe_key=dedupe_key,
        )
        if not inserted:
            continue
        state = apply_affordability_event(
            state,
            ledger_event,
            current_signal_ttl_hours=max(
                1, _env_int("AFFORDABILITY_CURRENT_SIGNAL_TTL_HOURS", 24)
            ),
            constraint_ttl_hours=max(
                1, _env_int("AFFORDABILITY_CONSTRAINT_TTL_HOURS", 72)
            ),
        )
        changed = True

    if changed:
        await save_affordability_state(
            creator_id=creator_id,
            fan_id=fan_id,
            state=state,
        )
        print(
            f"[AFFORDABILITY] fan={fan_id} status={state.status.value} "
            f"available={state.current_available_cents} "
            f"limit={state.current_limit_cents} "
            f"constraint={state.temporary_constraint}"
        )

    return state.to_context()


async def record_confirmed_purchase(
    *,
    creator_id: str,
    fan_id: str,
    amount_cents: int,
    source_ref: str,
    occurred_at: datetime | None = None,
    metadata: dict | None = None,
) -> dict:
    """Record authoritative payment evidence idempotently."""

    if not affordability_enabled():
        return {}
    event = AffordabilityEvent(
        event_type=AffordabilityEventType.PURCHASE_CONFIRMED,
        authority=AffordabilityAuthority.PAYMENT_CONFIRMED,
        amount_cents=max(0, int(amount_cents)),
        occurred_at=occurred_at or datetime.now(timezone.utc),
        source_ref=source_ref,
        metadata=metadata or {},
    )
    state = await get_affordability_state(fan_id)
    inserted = await insert_affordability_event(
        creator_id=creator_id,
        fan_id=fan_id,
        event=event,
        dedupe_key=_dedupe_key(fan_id, event),
    )
    if inserted:
        state = apply_affordability_event(state, event)
        await save_affordability_state(
            creator_id=creator_id,
            fan_id=fan_id,
            state=state,
        )
    return state.to_context()


def _to_affordability_event(
    event: CommercialEvent,
    *,
    occurred_at: datetime,
    source_ref: str | None,
    source_message_id: str | None,
    offered_price_cents: int | None,
    timezone_name: str,
    payday_send_hour: int,
) -> AffordabilityEvent | None:
    common = {
        "authority": AffordabilityAuthority.CHAT_EXPLICIT,
        "raw_expression": event.raw_expression,
        "confidence": event.confidence,
        "occurred_at": occurred_at,
        "source_ref": source_ref,
        "source_message_id": source_message_id,
        "metadata": dict(event.metadata or {}),
    }

    if event.type == EventType.BUDGET_STATED:
        return AffordabilityEvent(
            event_type=AffordabilityEventType.CURRENT_AMOUNT_STATED,
            amount_cents=event.amount_cents,
            **common,
        )
    if event.type == EventType.BUDGET_LIMIT_STATED:
        return AffordabilityEvent(
            event_type=AffordabilityEventType.CURRENT_LIMIT_STATED,
            amount_cents=event.amount_cents,
            **common,
        )
    if event.type == EventType.PACKAGE_SELECTED:
        return AffordabilityEvent(
            event_type=AffordabilityEventType.OFFER_SELECTED,
            amount_cents=event.amount_cents,
            **common,
        )
    if event.type == EventType.COUNTEROFFER_STATED:
        return AffordabilityEvent(
            event_type=AffordabilityEventType.COUNTEROFFER_STATED,
            amount_cents=event.amount_cents,
            **common,
        )
    if event.type == EventType.OFFER_DECLINED:
        return AffordabilityEvent(
            event_type=AffordabilityEventType.OFFER_DECLINED,
            amount_cents=offered_price_cents,
            **common,
        )
    if event.type == EventType.MONEY_UNAVAILABLE:
        return AffordabilityEvent(
            event_type=AffordabilityEventType.MONEY_UNAVAILABLE,
            **common,
        )
    if event.type == EventType.MONEY_AVAILABLE:
        return AffordabilityEvent(
            event_type=AffordabilityEventType.MONEY_AVAILABLE,
            **common,
        )
    if event.type == EventType.PAYDAY_MENTIONED:
        payday_at, resolved_confidence = resolve_payday(
            event.raw_expression,
            send_hour=payday_send_hour,
            timezone_name=timezone_name,
        )
        metadata = dict(common["metadata"])
        metadata["payday_at"] = payday_at.isoformat() if payday_at else None
        return AffordabilityEvent(
            event_type=AffordabilityEventType.PAYDAY_MENTIONED,
            confidence=max(event.confidence, resolved_confidence),
            authority=AffordabilityAuthority.CHAT_EXPLICIT,
            raw_expression=event.raw_expression,
            occurred_at=occurred_at,
            source_ref=source_ref,
            source_message_id=source_message_id,
            metadata=metadata,
        )

    # A chat claim that something was purchased is never authoritative.
    return None


def _safe_source_ref(source_ref: str | None) -> str | None:
    if not source_ref:
        return None
    text = str(source_ref)
    if text.startswith(("assisted:", "auto:", "auto-decline:")):
        return "message:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
    return text[:500]


def _dedupe_key(fan_id: str, event: AffordabilityEvent) -> str:
    stable_source = event.source_message_id or event.source_ref
    if not stable_source:
        stable_source = event.occurred_at.strftime("%Y-%m-%dT%H")
    raw = "|".join(
        [
            fan_id,
            event.event_type.value,
            str(event.amount_cents or ""),
            event.raw_expression.strip().lower(),
            str(stable_source),
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
