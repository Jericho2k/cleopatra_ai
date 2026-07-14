"""Supabase persistence for the immutable affordability ledger and snapshot."""

from __future__ import annotations

import asyncio
from typing import Any

from core.supabase import get_supabase
from models.affordability import AffordabilityEvent, AffordabilityState, state_from_row


async def get_affordability_state(fan_id: str) -> AffordabilityState:
    def _get() -> dict[str, Any] | None:
        response = (
            get_supabase()
            .table("fan_affordability_states")
            .select("*")
            .eq("fan_id", fan_id)
            .limit(1)
            .execute()
        )
        return (response.data or [None])[0]

    try:
        return state_from_row(await asyncio.to_thread(_get))
    except Exception as exc:
        print(f"[AFFORDABILITY] state read failed fan={fan_id}: {exc}")
        return AffordabilityState()


async def insert_affordability_event(
    *,
    creator_id: str,
    fan_id: str,
    event: AffordabilityEvent,
    dedupe_key: str,
) -> bool:
    """Insert once and return True only when this event was newly persisted."""

    payload = {
        "creator_id": creator_id,
        "fan_id": fan_id,
        "event_type": event.event_type.value,
        "authority": event.authority.value,
        "amount_cents": event.amount_cents,
        "raw_expression": event.raw_expression or None,
        "confidence": event.confidence,
        "occurred_at": event.occurred_at.isoformat(),
        "expires_at": event.expires_at.isoformat() if event.expires_at else None,
        "source_message_id": event.source_message_id,
        "source_ref": event.source_ref,
        "metadata": event.metadata,
        "dedupe_key": dedupe_key,
    }

    def _insert() -> bool:
        db = get_supabase()
        existing = (
            db.table("fan_affordability_events")
            .select("id")
            .eq("dedupe_key", dedupe_key)
            .limit(1)
            .execute()
        )
        if existing.data:
            return False
        try:
            db.table("fan_affordability_events").insert(payload).execute()
            return True
        except Exception:
            existing_after = (
                db.table("fan_affordability_events")
                .select("id")
                .eq("dedupe_key", dedupe_key)
                .limit(1)
                .execute()
            )
            if existing_after.data:
                return False
            raise

    return await asyncio.to_thread(_insert)


async def save_affordability_state(
    *, creator_id: str, fan_id: str, state: AffordabilityState
) -> None:
    payload = {
        "creator_id": creator_id,
        "fan_id": fan_id,
        "status": state.status.value,
        "current_available_cents": state.current_available_cents,
        "current_limit_cents": state.current_limit_cents,
        "current_signal_expires_at": (
            state.current_signal_expires_at.isoformat()
            if state.current_signal_expires_at
            else None
        ),
        "temporary_constraint": state.temporary_constraint,
        "constraint_until": (
            state.constraint_until.isoformat() if state.constraint_until else None
        ),
        "payday_raw": state.payday_raw,
        "payday_at": state.payday_at.isoformat() if state.payday_at else None,
        "payday_confidence": state.payday_confidence,
        "latest_offer_selected_cents": state.latest_offer_selected_cents,
        "latest_counteroffer_cents": state.latest_counteroffer_cents,
        "latest_rejected_price_cents": state.latest_rejected_price_cents,
        "last_confirmed_purchase_cents": state.last_confirmed_purchase_cents,
        "highest_confirmed_purchase_cents": state.highest_confirmed_purchase_cents,
        "confirmed_purchase_count": state.confirmed_purchase_count,
        "confirmed_purchase_total_cents": state.confirmed_purchase_total_cents,
        "last_confirmed_purchase_at": (
            state.last_confirmed_purchase_at.isoformat()
            if state.last_confirmed_purchase_at
            else None
        ),
        "reason_codes": state.reason_codes,
        "state_version": state.state_version,
        "updated_at": state.updated_at.isoformat(),
    }
    await asyncio.to_thread(
        lambda: get_supabase()
        .table("fan_affordability_states")
        .upsert(payload, on_conflict="fan_id")
        .execute()
    )


async def get_affordability_events(
    fan_id: str, *, limit: int = 50
) -> list[dict[str, Any]]:
    def _get() -> list[dict[str, Any]]:
        response = (
            get_supabase()
            .table("fan_affordability_events")
            .select("*")
            .eq("fan_id", fan_id)
            .order("occurred_at", desc=True)
            .limit(max(1, min(limit, 200)))
            .execute()
        )
        return list(response.data or [])

    return await asyncio.to_thread(_get)
