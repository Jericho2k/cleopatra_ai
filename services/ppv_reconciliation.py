"""Restart-safe verification and expiry handling for locked PPV messages."""
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import httpx

from core.supabase import get_supabase
from db.commercial_queries import (
    get_creator_policy,
    get_fan_state,
    save_fan_state,
    schedule_action,
)
from db.queries import get_fan_session, save_fan_session
from models.commercial import FanStatus
from services.followup_lifecycle import (
    abandoned_ppv_payload,
    followup_at,
    next_reconcile_at,
    payment_expires_at,
    pending_reference,
)
from services.session_lifecycle import mark_step_declined


class PPVReconcileDisposition(str, Enum):
    PURCHASED = "PURCHASED"
    PENDING = "PENDING"
    ABANDONED = "ABANDONED"
    STALE = "STALE"


@dataclass(frozen=True)
class PPVReconcileResult:
    disposition: PPVReconcileDisposition
    reason: str
    retry_at: datetime | None = None


def _matching_purchase(transactions: list[dict], expected_price: float) -> float | None:
    for transaction in transactions:
        if transaction.get("type") != 2110:
            continue
        gross = float(transaction.get("totalGross") or 0)
        if abs(gross - expected_price) / max(expected_price, 1.0) < 0.1:
            return gross
    return None


async def _load_platform_context(creator_id: str, fan_id: str) -> tuple[dict, dict]:
    def _load() -> tuple[dict, dict]:
        db = get_supabase()
        creator = (
            db.table("creators")
            .select("apifansly_account_id, fansly_account_id")
            .eq("id", creator_id)
            .single()
            .execute()
        ).data or {}
        fan = (
            db.table("fans")
            .select("platform_fan_id, not_sold_log, pending_ppv_check")
            .eq("id", fan_id)
            .single()
            .execute()
        ).data or {}
        return creator, fan

    return await asyncio.to_thread(_load)


async def _fetch_purchase_amount(
    *,
    apifansly_id: str,
    platform_fan_id: str,
    pending: dict[str, Any],
) -> float | None:
    api_key = os.environ.get("APIFANSLY_API_KEY")
    if not apifansly_id or not platform_fan_id or not api_key:
        raise RuntimeError("Fansly purchase verification is not configured")

    sent_at = pending.get("sent_at")
    try:
        sent_dt = datetime.fromisoformat(str(sent_at).replace("Z", "+00:00"))
        after_ms = int(sent_dt.timestamp() * 1000)
    except (TypeError, ValueError):
        raise RuntimeError("Pending PPV has no valid sent_at") from None

    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"https://v1.apifansly.com/api/fansly/{apifansly_id}/earnings/fans/{platform_fan_id}/stats",
            headers={"x-api-key": api_key},
            params={"after": after_ms},
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()
    transactions = data.get("data", {}).get("data", {}).get("response", [])
    return _matching_purchase(transactions, float(pending.get("price") or 0))


async def _persist_pending_check(
    fan_id: str,
    pending: dict[str, Any],
) -> None:
    def _update() -> None:
        get_supabase().table("fans").update({"pending_ppv_check": pending}).eq(
            "id", fan_id
        ).execute()

    await asyncio.to_thread(_update)


async def _finalize_abandonment(
    *,
    creator_id: str,
    fan_id: str,
    pending: dict[str, Any],
    fan_row: dict,
    now: datetime,
) -> bool:
    reference = pending_reference(pending)
    media_id = str(pending.get("media_id") or "")
    expected_price = float(pending.get("price") or 0)

    def _update_fan() -> bool:
        db = get_supabase()
        latest = (
            db.table("fans")
            .select("pending_ppv_check, not_sold_log")
            .eq("id", fan_id)
            .single()
            .execute()
        ).data or {}
        current = latest.get("pending_ppv_check") or {}
        if not current or pending_reference(current) != reference:
            return False
        not_sold = list(latest.get("not_sold_log") or fan_row.get("not_sold_log") or [])
        if not any(str(entry.get("payment_reference") or "") == reference for entry in not_sold):
            not_sold.append({
                "date": now.strftime("%d.%m.%Y"),
                "item": f"PPV media {media_id}",
                "media_id": media_id,
                "amount": expected_price,
                "reason": "payment window expired without a confirmed unlock",
                "payment_reference": reference,
                "chatter": "AI",
            })
        db.table("fans").update({
            "not_sold_log": not_sold,
            "pending_ppv_check": None,
        }).eq("id", fan_id).execute()
        return True

    if not await asyncio.to_thread(_update_fan):
        return False

    session = await get_fan_session(fan_id)
    if session and session.get("awaiting_purchase_index") is not None:
        session = mark_step_declined(
            session,
            reason="payment_window_expired",
            pause=False,
        )
        # Preserve the exact plan for a delayed platform unlock. Auto mode treats
        # an abandoned snapshot as non-executable and may replace it only after a
        # new explicit package selection.
        await save_fan_session(fan_id, session)

    policy = await get_creator_policy(creator_id)
    state = await get_fan_state(fan_id)
    state.status = FanStatus.OFFER_PENDING if state.offered_packages else FanStatus.IDLE
    state.last_declined_price_cents = int(round(expected_price * 100))
    state.last_abandoned_ppv_at = now
    state.last_abandoned_media_id = media_id or None

    if policy.abandoned_ppv_followup_enabled:
        execute_at = followup_at(now, policy.abandoned_ppv_followup_delay_hours)
        payload = abandoned_ppv_payload(
            pending,
            desired_experience=state.desired_experience,
            selected_package_id=state.selected_package_id,
        )
        dedupe_key = f"abandoned-ppv:{fan_id}:{reference}"
        state.next_followup_at = execute_at
        state.next_followup_type = "ABANDONED_PPV_FOLLOWUP"
        state.next_followup_payload = payload
        state.next_followup_dedupe_key = dedupe_key
    else:
        state.next_followup_at = None
        state.next_followup_type = None
        state.next_followup_payload = {}
        state.next_followup_dedupe_key = None
    await save_fan_state(fan_id, creator_id, state)
    if policy.abandoned_ppv_followup_enabled:
        try:
            await schedule_action(
                creator_id=creator_id,
                fan_id=fan_id,
                action_type="ABANDONED_PPV_FOLLOWUP",
                execute_at=state.next_followup_at,
                payload=state.next_followup_payload,
                dedupe_key=state.next_followup_dedupe_key,
            )
        except Exception as exc:
            # The persisted obligation is authoritative; the worker repairs the
            # missing action after a restart or transient database failure.
            print(
                f"[FOLLOWUP REPAIR NEEDED] fan={fan_id} "
                f"type=ABANDONED_PPV_FOLLOWUP error={exc}"
            )
    return True


async def reconcile_pending_ppv(
    *,
    creator_id: str,
    fan_id: str,
    expected_reference: str | None = None,
    now: datetime | None = None,
) -> PPVReconcileResult:
    now = now or datetime.now(timezone.utc)
    policy = await get_creator_policy(creator_id)
    creator_row, fan_row = await _load_platform_context(creator_id, fan_id)
    pending = fan_row.get("pending_ppv_check") or {}
    if not pending:
        return PPVReconcileResult(PPVReconcileDisposition.STALE, "no pending PPV")

    reference = pending_reference(pending)
    if expected_reference and reference != expected_reference:
        return PPVReconcileResult(
            PPVReconcileDisposition.STALE,
            "pending PPV was superseded",
        )

    amount = await _fetch_purchase_amount(
        apifansly_id=str(creator_row.get("apifansly_account_id") or ""),
        platform_fan_id=str(fan_row.get("platform_fan_id") or ""),
        pending=pending,
    )
    if amount is not None:
        from services.suggestions import record_ppv_purchase

        await record_ppv_purchase(fan_id, str(pending.get("media_id") or ""), amount)
        return PPVReconcileResult(PPVReconcileDisposition.PURCHASED, "purchase confirmed")

    expires_at = payment_expires_at(
        pending,
        payment_window_hours=policy.ppv_payment_window_hours,
    )
    if expires_at is None:
        raise RuntimeError("Pending PPV has no deterministic expiry")

    if now < expires_at:
        pending.update({
            "reference": reference,
            "expires_at": expires_at.isoformat(),
            "last_verified_at": now.isoformat(),
            "verification_attempts": int(pending.get("verification_attempts") or 0) + 1,
        })
        await _persist_pending_check(fan_id, pending)
        return PPVReconcileResult(
            PPVReconcileDisposition.PENDING,
            "not purchased yet; payment window remains open",
            retry_at=next_reconcile_at(
                now,
                expires_at=expires_at,
                recheck_minutes=policy.ppv_recheck_minutes,
            ),
        )

    finalized = await _finalize_abandonment(
        creator_id=creator_id,
        fan_id=fan_id,
        pending=pending,
        fan_row=fan_row,
        now=now,
    )
    if not finalized:
        return PPVReconcileResult(
            PPVReconcileDisposition.STALE,
            "pending PPV changed during expiry reconciliation",
        )
    return PPVReconcileResult(
        PPVReconcileDisposition.ABANDONED,
        "payment window expired without confirmed purchase",
    )
