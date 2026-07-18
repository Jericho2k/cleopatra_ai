"""Durable, retry-safe persistence for a PPV that was already delivered.

Nothing in this module calls the platform delivery API. Retrying these
operations can repair local state without risking a duplicate PPV send.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any

from core.supabase import get_supabase
from db.commercial_queries import (
    get_creator_policy,
    get_fan_state,
    save_fan_state,
    schedule_action,
)
from db.queries import get_fan_session, save_fan_session, save_message
from models.commercial import FanStatus
from services.db_reliability import retry_transient_db_operation
from services.followup_lifecycle import next_reconcile_at
from services.session_lifecycle import mark_step_sent
from services.vault_operations import normalize_media_ids


async def save_ppv_message_receipt(
    *,
    fan_id: str,
    creator_id: str,
    content: str,
    was_ai_suggested: bool,
    platform_message_id: str | None,
    media_context: dict[str, Any],
) -> str | None:
    """Persist one local receipt, recovering an insert whose response was lost."""
    ppv = media_context.get("ppv") or {}
    reference = str(ppv.get("payment_reference") or "").strip()
    if not reference:
        raise ValueError("PPV receipt requires a payment reference")

    async def _ensure() -> str | None:
        def _existing() -> str | None:
            result = (
                get_supabase().table("messages")
                .select("id")
                .eq("fan_id", fan_id)
                .eq("creator_id", creator_id)
                .eq("role", "creator")
                .eq("media_context->ppv->>payment_reference", reference)
                .limit(1)
                .execute()
            )
            rows = result.data or []
            return str(rows[0]["id"]) if rows else None

        existing_id = await asyncio.to_thread(_existing)
        if existing_id:
            return existing_id
        return await save_message(
            fan_id=fan_id,
            creator_id=creator_id,
            role="creator",
            content=content,
            was_ai_suggested=was_ai_suggested,
            fansly_message_id=platform_message_id,
            media_context=media_context,
        )

    return await retry_transient_db_operation(
        _ensure,
        label=f"PPV message receipt fan={fan_id}",
        log_prefix="PPV PERSIST RETRY",
    )


async def persist_ppv_reconciliation(
    *,
    creator_id: str,
    fan_id: str,
    pending: dict[str, Any],
    session: dict[str, Any] | None = None,
    platform_message_id: str | None = None,
) -> tuple[dict[str, Any] | None, datetime]:
    """Attach a delivered PPV to payment state and a durable reconcile action."""
    reference = str(pending.get("reference") or "").strip()
    if not reference:
        raise ValueError("PPV reconciliation requires a payment reference")

    sent_at = datetime.fromisoformat(
        str(pending.get("sent_at") or "").replace("Z", "+00:00")
    )
    if sent_at.tzinfo is None:
        sent_at = sent_at.replace(tzinfo=timezone.utc)

    policy = await retry_transient_db_operation(
        lambda: get_creator_policy(creator_id),
        label=f"PPV policy fan={fan_id}",
        log_prefix="PPV PERSIST RETRY",
    )
    expires_at = sent_at + timedelta(hours=policy.ppv_payment_window_hours)
    pending = {
        **pending,
        "reference": reference,
        "expires_at": str(pending.get("expires_at") or expires_at.isoformat()),
        "verification_attempts": int(pending.get("verification_attempts") or 0),
        "platform_message_id": pending.get("platform_message_id") or platform_message_id,
    }

    async def _save_pending() -> None:
        await asyncio.to_thread(
            lambda: get_supabase().table("fans")
            .update({"pending_ppv_check": pending})
            .eq("id", fan_id)
            .execute()
        )

    await retry_transient_db_operation(
        _save_pending,
        label=f"pending PPV fan={fan_id}",
        log_prefix="PPV PERSIST RETRY",
    )

    if session is None:
        session = await retry_transient_db_operation(
            lambda: get_fan_session(fan_id),
            label=f"PPV session read fan={fan_id}",
            log_prefix="PPV PERSIST RETRY",
        )
    step_index = pending.get("step_index")
    if session and step_index is not None:
        session = mark_step_sent(
            session,
            step_index=int(step_index),
            message_id=pending.get("platform_message_id") or platform_message_id,
        )
        await retry_transient_db_operation(
            lambda: save_fan_session(fan_id, session),
            label=f"PPV session write fan={fan_id}",
            log_prefix="PPV PERSIST RETRY",
        )

    state = await retry_transient_db_operation(
        lambda: get_fan_state(fan_id),
        label=f"PPV commercial state read fan={fan_id}",
        log_prefix="PPV PERSIST RETRY",
    )
    state.status = FanStatus.PAYMENT_PENDING
    await retry_transient_db_operation(
        lambda: save_fan_state(fan_id, creator_id, state),
        label=f"PPV commercial state write fan={fan_id}",
        log_prefix="PPV PERSIST RETRY",
    )

    persisted_expires_at = datetime.fromisoformat(
        str(pending["expires_at"]).replace("Z", "+00:00")
    )
    reconcile_at = next_reconcile_at(
        datetime.now(timezone.utc),
        expires_at=persisted_expires_at,
        recheck_minutes=policy.ppv_recheck_minutes,
    )
    await retry_transient_db_operation(
        lambda: schedule_action(
            creator_id=creator_id,
            fan_id=fan_id,
            action_type="PPV_RECONCILE",
            execute_at=reconcile_at,
            payload={"payment_reference": reference},
            dedupe_key=f"ppv-reconcile:{fan_id}:{reference}",
        ),
        label=f"PPV reconcile action fan={fan_id}",
        log_prefix="PPV PERSIST RETRY",
    )
    return session, reconcile_at


def pending_from_message_receipt(
    message: dict[str, Any],
    *,
    payment_window_hours: int,
    local_test_fan: bool = False,
) -> dict[str, Any]:
    """Reconstruct the pending record from an immutable local PPV receipt."""
    ppv = (message.get("media_context") or {}).get("ppv") or {}
    reference = str(ppv.get("payment_reference") or "").strip()
    media_ids = normalize_media_ids(ppv.get("media_ids") or [ppv.get("media_id")])
    if not reference or not media_ids:
        raise ValueError("No recoverable PPV receipt was found")
    sent_at = datetime.fromisoformat(str(message.get("sent_at") or "").replace("Z", "+00:00"))
    if sent_at.tzinfo is None:
        sent_at = sent_at.replace(tzinfo=timezone.utc)
    price = float(ppv.get("price") or 0)
    platform_message_id = message.get("fansly_message_id")
    if not platform_message_id and local_test_fan:
        platform_message_id = f"local-test:{reference}"
    return {
        "reference": reference,
        "media_id": media_ids[0],
        "media_ids": media_ids,
        "set_id": ppv.get("set_id"),
        "step_index": ppv.get("step_index"),
        "price": price,
        "price_cents": int(ppv.get("price_cents") or round(price * 100)),
        "source": ppv.get("source") or "auto",
        "sent_at": sent_at.isoformat(),
        "expires_at": (sent_at + timedelta(hours=payment_window_hours)).isoformat(),
        "verification_attempts": 0,
        "platform_message_id": platform_message_id,
    }
