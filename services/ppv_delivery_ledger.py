"""Durable state transitions for locked-PPV delivery.

The partial unique index in ``ppv_delivery_ledger_v1.sql`` is the concurrency
boundary. A platform send must never happen before ``claim_delivery`` succeeds.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from core.supabase import get_supabase
from services.db_reliability import retry_transient_db_operation
from services.vault_operations import normalize_media_ids


ACTIVE_STATUSES = {"claimed", "delivered_pending"}
TERMINAL_STATUSES = {"purchased", "abandoned", "voided", "failed"}
ALL_STATUSES = ACTIVE_STATUSES | TERMINAL_STATUSES


class PPVDeliveryClaimError(RuntimeError):
    """Raised when an atomic PPV delivery claim cannot be acquired."""


async def claim_delivery(
    *,
    reference: str,
    creator_id: str,
    fan_id: str,
    media_ids: list[str],
    price_cents: int,
    source: str,
    set_id: str | None,
    step_index: int | None,
) -> dict[str, Any]:
    exact_media_ids = normalize_media_ids(media_ids)
    row = {
        "reference": reference,
        "creator_id": creator_id,
        "fan_id": fan_id,
        "status": "claimed",
        "media_ids": exact_media_ids,
        "price_cents": int(price_cents),
        "source": source,
        "set_id": set_id,
        "step_index": step_index,
    }

    try:
        result = await asyncio.to_thread(
            lambda: get_supabase().table("ppv_deliveries").insert(row).execute()
        )
    except Exception as exc:
        # Do not retry an unknown insert outcome: the row may already exist and
        # retrying the surrounding delivery would risk a duplicate live send.
        raise PPVDeliveryClaimError(
            "another locked PPV is already being delivered to this fan"
        ) from exc
    return (result.data or [row])[0]


async def transition_delivery(
    reference: str,
    status: str,
    *,
    platform_message_id: str | None = None,
    amount_paid_cents: int | None = None,
    error: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    if status not in ALL_STATUSES:
        raise ValueError(f"unsupported PPV delivery status: {status}")

    now = datetime.now(timezone.utc).isoformat()
    update: dict[str, Any] = {
        "status": status,
        "updated_at": now,
    }
    timestamp_column = {
        "delivered_pending": "delivered_at",
        "purchased": "purchased_at",
        "abandoned": "abandoned_at",
        "voided": "voided_at",
        "failed": "failed_at",
    }.get(status)
    if timestamp_column:
        update[timestamp_column] = now
    if platform_message_id:
        update["platform_message_id"] = platform_message_id
    if amount_paid_cents is not None:
        update["amount_paid_cents"] = int(amount_paid_cents)
    if error:
        update["last_error"] = str(error)[:1000]
    if metadata:
        update["metadata"] = metadata

    async def _update() -> None:
        await asyncio.to_thread(
            lambda: get_supabase().table("ppv_deliveries")
            .update(update)
            .eq("reference", reference)
            .execute()
        )

    await retry_transient_db_operation(
        _update,
        label=f"PPV delivery ledger reference={reference}",
        log_prefix="PPV LEDGER RETRY",
    )


async def list_fan_deliveries(creator_id: str, fan_id: str) -> list[dict[str, Any]]:
    result = await asyncio.to_thread(
        lambda: get_supabase().table("ppv_deliveries")
        .select(
            "reference, status, media_ids, price_cents, source, set_id, "
            "platform_message_id, claimed_at, delivered_at, purchased_at, "
            "abandoned_at, voided_at, failed_at"
        )
        .eq("creator_id", creator_id)
        .eq("fan_id", fan_id)
        .order("claimed_at", desc=True)
        .execute()
    )
    return result.data or []
