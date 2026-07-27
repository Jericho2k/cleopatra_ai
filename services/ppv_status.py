"""Authoritative per-fan PPV media status projection."""
from __future__ import annotations

from typing import Any

from services.vault_operations import normalize_media_ids


def _ids(value: dict[str, Any]) -> list[str]:
    media_ids = normalize_media_ids(
        value.get("media_ids") or [value.get("media_id")]
    )
    if media_ids:
        return media_ids
    item = str(value.get("item") or "")
    prefix = "PPV media "
    if item.startswith(prefix):
        return normalize_media_ids([item[len(prefix):].strip()])
    return []


def build_media_status_by_id(
    *,
    deliveries: list[dict[str, Any]],
    message_rows: list[dict[str, Any]],
    sales_log: list[dict[str, Any]],
    not_sold_log: list[dict[str, Any]],
    pending_ppv: dict[str, Any] | None,
) -> dict[str, str]:
    """Project lifecycle records and legacy receipts into one UI status.

    ``deliveries`` must be newest first. Sales and the current pending record
    remain authoritative while older pre-ledger history is backfilled lazily.
    """
    status_by_id: dict[str, str] = {}

    # Legacy receipts predate the ledger. Only actual, non-voided platform
    # receipts count as sent.
    for message in message_rows:
        ppv = (message.get("media_context") or {}).get("ppv") or {}
        if not ppv or ppv.get("delivery_status") == "voided":
            continue
        for media_id in _ids(ppv):
            status_by_id.setdefault(media_id, "sent")

    for entry in not_sold_log:
        for media_id in _ids(entry):
            status_by_id[media_id] = "abandoned"

    ledger_mapping = {
        "claimed": "payment_pending",
        "delivered_pending": "payment_pending",
        "purchased": "sold",
        "abandoned": "abandoned",
        "voided": "voided",
        "failed": "voided",
    }
    seen_ledger_media: set[str] = set()
    for delivery in deliveries:
        projected = ledger_mapping.get(str(delivery.get("status") or ""))
        if not projected:
            continue
        for media_id in _ids(delivery):
            if media_id in seen_ledger_media:
                continue
            status_by_id[media_id] = projected
            seen_ledger_media.add(media_id)

    if pending_ppv:
        for media_id in _ids(pending_ppv):
            status_by_id[media_id] = "payment_pending"

    # A confirmed purchase is terminal and outranks every other historical
    # state, including a later failed attempt to resend the same media.
    for sale in sales_log:
        for media_id in _ids(sale):
            status_by_id[media_id] = "sold"

    return status_by_id
