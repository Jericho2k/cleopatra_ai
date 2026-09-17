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

#: The one status that records money. Nothing may move a row out of it.
PAID_STATUS = "purchased"

#: Which prior statuses each transition may move a row out of.
#:
#: Finding I of docs/autonomy_architecture_review.md. ``transition_delivery``
#: used to update by reference alone, while ``abandon_delivery_if_active`` in
#: this same module used a status predicate. Four writers reach this function
#: from different clocks:
#:
#:   * ``services/ppv_delivery.py`` writes ``delivered_pending`` AFTER lock
#:     verification, which takes a round trip to the platform;
#:   * ``services/suggestions.py::record_ppv_purchase`` writes ``purchased``
#:     from a purchase event;
#:   * ``services/ppv_reconciliation.py`` writes ``abandoned`` when the payment
#:     window expires;
#:   * ``services/ppv_recovery.py`` writes ``voided`` when an operator confirms
#:     nothing was sent.
#:
#: Unordered, any of the last three could land before a slow
#: ``delivered_pending`` and be silently undone by it; expiry could land after a
#: purchase and reverse paid state. That is the interleaving the review names:
#: *a delayed pending update must not reverse a confirmed purchase*.
#:
#: ``purchased`` is absent on purpose and handled in ``_allowed_prior``: money
#: arriving is a fact about the world, and refusing to record it would hide a
#: real payment from the operator. Every other transition may only move an
#: ACTIVE row.
_ALLOWED_PRIOR: dict[str, frozenset[str]] = {
    "delivered_pending": frozenset(ACTIVE_STATUSES),
    "abandoned": frozenset(ACTIVE_STATUSES),
    "voided": frozenset(ACTIVE_STATUSES),
    "failed": frozenset(ACTIVE_STATUSES),
}


def _allowed_prior(status: str) -> frozenset[str]:
    """Which statuses ``status`` may be reached from.

    A purchase may arrive after an expiry or a void — a fan can pay an offer
    this backend has already given up on — so it is reachable from anything
    except itself, where it is a no-op. Everything else stops at an active row.
    """
    if status == PAID_STATUS:
        return frozenset(ALL_STATUSES - {PAID_STATUS})
    return _ALLOWED_PRIOR.get(status, frozenset(ACTIVE_STATUSES))


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
) -> bool:
    """Move one delivery to ``status``, but only from a status it may leave.

    Returns whether the ledger now says ``status`` — true when this call moved
    the row, and also true when the row was already there, because a duplicate
    event and a lost response both mean the same thing about the world. False
    means the transition was refused: the row is terminal, or it does not
    exist. Callers may ignore the result; the guard protects the ledger either
    way, and no existing caller changed behaviour when it was added.

    The predicate is the point. See ``_ALLOWED_PRIOR`` for which writers race
    here and why a confirmed purchase must survive all of them.
    """
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

    allowed = sorted(_allowed_prior(status))

    async def _update() -> list[dict[str, Any]]:
        result = await asyncio.to_thread(
            lambda: get_supabase().table("ppv_deliveries")
            .update(update)
            .eq("reference", reference)
            .in_("status", allowed)
            .execute()
        )
        return list(result.data or [])

    moved = await retry_transient_db_operation(
        _update,
        label=f"PPV delivery ledger reference={reference}",
        log_prefix="PPV LEDGER RETRY",
    )
    if moved:
        return True

    # Nothing matched. Three different things look identical from here, and an
    # operator needs them told apart, so the current row is read once on this
    # rare path rather than every transition guessing.
    current = await _current_status(reference)
    if current is None:
        print(
            f"[PPV LEDGER] reference={reference} target={status} "
            "no_such_delivery=true"
        )
        return False
    if current == status:
        # Already there. Either a duplicate event, or the previous attempt
        # applied and its response was lost before the retry. Both mean the
        # ledger says what this call wanted it to say.
        print(
            f"[PPV LEDGER] reference={reference} target={status} "
            "already_in_target_status=true"
        )
        return True
    print(
        f"[PPV LEDGER REFUSED] reference={reference} "
        f"target={status} current={current} "
        + (
            "reason=would_reverse_a_confirmed_purchase"
            if current == PAID_STATUS
            else "reason=row_is_no_longer_active"
        )
    )
    return False


async def _current_status(reference: str) -> str | None:
    """The delivery's status right now, or ``None`` if there is no such row.

    Only called when a guarded update matched nothing. A read that itself fails
    returns ``None``: this exists to make a log line accurate, and must never
    be the thing that raises out of a transition.
    """
    try:
        result = await asyncio.to_thread(
            lambda: get_supabase().table("ppv_deliveries")
            .select("status")
            .eq("reference", reference)
            .limit(1)
            .execute()
        )
    except Exception as exc:  # pragma: no cover - diagnostics only
        print(f"[PPV LEDGER] reference={reference} status_read_failed={exc}")
        return None
    rows = list(result.data or [])
    return str(rows[0].get("status")) if rows else None


async def delivery_is_paid(reference: str) -> bool:
    """Whether the ledger already records money for this delivery.

    The cheap pre-check a caller does before writing commercial state that
    contradicts a purchase. It is not a substitute for the predicate inside
    ``transition_delivery`` — a purchase can still land in the window between
    this read and that write — but it closes the common case, where the
    purchase was already recorded minutes earlier and the expiry sweep simply
    had not looked.
    """
    return await _current_status(reference) == PAID_STATUS


async def abandon_delivery_if_active(reference: str) -> bool:
    """Atomically expire an offer unless a purchase won the race.

    Kept separate from ``transition_delivery(reference, "abandoned")``, which
    now carries the same predicate, because the two answer different questions.
    This one reports whether THIS call claimed the expiry, so a caller can bail
    out when another worker got there first; the other reports whether the
    ledger ends up saying ``abandoned``, which an already-abandoned row also
    satisfies.
    """
    now = datetime.now(timezone.utc).isoformat()

    def _update() -> bool:
        result = (
            get_supabase().table("ppv_deliveries")
            .update({
                "status": "abandoned",
                "abandoned_at": now,
                "updated_at": now,
            })
            .eq("reference", reference)
            .in_("status", list(ACTIVE_STATUSES))
            .execute()
        )
        return bool(result.data)

    return await asyncio.to_thread(_update)


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
