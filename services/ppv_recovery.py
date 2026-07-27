"""Operator resolutions for ambiguous PPV delivery outcomes."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from core.supabase import get_supabase
from db.commercial_queries import (
    cancel_actions_for_fan,
    get_creator_policy,
    get_fan_state,
    save_fan_state,
)
from db.queries import clear_fan_review, get_fan_session, save_fan_session
from models.commercial import FanStatus
from services.db_reliability import retry_transient_db_operation
from services.ppv_persistence import (
    pending_from_message_receipt,
    persist_ppv_reconciliation,
)
from services.vault_operations import normalize_media_ids


AMBIGUOUS_PPV_REVIEW_REASONS = {
    "ppv_sent_but_reconciliation_not_persisted",
    "delivery_sent_but_not_persisted",
    "ppv_purchase_verification_unavailable",
    "ppv_sent_missing_message_id",
}


class PPVRecoveryError(RuntimeError):
    pass


async def _load_recovery_context(fan_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    def _load() -> tuple[
        dict[str, Any],
        list[dict[str, Any]],
        list[dict[str, Any]],
    ]:
        db = get_supabase()
        fan = (
            db.table("fans")
            .select(
                "creator_id, platform_fan_id, pending_ppv_check, "
                "needs_human_review, review_reason"
            )
            .eq("id", fan_id)
            .single()
            .execute()
        ).data or {}
        messages = (
            db.table("messages")
            .select("id, sent_at, fansly_message_id, media_context")
            .eq("fan_id", fan_id)
            .eq("role", "creator")
            .not_.is_("media_context", "null")
            .order("sent_at", desc=True)
            .limit(50)
            .execute()
        ).data or []
        deliveries = (
            db.table("ppv_deliveries")
            .select(
                "reference, status, media_ids, price_cents, source, set_id, "
                "step_index, platform_message_id, claimed_at, delivered_at"
            )
            .eq("fan_id", fan_id)
            .in_("status", ["claimed", "delivered_pending"])
            .order("claimed_at", desc=True)
            .limit(1)
            .execute()
        ).data or []
        return fan, messages, deliveries

    fan, messages, deliveries = await retry_transient_db_operation(
        lambda: asyncio.to_thread(_load),
        label=f"PPV recovery context fan={fan_id}",
        log_prefix="PPV RECOVERY RETRY",
    )
    if not fan.get("creator_id"):
        raise PPVRecoveryError("fan was not found")
    for message in messages:
        ppv = (message.get("media_context") or {}).get("ppv") or {}
        if ppv.get("payment_reference") and ppv.get("delivery_status") != "voided":
            return fan, message
    if deliveries:
        delivery = deliveries[0]
        media_ids = normalize_media_ids(delivery.get("media_ids") or [])
        sent_at = delivery.get("delivered_at") or delivery.get("claimed_at")
        return fan, {
            "id": None,
            "sent_at": sent_at,
            "fansly_message_id": delivery.get("platform_message_id"),
            "media_context": {
                "ppv": {
                    "payment_reference": delivery.get("reference"),
                    "media_id": media_ids[0] if media_ids else None,
                    "media_ids": media_ids,
                    "price": float(delivery.get("price_cents") or 0) / 100,
                    "price_cents": int(delivery.get("price_cents") or 0),
                    "source": delivery.get("source"),
                    "set_id": delivery.get("set_id"),
                    "step_index": delivery.get("step_index"),
                }
            },
        }
    raise PPVRecoveryError("no recoverable PPV receipt was found")


async def _load_fan_review(fan_id: str) -> dict[str, Any]:
    def _load() -> dict[str, Any]:
        return (
            get_supabase().table("fans")
            .select("creator_id, needs_human_review, review_reason")
            .eq("id", fan_id)
            .single()
            .execute()
        ).data or {}

    fan = await retry_transient_db_operation(
        lambda: asyncio.to_thread(_load),
        label=f"review state fan={fan_id}",
        log_prefix="REVIEW RETRY",
    )
    if not fan.get("creator_id"):
        raise PPVRecoveryError("fan was not found")
    return fan


async def repair_ppv_reconciliation(
    fan_id: str,
    *,
    clear_review: bool = True,
) -> dict[str, Any]:
    """Repair payment tracking from the local receipt without resending media."""
    fan, message = await _load_recovery_context(fan_id)
    creator_id = str(fan["creator_id"])
    policy = await retry_transient_db_operation(
        lambda: get_creator_policy(creator_id),
        label=f"PPV recovery policy fan={fan_id}",
        log_prefix="PPV RECOVERY RETRY",
    )
    pending = fan.get("pending_ppv_check") or pending_from_message_receipt(
        message,
        payment_window_hours=policy.ppv_payment_window_hours,
        local_test_fan=str(fan.get("platform_fan_id") or "").startswith("test_"),
    )
    session = await retry_transient_db_operation(
        lambda: get_fan_session(fan_id),
        label=f"PPV recovery session fan={fan_id}",
        log_prefix="PPV RECOVERY RETRY",
    )
    session, reconcile_at, attached = await persist_ppv_reconciliation(
        creator_id=creator_id,
        fan_id=fan_id,
        pending=pending,
        session=session,
        platform_message_id=pending.get("platform_message_id"),
    )
    if not attached:
        raise PPVRecoveryError("PPV delivery was already resolved")
    if clear_review:
        await retry_transient_db_operation(
            lambda: clear_fan_review(fan_id),
            label=f"clear PPV review fan={fan_id}",
            log_prefix="PPV RECOVERY RETRY",
        )
    print(
        f"[PPV RECOVERY] fan={fan_id} resolution=repair "
        f"reference={pending['reference']} reconcile_at={reconcile_at.isoformat()}"
    )
    return {
        "status": "repaired",
        "fan_id": fan_id,
        "media_id": pending.get("media_id"),
        "media_ids": pending.get("media_ids") or [],
        "price": pending.get("price"),
        "payment_reference": pending.get("reference"),
        "reconcile_at": reconcile_at.isoformat(),
        "session_payment_state": (session or {}).get("payment_state"),
    }


async def _mark_receipt_not_sent(fan_id: str) -> dict[str, Any]:
    fan, message = await _load_recovery_context(fan_id)
    creator_id = str(fan["creator_id"])
    context = dict(message.get("media_context") or {})
    ppv = dict(context.get("ppv") or {})
    reference = str(ppv.get("payment_reference") or "")
    ppv.update({
        "delivery_status": "voided",
        "void_reason": "operator_confirmed_not_sent",
        "voided_at": datetime.now(timezone.utc).isoformat(),
    })
    context["ppv"] = ppv

    if message.get("id"):
        async def _void_message() -> None:
            await asyncio.to_thread(
                lambda: get_supabase().table("messages")
                .update({"media_context": context})
                .eq("id", message["id"])
                .execute()
            )

        await retry_transient_db_operation(
            _void_message,
            label=f"void PPV receipt fan={fan_id}",
            log_prefix="PPV RECOVERY RETRY",
        )
    if reference:
        from services.ppv_delivery_ledger import transition_delivery

        await transition_delivery(
            reference,
            "voided",
            error="operator_confirmed_not_sent",
        )

    current_pending = fan.get("pending_ppv_check") or {}
    if not current_pending or str(current_pending.get("reference") or "") == reference:
        async def _clear_pending() -> None:
            await asyncio.to_thread(
                lambda: get_supabase().table("fans")
                .update({"pending_ppv_check": None})
                .eq("id", fan_id)
                .execute()
            )

        await retry_transient_db_operation(
            _clear_pending,
            label=f"clear void PPV pending state fan={fan_id}",
            log_prefix="PPV RECOVERY RETRY",
        )

    session = await retry_transient_db_operation(
        lambda: get_fan_session(fan_id),
        label=f"void PPV session read fan={fan_id}",
        log_prefix="PPV RECOVERY RETRY",
    )
    media_ids = normalize_media_ids(ppv.get("media_ids") or [ppv.get("media_id")])
    if session:
        plan = session.get("plan") or []
        for item in plan:
            item_ids = normalize_media_ids(item.get("media_ids") or [item.get("media_id")])
            if set(item_ids) & set(media_ids):
                item["sent"] = False
                item.pop("sent_at", None)
                item.pop("message_id", None)
        session["awaiting_purchase_index"] = None
        session["payment_state"] = "OFFER_SELECTED"
        session["updated_at"] = datetime.now(timezone.utc).isoformat()
        await retry_transient_db_operation(
            lambda: save_fan_session(fan_id, session),
            label=f"void PPV session write fan={fan_id}",
            log_prefix="PPV RECOVERY RETRY",
        )

    state = await retry_transient_db_operation(
        lambda: get_fan_state(fan_id),
        label=f"void PPV state read fan={fan_id}",
        log_prefix="PPV RECOVERY RETRY",
    )
    state.status = (
        FanStatus.OFFER_SELECTED
        if state.selected_package_id
        else FanStatus.OFFER_PENDING
        if state.offered_packages
        else FanStatus.IDLE
    )
    await retry_transient_db_operation(
        lambda: save_fan_state(fan_id, creator_id, state),
        label=f"void PPV state write fan={fan_id}",
        log_prefix="PPV RECOVERY RETRY",
    )
    await retry_transient_db_operation(
        lambda: cancel_actions_for_fan(fan_id, "PPV_RECONCILE"),
        label=f"void PPV reconcile cancellation fan={fan_id}",
        log_prefix="PPV RECOVERY RETRY",
    )
    await retry_transient_db_operation(
        lambda: clear_fan_review(fan_id),
        label=f"clear void PPV review fan={fan_id}",
        log_prefix="PPV RECOVERY RETRY",
    )
    print(f"[PPV RECOVERY] fan={fan_id} resolution=not_sent reference={reference}")
    return {
        "status": "marked_not_sent",
        "fan_id": fan_id,
        "media_id": media_ids[0] if media_ids else None,
        "media_ids": media_ids,
        "payment_reference": reference,
    }


async def resolve_fan_review(
    fan_id: str,
    *,
    resolution: str,
    amount: float | None = None,
) -> dict[str, Any]:
    if resolution == "repair_ppv":
        return await repair_ppv_reconciliation(fan_id)
    if resolution == "mark_purchased":
        repaired = await repair_ppv_reconciliation(fan_id, clear_review=False)
        from services.suggestions import record_ppv_purchase

        purchase_amount = amount if amount is not None else repaired.get("price")
        await record_ppv_purchase(
            fan_id,
            str(repaired.get("media_id") or ""),
            purchase_amount,
        )
        await retry_transient_db_operation(
            lambda: clear_fan_review(fan_id),
            label=f"clear purchased PPV review fan={fan_id}",
            log_prefix="PPV RECOVERY RETRY",
        )
        print(
            f"[PPV RECOVERY] fan={fan_id} resolution=purchased "
            f"media={repaired.get('media_id')} amount={purchase_amount}"
        )
        return {**repaired, "status": "purchase_recorded", "amount": purchase_amount}
    if resolution == "mark_not_sent":
        return await _mark_receipt_not_sent(fan_id)
    if resolution == "mark_not_purchased":
        fan = await _load_fan_review(fan_id)
        if str(fan.get("review_reason") or "") != "ppv_purchase_verification_unavailable":
            raise PPVRecoveryError("This conversation is not awaiting purchase verification")
        from services.ppv_reconciliation import finalize_pending_ppv_as_not_purchased

        finalized = await finalize_pending_ppv_as_not_purchased(
            creator_id=str(fan["creator_id"]),
            fan_id=fan_id,
        )
        if not finalized:
            raise PPVRecoveryError("The pending PPV changed before it could be resolved")
        await retry_transient_db_operation(
            lambda: cancel_actions_for_fan(fan_id, "PPV_RECONCILE"),
            label=f"cancel unresolved PPV reconcile fan={fan_id}",
            log_prefix="PPV RECOVERY RETRY",
        )
        await retry_transient_db_operation(
            lambda: clear_fan_review(fan_id),
            label=f"clear not-purchased PPV review fan={fan_id}",
            log_prefix="PPV RECOVERY RETRY",
        )
        print(f"[PPV RECOVERY] fan={fan_id} resolution=not_purchased")
        return {"status": "not_purchased", "fan_id": fan_id}
    if resolution == "resume_ai":
        fan = await _load_fan_review(fan_id)
        reason = str(fan.get("review_reason") or "")
        if reason in AMBIGUOUS_PPV_REVIEW_REASONS:
            raise PPVRecoveryError(
                "This PPV has an ambiguous delivery outcome. Repair it, mark it purchased, "
                "or confirm the appropriate not-sent/not-purchased outcome before resuming AI."
            )
        await retry_transient_db_operation(
            lambda: clear_fan_review(fan_id),
            label=f"clear review fan={fan_id}",
            log_prefix="REVIEW RETRY",
        )
        return {"status": "resumed", "fan_id": fan_id}
    raise PPVRecoveryError("invalid review resolution")
