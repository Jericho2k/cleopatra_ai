"""Authoritative locked-PPV delivery for auto and operator workflows."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import uuid

import httpx

from core.supabase import get_supabase
from db.commercial_queries import get_creator_policy
from db.queries import (
    freeze_fan_for_review,
    get_fan_session,
)
from services.db_reliability import retry_transient_db_operation
from services.apifansly import (
    headers as apifansly_headers,
    raise_for_response as raise_for_apifansly_response,
    url as apifansly_url,
)
from services.ppv_persistence import (
    persist_ppv_reconciliation,
    save_ppv_message_receipt,
)
from services.vault_operations import normalize_media_ids


class PPVDeliveryError(RuntimeError):
    pass


async def send_locked_ppv(
    *,
    creator_id: str,
    fan_id: str,
    media_ids: list[str],
    price_cents: int,
    message_content: str,
    source: str,
    was_ai_suggested: bool,
    set_id: str | None = None,
    step_index: int | None = None,
) -> dict:
    """Send once, then durably attach reconciliation and payment state.

    Platform acceptance happens before local ``PAYMENT_PENDING`` persistence.
    If persistence fails after the live send, the fan is frozen for human review
    so a retry cannot duplicate-charge or duplicate-send content.
    """
    exact_media_ids = normalize_media_ids(media_ids)
    if not exact_media_ids:
        raise PPVDeliveryError("at least one media item is required")
    if int(price_cents) <= 0:
        raise PPVDeliveryError("price must be greater than zero")

    db = get_supabase()
    fan_row = await asyncio.to_thread(
        lambda: db.table("fans")
        .select("creator_id, fansly_group_id, platform_fan_id, pending_ppv_check")
        .eq("id", fan_id)
        .single()
        .execute()
    )
    fan = fan_row.data or {}
    if str(fan.get("creator_id") or "") != str(creator_id):
        raise PPVDeliveryError("fan does not belong to this creator")
    if fan.get("pending_ppv_check"):
        raise PPVDeliveryError("this fan already has a locked PPV awaiting payment")

    creator_row = await asyncio.to_thread(
        lambda: db.table("creators")
        .select("apifansly_account_id")
        .eq("id", creator_id)
        .single()
        .execute()
    )
    account_id = (creator_row.data or {}).get("apifansly_account_id")
    group_id = fan.get("fansly_group_id")
    if not group_id and account_id and fan.get("platform_fan_id"):
        from main import get_or_fetch_group_id

        group_id = await get_or_fetch_group_id(
            str(account_id),
            str(fan["platform_fan_id"]),
            fan_id,
        )
    if not group_id or not account_id:
        await freeze_fan_for_review(fan_id, "ppv_delivery_route_missing")
        raise PPVDeliveryError("no live delivery route for this fan")

    session = await get_fan_session(fan_id)
    if session and step_index is None:
        raise PPVDeliveryError(
            "this fan already has a planned paid session; resolve that session before sending a manual PPV"
        )
    if step_index is not None:
        if not session:
            raise PPVDeliveryError("the prepared paid-session step no longer exists")
        current_index = int(session.get("current_index", 0) or 0)
        if current_index != int(step_index) or session.get("awaiting_purchase_index") is not None:
            raise PPVDeliveryError("the prepared paid-session step is stale")
        plan = session.get("plan") or []
        if current_index >= len(plan):
            raise PPVDeliveryError("the prepared paid-session step is no longer available")
        planned_ids = normalize_media_ids(plan[current_index].get("media_ids") or [])
        planned_cents = int(
            plan[current_index].get("price_cents")
            or round(float(plan[current_index].get("price") or 0) * 100)
        )
        if planned_ids != exact_media_ids or planned_cents != int(price_cents):
            raise PPVDeliveryError("prepared media or price no longer matches the session")

    reference = uuid.uuid4().hex
    price_dollars = float(price_cents) / 100.0
    content = message_content.strip() or "just for you..."
    platform_message_id: str | None = None
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                apifansly_url(f"{account_id}/chats/{group_id}/messages"),
                headers=apifansly_headers(json_content=True),
                json={
                    "content": content,
                    "mediaIds": exact_media_ids,
                    "mediaId": exact_media_ids[0],
                    "access_type": "ppv",
                    "price": price_dollars,
                },
                timeout=10,
            )
        raise_for_apifansly_response(
            response,
            operation="locked PPV delivery",
            account_id=account_id,
        )
        try:
            body = response.json()
            platform_message_id = str(
                body.get("id") or body.get("data", {}).get("id") or ""
            ) or None
        except Exception:
            platform_message_id = None
    except Exception as exc:
        await freeze_fan_for_review(fan_id, "ppv_send_failed")
        raise PPVDeliveryError(f"platform rejected PPV delivery: {exc}") from exc

    sent_at = datetime.now(timezone.utc)
    try:
        policy = await retry_transient_db_operation(
            lambda: get_creator_policy(creator_id),
            label=f"PPV delivery policy fan={fan_id}",
            log_prefix="PPV PERSIST RETRY",
        )
        expires_at = sent_at + timedelta(hours=policy.ppv_payment_window_hours)
        pending = {
            "reference": reference,
            "media_id": exact_media_ids[0],
            "media_ids": exact_media_ids,
            "set_id": set_id,
            "step_index": step_index,
            "price": price_dollars,
            "price_cents": int(price_cents),
            "source": source,
            "sent_at": sent_at.isoformat(),
            "expires_at": expires_at.isoformat(),
            "verification_attempts": 0,
            "platform_message_id": platform_message_id,
        }
        media_context = {
            "ppv": {
                "media_ids": exact_media_ids,
                "media_id": exact_media_ids[0],
                "price": price_dollars,
                "price_cents": int(price_cents),
                "access_type": "ppv",
                "set_id": set_id,
                "step_index": step_index,
                "payment_reference": reference,
                "source": source,
            }
        }
        await save_ppv_message_receipt(
            fan_id=fan_id,
            creator_id=creator_id,
            content=content,
            was_ai_suggested=was_ai_suggested,
            platform_message_id=platform_message_id,
            media_context=media_context,
        )
        session, reconcile_at = await persist_ppv_reconciliation(
            creator_id=creator_id,
            fan_id=fan_id,
            pending=pending,
            session=session,
            platform_message_id=platform_message_id,
        )
        print(
            f"[PPV PERSIST] fan={fan_id} reference={reference} "
            f"state=PAYMENT_PENDING reconcile_at={reconcile_at.isoformat()}"
        )
    except Exception as exc:
        await freeze_fan_for_review(fan_id, "ppv_sent_but_reconciliation_not_persisted")
        raise PPVDeliveryError(
            "PPV was sent but local reconciliation could not be persisted"
        ) from exc

    return {
        "status": "sent",
        "reference": reference,
        "platform_message_id": platform_message_id,
        "media_ids": exact_media_ids,
        "price_cents": int(price_cents),
        "sent_at": sent_at.isoformat(),
        "expires_at": expires_at.isoformat(),
        "source": source,
    }


async def cancel_pending_ppv_approvals(fan_id: str, *, reason: str) -> int:
    db = get_supabase()
    result = await asyncio.to_thread(
        lambda: db.table("ppv_approval_requests")
        .update({
            "status": "cancelled",
            "last_error": reason,
            "resolved_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
        .eq("fan_id", fan_id)
        .eq("status", "pending")
        .execute()
    )
    return len(result.data or [])


async def create_ppv_approval_request(
    *,
    creator_id: str,
    fan_id: str,
    message_content: str,
    media_ids: list[str],
    price_cents: int,
    set_id: str | None,
    step_index: int | None,
    approved_experience: str | None,
) -> dict:
    exact_ids = normalize_media_ids(media_ids)
    if not exact_ids or price_cents <= 0:
        raise PPVDeliveryError("approval requires exact media and price")
    db = get_supabase()
    existing = await asyncio.to_thread(
        lambda: db.table("ppv_approval_requests")
        .select("*")
        .eq("fan_id", fan_id)
        .eq("status", "pending")
        .limit(1)
        .execute()
    )
    if existing.data:
        return existing.data[0]
    row = {
        "creator_id": creator_id,
        "fan_id": fan_id,
        "status": "pending",
        "source": "auto",
        "message_content": message_content.strip(),
        "media_ids": exact_ids,
        "price_cents": int(price_cents),
        "set_id": set_id,
        "step_index": step_index,
        "approved_experience": approved_experience,
    }
    created = await asyncio.to_thread(
        lambda: db.table("ppv_approval_requests").insert(row).execute()
    )
    return (created.data or [row])[0]


async def list_ppv_approval_requests(
    creator_id: str,
    *,
    status: str = "pending",
) -> list[dict]:
    db = get_supabase()
    result = await asyncio.to_thread(
        lambda: db.table("ppv_approval_requests")
        .select("*, fans(display_name)")
        .eq("creator_id", creator_id)
        .eq("status", status)
        .order("created_at", desc=True)
        .execute()
    )
    return result.data or []


async def approve_ppv_request(
    request_id: str,
    *,
    resolved_by: str | None = None,
) -> dict:
    """Atomically claim and send exactly what the auto engine prepared."""
    db = get_supabase()
    now = datetime.now(timezone.utc).isoformat()
    claimed = await asyncio.to_thread(
        lambda: db.table("ppv_approval_requests")
        .update({"status": "sending", "updated_at": now, "resolved_by": resolved_by})
        .eq("id", request_id)
        .eq("status", "pending")
        .execute()
    )
    if not claimed.data:
        raise PPVDeliveryError("approval is no longer pending")
    request = claimed.data[0]
    try:
        delivery = await send_locked_ppv(
            creator_id=str(request["creator_id"]),
            fan_id=str(request["fan_id"]),
            media_ids=request.get("media_ids") or [],
            price_cents=int(request.get("price_cents") or 0),
            message_content=str(request.get("message_content") or ""),
            source="auto_approved",
            was_ai_suggested=True,
            set_id=request.get("set_id"),
            step_index=request.get("step_index"),
        )
    except Exception as exc:
        await asyncio.to_thread(
            lambda: db.table("ppv_approval_requests")
            .update({
                "status": "failed",
                "last_error": str(exc),
                "resolved_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            })
            .eq("id", request_id)
            .execute()
        )
        raise

    await asyncio.to_thread(
        lambda: db.table("ppv_approval_requests")
        .update({
            "status": "sent",
            "resolved_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "last_error": None,
        })
        .eq("id", request_id)
        .execute()
    )
    return {"request_id": request_id, **delivery}


async def reject_ppv_request(
    request_id: str,
    *,
    resolved_by: str | None = None,
) -> dict:
    db = get_supabase()
    now = datetime.now(timezone.utc).isoformat()
    result = await asyncio.to_thread(
        lambda: db.table("ppv_approval_requests")
        .update({
            "status": "rejected",
            "resolved_at": now,
            "updated_at": now,
            "resolved_by": resolved_by,
        })
        .eq("id", request_id)
        .eq("status", "pending")
        .execute()
    )
    if not result.data:
        raise PPVDeliveryError("approval is no longer pending")
    return {"status": "rejected", "request_id": request_id}
