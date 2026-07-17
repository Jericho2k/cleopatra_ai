"""Deterministic paid-session lifecycle helpers.

A paid step is sent once, then the session waits for purchase/decline before
advancing. These helpers are pure so the behavior is easy to test and reuse by
webhooks, API endpoints and the auto-mode loop.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any


class SessionLifecycleError(ValueError):
    pass


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_session(session: dict[str, Any] | None) -> dict[str, Any] | None:
    if not session:
        return None
    result = deepcopy(session)
    result.setdefault("status", "active")
    result.setdefault("plan", [])
    result.setdefault("current_index", 0)
    result.setdefault("awaiting_purchase_index", None)
    result.setdefault("post_ppv_cooldown", False)
    result.setdefault("cooldown_messages_remaining", 0)
    result.setdefault("revenue_cents", 0)
    if "payment_state" not in result:
        if result.get("status") == "completed":
            result["payment_state"] = "COMPLETED"
        elif result.get("status") in {"abandoned", "paused"}:
            result["payment_state"] = str(result.get("status")).upper()
        elif result.get("awaiting_purchase_index") is not None:
            result["payment_state"] = "PAYMENT_PENDING"
        elif int(result.get("revenue_cents", 0) or 0) > 0:
            result["payment_state"] = "ACTIVE"
        else:
            result["payment_state"] = "OFFER_SELECTED"
    return result


def current_step(session: dict[str, Any] | None) -> dict[str, Any] | None:
    normalized = normalize_session(session)
    if not normalized or normalized.get("status") != "active":
        return None
    idx = int(normalized.get("current_index", 0) or 0)
    plan = normalized.get("plan") or []
    if idx < 0 or idx >= len(plan):
        return None
    return plan[idx]


def has_remaining_steps(session: dict[str, Any] | None) -> bool:
    normalized = normalize_session(session)
    if not normalized or normalized.get("status") != "active":
        return False
    return int(normalized.get("current_index", 0) or 0) < len(normalized.get("plan") or [])


def has_pending_purchase(session: dict[str, Any] | None) -> bool:
    normalized = normalize_session(session)
    if not normalized or normalized.get("status") != "active":
        return False
    idx = normalized.get("awaiting_purchase_index")
    if idx is None:
        return False
    plan = normalized.get("plan") or []
    return 0 <= int(idx) < len(plan) and not bool(plan[int(idx)].get("purchased"))


def is_cooldown_active(session: dict[str, Any] | None) -> bool:
    normalized = normalize_session(session)
    return bool(
        normalized
        and normalized.get("status") == "active"
        and normalized.get("post_ppv_cooldown")
        and int(normalized.get("cooldown_messages_remaining", 0) or 0) > 0
    )


def mark_step_sent(
    session: dict[str, Any],
    *,
    step_index: int | None = None,
    message_id: str | None = None,
) -> dict[str, Any]:
    result = normalize_session(session)
    if result is None:
        raise SessionLifecycleError("No active session")
    if result.get("status") != "active":
        raise SessionLifecycleError(f"Session is {result.get('status')}")
    if has_pending_purchase(result):
        return result  # idempotent: never send a second paid step while waiting

    idx = int(result.get("current_index", 0) if step_index is None else step_index)
    plan = result.get("plan") or []
    if idx < 0 or idx >= len(plan):
        raise SessionLifecycleError("No session step available")

    item = plan[idx]
    item["sent"] = True
    item["sent_at"] = item.get("sent_at") or utc_now_iso()
    if message_id:
        item["message_id"] = message_id
    result["awaiting_purchase_index"] = idx
    result["payment_state"] = "PAYMENT_PENDING"
    result["updated_at"] = utc_now_iso()
    return result


def mark_step_purchased(
    session: dict[str, Any],
    *,
    media_id: str | None = None,
    set_id: str | None = None,
    amount_cents: int | None = None,
    cooldown_messages: int = 2,
) -> tuple[dict[str, Any], bool]:
    """Mark one sent step purchased and advance only now.

    Returns ``(updated_session, completed)``. Repeated purchase webhooks are
    idempotent and do not double-count revenue.
    """
    result = normalize_session(session)
    if result is None:
        raise SessionLifecycleError("No active session")
    plan = result.get("plan") or []

    idx = result.get("awaiting_purchase_index")
    if idx is None:
        idx = _find_step_index(plan, media_id=media_id, set_id=set_id)
    if idx is None or int(idx) < 0 or int(idx) >= len(plan):
        raise SessionLifecycleError("Purchase does not match a session step")
    idx = int(idx)
    item = plan[idx]

    if not item.get("purchased"):
        item["purchased"] = True
        item["purchased_at"] = utc_now_iso()
        cents = amount_cents
        if cents is None:
            cents = int(round(float(item.get("price") or 0) * 100))
        item["purchase_amount_cents"] = max(0, int(cents or 0))
        result["revenue_cents"] = int(result.get("revenue_cents", 0) or 0) + item["purchase_amount_cents"]

    result["awaiting_purchase_index"] = None
    result["current_index"] = max(int(result.get("current_index", 0) or 0), idx + 1)
    completed = result["current_index"] >= len(plan)
    if completed:
        result["status"] = "completed"
        result["payment_state"] = "COMPLETED"
        result["completed_at"] = result.get("completed_at") or utc_now_iso()
        result["post_ppv_cooldown"] = False
        result["cooldown_messages_remaining"] = 0
    else:
        result["status"] = "active"
        result["payment_state"] = "ACTIVE"
        result["post_ppv_cooldown"] = cooldown_messages > 0
        result["cooldown_messages_remaining"] = max(0, int(cooldown_messages))
    result["updated_at"] = utc_now_iso()
    return result, completed


def mark_step_declined(
    session: dict[str, Any],
    *,
    reason: str = "declined",
    pause: bool = False,
) -> dict[str, Any]:
    result = normalize_session(session)
    if result is None:
        raise SessionLifecycleError("No active session")
    idx = result.get("awaiting_purchase_index")
    if idx is not None:
        plan = result.get("plan") or []
        if 0 <= int(idx) < len(plan):
            plan[int(idx)]["declined"] = True
            plan[int(idx)]["declined_at"] = utc_now_iso()
    result["awaiting_purchase_index"] = None
    result["status"] = "paused" if pause else "abandoned"
    result["payment_state"] = "PAUSED" if pause else "ABANDONED"
    result["end_reason"] = reason
    result["ended_at"] = utc_now_iso()
    result["post_ppv_cooldown"] = False
    result["cooldown_messages_remaining"] = 0
    return result


def resume_session(session: dict[str, Any]) -> dict[str, Any]:
    result = normalize_session(session)
    if result is None:
        raise SessionLifecycleError("No session to resume")
    if result.get("status") not in {"paused", "active"}:
        raise SessionLifecycleError(f"Cannot resume {result.get('status')} session")
    result["status"] = "active"
    result["payment_state"] = "OFFER_SELECTED"
    result["resumed_at"] = utc_now_iso()
    result["updated_at"] = utc_now_iso()
    return result


def decrement_cooldown(session: dict[str, Any]) -> dict[str, Any]:
    result = normalize_session(session)
    if result is None or not result.get("post_ppv_cooldown"):
        return result or {}
    remaining = max(0, int(result.get("cooldown_messages_remaining", 0) or 0) - 1)
    result["cooldown_messages_remaining"] = remaining
    if remaining == 0:
        result["post_ppv_cooldown"] = False
    result["updated_at"] = utc_now_iso()
    return result


def _find_step_index(
    plan: list[dict[str, Any]],
    *,
    media_id: str | None,
    set_id: str | None,
) -> int | None:
    for index, item in enumerate(plan):
        media_ids = [str(value) for value in (item.get("media_ids") or [])]
        if media_id and (str(item.get("media_id")) == str(media_id) or str(media_id) in media_ids):
            return index
        if set_id and str(item.get("set_id")) == str(set_id):
            return index
    return None
