"""Small operational surface required to supervise a controlled full-auto pilot."""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any

from core.supabase import get_supabase
from db.commercial_queries import (
    cancel_actions_for_fan,
    get_fan_state,
    get_scheduled_actions_for_fan,
    save_fan_state,
)
from db.queries import get_fan_session
from services.db_reliability import (
    is_transient_db_error,
    retry_transient_db_operation,
)


FOLLOWUP_ACTIONS = {
    "PAYDAY_REENGAGEMENT",
    "POST_SESSION_FOLLOWUP",
    "ABANDONED_PPV_FOLLOWUP",
    "OFFER_EXPIRY",
    "ABANDONED_OFFER_FOLLOWUP",
    "INACTIVITY_REENGAGEMENT",
}


class FullAutoStatusUnavailable(RuntimeError):
    """The operational snapshot could not be read after transient DB retries."""


def _is_transient_operation_error(exc: BaseException) -> bool:
    return is_transient_db_error(exc)


async def _retry_transient_operation(
    operation: Callable[[], Awaitable[Any]],
    *,
    label: str,
    attempts: int = 3,
    delay_seconds: float = 0.15,
) -> Any:
    try:
        return await retry_transient_db_operation(
            operation,
            label=label,
            attempts=attempts,
            delay_seconds=delay_seconds,
            log_prefix="FULL AUTO STATUS",
        )
    except Exception as exc:
        if _is_transient_operation_error(exc):
            raise FullAutoStatusUnavailable(
                f"{label} unavailable after {attempts} attempts"
            ) from exc
        raise


def ppv_media_bundles(messages: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Index every PPV media id to the complete immutable bundle it belonged to."""
    bundles: dict[str, list[str]] = {}
    for message in messages:
        ppv = (message.get("media_context") or {}).get("ppv") or {}
        media_ids = list(dict.fromkeys(
            str(value)
            for value in (ppv.get("media_ids") or [ppv.get("media_id")])
            if value
        ))
        for media_id in media_ids:
            bundles.setdefault(media_id, media_ids)
    return bundles


async def _safe_context(awaitable: Any) -> dict[str, Any]:
    try:
        value = await awaitable
        return value if isinstance(value, dict) else {}
    except Exception as exc:
        print(f"[FULL AUTO STATUS] optional context unavailable: {exc}")
        return {}


def summarize_operation_rows(
    states: list[dict[str, Any]],
    fans: list[dict[str, Any]],
    actions: list[dict[str, Any]],
) -> dict[str, int]:
    return {
        "payment_pending": sum(row.get("status") == "PAYMENT_PENDING" for row in states),
        "followups_pending": sum(bool(row.get("next_followup_at")) for row in states),
        "human_review": sum(bool(row.get("needs_human_review")) for row in fans),
        "failed_actions": sum(row.get("status") == "FAILED" for row in actions),
        "processing_actions": sum(row.get("status") == "PROCESSING" for row in actions),
    }


async def get_fan_full_auto_snapshot(fan_id: str) -> dict[str, Any]:
    def _fan() -> dict:
        return (
            get_supabase().table("fans")
            .select(
                "id, creator_id, auto_mode, needs_human_review, review_reason, "
                "pending_ppv_check, last_active, total_spent, spend_tier"
            )
            .eq("id", fan_id)
            .single()
            .execute()
        ).data or {}

    fan = await _retry_transient_operation(
        lambda: asyncio.to_thread(_fan),
        label="fan profile",
    )
    if not fan:
        return {"status": "not_found", "fan_id": fan_id}

    creator_id = str(fan.get("creator_id") or "")

    def _creator() -> dict:
        return (
            get_supabase().table("creators")
            .select("auto_mode, auto_audience_policy")
            .eq("id", creator_id)
            .single()
            .execute()
        ).data or {}

    def _memberships() -> list[dict]:
        return (
            get_supabase().table("fan_list_members")
            .select("list_id, fan_lists(exclude_from_auto)")
            .eq("fan_id", fan_id)
            .execute()
        ).data or []

    def _approved_set_count() -> int:
        response = (
            get_supabase().table("vault_sets")
            .select("id", count="exact", head=True)
            .eq("creator_id", creator_id)
            .eq("status", "approved")
            .execute()
        )
        return int(response.count or 0)

    def _creator_message_count() -> int:
        response = (
            get_supabase().table("messages")
            .select("id", count="exact", head=True)
            .eq("fan_id", fan_id)
            .eq("creator_id", creator_id)
            .eq("role", "creator")
            .execute()
        )
        return int(response.count or 0)

    def _ppv_messages() -> list[dict]:
        return (
            get_supabase().table("messages")
            .select("media_context")
            .eq("fan_id", fan_id)
            .eq("creator_id", creator_id)
            .eq("role", "creator")
            .not_.is_("media_context", "null")
            .order("sent_at", desc=True)
            .limit(1000)
            .execute()
        ).data or []

    from db.fan_intelligence_queries import get_fan_intelligence_context
    from services.affordability import get_affordability_context
    from services.fan_lifecycle import get_fan_lifecycle_context
    from services.price_learning import get_price_learning_context

    # These calls intentionally run sequentially. The Supabase client is shared,
    # and fanning many reads out across threads caused one HTTP/2 disconnect to
    # take down both the dashboard snapshot and a concurrent PPV persistence write.
    creator = await _retry_transient_operation(
        lambda: asyncio.to_thread(_creator), label="creator settings"
    )
    state = await _retry_transient_operation(
        lambda: get_fan_state(fan_id), label="commercial state"
    )
    session = await _retry_transient_operation(
        lambda: get_fan_session(fan_id), label="session state"
    )
    actions = await _retry_transient_operation(
        lambda: get_scheduled_actions_for_fan(fan_id), label="scheduled actions"
    )
    memberships = await _retry_transient_operation(
        lambda: asyncio.to_thread(_memberships), label="fan lists"
    )
    approved_set_count = await _retry_transient_operation(
        lambda: asyncio.to_thread(_approved_set_count),
        label="approved set availability",
    )
    creator_message_count = await _retry_transient_operation(
        lambda: asyncio.to_thread(_creator_message_count),
        label="creator message count",
    )
    ppv_messages = await _retry_transient_operation(
        lambda: asyncio.to_thread(_ppv_messages), label="PPV history"
    )
    fan_intelligence = await _safe_context(get_fan_intelligence_context(fan_id))
    buyer_lifecycle = await _safe_context(get_fan_lifecycle_context(fan_id))
    affordability = await _safe_context(get_affordability_context(fan_id))
    price_learning = await _safe_context(get_price_learning_context(fan_id))
    from services.auto_audience import AutoAudiencePolicy, evaluate_auto_eligibility

    try:
        audience_policy = AutoAudiencePolicy(**(creator.get("auto_audience_policy") or {}))
    except Exception:
        audience_policy = AutoAudiencePolicy()
    list_ids = {
        str(row.get("list_id")) for row in memberships if row.get("list_id")
    }
    legacy_exclusions = {
        str(row.get("list_id"))
        for row in memberships
        if row.get("list_id") and (row.get("fan_lists") or {}).get("exclude_from_auto")
    }
    audience_policy.exclude_list_ids = list(
        dict.fromkeys([*audience_policy.exclude_list_ids, *legacy_exclusions])
    )
    fan_auto = fan.get("auto_mode")
    creator_auto = bool(creator.get("auto_mode", False))
    eligibility = evaluate_auto_eligibility(
        creator_auto=creator_auto,
        fan_auto_override=fan_auto,
        needs_human_review=bool(fan.get("needs_human_review")),
        policy=audience_policy,
        fan_list_ids=list_ids,
        total_spent=int(fan.get("total_spent") or 0),
        spend_tier=str(fan.get("spend_tier") or "cold"),
        is_new_fan=creator_message_count == 0,
    )
    auto_available = approved_set_count > 0
    effective_auto = eligibility.eligible and auto_available
    pending = fan.get("pending_ppv_check") or None

    return {
        "status": "ok",
        "fan_id": fan_id,
        "creator_id": creator_id,
        "effective_auto_mode": effective_auto,
        "fan_auto_mode": fan_auto,
        "creator_auto_mode": creator_auto,
        "auto_mode_reason": (
            eligibility.reason if auto_available else "no_approved_sets"
        ),
        "auto_available": auto_available,
        "approved_sets": approved_set_count,
        "needs_human_review": bool(fan.get("needs_human_review")),
        "review_reason": fan.get("review_reason"),
        "commercial_state": state.model_dump(mode="json"),
        "session": session,
        "pending_ppv": pending,
        "scheduled_actions": actions,
        "last_active": fan.get("last_active"),
        "ppv_media_bundles": ppv_media_bundles(ppv_messages),
        "fan_intelligence": fan_intelligence,
        "buyer_lifecycle": buyer_lifecycle,
        "affordability": affordability,
        "price_learning": price_learning,
    }


async def get_creator_full_auto_health(creator_id: str) -> dict[str, Any]:
    def _load() -> tuple[list[dict], list[dict], list[dict]]:
        db = get_supabase()
        states = (
            db.table("fan_commercial_states")
            .select(
                "fan_id, status, next_followup_at, next_followup_type, "
                "last_abandoned_ppv_at, updated_at"
            )
            .eq("creator_id", creator_id)
            .execute()
        ).data or []
        fans = (
            db.table("fans")
            .select("id, display_name, auto_mode, needs_human_review, review_reason")
            .eq("creator_id", creator_id)
            .execute()
        ).data or []
        actions = (
            db.table("scheduled_actions")
            .select("id, fan_id, action_type, execute_at, status, attempts, last_error")
            .eq("creator_id", creator_id)
            .in_("status", ["PENDING", "PROCESSING", "FAILED"])
            .order("execute_at")
            .execute()
        ).data or []
        return states, fans, actions

    states, fans, actions = await _retry_transient_operation(
        lambda: asyncio.to_thread(_load),
        label="creator full-auto health",
    )
    names = {
        str(row.get("id")): row.get("display_name") or str(row.get("id"))
        for row in fans
    }
    state_by_fan = {str(row.get("fan_id")): row for row in states}
    actionable_ids = {
        str(row.get("fan_id"))
        for row in states
        if row.get("status") == "PAYMENT_PENDING" or row.get("next_followup_at")
    }
    actionable_ids.update(
        str(row.get("id")) for row in fans if row.get("needs_human_review")
    )
    actionable_ids.update(
        str(row.get("fan_id")) for row in actions if row.get("status") == "FAILED"
    )
    return {
        "creator_id": creator_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": summarize_operation_rows(states, fans, actions),
        "fans": [
            {
                "fan_id": fan_id,
                "display_name": names.get(fan_id, fan_id),
                "commercial_state": state_by_fan.get(fan_id, {}).get("status", "IDLE"),
                "next_followup_at": state_by_fan.get(fan_id, {}).get("next_followup_at"),
                "next_followup_type": state_by_fan.get(fan_id, {}).get("next_followup_type"),
                "needs_human_review": next(
                    (
                        bool(row.get("needs_human_review"))
                        for row in fans
                        if str(row.get("id")) == fan_id
                    ),
                    False,
                ),
                "failed_actions": [
                    row for row in actions
                    if str(row.get("fan_id")) == fan_id and row.get("status") == "FAILED"
                ],
            }
            for fan_id in sorted(actionable_ids)
        ],
    }


async def cancel_fan_followup(fan_id: str, action_type: str | None = None) -> dict:
    state = await get_fan_state(fan_id)
    selected_type = action_type or state.next_followup_type
    if selected_type not in FOLLOWUP_ACTIONS:
        return {"status": "no_followup", "fan_id": fan_id}
    await cancel_actions_for_fan(fan_id, selected_type)
    if state.next_followup_type == selected_type:
        state.next_followup_at = None
        state.next_followup_type = None
        state.next_followup_payload = {}
        state.next_followup_dedupe_key = None

        def _creator_id() -> str:
            row = (
                get_supabase().table("fans")
                .select("creator_id")
                .eq("id", fan_id)
                .single()
                .execute()
            ).data or {}
            return str(row.get("creator_id") or "")

        creator_id = await asyncio.to_thread(_creator_id)
        if creator_id:
            await save_fan_state(fan_id, creator_id, state)
    return {"status": "cancelled", "fan_id": fan_id, "action_type": selected_type}
