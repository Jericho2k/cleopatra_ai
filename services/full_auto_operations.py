"""Small operational surface required to supervise a controlled full-auto pilot."""
from __future__ import annotations

import asyncio
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


FOLLOWUP_ACTIONS = {
    "PAYDAY_REENGAGEMENT",
    "POST_SESSION_FOLLOWUP",
    "ABANDONED_PPV_FOLLOWUP",
}


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
                "pending_ppv_check, last_active"
            )
            .eq("id", fan_id)
            .single()
            .execute()
        ).data or {}

    fan = await asyncio.to_thread(_fan)
    if not fan:
        return {"status": "not_found", "fan_id": fan_id}

    creator_id = str(fan.get("creator_id") or "")

    def _creator() -> dict:
        return (
            get_supabase().table("creators")
            .select("auto_mode")
            .eq("id", creator_id)
            .single()
            .execute()
        ).data or {}

    creator, state, session, actions = await asyncio.gather(
        asyncio.to_thread(_creator),
        get_fan_state(fan_id),
        get_fan_session(fan_id),
        get_scheduled_actions_for_fan(fan_id),
    )
    fan_auto = fan.get("auto_mode")
    creator_auto = bool(creator.get("auto_mode", False))
    effective_auto = creator_auto if fan_auto is None else bool(fan_auto)
    pending = fan.get("pending_ppv_check") or None

    return {
        "status": "ok",
        "fan_id": fan_id,
        "creator_id": creator_id,
        "effective_auto_mode": effective_auto,
        "fan_auto_mode": fan_auto,
        "creator_auto_mode": creator_auto,
        "needs_human_review": bool(fan.get("needs_human_review")),
        "review_reason": fan.get("review_reason"),
        "commercial_state": state.model_dump(mode="json"),
        "session": session,
        "pending_ppv": pending,
        "scheduled_actions": actions,
        "last_active": fan.get("last_active"),
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

    states, fans, actions = await asyncio.to_thread(_load)
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
