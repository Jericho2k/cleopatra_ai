"""Best-effort orchestration for deterministic next-best-action planning."""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

from db.session_strategy_queries import (
    get_session_strategy,
    insert_session_strategy_audit,
    save_session_strategy,
)
from models.session_strategy import SessionStrategy, derive_session_strategy


def adaptive_planner_enabled() -> bool:
    return os.getenv("ADAPTIVE_SESSION_PLANNER_ENABLED", "false").strip().lower() in {
        "1", "true", "yes", "on"
    }


async def get_adaptive_session_context(fan_id: str) -> dict[str, Any]:
    if not adaptive_planner_enabled():
        return {}
    row = await get_session_strategy(fan_id)
    if not row:
        return {}
    return {key: value for key, value in row.items() if key not in {"fan_id", "creator_id", "created_at"}}


async def plan_next_action(
    *,
    creator_id: str,
    fan_id: str,
    situation: dict[str, Any] | None = None,
    commercial_decision: dict[str, Any] | None = None,
    lifecycle: dict[str, Any] | None = None,
    affordability: dict[str, Any] | None = None,
    price_learning: dict[str, Any] | None = None,
    active_session: dict[str, Any] | None = None,
    conversation_stage: str | None = None,
    conversation_director: dict[str, Any] | None = None,
    trigger_type: str = "message",
) -> dict[str, Any]:
    if not adaptive_planner_enabled():
        return {}

    strategy = derive_session_strategy(
        situation=situation,
        commercial_decision=commercial_decision,
        lifecycle=lifecycle,
        affordability=affordability,
        price_learning=price_learning,
        active_session=active_session,
        conversation_stage=conversation_stage,
        conversation_director=conversation_director,
    )
    context = strategy.to_context()
    try:
        existing = await get_session_strategy(fan_id)
        await save_session_strategy(creator_id=creator_id, fan_id=fan_id, strategy=context)
        if _fingerprint(existing) != _fingerprint(context):
            await insert_session_strategy_audit(
                {
                    "creator_id": creator_id,
                    "fan_id": fan_id,
                    "trigger_type": trigger_type,
                    "goal": context["goal"],
                    "phase": context["phase"],
                    "next_action": context["next_action"],
                    "strategy": context,
                    "dedupe_key": _dedupe_key(fan_id, trigger_type, context),
                }
            )
        print(
            f"[ADAPTIVE PLANNER] fan={fan_id} goal={context['goal']} "
            f"action={context['next_action']} reasons={','.join(context['reason_codes'])}"
        )
    except Exception as exc:
        # Planning guidance must never block a reply. Commercial policy still acts.
        print(f"[ADAPTIVE PLANNER] persistence failed fan={fan_id}: {exc}")
    return context


def _fingerprint(value: dict[str, Any] | SessionStrategy | None) -> str:
    if isinstance(value, SessionStrategy):
        value = value.to_context()
    value = value or {}
    payload = {
        key: value.get(key)
        for key in (
            "goal", "phase", "next_action", "writer_goal", "writer_avoid",
            "approved_offer_ids", "approved_offer_prices_cents",
            "selected_offer_price_cents", "reason_codes",
        )
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _dedupe_key(fan_id: str, trigger_type: str, context: dict[str, Any]) -> str:
    raw = f"{fan_id}|{trigger_type}|{_fingerprint(context)}"
    return hashlib.sha256(raw.encode()).hexdigest()
