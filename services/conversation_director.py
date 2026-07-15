"""Best-effort persistent multi-turn conversation progression."""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

from db.conversation_director_queries import (
    get_conversation_director,
    insert_conversation_director_audit,
    save_conversation_director,
)
from models.conversation_director import (
    ConversationDirectorState,
    advance_conversation_director,
)


def conversation_director_enabled() -> bool:
    return os.getenv("CONVERSATION_DIRECTOR_ENABLED", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


async def direct_conversation(
    *,
    creator_id: str,
    fan_id: str,
    conversation_history: list[Any] | None = None,
    latest_fan_message: str = "",
    situation: dict[str, Any] | None = None,
    commercial_decision: dict[str, Any] | None = None,
    lifecycle: dict[str, Any] | None = None,
    active_session: dict[str, Any] | None = None,
    conversation_stage: str | None = None,
    trigger_type: str = "message",
) -> dict[str, Any]:
    if not conversation_director_enabled():
        return {}

    history = conversation_history or []
    fan_turn_count = sum(
        1 for message in history if str(getattr(message, "role", "")).lower() == "fan"
    )
    creator_turn_count = sum(
        1
        for message in history
        if str(getattr(message, "role", "")).lower() == "creator"
    )

    latest_already_present = bool(
        history
        and str(getattr(history[-1], "role", "")).lower() == "fan"
        and str(getattr(history[-1], "content", "")).strip()
        == str(latest_fan_message or "").strip()
    )
    if latest_fan_message and not latest_already_present:
        fan_turn_count += 1

    previous = await get_conversation_director(fan_id)
    state = advance_conversation_director(
        previous=previous,
        situation=situation,
        commercial_decision=commercial_decision,
        lifecycle=lifecycle,
        active_session=active_session,
        conversation_stage=conversation_stage,
        fan_turn_count=fan_turn_count,
        creator_turn_count=creator_turn_count,
    )
    context = state.to_context()

    try:
        await save_conversation_director(
            creator_id=creator_id,
            fan_id=fan_id,
            state=context,
        )
        if _fingerprint(previous) != _fingerprint(context):
            await insert_conversation_director_audit(
                {
                    "creator_id": creator_id,
                    "fan_id": fan_id,
                    "trigger_type": trigger_type,
                    "phase": context["phase"],
                    "action": context["action"],
                    "state": context,
                    "dedupe_key": _dedupe_key(fan_id, trigger_type, context),
                }
            )
        print(
            f"[CONVERSATION DIRECTOR] fan={fan_id} "
            f"phase={context['phase']} action={context['action']} "
            f"turns={context['turns_in_phase']} "
            f"engagement={context['engagement_score']} "
            f"reason={context['transition_reason']}"
        )
    except Exception as exc:
        print(f"[CONVERSATION DIRECTOR] persistence failed fan={fan_id}: {exc}")

    return context


def _fingerprint(value: dict[str, Any] | ConversationDirectorState | None) -> str:
    if isinstance(value, ConversationDirectorState):
        value = value.to_context()
    value = value or {}
    payload = {
        key: value.get(key)
        for key in (
            "phase",
            "action",
            "turns_in_phase",
            "same_action_streak",
            "qualification_complete",
            "offer_eligible",
            "transition_reason",
        )
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()


def _dedupe_key(
    fan_id: str,
    trigger_type: str,
    context: dict[str, Any],
) -> str:
    raw = f"{fan_id}|{trigger_type}|{_fingerprint(context)}"
    return hashlib.sha256(raw.encode()).hexdigest()
