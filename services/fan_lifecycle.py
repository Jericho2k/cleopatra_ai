"""Deterministic first-purchase, repeat-buyer, and VIP lifecycle service."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any

from db.commercial_queries import get_fan_state
from db.fan_lifecycle_queries import (
    get_lifecycle_policy,
    get_lifecycle_state,
    get_purchase_aggregates,
    insert_lifecycle_transition,
    lifecycle_row_to_context,
    save_lifecycle_state,
)
from db.queries import get_fan_by_id
from models.fan_lifecycle import LifecycleInputs, derive_lifecycle


def lifecycle_enabled() -> bool:
    return os.getenv("FAN_LIFECYCLE_ENABLED", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        dumped = value.model_dump(mode="json")
        return dumped if isinstance(dumped, dict) else {}
    return {}


def _parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _purchase_intent_signal(
    situation: dict[str, Any],
    decision: dict[str, Any],
    commercial_state: Any,
    active_session: dict[str, Any] | None,
) -> bool:
    purchase_signal = str(situation.get("purchase_signal") or "").lower()
    offer_response = str(situation.get("offer_response") or "").lower()
    action = str(decision.get("action") or "").upper()
    state_status = str(getattr(commercial_state, "status", "") or "").upper()

    return bool(
        purchase_signal in {"ready_to_buy", "money_available", "bought"}
        or offer_response == "accepted"
        or action
        in {
            "PRESENT_SESSION_OPTIONS",
            "CREATE_PAID_SESSION",
            "SEND_NEXT_PPV_STEP",
            "RESUME_PREVIOUS_OFFER",
        }
        or state_status in {"OFFER_PENDING", "PAID_SESSION_ACTIVE"}
        or bool(active_session)
    )


def _transition_key(
    *,
    fan_id: str,
    from_stage: str | None,
    to_stage: str,
    purchase_count: int,
    total_spent_cents: int,
    trigger_type: str,
) -> str:
    raw = "|".join(
        [
            fan_id,
            from_stage or "",
            to_stage,
            str(purchase_count),
            str(total_spent_cents),
            trigger_type,
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


async def get_fan_lifecycle_context(fan_id: str) -> dict[str, Any]:
    if not lifecycle_enabled():
        return {}
    row = await get_lifecycle_state(fan_id)
    return lifecycle_row_to_context(row)


async def refresh_fan_lifecycle(
    *,
    creator_id: str,
    fan_id: str,
    situation: dict[str, Any] | None = None,
    commercial_decision: Any = None,
    active_session: dict[str, Any] | None = None,
    fan_profile: Any = None,
    trigger_type: str = "message",
) -> dict[str, Any]:
    """Refresh lifecycle best-effort; lifecycle failure must never block a reply."""

    if not lifecycle_enabled():
        return {}

    try:
        existing = await get_lifecycle_state(fan_id)
        policy = await get_lifecycle_policy(creator_id)
        purchases = await get_purchase_aggregates(fan_id)
        profile = fan_profile or await get_fan_by_id(fan_id)
        commercial_state = await get_fan_state(fan_id)
        decision = _mapping(commercial_decision)
        situation_map = situation or {}

        lifecycle = derive_lifecycle(
            LifecycleInputs(
                purchase_count=int(purchases.get("purchase_count") or 0),
                purchase_revenue_cents=int(
                    purchases.get("purchase_revenue_cents") or 0
                ),
                fan_total_spent_cents=int(
                    purchases.get("fan_total_spent_cents") or 0
                ),
                first_purchase_at=purchases.get("first_purchase_at"),
                last_purchase_at=purchases.get("last_purchase_at"),
                purchase_intent_signal=_purchase_intent_signal(
                    situation_map,
                    decision,
                    commercial_state,
                    active_session,
                ),
                existing_intent_expires_at=_parse_datetime(
                    (existing or {}).get("intent_expires_at")
                ),
                active_paid_session=bool(active_session),
                sales_paused=bool(getattr(profile, "sale_paused_at", None)),
                needs_human_review=bool(
                    getattr(profile, "needs_human_review", False)
                ),
            ),
            policy,
        )

        previous_stage = str((existing or {}).get("stage") or "") or None
        await save_lifecycle_state(
            creator_id=creator_id,
            fan_id=fan_id,
            lifecycle=lifecycle,
        )

        if previous_stage != lifecycle.stage.value:
            metadata = {
                "flags": lifecycle.flags,
                "intent_expires_at": (
                    lifecycle.intent_expires_at.isoformat()
                    if lifecycle.intent_expires_at
                    else None
                ),
            }
            await insert_lifecycle_transition(
                {
                    "creator_id": creator_id,
                    "fan_id": fan_id,
                    "from_stage": previous_stage,
                    "to_stage": lifecycle.stage.value,
                    "trigger_type": trigger_type,
                    "reason_codes": lifecycle.reason_codes,
                    "purchase_count": lifecycle.purchase_count,
                    "total_spent_cents": lifecycle.total_spent_cents,
                    "metadata": metadata,
                    "dedupe_key": _transition_key(
                        fan_id=fan_id,
                        from_stage=previous_stage,
                        to_stage=lifecycle.stage.value,
                        purchase_count=lifecycle.purchase_count,
                        total_spent_cents=lifecycle.total_spent_cents,
                        trigger_type=trigger_type,
                    ),
                }
            )
            print(
                f"[LIFECYCLE] fan={fan_id} {previous_stage or 'NONE'} "
                f"-> {lifecycle.stage.value} ({','.join(lifecycle.reason_codes)})"
            )

        return lifecycle.to_context()
    except Exception as exc:
        print(f"[LIFECYCLE] refresh failed fan={fan_id}: {exc}")
        return await get_fan_lifecycle_context(fan_id)
