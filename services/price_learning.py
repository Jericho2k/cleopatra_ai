"""Price-learning orchestration over affordability and lifecycle evidence."""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

from db.affordability_queries import get_affordability_events
from db.pricing_policy_queries import get_effective_price_learning_policy
from db.price_learning_queries import (
    get_price_learning_policy,
    get_price_learning_profile,
    insert_price_learning_audit,
    save_price_learning_profile,
)
from models.price_learning import (
    PriceLearningProfile,
    derive_price_learning_profile,
    select_recommended_packages,
)
from services.affordability import get_affordability_context
from services.fan_lifecycle import get_fan_lifecycle_context


def price_learning_enabled() -> bool:
    return os.getenv("PRICE_LEARNING_ENABLED", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


async def get_price_learning_context(fan_id: str) -> dict[str, Any]:
    if not price_learning_enabled():
        return {}
    return (await get_price_learning_profile(fan_id)).to_context()


async def refresh_price_learning(
    *,
    creator_id: str,
    fan_id: str,
    affordability: dict[str, Any] | None = None,
    lifecycle: dict[str, Any] | None = None,
    trigger_type: str = "message",
) -> dict[str, Any]:
    """Refresh best-effort; pricing intelligence must never block a reply."""

    if not price_learning_enabled():
        return {}
    try:
        affordability_context = affordability or await get_affordability_context(fan_id)
        lifecycle_context = lifecycle or await get_fan_lifecycle_context(fan_id)
        events = await get_affordability_events(fan_id, limit=200)
        policy = await get_effective_price_learning_policy(creator_id)
        existing = await get_price_learning_profile(fan_id)
        profile = derive_price_learning_profile(
            events,
            affordability=affordability_context,
            lifecycle=lifecycle_context,
            policy=policy,
        )
        await save_price_learning_profile(
            creator_id=creator_id,
            fan_id=fan_id,
            profile=profile,
        )
        if _fingerprint(existing) != _fingerprint(profile):
            context = profile.to_context()
            await insert_price_learning_audit(
                {
                    "creator_id": creator_id,
                    "fan_id": fan_id,
                    "trigger_type": trigger_type,
                    "mode": profile.mode.value,
                    "confidence": profile.confidence.value,
                    "recommended_floor_cents": profile.recommended_floor_cents,
                    "recommended_target_cents": profile.recommended_target_cents,
                    "recommended_ceiling_cents": profile.recommended_ceiling_cents,
                    "reason_codes": profile.reason_codes,
                    "evidence_summary": profile.evidence_summary,
                    "dedupe_key": _dedupe_key(fan_id, trigger_type, context),
                }
            )
        print(
            f"[PRICE LEARNING] fan={fan_id} mode={profile.mode.value} "
            f"target={profile.recommended_target_cents} "
            f"confidence={profile.confidence.value}"
        )
        return profile.to_context()
    except Exception as exc:
        print(f"[PRICE LEARNING] refresh failed fan={fan_id}: {exc}")
        return (await get_price_learning_profile(fan_id)).to_context()


def _fingerprint(profile: PriceLearningProfile) -> str:
    payload = {
        "mode": profile.mode.value,
        "confidence": profile.confidence.value,
        "stage": profile.lifecycle_stage,
        "floor": profile.recommended_floor_cents,
        "target": profile.recommended_target_cents,
        "ceiling": profile.recommended_ceiling_cents,
        "reasons": profile.reason_codes,
        "summary": profile.evidence_summary,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _dedupe_key(fan_id: str, trigger_type: str, context: dict[str, Any]) -> str:
    payload = json.dumps(context, sort_keys=True, default=str)
    raw = f"{fan_id}|{trigger_type}|{payload}"
    return hashlib.sha256(raw.encode()).hexdigest()


__all__ = [
    "get_price_learning_context",
    "price_learning_enabled",
    "refresh_price_learning",
    "select_recommended_packages",
]
