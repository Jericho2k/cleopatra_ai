"""Supabase persistence for deterministic price-learning profiles."""

from __future__ import annotations

import asyncio
import os
from typing import Any

from core.supabase import get_supabase
from models.price_learning import (
    PriceLearningPolicy,
    PriceLearningProfile,
    profile_from_row,
)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)) or default)
    except (TypeError, ValueError):
        return default


def _default_policy() -> PriceLearningPolicy:
    return PriceLearningPolicy(
        min_offer_cents=max(0, _env_int("PRICE_LEARNING_MIN_OFFER_CENTS", 500)),
        max_offer_cents=max(100, _env_int("PRICE_LEARNING_MAX_OFFER_CENTS", 50_000)),
        first_purchase_target_cents=max(
            0, _env_int("PRICE_LEARNING_FIRST_PURCHASE_TARGET_CENTS", 2_500)
        ),
        repeat_buyer_uplift_bps=max(
            0, _env_int("PRICE_LEARNING_REPEAT_UPLIFT_BPS", 1_000)
        ),
        vip_uplift_bps=max(0, _env_int("PRICE_LEARNING_VIP_UPLIFT_BPS", 1_500)),
        max_step_up_bps=max(
            0, _env_int("PRICE_LEARNING_MAX_STEP_UP_BPS", 2_500)
        ),
        range_width_bps=max(
            0, _env_int("PRICE_LEARNING_RANGE_WIDTH_BPS", 2_000)
        ),
        price_step_cents=max(1, _env_int("PRICE_LEARNING_PRICE_STEP_CENTS", 500)),
        evidence_lookback_days=max(
            1, _env_int("PRICE_LEARNING_LOOKBACK_DAYS", 365)
        ),
    )


async def get_price_learning_policy(creator_id: str) -> PriceLearningPolicy:
    def _get() -> dict[str, Any] | None:
        response = (
            get_supabase()
            .table("creator_price_learning_policies")
            .select("*")
            .eq("creator_id", creator_id)
            .limit(1)
            .execute()
        )
        return (response.data or [None])[0]

    try:
        row = await asyncio.to_thread(_get)
    except Exception as exc:
        print(f"[PRICE LEARNING] policy read failed creator={creator_id}: {exc}")
        return _default_policy()
    if not row:
        return _default_policy()
    payload = dict(row)
    payload.pop("creator_id", None)
    payload.pop("updated_at", None)
    try:
        return PriceLearningPolicy.model_validate(
            {**_default_policy().model_dump(), **payload}
        )
    except Exception:
        return _default_policy()


async def get_price_learning_profile(fan_id: str) -> PriceLearningProfile:
    def _get() -> dict[str, Any] | None:
        response = (
            get_supabase()
            .table("fan_price_learning_profiles")
            .select("*")
            .eq("fan_id", fan_id)
            .limit(1)
            .execute()
        )
        return (response.data or [None])[0]

    try:
        return profile_from_row(await asyncio.to_thread(_get))
    except Exception as exc:
        print(f"[PRICE LEARNING] profile read failed fan={fan_id}: {exc}")
        return PriceLearningProfile()


async def save_price_learning_profile(
    *, creator_id: str, fan_id: str, profile: PriceLearningProfile
) -> None:
    payload = {
        "creator_id": creator_id,
        "fan_id": fan_id,
        **profile.to_context(),
        "state_version": profile.state_version,
    }
    await asyncio.to_thread(
        lambda: get_supabase()
        .table("fan_price_learning_profiles")
        .upsert(payload, on_conflict="fan_id")
        .execute()
    )


async def insert_price_learning_audit(payload: dict[str, Any]) -> None:
    def _insert() -> None:
        db = get_supabase()
        existing = (
            db.table("fan_price_learning_audits")
            .select("id")
            .eq("dedupe_key", payload["dedupe_key"])
            .limit(1)
            .execute()
        )
        if existing.data:
            return
        try:
            db.table("fan_price_learning_audits").insert(payload).execute()
        except Exception:
            existing_after = (
                db.table("fan_price_learning_audits")
                .select("id")
                .eq("dedupe_key", payload["dedupe_key"])
                .limit(1)
                .execute()
            )
            if not existing_after.data:
                raise

    await asyncio.to_thread(_insert)
