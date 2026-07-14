"""DB-backed agency -> creator -> environment pricing-policy resolution."""

from __future__ import annotations

import asyncio
import os
from typing import Any

from core.supabase import get_supabase
from models.price_learning import PriceLearningPolicy


_POLICY_FIELDS = set(PriceLearningPolicy.model_fields)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)) or default)
    except (TypeError, ValueError):
        return default


def environment_price_learning_policy() -> PriceLearningPolicy:
    return PriceLearningPolicy(
        min_offer_cents=max(0, _env_int("PRICE_LEARNING_MIN_OFFER_CENTS", 500)),
        max_offer_cents=max(100, _env_int("PRICE_LEARNING_MAX_OFFER_CENTS", 50_000)),
        first_purchase_target_cents=max(0, _env_int("PRICE_LEARNING_FIRST_PURCHASE_TARGET_CENTS", 2_500)),
        repeat_buyer_uplift_bps=max(0, _env_int("PRICE_LEARNING_REPEAT_UPLIFT_BPS", 1_000)),
        vip_uplift_bps=max(0, _env_int("PRICE_LEARNING_VIP_UPLIFT_BPS", 1_500)),
        max_step_up_bps=max(0, _env_int("PRICE_LEARNING_MAX_STEP_UP_BPS", 2_500)),
        range_width_bps=max(0, _env_int("PRICE_LEARNING_RANGE_WIDTH_BPS", 2_000)),
        price_step_cents=max(1, _env_int("PRICE_LEARNING_PRICE_STEP_CENTS", 500)),
        evidence_lookback_days=max(1, _env_int("PRICE_LEARNING_LOOKBACK_DAYS", 365)),
    )


async def get_effective_price_learning_policy(creator_id: str) -> PriceLearningPolicy:
    """Resolve environment fallback, then agency defaults, then creator overrides."""

    base = environment_price_learning_policy().model_dump()

    def _get() -> tuple[dict[str, Any], dict[str, Any]]:
        db = get_supabase()
        membership = (
            db.table("creator_pricing_scope_memberships")
            .select("agency_scope_id")
            .eq("creator_id", creator_id)
            .limit(1)
            .execute()
        )
        agency_scope_id = ((membership.data or [{}])[0]).get("agency_scope_id")
        agency_settings: dict[str, Any] = {}
        if agency_scope_id:
            row = (
                db.table("price_learning_policy_scopes")
                .select("settings")
                .eq("scope_type", "AGENCY")
                .eq("scope_id", str(agency_scope_id))
                .limit(1)
                .execute()
            )
            agency_settings = ((row.data or [{}])[0]).get("settings") or {}
        creator_row = (
            db.table("price_learning_policy_scopes")
            .select("settings")
            .eq("scope_type", "CREATOR")
            .eq("scope_id", str(creator_id))
            .limit(1)
            .execute()
        )
        creator_settings = ((creator_row.data or [{}])[0]).get("settings") or {}
        return agency_settings, creator_settings

    try:
        agency_settings, creator_settings = await asyncio.to_thread(_get)
    except Exception as exc:
        print(f"[PRICING POLICY] scoped read failed creator={creator_id}: {exc}")
        return PriceLearningPolicy.model_validate(base)

    effective = dict(base)
    effective.update(_clean(agency_settings))
    effective.update(_clean(creator_settings))
    try:
        return PriceLearningPolicy.model_validate(effective)
    except Exception as exc:
        print(f"[PRICING POLICY] invalid scoped settings creator={creator_id}: {exc}")
        return PriceLearningPolicy.model_validate(base)


def _clean(settings: Any) -> dict[str, Any]:
    if not isinstance(settings, dict):
        return {}
    return {key: value for key, value in settings.items() if key in _POLICY_FIELDS and value is not None}
