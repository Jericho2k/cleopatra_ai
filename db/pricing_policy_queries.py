"""DB-backed agency -> creator -> environment pricing-policy resolution."""

from __future__ import annotations

import asyncio
import os
import time
from datetime import datetime, timezone
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
        cold_start_probe_bps=min(
            10_000, max(0, _env_int("PRICE_LEARNING_COLD_START_PROBE_BPS", 2_500))
        ),
        effortless_purchase_streak=max(
            1, _env_int("PRICE_LEARNING_EFFORTLESS_PURCHASE_STREAK", 2)
        ),
        customer_price_step_cents=max(
            1, _env_int("PRICE_LEARNING_CUSTOMER_PRICE_STEP_CENTS", 500)
        ),
    )


# Scoped pricing settings change when an operator edits them, not per message,
# but every inbound message now needs them to price an offer. A short in-process
# TTL keeps that from adding two Supabase round trips to every turn.
_POLICY_CACHE_TTL_SECONDS = max(
    0, _env_int("PRICE_LEARNING_POLICY_CACHE_SECONDS", 60)
)
_policy_cache: dict[str, tuple[float, PriceLearningPolicy]] = {}


def clear_price_learning_policy_cache(creator_id: str | None = None) -> None:
    """Drop cached scoped settings after an operator writes new ones."""
    if creator_id is None:
        _policy_cache.clear()
    else:
        _policy_cache.pop(str(creator_id), None)


async def get_effective_price_learning_policy(creator_id: str) -> PriceLearningPolicy:
    """Resolve environment fallback, then agency defaults, then creator overrides."""

    cache_key = str(creator_id)
    cached = _policy_cache.get(cache_key)
    if cached and (time.monotonic() - cached[0]) < _POLICY_CACHE_TTL_SECONDS:
        return cached[1]

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
        return _remember(cache_key, PriceLearningPolicy.model_validate(base))

    effective = dict(base)
    effective.update(_clean(agency_settings))
    effective.update(_clean(creator_settings))
    try:
        return _remember(cache_key, PriceLearningPolicy.model_validate(effective))
    except Exception as exc:
        print(f"[PRICING POLICY] invalid scoped settings creator={creator_id}: {exc}")
        return _remember(cache_key, PriceLearningPolicy.model_validate(base))


def _remember(cache_key: str, policy: PriceLearningPolicy) -> PriceLearningPolicy:
    if _POLICY_CACHE_TTL_SECONDS > 0:
        _policy_cache[cache_key] = (time.monotonic(), policy)
    return policy


def _clean(settings: Any) -> dict[str, Any]:
    if not isinstance(settings, dict):
        return {}
    return {key: value for key, value in settings.items() if key in _POLICY_FIELDS and value is not None}


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------
#
# The read path above has shipped since adaptive_planning_v1; the write path had
# not, so the two scope tables were only ever fillable by hand in the Supabase
# SQL editor. These are the smallest properly authorized writers for them.
#
# Authorization is NOT here. It lives on the route, which proves the caller may
# administer the creator before calling either of these, exactly as every other
# creator-scoped mutation does. What lives here is the guarantee that a write
# can only ever contain real policy fields with sane values: unknown keys are
# dropped, and every value is validated by PriceLearningPolicy before it is
# stored, so a malformed settings row can never be the reason an offer is
# mispriced at run time.


def _scope_type(value: str) -> str:
    scope = str(value or "").strip().upper()
    if scope not in {"AGENCY", "CREATOR"}:
        raise ValueError("scope_type must be AGENCY or CREATOR")
    return scope


def validate_policy_settings(settings: Any) -> dict[str, Any]:
    """Keep only real policy fields, and prove they make a valid policy.

    Validated against the whole policy — environment defaults filled in for
    anything absent — because the fields constrain each other. Storing a
    ``min_offer_cents`` above ``max_offer_cents`` would be accepted by a
    field-by-field check and would then fail at the moment an offer is built.
    """
    cleaned = _clean(settings)
    base = environment_price_learning_policy().model_dump()
    base.update(cleaned)
    policy = PriceLearningPolicy.model_validate(base)
    # The model validates each field's own range. The one constraint it cannot
    # express field-by-field is the relationship between the two offer bounds,
    # and an inverted pair would be accepted here and then fail at the moment an
    # offer is built — the worst possible place to find out.
    if policy.min_offer_cents > policy.max_offer_cents:
        raise ValueError(
            "min_offer_cents cannot exceed max_offer_cents "
            f"({policy.min_offer_cents} > {policy.max_offer_cents})"
        )
    return cleaned


async def get_policy_scope_settings(scope_type: str, scope_id: str) -> dict[str, Any]:
    """The raw stored settings for one scope. ``{}`` when none exist."""
    scope = _scope_type(scope_type)

    def _get() -> dict[str, Any]:
        response = (
            get_supabase().table("price_learning_policy_scopes")
            .select("settings")
            .eq("scope_type", scope)
            .eq("scope_id", str(scope_id))
            .limit(1)
            .execute()
        )
        rows = response.data or []
        return (rows[0] or {}).get("settings") or {}

    try:
        return await asyncio.to_thread(_get)
    except Exception as exc:
        print(f"[PRICING POLICY] read failed {scope}/{scope_id}: {exc}")
        return {}


async def save_policy_scope_settings(
    scope_type: str,
    scope_id: str,
    settings: dict[str, Any],
) -> dict[str, Any]:
    """Replace one scope's settings. Returns what was stored."""
    scope = _scope_type(scope_type)
    cleaned = validate_policy_settings(settings)

    def _write() -> None:
        (
            get_supabase().table("price_learning_policy_scopes")
            .upsert(
                {
                    "scope_type": scope,
                    "scope_id": str(scope_id),
                    "settings": cleaned,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
                on_conflict="scope_type,scope_id",
            )
            .execute()
        )

    await asyncio.to_thread(_write)
    # Every creator under this scope may now price differently, and the cache is
    # keyed by creator, so the safe move is to drop all of it. It refills from
    # one read per creator on the next inbound message.
    clear_price_learning_policy_cache()
    return cleaned


async def get_agency_scope_id(creator_id: str) -> str | None:
    """Which agency pricing scope this creator belongs to, if any."""

    def _get() -> str | None:
        response = (
            get_supabase().table("creator_pricing_scope_memberships")
            .select("agency_scope_id")
            .eq("creator_id", str(creator_id))
            .limit(1)
            .execute()
        )
        rows = response.data or []
        value = (rows[0] or {}).get("agency_scope_id") if rows else None
        return str(value) if value else None

    try:
        return await asyncio.to_thread(_get)
    except Exception as exc:
        print(f"[PRICING POLICY] membership read failed creator={creator_id}: {exc}")
        return None


async def set_agency_scope_membership(creator_id: str, agency_scope_id: str | None) -> None:
    """Place a creator in an agency pricing scope, or remove it from one."""

    def _write() -> None:
        (
            get_supabase().table("creator_pricing_scope_memberships")
            .upsert(
                {
                    "creator_id": str(creator_id),
                    "agency_scope_id": (
                        str(agency_scope_id) if agency_scope_id else None
                    ),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
                on_conflict="creator_id",
            )
            .execute()
        )

    await asyncio.to_thread(_write)
    clear_price_learning_policy_cache(creator_id)
