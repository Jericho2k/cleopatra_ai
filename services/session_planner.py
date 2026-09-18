"""Turn one accepted offer into the one locked PPV step that delivers it.

The commercial policy decides WHAT was accepted and at WHAT price. This service
turns that into a single purchase-gated step. It does not infer how much the fan
can spend, it does not choose a different price, and — since the two-package
flow was removed — it does not split one payment across several deliveries.

A "session" here is now one unlock. The next unlock is planned separately, later,
at a price decided then, only if the conversation actually gets there. That is
what makes the fan's spend incremental instead of a prepaid bundle, and it is why
nothing in this module knows how many steps might eventually follow.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from core.simulation_catalog import exclude_simulation_only, run_live_catalog_query
from core.supabase import get_supabase
from db.commercial_queries import get_fan_state
from db.queries import get_sent_ppv, save_fan_session
from models.commercial import FanStatus
from db.pricing_policy_queries import get_effective_price_learning_policy
from services.media_packages import (
    allocate_step_pricing,
    is_video_row,
    usable_sets,
)


async def plan_session_for_fan(
    creator_id: str,
    fan_id: str,
    *,
    accepted_set_id: str | None = None,
    accepted_price_cents: int | None = None,
    persist: bool = True,
) -> dict[str, Any]:
    """Build the single locked step that delivers the accepted offer.

    ``persist=False`` is the semantic runtime's pre-generation validation
    phase. It resolves the exact approved media and price without changing fan
    state. The runtime persists that already-reviewed plan only after writing
    succeeds and a final stale-state check passes.
    """
    # The creator policy is deliberately not read here any more. The only
    # thing this function took from it was the purchase-gating flag, which
    # CreatorPolicy forces True and nothing downstream read — so the read was a
    # round trip to Supabase on every plan, for a value that could not vary.
    state = await get_fan_state(fan_id)

    set_id = str(accepted_set_id or state.accepted_offer_set_id or "").strip()
    if not set_id:
        return {"status": "missing_accepted_offer", "session": None}

    price_cents = (
        accepted_price_cents
        or state.accepted_offer_price_cents
        or state.confirmed_budget_cents
    )
    if not price_cents or int(price_cents) <= 0:
        return {"status": "missing_confirmed_budget", "session": None}
    price_cents = int(price_cents)

    rows = await _load_approved_sets(creator_id)
    sent_ppv = await get_sent_ppv(fan_id)
    sent_media_ids = {
        str(media_id)
        for row in sent_ppv
        for media_id in (row.get("media_ids") or [row.get("media_id")])
        if media_id
    }
    sent_set_ids = {
        str(row.get("set_id"))
        for row in sent_ppv
        if row.get("set_id")
    }
    sellable = usable_sets(rows, sent_set_ids=sent_set_ids)
    for row in sellable:
        row["media_ids"] = [mid for mid in row.get("media_ids", []) if str(mid) not in sent_media_ids]
    sellable = [row for row in sellable if row.get("media_ids")]
    if not sellable:
        return {"status": "no_sets", "session": None}

    row = next((item for item in sellable if str(item["id"]) == set_id), None)
    if row is None:
        return {
            "status": "selected_set_unavailable",
            "session": None,
            "missing_set_ids": [set_id],
        }

    pricing_policy = await get_effective_price_learning_policy(creator_id)
    allocations = allocate_step_pricing(
        price_cents,
        [row],
        step_cents=pricing_policy.customer_price_step_cents,
    )
    if allocations is None:
        # The accepted price is not a valid price for this content. Fail closed:
        # an unlock must never be delivered at a price nobody approved.
        print(
            f"[SESSION] no valid allocation fan={fan_id} "
            f"price=${price_cents / 100:.2f} set={set_id}"
        )
        return {"status": "no_valid_allocation", "session": None}

    cents = allocations[0]
    media_ids = [str(value) for value in (row.get("media_ids") or []) if value]
    is_individual_video = is_video_row(row)
    step = {
        "step_number": 1,
        "step_count": 1,
        "media_ids": media_ids,
        "media_id": media_ids[0],
        "price": round(cents / 100, 2),
        "price_cents": cents,
        "set_id": str(row["id"]),
        "scene_key": row.get("title") or row.get("location") or "private",
        "location": row.get("location"),
        "outfit": row.get("outfit"),
        "explicit_min": row.get("explicit_min"),
        "explicit_max": row.get("explicit_max"),
        "description": (
            f"{row.get('title') or row.get('location') or 'private'} video"
            if is_individual_video
            else (
                f"{row.get('title') or row.get('location') or 'private'} "
                f"bundle ({len(media_ids)} pcs)"
            )
        ),
        "asset_type": "video" if is_individual_video else "photo_set",
        "sent": False,
        "purchased": False,
        "declined": False,
    }

    now = datetime.now(timezone.utc).isoformat()
    session = {
        "status": "active",
        "plan": [step],
        "current_index": 0,
        "awaiting_purchase_index": None,
        "started_at": now,
        "updated_at": now,
        "set_id": step["set_id"],
        "set_ids": [step["set_id"]],
        "scene_key": step["scene_key"],
        "commercial_offer_id": state.accepted_offer_id,
        "confirmed_budget_cents": cents,
        "revenue_cents": 0,
        "payment_state": "OFFER_SELECTED",
        # No require_purchase_before_next_step snapshot: it is forced True by
        # CreatorPolicy's own validator and nothing ever read the copy, so it
        # was a field in persisted JSON that could only ever say one thing.
    }
    if persist:
        await save_fan_session(fan_id, session)

        # A plan authorizes the locked PPV; it is not paid until the platform
        # confirms the unlock.
        state.status = FanStatus.OFFER_SELECTED
        state.confirmed_budget_cents = cents
        state.accepted_offer_set_id = step["set_id"]
        from db.commercial_queries import save_fan_state

        await save_fan_state(fan_id, creator_id, state)

    print(
        f"[SESSION] planned fan={fan_id} unlock={step['set_id']} "
        f"price=${cents / 100:.2f} persisted={persist}"
    )
    return {"status": "ok", "session": session}


async def _load_approved_sets(creator_id: str) -> list[dict[str, Any]]:
    def _get() -> list[dict[str, Any]]:
        def _build(apply_filter: bool):
            query = (
                get_supabase().table("vault_sets")
                .select(
                    "id, title, description, location, outfit, explicit_min, explicit_max, "
                    "media_ids, preview_media_id, suggested_price, tags, base_price_cents, "
                    "min_price_cents, max_price_cents, dynamic_pricing_enabled"
                )
                .eq("creator_id", creator_id)
                .eq("status", "approved")
            )
            # Owner-only mirrored test content is real inventory to the
            # simulator and invisible to everything else.
            if apply_filter:
                query = exclude_simulation_only(query)
            return query.execute()

        response = run_live_catalog_query(_build, label="session_planner.approved_sets")
        return response.data or []

    return await asyncio.to_thread(_get)
