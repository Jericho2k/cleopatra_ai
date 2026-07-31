"""Coherent multi-step paid-session planner.

The commercial policy selects a confirmed package/budget. This service turns
that contract into 1–4 approved, visually coherent PPV steps. It does not infer
how much the fan can spend and it does not choose a different price.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from core.supabase import get_supabase
from db.commercial_queries import get_creator_policy, get_fan_state
from db.queries import get_fan_by_id, get_sent_ppv, save_fan_session
from models.commercial import FanStatus
from services.media_packages import (
    allocate_budget,
    choose_sequence,
    explicitness,
    usable_sets,
)


async def plan_session_for_fan(
    creator_id: str,
    fan_id: str,
    *,
    selected_set_ids: list[str] | None = None,
    selected_price_cents: int | None = None,
    confirmed_kinks: list[str] | None = None,
) -> dict[str, Any]:
    policy = await get_creator_policy(creator_id)
    state = await get_fan_state(fan_id)
    fan = await get_fan_by_id(fan_id)

    authoritative_set_ids = [str(value) for value in (selected_set_ids or []) if value]
    if not authoritative_set_ids:
        authoritative_set_ids = [str(value) for value in state.selected_package_set_ids if value]
    if not authoritative_set_ids and state.selected_package_set_id:
        authoritative_set_ids = [str(state.selected_package_set_id)]

    budget_cents = (
        selected_price_cents
        or state.selected_package_price_cents
        or state.confirmed_budget_cents
    )
    if not budget_cents or int(budget_cents) <= 0:
        return {"status": "missing_confirmed_budget", "session": None}
    budget_cents = int(budget_cents)

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

    by_id = {str(row["id"]): row for row in sellable}
    if authoritative_set_ids:
        missing = [set_id for set_id in authoritative_set_ids if set_id not in by_id]
        if missing:
            return {
                "status": "selected_set_unavailable",
                "session": None,
                "missing_set_ids": missing,
            }
        sequence = [by_id[set_id] for set_id in authoritative_set_ids]
    else:
        preferred = list(confirmed_kinks or [])
        if not preferred and fan and getattr(fan, "ai_summary", None):
            preferred = list((fan.ai_summary or {}).get("kinks") or [])
        sequence = choose_sequence(
            sellable,
            target_cents=budget_cents,
            min_steps=policy.session_min_steps,
            max_steps=policy.session_max_steps,
            preferred_tags=preferred,
        )

    if not sequence:
        return {"status": "no_coherent_sequence", "session": None}

    # Always escalate within the already-coherent selected sequence.
    sequence = sorted(sequence, key=lambda row: (explicitness(row), str(row.get("id"))))
    allocations = allocate_budget(budget_cents, sequence)
    plan: list[dict[str, Any]] = []
    for index, (row, cents) in enumerate(zip(sequence, allocations, strict=True)):
        media_ids = [str(value) for value in (row.get("media_ids") or []) if value]
        is_individual_video = "individual_video" in (row.get("tags") or [])
        plan.append({
            "step_number": index + 1,
            "media_ids": media_ids,
            "media_id": media_ids[0],  # compatibility with current executor
            "price": round(cents / 100, 2),
            "price_cents": cents,
            "set_id": str(row["id"]),
            "scene_key": row.get("title") or row.get("location") or f"step {index + 1}",
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
        })

    now = datetime.now(timezone.utc).isoformat()
    session = {
        "status": "active",
        "plan": plan,
        "current_index": 0,
        "awaiting_purchase_index": None,
        "started_at": now,
        "updated_at": now,
        "fan_kinks": list(confirmed_kinks or []),
        "set_id": plan[0]["set_id"],
        "set_ids": [item["set_id"] for item in plan],
        "scene_key": plan[0]["scene_key"],
        "commercial_package_id": state.selected_package_id,
        "confirmed_budget_cents": budget_cents,
        "total_budget_cents": budget_cents,
        "revenue_cents": 0,
        "payment_state": "OFFER_SELECTED",
        "post_ppv_cooldown": False,
        "cooldown_messages_remaining": 0,
        "require_purchase_before_next_step": policy.require_purchase_before_next_step,
    }
    await save_fan_session(fan_id, session)

    # A plan authorizes the first locked PPV; it is not a paid session until
    # the platform confirms an unlock.
    state.status = FanStatus.OFFER_SELECTED
    state.confirmed_budget_cents = budget_cents
    state.selected_package_set_ids = [item["set_id"] for item in plan]
    from db.commercial_queries import save_fan_state
    await save_fan_state(fan_id, creator_id, state)

    print(
        f"[SESSION] planned fan={fan_id} steps={len(plan)} "
        f"budget=${budget_cents / 100:.2f} sets={session['set_ids']}"
    )
    return {"status": "ok", "session": session}


async def _load_approved_sets(creator_id: str) -> list[dict[str, Any]]:
    def _get() -> list[dict[str, Any]]:
        response = (
            get_supabase().table("vault_sets")
            .select(
                "id, title, description, location, outfit, explicit_min, explicit_max, "
                "media_ids, preview_media_id, suggested_price, tags, base_price_cents, "
                "min_price_cents, max_price_cents, dynamic_pricing_enabled"
            )
            .eq("creator_id", creator_id)
            .eq("status", "approved")
            .execute()
        )
        return response.data or []

    return await asyncio.to_thread(_get)
