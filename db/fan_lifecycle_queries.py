"""Persistence and authoritative purchase aggregates for buyer lifecycle."""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from typing import Any

from core.supabase import get_supabase
from models.fan_lifecycle import BuyerLifecycleStage, DerivedLifecycle, LifecyclePolicy


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)) or default)
    except (TypeError, ValueError):
        return default


def _default_policy() -> LifecyclePolicy:
    return LifecyclePolicy(
        vip_spend_cents=max(
            0, _env_int("LIFECYCLE_VIP_SPEND_CENTS", 50_000)
        ),
        vip_purchase_count=max(
            1, _env_int("LIFECYCLE_VIP_PURCHASE_COUNT", 5)
        ),
        repeat_buyer_purchase_count=max(
            2, _env_int("LIFECYCLE_REPEAT_BUYER_PURCHASE_COUNT", 2)
        ),
        first_purchase_intent_ttl_hours=min(
            720,
            max(
                1,
                _env_int("LIFECYCLE_FIRST_PURCHASE_INTENT_TTL_HOURS", 72),
            ),
        ),
    )


def _parse_sales_date(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    for fmt in ("%d.%m.%Y", "%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"):
        try:
            parsed = datetime.strptime(text[:19], fmt)
            return parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


async def get_lifecycle_policy(creator_id: str) -> LifecyclePolicy:
    def _get() -> dict[str, Any] | None:
        response = (
            get_supabase()
            .table("creator_lifecycle_policies")
            .select("*")
            .eq("creator_id", creator_id)
            .limit(1)
            .execute()
        )
        return (response.data or [None])[0]

    try:
        row = await asyncio.to_thread(_get)
    except Exception as exc:
        print(f"[LIFECYCLE] policy read failed creator={creator_id}: {exc}")
        return _default_policy()

    if not row:
        return _default_policy()
    row.pop("creator_id", None)
    row.pop("updated_at", None)
    try:
        defaults = _default_policy().model_dump()
        return LifecyclePolicy.model_validate({**defaults, **row})
    except Exception:
        return _default_policy()


async def get_lifecycle_state(fan_id: str) -> dict[str, Any] | None:
    def _get() -> dict[str, Any] | None:
        response = (
            get_supabase()
            .table("fan_lifecycle_states")
            .select("*")
            .eq("fan_id", fan_id)
            .limit(1)
            .execute()
        )
        return (response.data or [None])[0]

    try:
        return await asyncio.to_thread(_get)
    except Exception as exc:
        print(f"[LIFECYCLE] state read failed fan={fan_id}: {exc}")
        return None


async def get_purchase_aggregates(fan_id: str) -> dict[str, Any]:
    """Read confirmed sales from the existing idempotent sales log.

    `fans.total_spent` remains authoritative for aggregate value because it can
    include tips and other platform spend. `sales_log` supplies confirmed PPV
    purchase frequency without trusting chat claims.
    """

    def _get() -> dict[str, Any]:
        response = (
            get_supabase()
            .table("fans")
            .select("total_spent, sales_log")
            .eq("id", fan_id)
            .single()
            .execute()
        )
        row = response.data or {}
        sales = list(row.get("sales_log") or [])

        unique: dict[str, dict[str, Any]] = {}
        for entry in sales:
            if not isinstance(entry, dict):
                continue
            media_id = str(entry.get("media_id") or "").strip()
            key = media_id or "|".join(
                [
                    str(entry.get("date") or ""),
                    str(entry.get("item") or ""),
                    str(entry.get("amount") or ""),
                ]
            )
            unique.setdefault(key, entry)

        amounts: list[int] = []
        dates: list[datetime] = []
        for entry in unique.values():
            try:
                amounts.append(max(0, int(round(float(entry.get("amount") or 0) * 100))))
            except (TypeError, ValueError):
                amounts.append(0)
            parsed = _parse_sales_date(entry.get("date"))
            if parsed:
                dates.append(parsed)

        try:
            total_spent_cents = max(
                0,
                int(round(float(row.get("total_spent") or 0) * 100)),
            )
        except (TypeError, ValueError):
            total_spent_cents = 0

        return {
            "purchase_count": len(unique),
            "purchase_revenue_cents": sum(amounts),
            "fan_total_spent_cents": total_spent_cents,
            "first_purchase_at": min(dates) if dates else None,
            "last_purchase_at": max(dates) if dates else None,
        }

    return await asyncio.to_thread(_get)


async def save_lifecycle_state(
    *,
    creator_id: str,
    fan_id: str,
    lifecycle: DerivedLifecycle,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    payload = {
        "creator_id": creator_id,
        "fan_id": fan_id,
        "stage": lifecycle.stage.value,
        "purchase_count": lifecycle.purchase_count,
        "purchase_revenue_cents": lifecycle.purchase_revenue_cents,
        "total_spent_cents": lifecycle.total_spent_cents,
        "first_purchase_at": (
            lifecycle.first_purchase_at.isoformat()
            if lifecycle.first_purchase_at
            else None
        ),
        "last_purchase_at": (
            lifecycle.last_purchase_at.isoformat()
            if lifecycle.last_purchase_at
            else None
        ),
        "intent_expires_at": (
            lifecycle.intent_expires_at.isoformat()
            if lifecycle.intent_expires_at
            else None
        ),
        "flags": lifecycle.flags,
        "reason_codes": lifecycle.reason_codes,
        "state_version": 1,
        "updated_at": now,
    }

    await asyncio.to_thread(
        lambda: get_supabase()
        .table("fan_lifecycle_states")
        .upsert(payload, on_conflict="fan_id")
        .execute()
    )


async def insert_lifecycle_transition(payload: dict[str, Any]) -> None:
    def _insert() -> None:
        db = get_supabase()
        existing = (
            db.table("fan_lifecycle_transitions")
            .select("id")
            .eq("dedupe_key", payload["dedupe_key"])
            .limit(1)
            .execute()
        )
        if existing.data:
            return
        try:
            db.table("fan_lifecycle_transitions").insert(payload).execute()
        except Exception:
            existing = (
                db.table("fan_lifecycle_transitions")
                .select("id")
                .eq("dedupe_key", payload["dedupe_key"])
                .limit(1)
                .execute()
            )
            if not existing.data:
                raise

    await asyncio.to_thread(_insert)


def lifecycle_row_to_context(row: dict[str, Any] | None) -> dict[str, Any]:
    if not row:
        return {}
    stage = str(row.get("stage") or BuyerLifecycleStage.PROSPECT.value)
    return {
        "stage": stage,
        "purchase_count": int(row.get("purchase_count") or 0),
        "purchase_revenue_cents": int(row.get("purchase_revenue_cents") or 0),
        "total_spent_cents": int(row.get("total_spent_cents") or 0),
        "first_purchase_at": row.get("first_purchase_at"),
        "last_purchase_at": row.get("last_purchase_at"),
        "intent_expires_at": row.get("intent_expires_at"),
        "flags": row.get("flags") or {},
        "reason_codes": row.get("reason_codes") or [],
        "updated_at": row.get("updated_at"),
    }
