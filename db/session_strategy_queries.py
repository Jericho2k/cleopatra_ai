"""Persistence for current and historical adaptive session strategies."""

from __future__ import annotations

import asyncio
from typing import Any

from core.supabase import get_supabase


async def get_session_strategy(fan_id: str) -> dict[str, Any]:
    def _get() -> dict[str, Any] | None:
        result = (
            get_supabase()
            .table("fan_session_strategies")
            .select("*")
            .eq("fan_id", fan_id)
            .limit(1)
            .execute()
        )
        return (result.data or [None])[0]

    try:
        return (await asyncio.to_thread(_get)) or {}
    except Exception as exc:
        print(f"[ADAPTIVE PLANNER] strategy read failed fan={fan_id}: {exc}")
        return {}


async def save_session_strategy(
    *, creator_id: str, fan_id: str, strategy: dict[str, Any]
) -> None:
    payload = {"creator_id": creator_id, "fan_id": fan_id, **strategy}
    await asyncio.to_thread(
        lambda: get_supabase()
        .table("fan_session_strategies")
        .upsert(payload, on_conflict="fan_id")
        .execute()
    )


async def insert_session_strategy_audit(payload: dict[str, Any]) -> None:
    def _insert() -> None:
        db = get_supabase()
        existing = (
            db.table("fan_session_strategy_audits")
            .select("id")
            .eq("dedupe_key", payload["dedupe_key"])
            .limit(1)
            .execute()
        )
        if existing.data:
            return
        db.table("fan_session_strategy_audits").insert(payload).execute()

    await asyncio.to_thread(_insert)
