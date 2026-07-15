"""Persistence for current conversation progression and audit history."""

from __future__ import annotations

import asyncio
from typing import Any

from core.supabase import get_supabase


async def get_conversation_director(fan_id: str) -> dict[str, Any]:
    def _get() -> dict[str, Any] | None:
        result = (
            get_supabase()
            .table("fan_conversation_directors")
            .select("*")
            .eq("fan_id", fan_id)
            .limit(1)
            .execute()
        )
        return (result.data or [None])[0]

    try:
        return (await asyncio.to_thread(_get)) or {}
    except Exception as exc:
        print(f"[CONVERSATION DIRECTOR] read failed fan={fan_id}: {exc}")
        return {}


async def save_conversation_director(
    *,
    creator_id: str,
    fan_id: str,
    state: dict[str, Any],
) -> None:
    payload = {"creator_id": creator_id, "fan_id": fan_id, **state}
    await asyncio.to_thread(
        lambda: get_supabase()
        .table("fan_conversation_directors")
        .upsert(payload, on_conflict="fan_id")
        .execute()
    )


async def insert_conversation_director_audit(payload: dict[str, Any]) -> None:
    def _insert() -> None:
        db = get_supabase()
        existing = (
            db.table("fan_conversation_director_audits")
            .select("id")
            .eq("dedupe_key", payload["dedupe_key"])
            .limit(1)
            .execute()
        )
        if existing.data:
            return
        db.table("fan_conversation_director_audits").insert(payload).execute()

    await asyncio.to_thread(_insert)
