"""Persistence for the current conversational scene.

Deliberately best-effort in the same way ``conversation_director_queries`` is:
a scene that cannot be read is a scene that starts fresh, not a reply that
never goes out. It is choreography, and choreography must never be able to
block a conversation.
"""

from __future__ import annotations

import asyncio
from typing import Any

from core.supabase import get_supabase


async def get_scene(fan_id: str) -> dict[str, Any]:
    def _get() -> dict[str, Any] | None:
        result = (
            get_supabase()
            .table("fan_experience_scenes")
            .select("*")
            .eq("fan_id", fan_id)
            .limit(1)
            .execute()
        )
        return (result.data or [None])[0]

    try:
        return (await asyncio.to_thread(_get)) or {}
    except Exception as exc:
        print(f"[EXPERIENCE] scene read failed fan={fan_id}: {exc}")
        return {}


async def save_scene(*, creator_id: str, fan_id: str, scene: dict[str, Any]) -> None:
    payload = {"creator_id": creator_id, "fan_id": fan_id, **scene}
    await asyncio.to_thread(
        lambda: get_supabase()
        .table("fan_experience_scenes")
        .upsert(payload, on_conflict="fan_id")
        .execute()
    )


async def get_set_scene_row(creator_id: str, set_id: str) -> dict[str, Any]:
    """The approved experience metadata for one vault set.

    Reads only the columns the scene is allowed to be built from. Notably NOT
    price columns: the scene never carries money, so it must not be able to
    leak one through a metadata read.
    """
    if not set_id:
        return {}

    def _get() -> dict[str, Any] | None:
        result = (
            get_supabase()
            .table("vault_sets")
            .select(
                "id, title, description, location, outfit, explicit_min, explicit_max, "
                "tags, scene_key, scene_premise, intensity_level, reveals, setup_line, "
                "continuation, paid_sellable"
            )
            .eq("creator_id", creator_id)
            .eq("id", set_id)
            .limit(1)
            .execute()
        )
        return (result.data or [None])[0]

    try:
        return (await asyncio.to_thread(_get)) or {}
    except Exception as exc:
        print(f"[EXPERIENCE] set metadata read failed set={set_id}: {exc}")
        return {}
