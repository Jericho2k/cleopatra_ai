"""All database reads and writes. No AI logic."""

import asyncio
from datetime import datetime

from core.supabase import get_supabase
from models.schemas import ExchangeExample, Fan, Message, Persona


def _row_to_fan(row: dict) -> Fan:
    last_active = row.get("last_active")
    if last_active is not None and isinstance(last_active, str):
        last_active = datetime.fromisoformat(last_active.replace("Z", "+00:00"))
    return Fan(
        id=str(row["id"]),
        display_name=row["display_name"],
        total_spent=row.get("total_spent", 0),
        spend_tier=row.get("spend_tier", "cold"),
        last_active=last_active,
        preferences=row.get("preferences") or [],
        notes=row.get("notes", ""),
    )


def _row_to_message(row: dict) -> Message:
    sent_at = row.get("sent_at")
    if sent_at is not None and isinstance(sent_at, str):
        sent_at = datetime.fromisoformat(sent_at.replace("Z", "+00:00"))
    return Message(
        role=row["role"],
        content=row["content"],
        sent_at=sent_at,
    )


async def get_fan(creator_id: str, platform_fan_id: str) -> Fan | None:
    def _get():
        r = get_supabase().table("fans").select("*").eq("creator_id", creator_id).eq("platform_fan_id", platform_fan_id).execute()
        if not r.data or len(r.data) == 0:
            return None
        return _row_to_fan(r.data[0])

    return await asyncio.to_thread(_get)


async def create_fan(creator_id: str, platform_fan_id: str, display_name: str) -> Fan:
    def _create():
        r = get_supabase().table("fans").insert({
            "creator_id": creator_id,
            "platform_fan_id": platform_fan_id,
            "display_name": display_name,
        }).execute()
        return _row_to_fan(r.data[0])

    return await asyncio.to_thread(_create)


async def update_fan_spend(fan_id: str, total_spent: int, spend_tier: str) -> None:
    def _update():
        get_supabase().table("fans").update({"total_spent": total_spent, "spend_tier": spend_tier}).eq("id", fan_id).execute()

    await asyncio.to_thread(_update)


async def get_conversation_history(fan_id: str, limit: int = 40) -> list[Message]:
    def _get():
        r = get_supabase().table("messages").select("role, content, sent_at").eq("fan_id", fan_id).order("sent_at", desc=False).limit(limit).execute()
        return [_row_to_message(row) for row in (r.data or [])]

    return await asyncio.to_thread(_get)


async def save_message(fan_id: str, creator_id: str, role: str, content: str, was_ai_suggested: bool = False) -> None:
    def _save():
        get_supabase().table("messages").insert({
            "fan_id": fan_id,
            "creator_id": creator_id,
            "role": role,
            "content": content,
            "was_ai_suggested": was_ai_suggested,
        }).execute()

    await asyncio.to_thread(_save)


async def get_creator_persona(creator_id: str) -> Persona | None:
    def _get():
        r = get_supabase().table("creators").select("persona").eq("id", creator_id).execute()
        if not r.data or len(r.data) == 0:
            return None
        raw = r.data[0].get("persona")
        if raw is None:
            return None
        return Persona.model_validate(raw)

    return await asyncio.to_thread(_get)


async def save_persona(creator_id: str, persona: Persona) -> None:
    def _save():
        get_supabase().table("creators").update({"persona": persona.model_dump()}).eq("id", creator_id).execute()

    await asyncio.to_thread(_save)


async def get_similar_exchanges(embedding: list[float], creator_id: str, limit: int = 5) -> list[ExchangeExample]:
    def _get():
        r = get_supabase().rpc("match_similar_exchanges", {
            "query_embedding": embedding,
            "p_creator_id": creator_id,
            "match_count": limit,
        }).execute()
        data = r.data or []
        return [
            ExchangeExample(fan_message=row["fan_message"], creator_reply=row["creator_response"])
            for row in data
        ]

    return await asyncio.to_thread(_get)


async def save_embedding(creator_id: str, fan_message: str, creator_response: str, stage: str, embedding: list[float]) -> None:
    def _save():
        get_supabase().table("message_embeddings").insert({
            "creator_id": creator_id,
            "fan_message": fan_message,
            "creator_response": creator_response,
            "conversation_stage": stage,
            "embedding": embedding,
        }).execute()

    await asyncio.to_thread(_save)


async def update_fan_memory(
    fan_id: str,
    notes: str,
    preferences: list[str],
    spend_tier: str,
) -> None:
    def _update():
        get_supabase().table("fans").update({
            "notes": notes,
            "preferences": preferences,
            "spend_tier": spend_tier,
        }).eq("id", fan_id).execute()

    await asyncio.to_thread(_update)
