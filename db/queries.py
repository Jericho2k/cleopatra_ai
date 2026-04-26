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
        auto_mode=row.get("auto_mode"),
        platform_fan_id=str(row["platform_fan_id"]) if row.get("platform_fan_id") is not None else None,
        fansly_group_id=str(row["fansly_group_id"]) if row.get("fansly_group_id") is not None else None,
        total_spent=row.get("total_spent", 0),
        spend_tier=row.get("spend_tier", "cold"),
        last_active=last_active,
        preferences=row.get("preferences") or [],
        notes=row.get("notes", ""),
        member_note=row.get("member_note", ""),
        model_note=row.get("model_note", ""),
        ai_summary=row.get("ai_summary"),
    )


def _row_to_message(row: dict) -> Message:
    sent_at = row.get("sent_at")
    if sent_at is not None and isinstance(sent_at, str):
        sent_at = datetime.fromisoformat(sent_at.replace("Z", "+00:00"))
    return Message(
        role=row["role"],
        content=row["content"],
        sent_at=sent_at,
        media_context=row.get("media_context"),
    )


async def get_fan(creator_id: str, platform_fan_id: str) -> Fan | None:
    def _get():
        r = get_supabase().table("fans").select("*").eq("creator_id", creator_id).eq("platform_fan_id", platform_fan_id).execute()
        if not r.data or len(r.data) == 0:
            return None
        return _row_to_fan(r.data[0])

    return await asyncio.to_thread(_get)


async def get_fan_by_id(fan_id: str) -> Fan | None:
    def _get():
        r = get_supabase().table("fans").select("*").eq("id", fan_id).execute()
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


async def increment_fan_total_spent(fan_id: str, amount: int) -> None:
    def _update():
        get_supabase().rpc(
            "increment_fan_spent",
            {
                "fan_id_input": fan_id,
                "amount_input": amount,
            },
        ).execute()

    await asyncio.to_thread(_update)


async def get_sent_ppv(fan_id: str) -> list[dict]:
    """Return list of PPV media already sent to this fan with purchase status."""
    def _get():
        r = get_supabase().table("messages") \
            .select("media_context, sent_at") \
            .eq("fan_id", fan_id) \
            .eq("role", "creator") \
            .not_.is_("media_context", "null") \
            .order("sent_at", desc=False) \
            .execute()
        sent = []
        for row in (r.data or []):
            mc = row.get("media_context") or {}
            ppv = mc.get("ppv")
            if ppv and ppv.get("media_id"):
                sent.append({
                    "media_id": ppv["media_id"],
                    "price": ppv.get("price", 0),
                    "purchased": ppv.get("purchased", False),
                    "sent_at": row.get("sent_at", ""),
                })
        return sent
    return await asyncio.to_thread(_get)


async def get_conversation_history(fan_id: str, limit: int = 40) -> list[Message]:
    def _get():
        r = get_supabase().table("messages").select("role, content, sent_at, media_context").eq("fan_id", fan_id).order("sent_at", desc=False).limit(limit).execute()
        return [_row_to_message(row) for row in (r.data or [])]

    return await asyncio.to_thread(_get)


async def save_message(
    fan_id: str,
    creator_id: str,
    role: str,
    content: str,
    was_ai_suggested: bool = False,
    fansly_message_id: str | None = None,
    media_context: dict | None = None,
) -> None:
    """media_context: e.g. {"attachments": [...]} for fan, {"ppv": {...}} for creator PPV."""

    def _save():
        row = {
            "fan_id": fan_id,
            "creator_id": creator_id,
            "role": role,
            "content": content,
            "was_ai_suggested": was_ai_suggested,
        }
        if fansly_message_id is not None:
            row["fansly_message_id"] = fansly_message_id
        if media_context is not None:
            row["media_context"] = media_context
        get_supabase().table("messages").insert(row).execute()

    await asyncio.to_thread(_save)


async def get_creator_fansly_account_id(creator_id: str) -> str | None:
    def _get():
        r = (
            get_supabase()
            .table("creators")
            .select("fansly_account_id")
            .eq("id", creator_id)
            .limit(1)
            .execute()
        )
        if not r.data:
            return None
        v = r.data[0].get("fansly_account_id")
        return str(v) if v is not None else None

    return await asyncio.to_thread(_get)


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


async def get_ppv_offers(creator_id: str) -> list[dict]:
    def _get():
        ppv = (
            get_supabase()
            .table("ppv_offers")
            .select("title, description, price")
            .eq("creator_id", creator_id)
            .execute()
        )
        vault = (
            get_supabase()
            .table("creator_vault_media")
            .select("title, description, price, fansly_media_id, account_media_id")
            .eq("creator_id", creator_id)
            .eq("is_active", True)
            .execute()
        )

        offers: list[dict] = []
        for row in ppv.data or []:
            offers.append(
                {
                    "title": row["title"],
                    "description": row.get("description", ""),
                    "price": row["price"],
                }
            )
        for row in vault.data or []:
            offers.append(
                {
                    "title": row["title"],
                    "description": row.get("description", ""),
                    "price": row.get("price", 0),
                    "media_id": row.get("account_media_id") or row.get("fansly_media_id"),
                }
            )
        return offers

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
            ExchangeExample(
                fan_message=row["fan_message"],
                creator_reply=row["creator_reply"],
            )
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
    member_note: str = "",
    model_note: str = "",
) -> None:
    def _update():
        get_supabase().table("fans").update({
            "notes": notes,
            "preferences": preferences,
            "spend_tier": spend_tier,
            "member_note": member_note,
            "model_note": model_note,
        }).eq("id", fan_id).execute()

    await asyncio.to_thread(_update)


async def update_fan_ai_summary(fan_id: str, summary: dict) -> None:
    def _update():
        get_supabase().table("fans").update({
            "ai_summary": summary,
        }).eq("id", fan_id).execute()

    await asyncio.to_thread(_update)
