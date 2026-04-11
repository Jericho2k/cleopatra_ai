"""Suggestion orchestration service.

Coordinates DB, stage classification, RAG, prompt building, and generation.
"""

import asyncio
import json
import os
import random
import re

import httpx

from ai.generator import generate_replies
from openai import AsyncOpenAI
from core.config import get_settings
from core.supabase import get_supabase
from ai.prompt_builder import build_prompt
from ai.situation_analyzer import analyze_situation
from ai.rag import find_similar_exchanges
from ai.stage_classifier import classify_stage
from db.queries import (
    get_conversation_history,
    get_creator_persona,
    get_fan,
    get_ppv_offers,
    save_message,
    update_fan_memory,
    update_fan_ai_summary,
)
from models.schemas import (
    ConversationContext,
    Fan,
    Message,
    Persona,
    SuggestionResponse,
)

together_client = AsyncOpenAI(
    base_url="https://api.together.xyz/v1",
    api_key=get_settings().TOGETHER_API_KEY,
)


async def get_suggestions(
    fan_id: str,
    creator_id: str,
    fan_message: str,
    creator_name: str = "a creator",
    save_fan_message: bool = True,
) -> SuggestionResponse:
    conversation_history = await get_conversation_history(fan_id)

    fan_profile = await get_fan(creator_id, fan_id)
    if fan_profile is None:
        fan_profile = Fan(id=fan_id, display_name=fan_id)

    creator_persona = await get_creator_persona(creator_id)
    if creator_persona is None:
        creator_persona = Persona()
    ppv_offers = await get_ppv_offers(creator_id)

    conversation_stage = classify_stage(conversation_history, fan_profile)

    similar_exchanges = await find_similar_exchanges(
        fan_message, creator_id, enabled=False
    )

    ctx_without_situation = ConversationContext(
        fan_message=fan_message,
        conversation_history=conversation_history,
        fan_profile=fan_profile,
        creator_persona=creator_persona,
        similar_exchanges=similar_exchanges,
        conversation_stage=conversation_stage,
        creator_name=creator_name,
        ppv_offers=ppv_offers,
    )

    situation = await analyze_situation(ctx_without_situation)

    ctx = ConversationContext(
        fan_message=fan_message,
        conversation_history=conversation_history,
        fan_profile=fan_profile,
        creator_persona=creator_persona,
        similar_exchanges=similar_exchanges,
        conversation_stage=conversation_stage,
        creator_name=creator_name,
        situation=situation,
        ppv_offers=ppv_offers,
    )

    prompt = build_prompt(ctx)
    replies = await generate_replies(prompt, creator_persona)

    if save_fan_message:
        await save_message(fan_id, creator_id, "fan", fan_message)

    if _should_update_memory(conversation_history):
        asyncio.create_task(_update_fan_memory(fan_id, creator_id, conversation_history, fan_profile.total_spent))
        asyncio.create_task(_update_fan_ai_summary(fan_id, conversation_history))

    return SuggestionResponse(suggestions=replies)


async def _update_fan_memory(
    fan_id: str,
    creator_id: str,
    conversation_history: list[Message],
    fan_total_spent: int,
) -> None:
    try:
        recent_messages = conversation_history[-20:]
        convo_lines: list[str] = []
        for msg in recent_messages:
            speaker = "Fan" if msg.role == "fan" else "Creator"
            convo_lines.append(f"{speaker}: {msg.content}")
        convo_text = "\n".join(convo_lines)

        system_prompt = (
            "You are a fan relationship analyst. Extract key facts about this fan "
            "from the conversation. Return only valid JSON, no markdown."
        )
        user_prompt = (
            "Based on the following conversation, return a JSON object with exactly "
            "these fields:\n"
            '{\n'
            '  "notes": "2-3 sentence summary of important facts about this fan",\n'
            '  "preferences": ["list of content preferences mentioned or implied"],\n'
            '  "spend_tier": "whale | active | casual | cold"\n'
            "}\n\n"
            "Conversation:\n"
            f"{convo_text}"
        )

        response = await together_client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )

        content = response.choices[0].message.content or ""
        lines = content.splitlines()
        cleaned_lines = [line for line in lines if not line.lstrip().startswith("```")]
        cleaned = "\n".join(cleaned_lines).strip() or content.strip()

        data = json.loads(cleaned)
        notes = data.get("notes", "")
        preferences = data.get("preferences") or []

        if not isinstance(preferences, list):
            preferences = []

        # Override AI's spend_tier with actual spend data
        actual_tier = "cold"
        if fan_total_spent >= 500:
            actual_tier = "whale"
        elif fan_total_spent >= 100:
            actual_tier = "active"
        elif fan_total_spent > 0:
            actual_tier = "casual"

        await update_fan_memory(
            fan_id=fan_id,
            notes=notes,
            preferences=preferences,
            spend_tier=actual_tier,  # use actual spend, not AI guess
        )
    except Exception:
        # Silent failure; this runs in the background
        return


async def _update_fan_ai_summary(
    fan_id: str,
    conversation_history: list[Message],
) -> None:
    try:
        convo_lines = []
        for msg in conversation_history[-20:]:
            speaker = "Fan" if msg.role == "fan" else "Creator"
            convo_lines.append(f"{speaker}: {msg.content}")
        convo_text = "\n".join(convo_lines)

        system_prompt = (
            "You are an expert fan relationship analyst for an OnlyFans agency. "
            "Analyze this conversation and extract a detailed psychological and behavioral profile of the fan. "
            "Return only valid JSON, no markdown, no explanation."
        )
        user_prompt = (
            "Analyze this conversation and return a JSON object with these fields:\n"
            "{\n"
            '  "real_name": "their real name if mentioned, otherwise null",\n'
            '  "location": "city/country if mentioned, otherwise null",\n'
            '  "occupation": "job or income signals if mentioned, otherwise null",\n'
            '  "relationship_status": "single/relationship/married/unknown",\n'
            '  "payday": "when they get paid if mentioned, otherwise null",\n'
            '  "kinks": ["list of explicit kinks, fetishes, or sexual preferences mentioned or clearly implied"],\n'
            '  "emotional_type": "one of: romantic | submissive | dominant | transactional | playful | mixed",\n'
            '  "spending_behavior": "description of how they spend — e.g. tips spontaneously, haggles on price, pays without hesitation",\n'
            '  "best_time_to_message": "time of day or days they seem most active, or null",\n'
            '  "reengagement_triggers": "what topics or messages get them most responsive",\n'
            '  "risk_signals": "any red flags like money problems, about to cancel, frustration — or null",\n'
            '  "summary": "3-4 sentence psychological profile of this fan — who they are, what they want, how to handle them"\n'
            "}\n\n"
            "Conversation:\n"
            f"{convo_text}"
        )

        response = await together_client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=1000,
        )

        content = response.choices[0].message.content or ""
        lines = content.splitlines()
        cleaned = "\n".join(
            line for line in lines if not line.lstrip().startswith("```")
        ).strip()

        data = json.loads(cleaned)
        await update_fan_ai_summary(fan_id=fan_id, summary=data)

    except Exception as e:
        print(f"[AI SUMMARY ERROR] fan={fan_id} error={e}")
        import traceback
        traceback.print_exc()


async def _send_auto_reply(fan_id: str, creator_id: str, reply: str) -> None:
    try:
        print(f"[AUTO REPLY] Starting delay for fan={fan_id} reply={reply[:50]}")
        delay = random.randint(45, 90)
        await asyncio.sleep(delay)

        parts = [p.strip() for p in reply.split("|") if p.strip()]

        db = get_supabase()
        fan_row = await asyncio.to_thread(
            lambda: db.table("fans")
            .select("fansly_group_id, platform_fan_id, creator_id")
            .eq("id", fan_id)
            .single()
            .execute()
        )

        creator_row = await asyncio.to_thread(
            lambda: db.table("creators")
            .select("fansly_account_id, apifansly_account_id")
            .eq("id", creator_id)
            .single()
            .execute()
        )

        group_id = (fan_row.data or {}).get("fansly_group_id")
        apifansly_account_id = (creator_row.data or {}).get("apifansly_account_id")

        for i, part in enumerate(parts):
            if i > 0:
                await asyncio.sleep(random.randint(5, 15))

            ppv_match = re.search(r"\[PPV:([^:]+):(\d+(?:\.\d+)?)\]", part)
            if ppv_match:
                text_out = part[: ppv_match.start()].strip()
                media_id = ppv_match.group(1)
                price = float(ppv_match.group(2))
                ppv_media_context = {
                    "ppv": {
                        "media_id": media_id,
                        "price": price,
                        "access_type": "ppv",
                    }
                }
            else:
                text_out = part
                ppv_media_context = None

            await save_message(
                fan_id=fan_id,
                creator_id=creator_id,
                role="creator",
                content=text_out,
                was_ai_suggested=True,
                media_context=ppv_media_context,
            )

            if group_id and apifansly_account_id:
                if ppv_match:
                    async with httpx.AsyncClient() as hc:
                        resp = await hc.post(
                            f"https://v1.apifansly.com/api/fansly/{apifansly_account_id}/chats/{group_id}/messages",
                            headers={
                                "x-api-key": os.environ.get("APIFANSLY_API_KEY"),
                                "Content-Type": "application/json",
                            },
                            json={
                                "content": text_out,
                                "mediaId": media_id,
                                "access_type": "ppv",
                                "price": price,
                            },
                            timeout=10,
                        )
                    print(
                        f"[AUTO REPLY] Sent part {i+1} PPV media={media_id} "
                        f"fansly status={resp.status_code} body={resp.text[:200]}"
                    )
                else:
                    from main import send_fansly_message

                    sent = await send_fansly_message(apifansly_account_id, str(group_id), part)
                    print(f"[AUTO REPLY] Sent part {i+1}: {part[:50]} fansly_sent={sent}")
            else:
                print(f"[AUTO REPLY] Sent part {i+1}: {part[:50]} (no fansly config)")

    except Exception as e:
        print(f"[AUTO REPLY ERROR] fan={fan_id} error={e}")
        import traceback
        traceback.print_exc()


def _should_update_memory(conversation_history: list[Message]) -> bool:
    count = len([m for m in conversation_history if m.role == "fan"])
    return count > 0 and count % 10 == 0

