"""Suggestion orchestration service.

Coordinates DB, stage classification, RAG, prompt building, and generation.
"""

import asyncio
import json

from ai.generator import client as together_client, generate_replies
from ai.prompt_builder import build_prompt
from ai.rag import find_similar_exchanges
from ai.stage_classifier import classify_stage
from db.queries import (
    get_conversation_history,
    get_creator_persona,
    get_fan,
    save_message,
    update_fan_memory,
)
from models.schemas import (
    ConversationContext,
    Fan,
    Message,
    Persona,
    SuggestionResponse,
)


async def get_suggestions(
    fan_id: str,
    creator_id: str,
    fan_message: str,
    creator_name: str = "a creator",
) -> SuggestionResponse:
    conversation_history = await get_conversation_history(fan_id)

    fan_profile = await get_fan(creator_id, fan_id)
    if fan_profile is None:
        fan_profile = Fan(id=fan_id, display_name=fan_id)

    creator_persona = await get_creator_persona(creator_id)
    if creator_persona is None:
        creator_persona = Persona()

    conversation_stage = classify_stage(conversation_history, fan_profile)

    similar_exchanges = await find_similar_exchanges(fan_message, creator_id)

    ctx = ConversationContext(
        fan_message=fan_message,
        conversation_history=conversation_history,
        fan_profile=fan_profile,
        creator_persona=creator_persona,
        similar_exchanges=similar_exchanges,
        conversation_stage=conversation_stage,
        creator_name=creator_name,
    )

    prompt = build_prompt(ctx)
    replies = await generate_replies(prompt, creator_persona)

    await save_message(fan_id, creator_id, "fan", fan_message)

    if _should_update_memory(conversation_history):
        asyncio.create_task(_update_fan_memory(fan_id, creator_id, conversation_history))

    return SuggestionResponse(suggestions=replies)


async def _update_fan_memory(
    fan_id: str,
    creator_id: str,
    conversation_history: list[Message],
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
            model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
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
        spend_tier = data.get("spend_tier", "cold")

        if not isinstance(preferences, list):
            preferences = []

        await update_fan_memory(
            fan_id=fan_id,
            notes=notes,
            preferences=preferences,
            spend_tier=spend_tier,
        )
    except Exception:
        # Silent failure; this runs in the background
        return


def _should_update_memory(conversation_history: list[Message]) -> bool:
    count = len([m for m in conversation_history if m.role == 'fan'])
    return count > 0 and count % 10 == 0

