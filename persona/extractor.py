"""Persona extraction and embedding helpers.

Uses Together AI for style analysis and OpenAI embeddings (via ai.rag).
"""

import json

from openai import AsyncOpenAI

from ai.rag import get_embedding
from core.config import get_settings
from db.queries import save_embedding, save_persona
from models.schemas import Persona


# Module-level Together client for persona extraction (chat completions)
persona_client = AsyncOpenAI(
    base_url="https://api.together.xyz/v1",
    api_key=get_settings().TOGETHER_API_KEY,
)


async def extract_persona(
    chat_logs: list[dict],
    creator_id: str,
) -> Persona:
    """Analyze creator messages and persist a Persona."""
    try:
        creator_messages = [
            entry.get("content", "")
            for entry in chat_logs
            if entry.get("role") == "creator" and entry.get("content")
        ][:200]

        if not creator_messages:
            persona = Persona()
            await save_persona(creator_id, persona)
            return persona

        numbered_samples = "\n".join(
            f"{i + 1}. {msg}" for i, msg in enumerate(creator_messages)
        )

        system_prompt = (
            "You are a writing style analyst. Analyze the messages and return only "
            "valid JSON. No explanation, no markdown, no code fences."
        )

        user_prompt = (
            "Analyze the following creator messages and return a single JSON object "
            "with exactly these fields:\n\n"
            "avg_message_length: \"short\" | \"medium\" | \"long\"\n"
            "sends_multiple_messages: true | false\n"
            "emoji_usage: \"none\" | \"rare\" | \"moderate\" | \"heavy\"\n"
            "signature_emojis: [list of up to 5 most used emojis]\n"
            "vocabulary: [list of up to 15 characteristic words or phrases]\n"
            "capitalization: \"none\" | \"lowercase\" | \"mixed\" | \"normal\"\n"
            "punctuation_style: short description string\n"
            "flirt_style: short description string\n"
            "upsell_style: short description string\n"
            "example_greetings: [3 example opening messages in their exact style]\n"
            "example_flirts: [3 example flirty messages in their exact style]\n"
            "dont_list: [5 phrases that would sound completely unlike them]\n\n"
            "Messages:\n"
            f"{numbered_samples}"
        )

        response = await persona_client.chat.completions.create(
            model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )

        content = response.choices[0].message.content or ""

        # Strip markdown fences if present
        lines = content.splitlines()
        cleaned_lines = [line for line in lines if not line.lstrip().startswith("```")]
        cleaned = "\n".join(cleaned_lines).strip() or content.strip()

        data = json.loads(cleaned)
        persona = Persona.model_validate(data)
        await save_persona(creator_id, persona)
        return persona
    except Exception:
        # Fallback: return default persona without crashing
        fallback = Persona()
        try:
            await save_persona(creator_id, fallback)
        except Exception:
            # Swallow persistence errors as well
            pass
        return fallback


async def embed_chat_logs(
    chat_logs: list[dict],
    creator_id: str,
) -> int:
    """Embed fan/creator exchange pairs and store them via db.queries."""
    embedded_count = 0

    for i in range(len(chat_logs) - 1):
        current = chat_logs[i]
        nxt = chat_logs[i + 1]
        if current.get("role") != "fan" or nxt.get("role") != "creator":
            continue

        fan_message = current.get("content") or ""
        creator_response = nxt.get("content") or ""
        if not fan_message or not creator_response:
            continue

        try:
            embedding = await get_embedding(fan_message)
            await save_embedding(
                creator_id=creator_id,
                fan_message=fan_message,
                creator_response=creator_response,
                stage="unknown",
                embedding=embedding,
            )
            embedded_count += 1
        except Exception:
            # Skip failures and continue with the next pair
            continue

    return embedded_count

