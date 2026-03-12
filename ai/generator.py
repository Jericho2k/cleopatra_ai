"""LLM reply generator for Cleopatra.

Calls Together AI via the OpenAI-compatible client and returns reply options.
"""

import json

from openai import AsyncOpenAI

from core.config import get_settings
from models.schemas import Persona


# Module level — create once
client = AsyncOpenAI(
    base_url="https://api.together.xyz/v1",
    api_key=get_settings().TOGETHER_API_KEY,
)


async def generate_replies(
    prompt_messages: list[dict],
    creator_persona: Persona,
) -> list[str]:
    bot_phrases = [
        "certainly",
        "of course",
        "i'd be happy",
        "as an ai",
        "i understand that",
        "great question",
        "absolutely",
        "i apologize",
    ]

    def _is_valid(reply: str) -> bool:
        lowered = reply.lower()
        if any(phrase in lowered for phrase in bot_phrases):
            return False
        if creator_persona.avg_message_length == "short":
            if len(reply.split()) > 25:
                return False
        return True

    for _ in range(3):
        try:
            response = await client.chat.completions.create(
                model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
                messages=prompt_messages,
                temperature=0.8,
                max_tokens=300,
            )
            content = response.choices[0].message.content or ""

            # Strip markdown code fences if present
            lines = content.splitlines()
            cleaned_lines = [line for line in lines if not line.lstrip().startswith("```")]
            cleaned = "\n".join(cleaned_lines).strip() or content.strip()

            replies = json.loads(cleaned)
            if not isinstance(replies, list):
                continue

            # Ensure list of strings
            replies = [r for r in replies if isinstance(r, str)]
            if not replies:
                continue

            valid = [r for r in replies if _is_valid(r)]
            if len(valid) >= 2:
                if len(valid) >= 3:
                    return valid[:3]
                # Pad to 3 with any other replies (even if invalid)
                result = list(valid)
                for r in replies:
                    if r not in result:
                        result.append(r)
                    if len(result) == 3:
                        break
                if len(result) == 3:
                    return result
        except Exception:
            continue

    # Fallback if all attempts fail
    return ["hey 😘", "omg haha", "tell me more"]

