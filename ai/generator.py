"""LLM reply generator for Cleopatra.

Calls Together AI via the OpenAI-compatible client and returns reply options.
"""

import json
import re

import anthropic
import os

from core.config import get_settings
from models.schemas import Persona


# Module level — create once
anthropic_client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

BANNED_PHRASES = [
    "hehe", "making me blush", "ur too sweet", "aww that's so sweet",
    "you're so sweet", "wired", "$500 yet", "too sweet",
]


def filter_suggestions(suggestions: list[str]) -> list[str]:
    filtered = []
    for s in suggestions:
        lower = s.lower()
        if not any(phrase in lower for phrase in BANNED_PHRASES):
            filtered.append(s)
    return filtered if filtered else suggestions  # fallback to unfiltered if all banned


def _clean_reply(reply: str) -> str:
    """Fix malformed split messages."""
    if "|" not in reply:
        return reply.strip()
    parts = [p.strip() for p in reply.split("|")]
    # Remove empty parts
    parts = [p for p in parts if p]
    if len(parts) == 0:
        return ""
    return " | ".join(parts)


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
        "hehe",
        "too sweet",
        "ur too sweet",
        "u r too sweet",
        "making me blush",
        "u make me blush",
        "ur making me blush",
        "omg you're curious",
        "i like that",
        "nice dreams",
        "friendly vibes",
        "that sounds nice",
        "sounds interesting",
        "that's nice",
        "what's your story",
        "gorgeous back at ya",
        "mind blowing yourself",
        "beautiful yourself",
        "stunning yourself",
    ]

    def _is_valid(reply: str) -> bool:
        lowered = reply.lower()
        if any(phrase in lowered for phrase in bot_phrases):
            return False
        if creator_persona.avg_message_length == "short":
            if len(reply.split()) > 25:
                return False
        return True

    for attempt in range(3):
        model = "claude-sonnet-4-20250514"
        try:
            response = await anthropic_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                system=prompt_messages[0]["content"],
                messages=[{"role": "user", "content": prompt_messages[1]["content"]}],
            )
            content = response.content[0].text
            # Strip any injected system content
            content = re.sub(r"<ethics_reminder>.*?</ethics_reminder>", "", content, flags=re.DOTALL)
            content = content.strip()

            # Strip markdown code fences if present
            lines = content.splitlines()
            cleaned_lines = [line for line in lines if not line.lstrip().startswith("```")]
            cleaned = "\n".join(cleaned_lines).strip() or content.strip()

            replies = json.loads(cleaned)
            if not isinstance(replies, list):
                continue

            # Ensure list of strings
            replies = [_clean_reply(r) for r in replies if isinstance(r, str)]
            replies = [r for r in replies if r]  # remove any that became empty
            if not replies:
                continue

            valid = [r for r in replies if _is_valid(r)]
            if len(valid) >= 2:
                if len(valid) >= 3:
                    return filter_suggestions(valid[:3])
                # Pad to 3 with any other replies (even if invalid)
                result = list(valid)
                for r in replies:
                    if r not in result:
                        result.append(r)
                    if len(result) == 3:
                        break
                if len(result) == 3:
                    return filter_suggestions(result)
        except Exception as e:
            print(f"[GENERATOR ERROR] attempt {attempt + 1} model={model} error={e}")
            continue

    # Fallback if all attempts fail
    return filter_suggestions(["hey 😘", "omg haha", "tell me more"])

