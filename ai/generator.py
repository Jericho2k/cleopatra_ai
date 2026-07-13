"""Provider-neutral LLM reply generator for Cleopatra."""

from __future__ import annotations

import json
import re
from typing import Any

from ai.model_providers import complete, get_runtime_target
from models.model_runtime import ModelTarget, ModelTelemetryContext
from models.schemas import Persona
from services.model_telemetry import record_model_failure, record_model_result

BANNED_PHRASES = [
    "hehe",
    "making me blush",
    "ur too sweet",
    "aww that's so sweet",
    "you're so sweet",
    "wired",
    "$500 yet",
    "too sweet",
]

BOT_PHRASES = [
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
    "hi yourself",
    "hello yourself",
    "gorgeous yourself",
    "beautiful yourself",
    "sexy yourself",
    "yourself",
    "stunning yourself",
    "interesting",
    "noted",
    "understood",
    "got it",
    "sure thing",
]


def filter_suggestions(suggestions: list[str]) -> list[str]:
    filtered = []
    for suggestion in suggestions:
        lower = suggestion.lower()
        if not any(phrase in lower for phrase in BANNED_PHRASES):
            filtered.append(suggestion)
    return filtered if filtered else suggestions


def _clean_reply(reply: str) -> str:
    """Fix malformed split messages."""
    if "|" not in reply:
        return reply.strip()
    parts = [part.strip() for part in reply.split("|")]
    parts = [part for part in parts if part]
    return " | ".join(parts)


def parse_reply_candidates(
    content: str,
    creator_persona: Persona,
) -> list[str]:
    """Parse model output into validated reply candidates.

    Invalid, malformed, or non-JSON model output must fail closed by
    returning an empty list. Full Auto must never send fallback filler.
    """
    if not content or not content.strip():
        return []

    # Remove injected/reminder blocks.
    cleaned = re.sub(
        r"<[a-z_]+_reminder>.*?</[a-z_]+_reminder>",
        "",
        content,
        flags=re.DOTALL,
    ).strip()

    # Remove Markdown code fences while preserving their contents.
    cleaned_lines = [
        line
        for line in cleaned.splitlines()
        if not line.lstrip().startswith("```")
    ]
    cleaned = "\n".join(cleaned_lines).strip()

    if not cleaned:
        return []

    try:
        payload = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError, ValueError):
        return []

    if not isinstance(payload, list):
        return []

    replies = [
        _clean_reply(reply)
        for reply in payload
        if isinstance(reply, str)
    ]
    replies = [reply for reply in replies if reply]

    if not replies:
        return []

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
        "hi yourself",
        "hello yourself",
        "gorgeous yourself",
        "beautiful yourself",
        "sexy yourself",
        "yourself",
        "stunning yourself",
        "interesting",
        "noted",
        "understood",
        "got it",
        "sure thing",
    ]

    def is_valid(reply: str) -> bool:
        lowered = reply.lower()

        if any(phrase in lowered for phrase in bot_phrases):
            return False

        if (
            creator_persona.avg_message_length == "short"
            and len(reply.split()) > 25
        ):
            return False

        return True

    valid = [reply for reply in replies if is_valid(reply)]

    if len(valid) >= 3:
        return filter_suggestions(valid[:3])

    if len(valid) >= 2:
        result = list(valid)

        for reply in replies:
            if reply not in result:
                result.append(reply)

            if len(result) == 3:
                break

        if len(result) == 3:
            return filter_suggestions(result)

    return []


async def generate_replies(
    prompt_messages: list[dict[str, Any]],
    creator_persona: Persona,
    *,
    telemetry_context: dict[str, Any] | None = None,
    target_override: ModelTarget | None = None,
) -> list[str]:
    """Generate exactly three reply candidates, failing closed on bad output."""

    target = target_override or get_runtime_target("CHAT")
    metadata = telemetry_context or {}
    context = ModelTelemetryContext(
        feature=str(metadata.get("feature") or "chat_reply"),
        creator_id=metadata.get("creator_id"),
        fan_id=metadata.get("fan_id"),
        evaluation_run_id=metadata.get("evaluation_run_id"),
        scenario_id=metadata.get("scenario_id"),
        metadata={
            key: value
            for key, value in metadata.items()
            if key
            not in {
                "feature",
                "creator_id",
                "fan_id",
                "evaluation_run_id",
                "scenario_id",
            }
        },
    )

    system = str(prompt_messages[0]["content"])
    messages = [{"role": "user", "content": str(prompt_messages[1]["content"])}]

    for attempt in range(3):
        try:
            result = await complete(
                target,
                system=system,
                messages=messages,
                max_tokens=1000,
            )
            try:
                replies = parse_reply_candidates(result.text, creator_persona)
            except Exception as parse_error:
                replies = []
                await record_model_result(
                    result,
                    context,
                    success=False,
                    retry_count=attempt,
                    parse_valid=False,
                    error=f"parse_error: {parse_error}",
                )
                print(
                    f"[GENERATOR ERROR] attempt {attempt + 1} "
                    f"model={target.model} parse_error={parse_error}"
                )
                continue

            await record_model_result(
                result,
                context,
                success=bool(replies),
                retry_count=attempt,
                parse_valid=bool(replies),
                error=None if replies else "reply candidates failed validation",
            )
            if replies:
                return replies
        except Exception as error:
            await record_model_failure(
                target,
                context,
                error=str(error),
                retry_count=attempt,
            )
            print(
                f"[GENERATOR ERROR] attempt {attempt + 1} "
                f"provider={target.provider} model={target.model} error={error}"
            )

    print("[GENERATOR ERROR] all attempts failed — returning no suggestions (fail closed)")
    return []
