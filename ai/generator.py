"""Provider-neutral LLM reply generator for Cleopatra."""

from __future__ import annotations

import json
import re
from typing import Any

from ai.model_providers import complete, get_runtime_target
from models.model_runtime import ModelTarget, ModelTelemetryContext
from models.schemas import Persona
from services.model_telemetry import record_model_failure, record_model_result
from services.model_availability import (
    record_model_transport_failure,
    record_model_transport_success,
)

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

def _same_model_target(left: ModelTarget, right: ModelTarget | None) -> bool:
    return bool(
        right
        and left.provider == right.provider
        and left.model == right.model
        and left.base_url == right.base_url
    )


def _telemetry_context_for_attempt(
    metadata: dict[str, Any],
    *,
    primary_target: ModelTarget,
    attempt_target: ModelTarget,
    fallback_target: ModelTarget | None,
    attempt: int,
) -> ModelTelemetryContext:
    fallback_used = not _same_model_target(primary_target, attempt_target)
    reserved = {
        "feature",
        "creator_id",
        "fan_id",
        "evaluation_run_id",
        "scenario_id",
    }
    attempt_metadata = {
        key: value
        for key, value in metadata.items()
        if key not in reserved
    }
    attempt_metadata.update(
        {
            "writer_attempt": attempt + 1,
            "writer_attempt_role": "fallback" if fallback_used else "primary",
            "writer_fallback_used": fallback_used,
            "writer_attempt_provider": attempt_target.provider,
            "writer_attempt_model": attempt_target.model,
            "writer_primary_provider": primary_target.provider,
            "writer_primary_model": primary_target.model,
            "writer_fallback_provider": (
                fallback_target.provider if fallback_target else None
            ),
            "writer_fallback_model": (
                fallback_target.model if fallback_target else None
            ),
        }
    )
    return ModelTelemetryContext(
        feature=str(metadata.get("feature") or "chat_reply"),
        creator_id=metadata.get("creator_id"),
        fan_id=metadata.get("fan_id"),
        evaluation_run_id=metadata.get("evaluation_run_id"),
        scenario_id=metadata.get("scenario_id"),
        metadata=attempt_metadata,
    )


async def generate_replies(
    prompt_messages: list[dict[str, Any]],
    creator_persona: Persona,
    *,
    telemetry_context: dict[str, Any] | None = None,
    target_override: ModelTarget | None = None,
    fallback_target_override: ModelTarget | None = None,
) -> list[str]:
    """Generate three candidates with a bounded primary-to-fallback plan.

    Ordinary routed turns try Kimi twice, then DeepSeek once. Complex and
    safety-sensitive routes use DeepSeek only. Every attempt is logged with the
    route, reason, model role, and whether fallback was used.
    """

    primary_target = target_override or get_runtime_target("CHAT")
    fallback_target = fallback_target_override
    if _same_model_target(primary_target, fallback_target):
        fallback_target = None

    attempt_targets = [primary_target, primary_target]
    attempt_targets.append(fallback_target or primary_target)

    metadata = dict(telemetry_context or {})
    system = str(prompt_messages[0]["content"])
    messages = [{"role": "user", "content": str(prompt_messages[1]["content"])}]

    for attempt, attempt_target in enumerate(attempt_targets):
        context = _telemetry_context_for_attempt(
            metadata,
            primary_target=primary_target,
            attempt_target=attempt_target,
            fallback_target=fallback_target,
            attempt=attempt,
        )
        try:
            result = await complete(
                attempt_target,
                system=system,
                messages=messages,
                max_tokens=1000,
            )
            record_model_transport_success(attempt_target.model)
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
                    f"model={attempt_target.model} parse_error={parse_error}"
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
                if not _same_model_target(primary_target, attempt_target):
                    print(
                        f"[WRITER ROUTE] fallback succeeded "
                        f"primary={primary_target.model} fallback={attempt_target.model}"
                    )
                return replies
        except Exception as error:
            record_model_transport_failure(attempt_target.model, error)
            await record_model_failure(
                attempt_target,
                context,
                error=str(error),
                retry_count=attempt,
            )
            print(
                f"[GENERATOR ERROR] attempt {attempt + 1} "
                f"provider={attempt_target.provider} "
                f"model={attempt_target.model} error={error}"
            )

    print("[GENERATOR ERROR] all attempts failed — returning no suggestions (fail closed)")
    return []
