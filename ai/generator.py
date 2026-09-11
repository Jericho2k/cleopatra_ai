"""Provider-neutral LLM reply generator for Cleopatra."""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
from dataclasses import dataclass, field
from typing import Any

from ai.model_providers import complete, get_runtime_target
from ai.prompt_blocks import flatten_message_content
from ai.session_affinity import writer_end_user_id, writer_session_id
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

# COST-001 — writer validation, in two tiers.
#
# The single list this replaces matched bare substrings, so it rejected ordinary
# creator copy: "yourself" killed "touch yourself" and "by yourself",
# "interesting" killed "that's interesting, tell me more", and "noted"/"got it"
# killed any reply containing them anywhere. Each rejection cost a full extra
# paid generation and often ended in no reply at all.
#
# Tier 1: unambiguous bot-speak, matched anywhere in the reply. Every entry is a
# multi-word phrase or a distinctive token, so it cannot fire on normal writing.
BOT_PHRASES = [
    "as an ai",
    "i'd be happy",
    "i understand that",
    "great question",
    "i apologize",
    "hehe",
    "too sweet",
    "ur too sweet",
    "u r too sweet",
    "making me blush",
    "u make me blush",
    "ur making me blush",
    "omg you're curious",
    "nice dreams",
    "friendly vibes",
    "that sounds nice",
    "sounds interesting",
    "that's nice",
    "what's your story",
    "gorgeous back at ya",
    "sure thing",
    # The "X yourself" echo the writer used to produce. These stay as phrases;
    # the bare word "yourself" does not, because it is ordinary copy here.
    "mind blowing yourself",
    "hi yourself",
    "hello yourself",
    "gorgeous yourself",
    "beautiful yourself",
    "sexy yourself",
    "stunning yourself",
]

# Tier 2: words that are only a bot-tell when they are the ENTIRE reply. A
# message that is just "noted" is filler; "i noted every word you said" is not.
BOT_STANDALONE_REPLIES = {
    "noted",
    "got it",
    "understood",
    "interesting",
    "certainly",
    "of course",
    "absolutely",
    "i like that",
    "sure",
    "ok",
    "okay",
}


def _is_standalone_filler(lowered: str) -> bool:
    """True when the whole reply is one of the tier-2 words, bar punctuation."""
    stripped = re.sub(r"[^a-z0-9' ]+", "", lowered).strip()
    stripped = re.sub(r"\s+", " ", stripped)
    return stripped in BOT_STANDALONE_REPLIES


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


PARSE_OK = "ok"
PARSE_UNPARSEABLE = "unparseable"
PARSE_ALL_REJECTED = "all_rejected"


@dataclass
class ParseOutcome:
    """Why a generation produced no usable replies.

    COST-001 turns on this distinction. "The model emitted garbage" and "the
    model emitted fine replies that our filter disliked" both used to surface as
    an empty list, so both triggered another full paid generation against the
    same model. Only the first is worth retrying there.
    """

    replies: list[str] = field(default_factory=list)
    reason: str = PARSE_OK


def parse_reply_candidates(
    content: str,
    creator_persona: Persona,
) -> list[str]:
    """Parse model output into validated reply candidates.

    Invalid, malformed, or non-JSON model output must fail closed by
    returning an empty list. Full Auto must never send fallback filler.
    """
    return parse_reply_outcome(content, creator_persona).replies


def parse_reply_outcome(
    content: str,
    creator_persona: Persona,
) -> ParseOutcome:
    """parse_reply_candidates, plus the reason nothing survived."""
    if not content or not content.strip():
        return ParseOutcome(reason=PARSE_UNPARSEABLE)

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
        return ParseOutcome(reason=PARSE_UNPARSEABLE)

    try:
        payload = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError, ValueError):
        return ParseOutcome(reason=PARSE_UNPARSEABLE)

    if not isinstance(payload, list):
        return ParseOutcome(reason=PARSE_UNPARSEABLE)

    replies = [
        _clean_reply(reply)
        for reply in payload
        if isinstance(reply, str)
    ]
    replies = [reply for reply in replies if reply]

    if not replies:
        return ParseOutcome(reason=PARSE_UNPARSEABLE)

    def is_valid(reply: str) -> bool:
        lowered = reply.lower()

        if any(phrase in lowered for phrase in BOT_PHRASES):
            return False

        if _is_standalone_filler(lowered):
            return False

        if (
            creator_persona.avg_message_length == "short"
            and len(reply.split()) > 25
        ):
            return False

        return True

    valid = [reply for reply in replies if is_valid(reply)]

    # COST-001 — one good candidate is a usable answer. The old rule required
    # three survivors, or two plus padding back up to three from the rejected
    # ones, and otherwise returned nothing. That both discarded working copy and
    # padded results with replies the validator had just refused.
    if valid:
        return ParseOutcome(replies=filter_suggestions(valid[:3]))

    return ParseOutcome(reason=PARSE_ALL_REJECTED)


def _log_unsuccessful_generation(
    *,
    attempt: int,
    target: ModelTarget,
    outcome: str,
    text: str,
    output_tokens: int,
) -> None:
    """Report a generation that came back HTTP-successful but unusable.

    Before this existed, an attempt whose parse outcome was ``unparseable`` —
    which is what an empty ``message.content`` produces — advanced to the next
    model in total silence. A reasoning-enabled writer that spent its whole
    completion budget on hidden reasoning therefore looked, in Railway, exactly
    like an attempt that never happened; the only visible line was the LAST
    model's transport error. That is the failure this line makes obvious.

    Deliberately content-free. Only the shape of the answer is logged: which
    attempt, which provider and model, why it was unusable, whether the message
    body was empty at all, and how many output tokens were billed for it. No fan
    message, no prompt, and no model output ever reaches the log.
    """
    # A logging helper on an error path must never be the thing that raises.
    body = str(text or "")
    print(
        f"[GENERATOR] attempt={attempt} provider={target.provider} "
        f"model={target.model} outcome={outcome} "
        f"content_empty={str(not body.strip()).lower()} "
        f"output_tokens={output_tokens}"
    )


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


# COST-001 — bounded jittered backoff between transport retries.
#
# Kimi is pinned to a single upstream with allow_fallbacks=False, so a retry
# cannot route around a throttled provider. Three immediate requests into a
# provider that just returned 429 is the most likely cascading-failure path in
# the system, and it repeats under every durable action retry.
_BACKOFF_BASE_SECONDS = float(os.getenv("WRITER_RETRY_BASE_SECONDS", "0.5"))
_BACKOFF_MAX_SECONDS = float(os.getenv("WRITER_RETRY_MAX_SECONDS", "8.0"))
# An upstream may advertise a very long Retry-After. Waiting minutes inside a
# request is worse than giving up, so honour it only up to this bound.
_RETRY_AFTER_CAP_SECONDS = float(os.getenv("WRITER_RETRY_AFTER_CAP_SECONDS", "30.0"))


async def _sleep(seconds: float) -> None:
    """Indirection so tests can assert on delays without waiting for them."""
    await asyncio.sleep(seconds)


def _retry_after_seconds(error: Exception) -> float | None:
    """Read Retry-After from a provider error, whatever SDK raised it.

    Duck-typed on purpose: the OpenAI and Anthropic SDKs both hang a response
    with headers off the exception, and importing either here would couple the
    generator to a specific client version.
    """
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    try:
        raw = headers.get("retry-after") or headers.get("Retry-After")
    except (AttributeError, TypeError):
        return None
    if raw is None:
        return None
    try:
        seconds = float(str(raw).strip())
    except (TypeError, ValueError):
        # The HTTP-date form is legal but providers do not use it here, and
        # guessing a date is worse than falling back to exponential backoff.
        return None
    if seconds < 0:
        return None
    return min(seconds, _RETRY_AFTER_CAP_SECONDS)


def _status_code(error: Exception) -> int | None:
    code = getattr(error, "status_code", None)
    if isinstance(code, int):
        return code
    response = getattr(error, "response", None)
    code = getattr(response, "status_code", None)
    return code if isinstance(code, int) else None


def _backoff_delay(attempt: int, error: Exception) -> float:
    """Delay before the attempt after a failed one. Full jitter, bounded.

    Retry-After wins when the provider sent one; otherwise exponential from
    _BACKOFF_BASE_SECONDS. Jitter is applied so a burst of concurrent writers
    does not re-converge on the same instant.
    """
    advertised = _retry_after_seconds(error)
    if advertised is not None:
        return advertised
    ceiling = min(_BACKOFF_BASE_SECONDS * (2 ** attempt), _BACKOFF_MAX_SECONDS)
    return random.uniform(0.0, ceiling)


async def generate_replies(
    prompt_messages: list[dict[str, Any]],
    creator_persona: Persona,
    *,
    telemetry_context: dict[str, Any] | None = None,
    target_override: ModelTarget | None = None,
    fallback_target_override: ModelTarget | None = None,
) -> list[str]:
    """Generate candidates with a bounded primary-to-fallback plan.

    Ordinary routed turns try Kimi twice, then DeepSeek once. Complex and
    safety-sensitive routes use DeepSeek only. Every attempt is logged with the
    route, reason, model role, and whether fallback was used. Routing itself is
    unchanged: no OpenRouter provider fallback is enabled here, and the only
    fallback ever used is the explicitly configured target.

    COST-001 — the three attempts are no longer interchangeable. What went wrong
    decides what happens next:

    * transport/provider failure — retryable, after bounded jittered backoff
      that honours Retry-After. Kimi is pinned to one upstream, so retrying
      without a delay just re-enters a throttled provider.
    * unparseable output — the model misbehaved; one more attempt on the same
      target is reasonable, with no delay since nothing is throttling us.
    * every candidate rejected by validation — the model answered fine and our
      filter refused it. Paying for an identical generation from the same model
      cannot help, so the same target is not tried again; only an explicitly
      configured fallback model is.
    """

    primary_target = target_override or get_runtime_target("CHAT")
    fallback_target = fallback_target_override
    if _same_model_target(primary_target, fallback_target):
        fallback_target = None

    attempt_targets = [primary_target, primary_target]
    attempt_targets.append(fallback_target or primary_target)

    metadata = dict(telemetry_context or {})
    # COST-002a — the system content is handed to the transport in whatever
    # shape build_prompt produced. Flattening it here is what used to discard
    # the cache_control marker before Anthropic ever saw it; the transport now
    # decides, because only it knows whether the provider consumes blocks.
    system = prompt_messages[0]["content"]
    messages = [
        {
            "role": "user",
            "content": flatten_message_content(prompt_messages[1]["content"]),
        }
    ]

    # One fan conversation keeps one affinity key for its whole life, so a
    # provider that caches prompt prefixes keeps serving the same warm cache.
    session_id = writer_session_id(
        metadata.get("creator_id"),
        metadata.get("fan_id"),
    )
    end_user_id = writer_end_user_id(
        metadata.get("creator_id"),
        metadata.get("fan_id"),
    )

    exhausted_targets: set[tuple[str, str]] = set()
    pending_backoff: float = 0.0

    for attempt, attempt_target in enumerate(attempt_targets):
        target_key = (attempt_target.provider, attempt_target.model)
        if target_key in exhausted_targets:
            # Validation already refused this model's output. Another identical
            # generation from it is pure cost (COST-001).
            print(
                f"[GENERATOR] skipping attempt {attempt + 1} on "
                f"{attempt_target.model}: validation already rejected its output"
            )
            continue

        if pending_backoff > 0:
            print(
                f"[GENERATOR] backing off {pending_backoff:.2f}s before attempt "
                f"{attempt + 1} on {attempt_target.model}"
            )
            await _sleep(pending_backoff)
            pending_backoff = 0.0

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
                session_id=session_id,
                end_user_id=end_user_id,
            )
            record_model_transport_success(attempt_target.model)
            try:
                outcome = parse_reply_outcome(result.text, creator_persona)
                replies = outcome.replies
                parse_reason = outcome.reason
            except Exception as parse_error:
                replies = []
                parse_reason = PARSE_UNPARSEABLE
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
                _log_unsuccessful_generation(
                    attempt=attempt + 1,
                    target=attempt_target,
                    outcome="parse_error",
                    text=result.text,
                    output_tokens=result.usage.output_tokens,
                )
                continue

            await record_model_result(
                result,
                context,
                success=bool(replies),
                retry_count=attempt,
                parse_valid=bool(replies),
                error=None if replies else f"reply candidates failed: {parse_reason}",
            )
            if replies:
                if not _same_model_target(primary_target, attempt_target):
                    print(
                        f"[WRITER ROUTE] fallback succeeded "
                        f"primary={primary_target.model} fallback={attempt_target.model}"
                    )
                return replies

            _log_unsuccessful_generation(
                attempt=attempt + 1,
                target=attempt_target,
                outcome=parse_reason,
                text=result.text,
                output_tokens=result.usage.output_tokens,
            )

            if parse_reason == PARSE_ALL_REJECTED:
                # Retire this model for this turn. A different, explicitly
                # configured fallback may still be tried below.
                exhausted_targets.add(target_key)
                print(
                    f"[GENERATOR] attempt {attempt + 1} model={attempt_target.model} "
                    "produced only rejected candidates; not retrying this model"
                )
        except Exception as error:
            record_model_transport_failure(attempt_target.model, error)
            status = _status_code(error)
            pending_backoff = _backoff_delay(attempt, error)
            await record_model_failure(
                attempt_target,
                context,
                error=str(error),
                retry_count=attempt,
            )
            print(
                f"[GENERATOR ERROR] attempt {attempt + 1} "
                f"provider={attempt_target.provider} "
                f"model={attempt_target.model} status={status} error={error}"
            )

    print("[GENERATOR ERROR] all attempts failed — returning no suggestions (fail closed)")
    return []
