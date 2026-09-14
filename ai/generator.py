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


# How many candidates a turn keeps when the caller does not say. Three is the
# historical behaviour and what Assisted wants: an operator picks from a list.
DEFAULT_MAX_CANDIDATES = 3


# ---------------------------------------------------------------------------
# Output contracts
# ---------------------------------------------------------------------------
#
# ``candidates`` is the historical shape: a JSON array of alternative replies,
# one of which is chosen. It is what Assisted needs, because a human picks.
#
# ``auto_messages`` is the Full Auto shape. A Full Auto turn has ONE reply, and
# that reply is one or several natural message bubbles. It is not a list of
# alternatives with the extras thrown away, and the array-of-one workaround kept
# leaking that meaning: a model that returned two perfectly good bubbles was
# logged as "returned 2 candidates for a 1-candidate turn" and had the second
# one deleted. The object form removes the ambiguity at the source.
CONTRACT_CANDIDATES = "candidates"
CONTRACT_AUTO_MESSAGES = "auto_messages"

# An upper bound on bubbles, not a target. One is fine, two or three are normal.
# Beyond this the model has stopped writing a text message and started writing a
# transcript, which is a structural failure and is retried rather than truncated.
MAX_AUTO_MESSAGE_BUBBLES = 6


def parse_reply_candidates(
    content: str,
    creator_persona: Persona,
    *,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> list[str]:
    """Parse model output into validated reply candidates.

    Invalid, malformed, or non-JSON model output must fail closed by
    returning an empty list. Full Auto must never send fallback filler.
    """
    return parse_reply_outcome(
        content, creator_persona, max_candidates=max_candidates
    ).replies


def _decoded_payload(content: str) -> Any | None:
    """Strip reminder blocks and code fences, then decode JSON. None on failure."""
    if not content or not content.strip():
        return None

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
        return None

    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def _is_valid_reply(reply: str, creator_persona: Persona) -> bool:
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


def parse_reply_outcome(
    content: str,
    creator_persona: Persona,
    *,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> ParseOutcome:
    """parse_reply_candidates, plus the reason nothing survived.

    ``max_candidates`` is the cardinality the CALLER will actually use. An
    Assisted turn keeps three, because a human picks between them.
    """
    payload = _decoded_payload(content)
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

    valid = [reply for reply in replies if _is_valid_reply(reply, creator_persona)]

    # COST-001 — one good candidate is a usable answer. The old rule required
    # three survivors, or two plus padding back up to three from the rejected
    # ones, and otherwise returned nothing. That both discarded working copy and
    # padded results with replies the validator had just refused.
    if valid:
        keep = max(1, int(max_candidates))
        if len(valid) > keep:
            print(
                f"[GENERATOR] writer returned {len(valid)} candidates for a "
                f"{keep}-candidate turn; keeping the first {keep}"
            )
        return ParseOutcome(replies=filter_suggestions(valid[:keep]))

    return ParseOutcome(reason=PARSE_ALL_REJECTED)


def parse_auto_messages_outcome(
    content: str,
    creator_persona: Persona,
) -> ParseOutcome:
    """Parse the Full Auto contract: ONE reply, in one or several bubbles.

    The expected shape is ``{"messages": ["first bubble", "second bubble"]}``.
    There are no alternatives to choose between, so nothing here truncates a
    valid second or third bubble — they are the rest of the same reply.

    Any other shape is a structural failure and is reported as
    ``PARSE_UNPARSEABLE`` so the caller retries the same model under its retry
    policy. Coercing an array, an object with a different key, or a bare string
    into "close enough" is how the candidate semantics leaked back in.
    """
    payload = _decoded_payload(content)
    if not isinstance(payload, dict):
        return ParseOutcome(reason=PARSE_UNPARSEABLE)

    raw = payload.get("messages")
    if not isinstance(raw, list) or not raw:
        return ParseOutcome(reason=PARSE_UNPARSEABLE)
    if any(not isinstance(bubble, str) for bubble in raw):
        return ParseOutcome(reason=PARSE_UNPARSEABLE)

    bubbles = [_clean_reply(bubble) for bubble in raw]
    bubbles = [bubble for bubble in bubbles if bubble]
    if not bubbles:
        return ParseOutcome(reason=PARSE_UNPARSEABLE)
    if len(bubbles) > MAX_AUTO_MESSAGE_BUBBLES:
        print(
            f"[GENERATOR] auto reply had {len(bubbles)} bubbles "
            f"(max {MAX_AUTO_MESSAGE_BUBBLES}); treating as malformed structure"
        )
        return ParseOutcome(reason=PARSE_UNPARSEABLE)

    reply = " | ".join(bubbles)
    # Bot-speak is judged per bubble; "is this whole reply filler" and the
    # persona length rule are judged on the reply, because that is what gets
    # sent. One rejected bubble rejects the reply: there is no alternative to
    # fall back to, and sending the rest would send a reply nobody wrote.
    if any(
        any(phrase in bubble.lower() for phrase in BOT_PHRASES)
        for bubble in bubbles
    ):
        return ParseOutcome(reason=PARSE_ALL_REJECTED)
    if not _is_valid_reply(reply, creator_persona):
        return ParseOutcome(reason=PARSE_ALL_REJECTED)

    return ParseOutcome(replies=[reply])


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


# ---------------------------------------------------------------------------
# Writer retry policy
# ---------------------------------------------------------------------------
#
# The fallback model is not a second opinion. It is a different voice, chosen
# because it is on a different provider, and reaching it means the conversation
# is no longer being written by the model the profile actually selected. Under
# the legacy plan one Kimi hiccup — a 429, a truncated body, a rejected
# candidate — was enough to hand the turn to Qwen, so "primary: Kimi" was true
# of the configuration and frequently false of the output.
#
# The persistent plan gives the primary model four real attempts, spaced far
# enough apart that a rate limit or a brief provider incident has time to clear,
# and only then falls back. The waits are constants here rather than sleeps
# scattered through the attempt loop, so the schedule is one thing to read, one
# thing to configure, and one thing for a test to patch.


def _wait_schedule(env_name: str, default: tuple[float, ...]) -> tuple[float, ...]:
    """Read a comma-separated wait schedule from the environment."""
    raw = os.getenv(env_name)
    if not raw:
        return default
    waits: list[float] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            waits.append(max(0.0, float(chunk)))
        except ValueError:
            return default
    return tuple(waits) or default


# Waits BEFORE primary attempts 2, 3 and 4. Attempt 1 is immediate.
PRIMARY_RETRY_WAIT_SECONDS: tuple[float, ...] = _wait_schedule(
    "WRITER_PRIMARY_RETRY_WAIT_SECONDS", (5.0, 30.0, 60.0)
)
PRIMARY_RETRY_ATTEMPTS = max(
    1, int(os.getenv("WRITER_PRIMARY_RETRY_ATTEMPTS", "4") or 4)
)


@dataclass(frozen=True)
class WriterRetryPolicy:
    """How hard a turn tries the primary writer before accepting the fallback."""

    label: str
    #: Attempts against the profile's primary model before any fallback.
    primary_attempts: int = 2
    #: Fixed waits before primary attempts 2..N. Empty means jittered backoff.
    primary_waits: tuple[float, ...] = ()
    #: Whether output the validator rejected is worth another primary attempt.
    retry_rejected_output: bool = False
    #: Whether a turn with no configured fallback repeats the primary once more.
    repeat_primary_without_fallback: bool = True
    #: Whether the primary's backoff is also applied before the fallback attempt.
    #: The fallback is a different provider, so the primary's rate limit says
    #: nothing about it; the legacy plan waits anyway and keeps doing so.
    backoff_before_fallback: bool = True

    def wait_before_primary_attempt(self, attempt_number: int) -> float:
        """Seconds to wait before primary attempt ``attempt_number`` (1-based)."""
        index = attempt_number - 2
        if index < 0 or index >= len(self.primary_waits):
            return 0.0
        return float(self.primary_waits[index])


# The frozen plan. ``cleo_legacy_v1`` and ``cleo_v2`` keep it exactly: two Kimi
# attempts with jittered backoff, then the configured fallback. Changing it
# would change the baseline those profiles exist to be.
LEGACY_WRITER_RETRY_POLICY = WriterRetryPolicy(label="legacy")

# ``cleo_v3``: Kimi is the writer, so Qwen is a last resort rather than a second
# attempt.
PERSISTENT_PRIMARY_RETRY_POLICY = WriterRetryPolicy(
    label="persistent_primary",
    primary_attempts=PRIMARY_RETRY_ATTEMPTS,
    primary_waits=PRIMARY_RETRY_WAIT_SECONDS,
    retry_rejected_output=True,
    repeat_primary_without_fallback=False,
    backoff_before_fallback=False,
)


# Statuses that no amount of waiting can fix. Sleeping 95 seconds before
# discovering the API key is still wrong helps nobody, so these skip straight to
# the fallback. A 408/409/425/429 and every 5xx are deliberately absent: those
# are exactly what the waits exist for.
_PERMANENT_STATUS_CODES = frozenset({400, 401, 402, 403, 404, 405, 422})

_PERMANENT_ERROR_MARKERS = (
    "api key",
    "api_key",
    "unauthorized",
    "invalid authentication",
    "authentication_error",
    "permission denied",
    "not configured",
    "no such model",
    "unknown model",
    "model not found",
    "unable to access non-serverless model",
)


def is_permanent_failure(error: Exception) -> bool:
    """Whether retrying this exact request against this model is pointless."""
    status = _status_code(error)
    if status is not None:
        if status in _PERMANENT_STATUS_CODES:
            return True
        # Any other status that is not a server error and not a rate limit is
        # still a client-side problem; retrying identical input will repeat it.
        if 400 <= status < 500 and status not in {408, 409, 425, 429}:
            return True
        return False
    text = str(error).lower()
    return any(marker in text for marker in _PERMANENT_ERROR_MARKERS)


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
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    output_contract: str = CONTRACT_CANDIDATES,
    retry_policy: WriterRetryPolicy = LEGACY_WRITER_RETRY_POLICY,
    profile_id: str = "",
) -> list[str]:
    """Generate the turn's copy with a bounded primary-to-fallback plan.

    ``retry_policy`` decides how many attempts the profile's PRIMARY model gets
    and how long the turn waits between them. ``LEGACY_WRITER_RETRY_POLICY`` is
    the frozen plan — two primary attempts with jittered backoff, then the
    configured fallback — and is what ``cleo_legacy_v1`` and ``cleo_v2`` run.
    ``PERSISTENT_PRIMARY_RETRY_POLICY`` gives the primary four attempts spaced
    by ``PRIMARY_RETRY_WAIT_SECONDS`` before the fallback is reached at all,
    because on ``cleo_v3`` Kimi IS the writer and Qwen is the last resort.

    What went wrong still decides what happens next:

    * transport/provider failure — retryable. Under the persistent policy the
      wait is the configured one (raised to a longer advertised Retry-After,
      capped); under the legacy policy it is bounded jittered backoff.
    * a failure that no wait can fix — a bad key, an unknown model, a rejected
      request — skips the remaining primary attempts and their sleeps entirely
      and goes straight to the fallback.
    * unparseable output — the model ignored the output contract. Retried on the
      same target, because that is a generation fault, not a routing one.
    * every candidate rejected by validation — under the legacy policy the model
      is retired for this turn (COST-001: an identical generation cannot help).
      Under the persistent policy it is retried, because four attempts at the
      profile's own writer is the point and sampling is not deterministic.

    ``output_contract`` selects how the model's text is read back:
    ``CONTRACT_CANDIDATES`` for a JSON array of alternatives (Assisted), or
    ``CONTRACT_AUTO_MESSAGES`` for the Full Auto object whose ``messages`` array
    is ONE reply's bubbles. The auto contract returns a single joined reply, so
    downstream code still receives "the reply to send" and never a choice.
    """

    primary_target = target_override or get_runtime_target("CHAT")
    fallback_target = fallback_target_override
    if _same_model_target(primary_target, fallback_target):
        fallback_target = None

    primary_attempts = max(1, int(retry_policy.primary_attempts))
    attempt_targets = [primary_target] * primary_attempts
    if fallback_target is not None:
        attempt_targets.append(fallback_target)
    elif retry_policy.repeat_primary_without_fallback:
        attempt_targets.append(primary_target)

    metadata = dict(telemetry_context or {})
    profile = str(
        profile_id or metadata.get("ai_stack_profile") or "unknown"
    )
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

    def _parse(text: str) -> ParseOutcome:
        if output_contract == CONTRACT_AUTO_MESSAGES:
            return parse_auto_messages_outcome(text, creator_persona)
        return parse_reply_outcome(
            text, creator_persona, max_candidates=max_candidates
        )

    exhausted_targets: set[tuple[str, str]] = set()
    pending_backoff: float = 0.0
    pending_reason: str = ""
    skip_remaining_primary = False
    primary_attempt_number = 0

    for attempt, attempt_target in enumerate(attempt_targets):
        is_primary = _same_model_target(primary_target, attempt_target)
        if is_primary:
            primary_attempt_number += 1
        target_key = (attempt_target.provider, attempt_target.model)

        if is_primary and skip_remaining_primary and primary_attempt_number > 1:
            # A failure no wait can repair. Do not spend the configured delay
            # rediscovering it; the fallback is the only thing left that can
            # still answer this turn.
            print(
                f"[WRITER RETRY] profile={profile} model={attempt_target.model} "
                f"attempt={primary_attempt_number} wait=0 "
                f"reason=skipped_permanent_failure:{pending_reason or 'unknown'}"
            )
            continue

        if target_key in exhausted_targets:
            # Validation already refused this model's output and the policy says
            # another identical generation from it is pure cost (COST-001).
            print(
                f"[GENERATOR] skipping attempt {attempt + 1} on "
                f"{attempt_target.model}: validation already rejected its output"
            )
            continue

        if is_primary and primary_attempt_number > 1:
            wait = retry_policy.wait_before_primary_attempt(primary_attempt_number)
            # A provider that asked for longer than the schedule gets it, up to
            # the cap; the schedule is a floor, never a way to ignore a 429.
            wait = max(wait, pending_backoff)
            print(
                f"[WRITER RETRY] profile={profile} model={attempt_target.model} "
                f"attempt={primary_attempt_number} wait={wait:g} "
                f"reason={pending_reason or 'unknown'}"
            )
            if wait > 0:
                await _sleep(wait)
            pending_backoff = 0.0
        elif pending_backoff > 0 and (is_primary or retry_policy.backoff_before_fallback):
            print(
                f"[GENERATOR] backing off {pending_backoff:.2f}s before attempt "
                f"{attempt + 1} on {attempt_target.model}"
            )
            await _sleep(pending_backoff)
            pending_backoff = 0.0

        if not is_primary:
            print(
                f"[WRITER FALLBACK] profile={profile} "
                f"primary={primary_target.model} fallback={attempt_target.model} "
                f"after_primary_attempts={primary_attempt_number} "
                f"reason={pending_reason or 'unknown'}"
            )

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
                outcome = _parse(result.text)
                replies = outcome.replies
                parse_reason = outcome.reason
            except Exception as parse_error:
                replies = []
                parse_reason = PARSE_UNPARSEABLE
                pending_reason = f"parse_error:{parse_error}"
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

            pending_reason = parse_reason
            _log_unsuccessful_generation(
                attempt=attempt + 1,
                target=attempt_target,
                outcome=parse_reason,
                text=result.text,
                output_tokens=result.usage.output_tokens,
            )

            if parse_reason == PARSE_ALL_REJECTED and not retry_policy.retry_rejected_output:
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
            permanent = is_permanent_failure(error)
            pending_reason = (
                f"{'permanent' if permanent else 'transport'}"
                f"{f':{status}' if status is not None else ''}"
            )
            pending_backoff = 0.0 if permanent else _backoff_delay(attempt, error)
            if permanent and is_primary:
                skip_remaining_primary = True
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
