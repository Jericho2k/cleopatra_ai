"""Provider-neutral LLM reply generator for Cleopatra."""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any

from ai import openrouter_routing, writer_recovery
from ai.generation_trace import GenerationTrace
from ai.model_providers import complete, get_runtime_target
from ai.prompt_blocks import flatten_message_content
from ai.session_affinity import writer_end_user_id, writer_session_id
from models.model_runtime import ModelTarget, ModelTelemetryContext
from models.schemas import Persona
from services.model_telemetry import (
    record_model_failure,
    record_model_result,
    record_writer_recovery_outcome,
)
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
    role: str = "",
    routed_provider: str = "",
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
            # Which rung of the recovery ladder this was, and which upstream it
            # was aimed at. "openrouter" names an aggregator and cannot answer
            # "did the Inceptron pin hold?", which is the question this pass
            # exists to make answerable.
            "writer_recovery_role": role or None,
            "writer_routed_provider": routed_provider or None,
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
# Used by the FROZEN legacy plan, and as a floor-raiser under the persistent
# plan when a provider advertises a Retry-After longer than the schedule. The
# persistent plan's own waits are fixed constants in ai/writer_recovery.py.
_BACKOFF_BASE_SECONDS = float(os.getenv("WRITER_RETRY_BASE_SECONDS", "0.5"))
_BACKOFF_MAX_SECONDS = float(os.getenv("WRITER_RETRY_MAX_SECONDS", "8.0"))
# An upstream may advertise a very long Retry-After. Waiting minutes inside a
# turn is worse than moving to another host, so honour it only up to this bound.
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
# The persistent plan keeps the profile's own writer for as long as it can
# plausibly be served: several cache-affine attempts on the preferred upstream,
# then the SAME model on another eligible host, and only then the fallback
# model. The shape of that ladder lives in ai/writer_recovery.py, so the
# schedule is one thing to read, one thing to configure, and one thing for a
# test to patch.

# Pinned attempts against the preferred upstream, and the waits between them.
# Attempt 1 is immediate; these are the waits BEFORE attempts 2..N.
KIMI_PINNED_ATTEMPTS = writer_recovery.pinned_attempts()
KIMI_PINNED_WAIT_SECONDS: tuple[float, ...] = writer_recovery.pinned_waits()
# Attempts at the same model on a different eligible host, once the preferred
# one is considered unhealthy for this turn.
KIMI_ALTERNATE_ATTEMPTS = writer_recovery.alternate_attempts()
KIMI_ALTERNATE_WAIT_SECONDS: tuple[float, ...] = writer_recovery.alternate_waits()


@dataclass(frozen=True)
class WriterRetryPolicy:
    """How hard a turn tries the primary writer before accepting the fallback."""

    label: str
    #: Cache-affine attempts against the profile's primary model on its
    #: preferred upstream.
    primary_attempts: int = 2
    #: Fixed waits before pinned attempts 2..N. Empty means jittered backoff.
    primary_waits: tuple[float, ...] = ()
    #: Attempts at the SAME model on another eligible host, after the pinned
    #: ones are spent. Zero keeps the frozen "pinned, then fallback" shape.
    alternate_provider_attempts: int = 0
    #: Waits before alternate attempts 1..N. The first is normally zero.
    alternate_provider_waits: tuple[float, ...] = ()
    #: Whether output the validator rejected is worth another primary attempt.
    retry_rejected_output: bool = False
    #: Whether a turn with no configured fallback repeats the primary once more.
    repeat_primary_without_fallback: bool = True
    #: Whether the primary's backoff is also applied before the fallback attempt.
    #: The fallback is a different provider, so the primary's rate limit says
    #: nothing about it; the legacy plan waits anyway and keeps doing so.
    backoff_before_fallback: bool = True

    def wait_before_primary_attempt(self, attempt_number: int) -> float:
        """Seconds to wait before pinned attempt ``attempt_number`` (1-based)."""
        index = attempt_number - 2
        if index < 0 or index >= len(self.primary_waits):
            return 0.0
        return float(self.primary_waits[index])

    def build_plan(
        self,
        primary_target: ModelTarget,
        fallback_target: ModelTarget | None,
    ) -> tuple[writer_recovery.WriterAttempt, ...]:
        """The whole ladder for one turn, resolved before any request is made."""
        return writer_recovery.build_attempt_plan(
            primary_target,
            fallback_target,
            pinned=self.primary_attempts,
            pinned_wait_schedule=self.primary_waits,
            alternate=self.alternate_provider_attempts,
            alternate_wait_schedule=self.alternate_provider_waits,
            repeat_primary_without_fallback=self.repeat_primary_without_fallback,
        )


# The frozen plan. ``cleo_legacy_v1`` and ``cleo_v2`` keep it exactly: two Kimi
# attempts with jittered backoff, then the configured fallback, and no provider
# failover at all. Changing it would change the baseline those profiles exist
# to be.
LEGACY_WRITER_RETRY_POLICY = WriterRetryPolicy(label="legacy")

# ``cleo_v3``: Kimi is the writer, so another Kimi host comes before another
# model, and Qwen is a last resort rather than a second attempt.
PERSISTENT_PRIMARY_RETRY_POLICY = WriterRetryPolicy(
    label="persistent_primary",
    primary_attempts=KIMI_PINNED_ATTEMPTS,
    primary_waits=KIMI_PINNED_WAIT_SECONDS,
    alternate_provider_attempts=KIMI_ALTERNATE_ATTEMPTS,
    alternate_provider_waits=KIMI_ALTERNATE_WAIT_SECONDS,
    retry_rejected_output=True,
    repeat_primary_without_fallback=False,
    backoff_before_fallback=False,
)


# Classification lives in ai/writer_recovery.py so the generator, the telemetry
# and the logs cannot disagree about what a failure was. Re-exported because
# callers and tests already import it from here.
is_permanent_failure = writer_recovery.is_permanent_failure


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
    return writer_recovery.status_code(error)


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


def _routed_provider_label(target: ModelTarget) -> str:
    """What to call the upstream this attempt is aimed at, in a log line.

    For a direct provider that is simply the provider. For OpenRouter it is the
    pin the request carries, because "openrouter" names an aggregator and says
    nothing about which host is actually serving — which is the entire subject
    of these log lines.
    """
    if target.provider != "openrouter":
        return target.provider
    mode = str((target.metadata or {}).get("openrouter_provider_mode") or "")
    if mode == openrouter_routing.PROVIDER_MODE_ALTERNATE:
        explicit = openrouter_routing.alternate_providers()
        return "+".join(explicit) if explicit else "any_eligible_except_preferred"
    pinned = openrouter_routing.pinned_providers(target.metadata)
    return "+".join(pinned) if pinned else "openrouter_default"


def _retry_log_prefix(target: ModelTarget) -> str:
    """``[KIMI RETRY]`` for the aggregator-routed writer, ``[WRITER RETRY]`` else.

    The incident vocabulary is about the Kimi/OpenRouter path specifically, and
    a safety turn retried on Together is not that. One prefix per thing.
    """
    return "[KIMI RETRY]" if target.provider == "openrouter" else "[WRITER RETRY]"


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
    trace: GenerationTrace | None = None,
) -> list[str]:
    """Generate the turn's copy with a bounded, deadline-owned recovery ladder.

    ``retry_policy`` decides how far the profile's own writer is pursued before
    a different model is accepted. ``LEGACY_WRITER_RETRY_POLICY`` is the frozen
    plan — two primary attempts with jittered backoff, then the configured
    fallback, and no provider failover — and is what ``cleo_legacy_v1`` and
    ``cleo_v2`` run. ``PERSISTENT_PRIMARY_RETRY_POLICY`` runs the V3 ladder:
    cache-affine attempts on the preferred upstream, the SAME model on another
    eligible host once that upstream is unhealthy for this turn, and the
    fallback model only when the writer itself cannot be served anywhere.

    What went wrong still decides what happens next:

    * a rate limit, a provider 5xx, a timeout or a transport failure — retryable.
      Under the persistent policy the wait is the configured one (raised to a
      longer advertised Retry-After, capped); repeated retryable failure on the
      preferred upstream is what makes it unhealthy for this turn and moves the
      request to another host.
    * a failure that no wait and no other host can fix — a bad key, an unknown
      model, a request our own code malformed — skips every remaining primary
      attempt and its sleeps entirely and goes straight to the fallback.
    * unparseable output — the model ignored the output contract. Retried on the
      same target, because that is a generation fault, not a routing one.
    * every candidate rejected by validation — under the legacy policy the model
      is retired for this turn (COST-001: an identical generation cannot help).
      Under the persistent policy it is retried, because pursuing the profile's
      own writer is the point and sampling is not deterministic.

    The whole turn is bounded by a backend-owned deadline derived from the
    ladder itself (``ai/writer_recovery.plan_deadline_seconds``). When it
    expires the turn ends in a reported total failure rather than running on:
    an abandoned generation must never arrive later as a reply nobody is
    expecting.

    ``output_contract`` selects how the model's text is read back:
    ``CONTRACT_CANDIDATES`` for a JSON array of alternatives (Assisted), or
    ``CONTRACT_AUTO_MESSAGES`` for the Full Auto object whose ``messages`` array
    is ONE reply's bubbles. The auto contract returns a single joined reply, so
    downstream code still receives "the reply to send" and never a choice.

    ``trace`` is an optional ``GenerationTrace`` the caller owns, into which this
    function records which rung of the ladder actually answered. The return type
    stays ``list[str]``, so nothing about the writer contract changes and callers
    that pass nothing are unaffected; the trace exists because the model the
    router ASKED for and the model that ANSWERED are different facts, and only
    the first of them used to survive as far as the persisted message
    (``docs/autonomy_architecture_review.md`` finding H). It is filled in on
    total failure too: "every attempt failed" is ground truth worth keeping.
    """

    primary_target = target_override or get_runtime_target("CHAT")
    fallback_target = fallback_target_override
    if _same_model_target(primary_target, fallback_target):
        fallback_target = None

    writer_recovery.warn_about_retired_configuration()

    plan = retry_policy.build_plan(primary_target, fallback_target)
    deadline_seconds = writer_recovery.plan_deadline_seconds(plan)
    started = time.monotonic()

    metadata = dict(telemetry_context or {})
    profile = str(
        profile_id or metadata.get("ai_stack_profile") or "unknown"
    )
    if trace is not None:
        # Recorded before the first attempt, so a turn that never reaches a
        # model still says which one it could not reach.
        trace.record_request(
            primary_target=primary_target,
            fallback_target=fallback_target,
            profile=profile,
            policy=retry_policy.label,
            deadline_seconds=deadline_seconds,
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

    def _elapsed() -> float:
        return time.monotonic() - started

    def _remaining() -> float:
        return deadline_seconds - _elapsed()

    exhausted_targets: set[tuple[str, str]] = set()
    pending_backoff: float = 0.0
    pending_reason: str = ""
    pending_failover_reason: str = ""
    skip_remaining_primary = False
    failover_announced = False
    pinned_attempts_made = 0
    primary_attempts_made = 0
    attempts_made = 0
    succeeded: writer_recovery.WriterAttempt | None = None
    deadline_exceeded = False

    async def _report(
        outcome: str,
        *,
        attempt: writer_recovery.WriterAttempt | None,
        upstream: str | None = None,
        error: str | None = None,
    ) -> None:
        await record_writer_recovery_outcome(
            outcome,
            target=(attempt.target if attempt else primary_target),
            context=_telemetry_context_for_attempt(
                metadata,
                primary_target=primary_target,
                attempt_target=(attempt.target if attempt else primary_target),
                fallback_target=fallback_target,
                attempt=(attempt.index - 1) if attempt else attempts_made,
            ),
            profile=profile,
            policy=retry_policy.label,
            elapsed_ms=int(_elapsed() * 1000),
            deadline_seconds=deadline_seconds,
            attempts=attempts_made,
            pinned_attempts=pinned_attempts_made,
            alternate_attempts=max(0, primary_attempts_made - pinned_attempts_made),
            role=(attempt.role if attempt else ""),
            upstream_provider=upstream,
            error=error,
        )

    for attempt in plan:
        target = attempt.target
        target_key = (target.provider, target.model)

        if attempt.is_primary_model and skip_remaining_primary:
            # A failure no wait and no other host can repair. Do not spend the
            # configured delay rediscovering it; the fallback is the only thing
            # left that can still answer this turn.
            print(
                f"{_retry_log_prefix(target)} profile={profile} "
                f"model={target.model} attempt={attempt.attempt_in_role} "
                f"role={attempt.role} wait=0 "
                f"reason=skipped_permanent_failure:{pending_reason or 'unknown'}"
            )
            continue

        if target_key in exhausted_targets:
            # Validation already refused this model's output and the policy says
            # another identical generation from it is pure cost (COST-001).
            print(
                f"[GENERATOR] skipping attempt {attempt.index} on "
                f"{target.model}: validation already rejected its output"
            )
            continue

        wait = attempt.wait_before
        if attempt.role == writer_recovery.ROLE_FALLBACK:
            wait = pending_backoff if retry_policy.backoff_before_fallback else 0.0
        elif attempt.attempt_in_role > 1:
            # A provider that asked for longer than the schedule gets it, up to
            # the cap; the schedule is a floor, never a way to ignore a 429.
            # Applies to the alternate host too: if IT rate-limits us, its
            # Retry-After is about it, and the second alternate attempt owes it
            # the same respect.
            #
            # The FIRST alternate attempt deliberately does not inherit this.
            # The backoff there came from the provider we have just given up
            # on, and its rate limit says nothing about a different host.
            wait = max(wait, pending_backoff)
        elif attempt.role == writer_recovery.ROLE_PINNED and not retry_policy.primary_waits:
            # The legacy plan has no schedule of its own: jittered backoff is
            # the whole of its spacing.
            wait = pending_backoff

        # The deadline is checked BEFORE the wait, so a turn never sleeps into
        # an expiry it could already see coming.
        if _remaining() <= wait:
            deadline_exceeded = True
            print(
                f"[WRITER DEADLINE] profile={profile} policy={retry_policy.label} "
                f"elapsed={_elapsed():.1f}s budget={deadline_seconds:.1f}s "
                f"attempts={attempts_made} "
                f"abandoned_before=attempt_{attempt.index}:{attempt.role} "
                f"reason={pending_reason or 'deadline'}"
            )
            break

        if attempt.role == writer_recovery.ROLE_ALTERNATE and not failover_announced:
            failover_announced = True
            print(
                f"[KIMI PROVIDER FAILOVER] profile={profile} "
                f"from={_routed_provider_label(primary_target)} "
                f"to={_routed_provider_label(target)} "
                f"model={target.model} "
                f"reason={pending_failover_reason or 'repeated_failure'} "
                f"after_pinned_attempts={pinned_attempts_made}"
            )
        elif attempt.role == writer_recovery.ROLE_FALLBACK:
            print(
                f"[WRITER FALLBACK] profile={profile} "
                f"primary={primary_target.model} fallback={target.model} "
                f"after_primary_attempts={primary_attempts_made} "
                f"reason={'kimi_exhausted' if primary_attempts_made else 'no_primary_attempt'}"
                f":{pending_reason or 'unknown'}"
            )
        elif attempt.attempt_in_role > 1 or attempt.role == writer_recovery.ROLE_ALTERNATE:
            print(
                f"{_retry_log_prefix(target)} profile={profile} "
                f"model={target.model} attempt={attempt.attempt_in_role} "
                f"role={attempt.role} wait={wait:g} "
                f"provider={_routed_provider_label(target)} "
                f"reason={pending_reason or 'unknown'}"
            )
        else:
            print(
                f"[WRITER PRIMARY] profile={profile} "
                f"provider={_routed_provider_label(target)} "
                f"model={target.model} attempt={attempt.attempt_in_role} "
                f"deadline={deadline_seconds:.0f}s"
            )

        if wait > 0:
            await _sleep(wait)
        pending_backoff = 0.0

        # Re-checked after the wait: the sleep is where most of a long ladder's
        # time goes, and a turn that woke up past its budget must not then
        # start a request nobody will be waiting for.
        remaining = _remaining()
        if remaining <= 0:
            deadline_exceeded = True
            print(
                f"[WRITER DEADLINE] profile={profile} policy={retry_policy.label} "
                f"elapsed={_elapsed():.1f}s budget={deadline_seconds:.1f}s "
                f"attempts={attempts_made} "
                f"abandoned_before=attempt_{attempt.index}:{attempt.role} "
                f"reason={pending_reason or 'deadline'}"
            )
            break

        attempts_made += 1
        if attempt.is_primary_model:
            primary_attempts_made += 1
            if attempt.role == writer_recovery.ROLE_PINNED:
                pinned_attempts_made += 1

        context = _telemetry_context_for_attempt(
            metadata,
            primary_target=primary_target,
            attempt_target=target,
            fallback_target=fallback_target,
            attempt=attempt.index - 1,
            role=attempt.role,
            routed_provider=_routed_provider_label(target),
        )
        try:
            # The per-attempt ceiling is whichever is tighter: the target's own
            # client timeout, or what is left of the turn's budget. Without the
            # second, one slow provider could consume a deadline the remaining
            # rungs were supposed to share.
            result = await asyncio.wait_for(
                complete(
                    target,
                    system=system,
                    messages=messages,
                    max_tokens=1000,
                    session_id=session_id,
                    end_user_id=end_user_id,
                ),
                timeout=remaining,
            )
            record_model_transport_success(target.model)
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
                    retry_count=attempt.index - 1,
                    parse_valid=False,
                    error=f"parse_error: {parse_error}",
                )
                print(
                    f"[GENERATOR ERROR] attempt {attempt.index} "
                    f"model={target.model} parse_error={parse_error}"
                )
                _log_unsuccessful_generation(
                    attempt=attempt.index,
                    target=target,
                    outcome="parse_error",
                    text=result.text,
                    output_tokens=result.usage.output_tokens,
                )
                continue

            await record_model_result(
                result,
                context,
                success=bool(replies),
                retry_count=attempt.index - 1,
                parse_valid=bool(replies),
                error=None if replies else f"reply candidates failed: {parse_reason}",
            )
            if replies:
                succeeded = attempt
                upstream = getattr(result, "upstream_provider", None)
                if attempt.role == writer_recovery.ROLE_ALTERNATE:
                    print(
                        f"[KIMI PROVIDER SUCCESS] profile={profile} "
                        f"provider={upstream or _routed_provider_label(target)} "
                        f"model={target.model} "
                        f"after_pinned_attempts={pinned_attempts_made}"
                    )
                elif attempt.role == writer_recovery.ROLE_FALLBACK:
                    print(
                        f"[WRITER ROUTE] fallback succeeded "
                        f"primary={primary_target.model} fallback={target.model}"
                    )
                if trace is not None:
                    trace.record_success(
                        target=target,
                        role=attempt.role,
                        attempt_index=attempt.index,
                        upstream_provider=upstream,
                        outcome=writer_recovery.outcome_for(attempt),
                        attempts=attempts_made,
                        pinned_attempts=pinned_attempts_made,
                        alternate_attempts=max(
                            0, primary_attempts_made - pinned_attempts_made
                        ),
                        elapsed_ms=int(_elapsed() * 1000),
                    )
                    print(trace.describe())
                await _report(
                    writer_recovery.outcome_for(attempt),
                    attempt=attempt,
                    upstream=upstream,
                )
                return replies

            pending_reason = writer_recovery.REASON_UNPARSEABLE
            if parse_reason == PARSE_ALL_REJECTED:
                pending_reason = writer_recovery.REASON_REJECTED
            pending_failover_reason = f"repeated_{pending_reason}"
            _log_unsuccessful_generation(
                attempt=attempt.index,
                target=target,
                outcome=parse_reason,
                text=result.text,
                output_tokens=result.usage.output_tokens,
            )

            if parse_reason == PARSE_ALL_REJECTED and not retry_policy.retry_rejected_output:
                # Retire this model for this turn. A different, explicitly
                # configured fallback may still be tried below.
                exhausted_targets.add(target_key)
                print(
                    f"[GENERATOR] attempt {attempt.index} model={target.model} "
                    "produced only rejected candidates; not retrying this model"
                )
        except asyncio.TimeoutError:
            # The turn's budget ran out inside this request, not the provider's
            # own client timeout (which surfaces as an SDK error below). The
            # request is already cancelled by wait_for, so nothing it would
            # have produced can arrive later.
            deadline_exceeded = True
            print(
                f"[WRITER DEADLINE] profile={profile} policy={retry_policy.label} "
                f"elapsed={_elapsed():.1f}s budget={deadline_seconds:.1f}s "
                f"attempts={attempts_made} "
                f"cancelled=attempt_{attempt.index}:{attempt.role} "
                f"model={target.model}"
            )
            await record_model_failure(
                target,
                context,
                error="writer turn deadline exceeded",
                retry_count=attempt.index - 1,
            )
            break
        except Exception as error:
            record_model_transport_failure(target.model, error)
            classification = writer_recovery.classify_failure(error)
            pending_reason = classification.label
            pending_failover_reason = (
                f"repeated_{classification.status or classification.reason}"
            )
            pending_backoff = (
                0.0 if not classification.retryable else _backoff_delay(attempt.index - 1, error)
            )
            if not classification.retryable and attempt.is_primary_model:
                skip_remaining_primary = True
            await record_model_failure(
                target,
                context,
                error=str(error),
                retry_count=attempt.index - 1,
            )
            print(
                f"[GENERATOR ERROR] attempt {attempt.index} "
                f"provider={target.provider} "
                f"routed_provider={_routed_provider_label(target)} "
                f"model={target.model} status={classification.status} "
                f"reason={classification.label} error={error}"
            )

    print(
        "[GENERATOR ERROR] all attempts failed — returning no suggestions "
        f"(fail closed) profile={profile} attempts={attempts_made} "
        f"deadline_exceeded={str(bool(deadline_exceeded)).lower()}"
    )
    failure_reason = (
        "writer turn deadline exceeded"
        if deadline_exceeded
        else (pending_reason or "all attempts failed")
    )
    if trace is not None:
        trace.record_failure(
            outcome=writer_recovery.OUTCOME_TOTAL_FAILURE,
            reason=failure_reason,
            attempts=attempts_made,
            pinned_attempts=pinned_attempts_made,
            alternate_attempts=max(0, primary_attempts_made - pinned_attempts_made),
            elapsed_ms=int(_elapsed() * 1000),
            deadline_exceeded=deadline_exceeded,
        )
        print(trace.describe())
    await _report(
        writer_recovery.OUTCOME_TOTAL_FAILURE,
        attempt=succeeded,
        error=failure_reason,
    )
    return []
