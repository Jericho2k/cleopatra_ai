"""How one writer turn recovers: same model elsewhere first, another model last.

THE INCIDENT THIS EXISTS FOR
----------------------------
V3 selected ``moonshotai/kimi-k2.6``, which is pinned to the OpenRouter
upstream ``Inceptron``. Inceptron answered 429. The turn then retried the SAME
rate-limited pool three more times on a 5s/30s/60s schedule — ninety-five
seconds of waiting for one provider to stop being busy — and only then reached
the Qwen fallback, which answered immediately. The reply the fan received was
therefore written by a different model than the one the profile selected, for
no better reason than that one host had a bad minute.

Two things were wrong, and they are separable:

1. *Nothing ever tried Kimi anywhere else.* The pin is correct for healthy
   traffic and wrong as a terminal condition. A model is not unavailable
   because one of its hosts is throttling.
2. *The third wait bought nothing.* After ~35 seconds of 429s from one
   provider, one more cache-affine attempt is worth less than a Kimi response
   from a different host.

THE LADDER
----------
::

    1  Kimi @ Inceptron                            (immediate)
       retryable failure  ->  wait ~5s
    2  Kimi @ Inceptron
       retryable failure  ->  wait ~30s
    3  Kimi @ Inceptron
       retryable failure  ->  Inceptron is unhealthy FOR THIS TURN
    4  Kimi @ any other eligible OpenRouter host   (immediate)
       retryable failure  ->  wait ~5s
    5  Kimi @ any other eligible OpenRouter host   (final bounded Kimi attempt)
       retryable failure  ->  Kimi cannot be served
    6  Qwen3.7-Plus on Together                    (emergency fallback)

Deliberately NOT a load balancer. Steps 1-3 are the only thing healthy traffic
ever executes, so production stays cache-affine to one provider exactly as
before; steps 4-5 are reached only after that provider has failed repeatedly
inside a single turn, and the decision is per-turn — the next turn starts at
step 1 again. See ``ai/openrouter_routing.py`` for how each step is expressed
to OpenRouter.

FAST FAILURE STILL EXISTS
-------------------------
A rejected key, an unknown model, a request our own code malformed — none of
that is repaired by waiting or by changing host, so it skips the whole ladder
and goes straight to the fallback. ``classify_failure`` is the single place
that decision is made, and it names the reason so telemetry and the logs agree.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from ai import openrouter_routing
from models.model_runtime import ModelTarget


# ---------------------------------------------------------------------------
# What went wrong
# ---------------------------------------------------------------------------

#: The upstream asked us to slow down. The waits exist for exactly this.
REASON_RATE_LIMITED = "rate_limited"
#: The upstream broke. Another attempt, or another host, may well work.
REASON_PROVIDER_5XX = "provider_5xx"
#: The request did not complete in the configured client timeout.
REASON_TIMEOUT = "timeout"
#: Connection reset, DNS, TLS, a truncated body — the request never landed.
REASON_TRANSPORT = "transport"
#: No amount of waiting and no other host can repair this.
REASON_PERMANENT = "permanent"
#: HTTP 200, but the model ignored the output contract.
REASON_UNPARSEABLE = "unparseable_output"
#: HTTP 200 and parseable, but validation refused every candidate.
REASON_REJECTED = "rejected_output"

# Statuses that no amount of waiting can fix. Sleeping 95 seconds before
# discovering the API key is still wrong helps nobody, and neither does asking
# a different provider to accept a model handle that does not exist. A
# 408/409/425/429 and every 5xx are deliberately absent: those are exactly what
# the waits and the alternate hosts exist for.
PERMANENT_STATUS_CODES = frozenset({400, 401, 402, 403, 404, 405, 422})

RETRYABLE_CLIENT_STATUS_CODES = frozenset({408, 409, 425, 429})

PERMANENT_ERROR_MARKERS = (
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

# Transport-level words that mean "the clock ran out", for SDKs that raise a
# timeout without a status code and without a distinguishable type.
_TIMEOUT_MARKERS = ("timed out", "timeout", "deadline exceeded")


@dataclass(frozen=True)
class FailureClassification:
    """Why an attempt failed, and what that justifies doing next."""

    reason: str
    status: int | None = None

    @property
    def retryable(self) -> bool:
        return self.reason != REASON_PERMANENT

    @property
    def label(self) -> str:
        """The compact form used in log lines and telemetry."""
        if self.status is None:
            return self.reason
        return f"{self.reason}:{self.status}"


def status_code(error: BaseException) -> int | None:
    """The HTTP status an SDK exception carries, whichever SDK raised it."""
    code = getattr(error, "status_code", None)
    if isinstance(code, int):
        return code
    response = getattr(error, "response", None)
    code = getattr(response, "status_code", None)
    return code if isinstance(code, int) else None


def classify_failure(error: BaseException) -> FailureClassification:
    """Name what happened, from the exception alone.

    Duck-typed rather than importing any provider SDK: the generator must stay
    provider-neutral, and coupling this to one client version is how a
    classification silently stops matching reality after an upgrade.
    """
    status = status_code(error)
    if status is not None:
        if status in PERMANENT_STATUS_CODES:
            return FailureClassification(REASON_PERMANENT, status)
        if status == 429:
            return FailureClassification(REASON_RATE_LIMITED, status)
        if status >= 500:
            return FailureClassification(REASON_PROVIDER_5XX, status)
        if status in RETRYABLE_CLIENT_STATUS_CODES:
            return FailureClassification(REASON_TIMEOUT if status == 408 else REASON_TRANSPORT, status)
        if 400 <= status < 500:
            # Any other 4xx is still a client-side problem. Retrying identical
            # input, here or anywhere else, will repeat it.
            return FailureClassification(REASON_PERMANENT, status)
        return FailureClassification(REASON_TRANSPORT, status)

    text = str(error).lower()
    if any(marker in text for marker in PERMANENT_ERROR_MARKERS):
        return FailureClassification(REASON_PERMANENT, None)
    # A timeout has no status because the response never arrived. It is the
    # most common non-429 way a busy provider shows up, so it must stay
    # retryable AND must be distinguishable in telemetry from a reset socket.
    if isinstance(error, TimeoutError) or any(
        marker in text for marker in _TIMEOUT_MARKERS
    ):
        return FailureClassification(REASON_TIMEOUT, None)
    return FailureClassification(REASON_TRANSPORT, None)


def is_permanent_failure(error: BaseException) -> bool:
    """Whether retrying this exact request — anywhere — is pointless."""
    return classify_failure(error).reason == REASON_PERMANENT


# ---------------------------------------------------------------------------
# The ladder
# ---------------------------------------------------------------------------

#: A cache-affine attempt against the preferred upstream. Normal production.
ROLE_PINNED = "pinned"
#: The same model, deliberately routed away from the preferred upstream.
ROLE_ALTERNATE = "alternate_provider"
#: A different model entirely. The last resort.
ROLE_FALLBACK = "fallback"


@dataclass(frozen=True)
class WriterAttempt:
    """One rung of the ladder, fully resolved before the turn starts."""

    index: int
    role: str
    target: ModelTarget
    #: Seconds to wait before this attempt. Attempt 1 is always immediate.
    wait_before: float
    #: Which attempt this is WITHIN its role, 1-based. Used by the log lines,
    #: which read as "Kimi retry 2", not "writer attempt 2".
    attempt_in_role: int

    @property
    def is_primary_model(self) -> bool:
        return self.role in {ROLE_PINNED, ROLE_ALTERNATE}


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


def _positive_int(env_name: str, default: int) -> int:
    try:
        return max(0, int(os.getenv(env_name, str(default)) or default))
    except (TypeError, ValueError):
        return default


def _wait_for(schedule: tuple[float, ...], attempt_in_role: int, *, first_free: bool) -> float:
    """The wait before ``attempt_in_role`` of a role, or 0 when unscheduled."""
    index = attempt_in_role - (2 if first_free else 1)
    if index < 0 or index >= len(schedule):
        return 0.0
    return float(schedule[index])


def pinned_attempts() -> int:
    """Cache-affine attempts against the preferred provider, before failover."""
    return max(1, _positive_int("WRITER_KIMI_PINNED_ATTEMPTS", 3))


def pinned_waits() -> tuple[float, ...]:
    """Waits BEFORE pinned attempts 2..N. Attempt 1 is immediate."""
    return _wait_schedule("WRITER_KIMI_PINNED_WAIT_SECONDS", (5.0, 30.0))


def alternate_attempts() -> int:
    """Attempts at the same model on a DIFFERENT eligible host.

    Whether failover happens at all is decided per-target at plan time by
    ``supports_provider_failover``, which reads the live switch — so
    ``OPENROUTER_PROVIDER_FAILOVER=false`` takes effect without a deploy even
    though this count is resolved once at import.
    """
    return _positive_int("WRITER_KIMI_ALTERNATE_ATTEMPTS", 2)


def alternate_waits() -> tuple[float, ...]:
    """Waits before alternate attempts 1..N.

    The first is deliberately zero. After the preferred provider has already
    burned ~35 seconds, another wait before asking a *different* host is pure
    latency: its rate limit has nothing to do with the one we just hit.
    """
    return _wait_schedule("WRITER_KIMI_ALTERNATE_WAIT_SECONDS", (0.0, 5.0))


def alternate_provider_target(target: ModelTarget) -> ModelTarget:
    """The same model, routed anywhere eligible except the preferred host.

    Returns the target unchanged for anything that is not reached through an
    aggregator: a direct provider has exactly one host, so "somewhere else" is
    not a thing that exists for it.
    """
    if target.provider != "openrouter":
        return target
    metadata = {
        **(target.metadata or {}),
        "openrouter_provider_mode": openrouter_routing.PROVIDER_MODE_ALTERNATE,
    }
    from dataclasses import replace

    return replace(target, metadata=metadata)


def supports_provider_failover(target: ModelTarget) -> bool:
    """Whether this target has other hosts to fail over to at all."""
    return (
        target.provider == "openrouter"
        and openrouter_routing.provider_failover_enabled()
    )


def build_attempt_plan(
    primary_target: ModelTarget,
    fallback_target: ModelTarget | None,
    *,
    pinned: int,
    pinned_wait_schedule: tuple[float, ...],
    alternate: int,
    alternate_wait_schedule: tuple[float, ...],
    repeat_primary_without_fallback: bool = False,
) -> tuple[WriterAttempt, ...]:
    """Resolve the whole ladder for one turn, before any request is made.

    Building it up front rather than deciding rung by rung is what makes the
    deadline computable, the telemetry comparable, and the schedule one thing
    to read instead of sleeps scattered through a loop.
    """

    plan: list[WriterAttempt] = []

    for number in range(1, max(1, pinned) + 1):
        plan.append(
            WriterAttempt(
                index=len(plan) + 1,
                role=ROLE_PINNED,
                target=primary_target,
                wait_before=_wait_for(pinned_wait_schedule, number, first_free=True),
                attempt_in_role=number,
            )
        )

    if alternate > 0 and supports_provider_failover(primary_target):
        alternate_target = alternate_provider_target(primary_target)
        for number in range(1, alternate + 1):
            plan.append(
                WriterAttempt(
                    index=len(plan) + 1,
                    role=ROLE_ALTERNATE,
                    target=alternate_target,
                    wait_before=_wait_for(
                        alternate_wait_schedule, number, first_free=False
                    ),
                    attempt_in_role=number,
                )
            )

    if fallback_target is not None:
        plan.append(
            WriterAttempt(
                index=len(plan) + 1,
                role=ROLE_FALLBACK,
                target=fallback_target,
                wait_before=0.0,
                attempt_in_role=1,
            )
        )
    elif repeat_primary_without_fallback:
        plan.append(
            WriterAttempt(
                index=len(plan) + 1,
                role=ROLE_PINNED,
                target=primary_target,
                wait_before=0.0,
                attempt_in_role=max(1, pinned) + 1,
            )
        )

    return tuple(plan)


# ---------------------------------------------------------------------------
# The backend-owned deadline
# ---------------------------------------------------------------------------
#
# Durable, pollable turns mean a browser timeout can no longer disagree with the
# backend — but they do not mean a writer task may run forever. The ceiling is
# derived from the ladder the deployment actually configured (its waits plus
# each target's own client timeout) rather than from a number somebody typed,
# so changing the schedule cannot leave the deadline behind.

#: Headroom over the arithmetic worst case, for prompt building, parsing, the
#: model gate and telemetry. Small on purpose: this is a ceiling, not a budget.
DEADLINE_MARGIN = 1.1


def plan_deadline_seconds(plan: tuple[WriterAttempt, ...]) -> float:
    """The hard ceiling for one writer turn, in seconds.

    ``WRITER_TURN_DEADLINE_SECONDS`` overrides it exactly, for a deployment
    that wants a tighter bound than the ladder implies. It is honoured as
    given: a value below one attempt's client timeout genuinely does cut
    attempts short, and silently raising it to fit would mean the configured
    ceiling was not the ceiling.
    """
    raw = os.getenv("WRITER_TURN_DEADLINE_SECONDS")
    if raw and raw.strip():
        try:
            configured = float(raw)
        except ValueError:
            configured = 0.0
        if configured > 0:
            return configured

    derived = sum(
        attempt.wait_before + float(attempt.target.timeout_seconds) for attempt in plan
    )
    return derived * DEADLINE_MARGIN


def writer_turn_deadline_seconds() -> float:
    """The deadline for the DEFAULT V3 ladder.

    What the Simulator's own turn deadline is derived from, so the two cannot
    drift apart: the surrounding turn must outlive the writer it is waiting on.
    """
    from ai.model_providers import find_catalog_target
    from ai.writer_router import (
        COMPLEX_WRITER_MODEL,
        COMPLEX_WRITER_PROVIDER,
        DEFAULT_WRITER_MODEL,
        DEFAULT_WRITER_PROVIDER,
    )

    primary = find_catalog_target(DEFAULT_WRITER_PROVIDER, DEFAULT_WRITER_MODEL)
    fallback = find_catalog_target(COMPLEX_WRITER_PROVIDER, COMPLEX_WRITER_MODEL)
    if primary is None:
        # No catalog entry to read a timeout from. The margin over the waits is
        # the only honest answer, and it is still finite.
        return DEADLINE_MARGIN * (
            sum(pinned_waits()) + sum(alternate_waits()) + 45.0
        )
    return plan_deadline_seconds(
        build_attempt_plan(
            primary,
            fallback,
            pinned=pinned_attempts(),
            pinned_wait_schedule=pinned_waits(),
            alternate=alternate_attempts(),
            alternate_wait_schedule=alternate_waits(),
        )
    )


# ---------------------------------------------------------------------------
# Operator telemetry vocabulary
# ---------------------------------------------------------------------------
#
# These strings are queried by hand in the model telemetry table, so they are
# fixed and name the production pin explicitly. They are OPERATOR facts: an
# agency is never told which provider or model answered (see
# services/ai_stack_visibility.py), and nothing here reaches a tenant surface.

OUTCOME_PINNED_FIRST_TRY = "kimi_inceptron_first_try_success"
OUTCOME_PINNED_RETRY = "kimi_inceptron_retry_success"
OUTCOME_ALTERNATE_PROVIDER = "kimi_alternate_provider_success"
OUTCOME_QWEN_FALLBACK = "qwen_emergency_fallback"
OUTCOME_TOTAL_FAILURE = "writer_total_failure"

OUTCOMES: tuple[str, ...] = (
    OUTCOME_PINNED_FIRST_TRY,
    OUTCOME_PINNED_RETRY,
    OUTCOME_ALTERNATE_PROVIDER,
    OUTCOME_QWEN_FALLBACK,
    OUTCOME_TOTAL_FAILURE,
)


def outcome_for(attempt: WriterAttempt | None) -> str:
    """Which of the five outcomes a successful attempt represents."""
    if attempt is None:
        return OUTCOME_TOTAL_FAILURE
    if attempt.role == ROLE_FALLBACK:
        return OUTCOME_QWEN_FALLBACK
    if attempt.role == ROLE_ALTERNATE:
        return OUTCOME_ALTERNATE_PROVIDER
    if attempt.attempt_in_role <= 1:
        return OUTCOME_PINNED_FIRST_TRY
    return OUTCOME_PINNED_RETRY


# ---------------------------------------------------------------------------
# Obsolete configuration
# ---------------------------------------------------------------------------
#
# WRITER_PRIMARY_RETRY_ATTEMPTS and WRITER_PRIMARY_RETRY_WAIT_SECONDS described
# the four-attempts-on-one-provider schedule this module replaces. A deployment
# that still sets them is configuring nothing, which is worse than configuring
# the wrong thing, so it is said once at startup rather than discovered during
# the next incident.

_RETIRED_VARIABLES: dict[str, str] = {
    "WRITER_PRIMARY_RETRY_ATTEMPTS": "WRITER_KIMI_PINNED_ATTEMPTS",
    "WRITER_PRIMARY_RETRY_WAIT_SECONDS": "WRITER_KIMI_PINNED_WAIT_SECONDS",
}

_warned = False


def warn_about_retired_configuration() -> list[str]:
    """Report, once per process, any retired retry variable still set."""
    global _warned
    stale = [name for name in _RETIRED_VARIABLES if (os.getenv(name) or "").strip()]
    if stale and not _warned:
        _warned = True
        for name in stale:
            print(
                f"[WRITER CONFIG] {name} no longer has any effect; "
                f"use {_RETIRED_VARIABLES[name]} instead"
            )
    return stale


def describe_plan(plan: tuple[WriterAttempt, ...]) -> list[dict[str, Any]]:
    """The ladder as data, for diagnostics and tests."""
    return [
        {
            "index": attempt.index,
            "role": attempt.role,
            "provider": attempt.target.provider,
            "model": attempt.target.model,
            "wait_before": attempt.wait_before,
            "attempt_in_role": attempt.attempt_in_role,
        }
        for attempt in plan
    ]
