"""Which model attempt actually produced the text that was sent.

Finding H of ``docs/autonomy_architecture_review.md``: the durable marker on a
creator message records ``route.primary_target`` — the model the router *asked
for*. ``ai/generator.generate_replies`` may succeed on a retry, on the same
model served by a different upstream host, or on the configured fallback, and
its return type is ``list[str]``, so which of those happened is dropped on the
floor before anything is persisted. Recovery telemetry exists, but it is a
separate stream: joining it back to one visible message after the fact needs a
timestamp guess, and a quality comparison built on a guess is not a comparison.

``GenerationTrace`` is the missing return channel. The generator takes one as an
optional keyword sink — the same shape as ``outcome_sink`` in
``services/suggestions._debounced_auto_reply`` — and fills it in on the way out,
on success and on total failure alike. Callers that pass nothing behave exactly
as before, so the recovery ladder keeps one contract and one set of call sites.

What the trace deliberately separates:

``requested_*``
    What routing chose before the turn ran. This is what the old marker stored,
    and it stays, because "we asked for Kimi and got Qwen" is the interesting
    sentence and it needs both halves.

``model``/``provider``/``upstream_provider``
    What answered. ``upstream_provider`` is the host behind an aggregator, which
    is the difference between "Kimi failed" and "one OpenRouter upstream failed"
    — a distinction the review asks for by name and the only one that survives
    to explain a voice change.

``attempts``/``role``/``outcome``
    How much of the ladder was spent getting there. A first-try success and a
    success after two rate limits and a host failover have the same text and
    very different meanings for latency, cost and provider health.

A trace is a record of what happened, never an instruction. Nothing reads it to
decide anything; it exists so that a conclusion drawn from a transcript can be
checked against the run that produced it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from models.model_runtime import ModelTarget


@dataclass
class GenerationTrace:
    """A mutable sink the writer fills in as one turn's ladder plays out.

    Created empty by the caller, handed to ``generate_replies``, and read after
    it returns. An untouched trace (``recorded`` false) means the writer was
    never reached — which is itself worth persisting, because a turn that sent
    nothing because no generation was attempted is a different failure from one
    whose every attempt failed.
    """

    #: True once the generator has reported an outcome into this trace.
    recorded: bool = False

    #: What routing asked for, before anything ran.
    requested_provider: str = ""
    requested_model: str = ""
    requested_fallback_provider: str = ""
    requested_fallback_model: str = ""

    #: Which stack profile and recovery policy governed the ladder.
    profile: str = ""
    policy: str = ""

    #: The attempt that actually answered. Empty when none did.
    provider: str = ""
    model: str = ""
    upstream_provider: str = ""
    role: str = ""
    attempt_index: int = 0

    #: How much of the ladder was spent.
    attempts: int = 0
    pinned_attempts: int = 0
    alternate_attempts: int = 0
    elapsed_ms: int = 0
    deadline_seconds: float = 0.0
    deadline_exceeded: bool = False

    #: ``ai.writer_recovery`` outcome label, e.g. ``kimi_inceptron_first_try_success``.
    outcome: str = ""
    #: Why the turn ended without usable text. Empty on success.
    failure_reason: str = ""

    @property
    def succeeded(self) -> bool:
        """Whether a model attempt produced the text that was actually used."""
        return bool(self.recorded and self.model)

    @property
    def served_by_requested_model(self) -> bool:
        """Whether the model that answered is the one routing asked for.

        False means the visible reply is not in the voice the router selected.
        That is the single most important thing a quality comparison can know
        about a message, and before this trace existed it was not recorded.
        """
        return bool(self.model) and self.model == self.requested_model

    def record_request(
        self,
        *,
        primary_target: ModelTarget,
        fallback_target: ModelTarget | None,
        profile: str,
        policy: str,
        deadline_seconds: float,
    ) -> None:
        """Note what was asked for, before the first attempt is made.

        Called even if every attempt then fails, so a total failure still says
        which model could not be reached.
        """
        self.recorded = True
        self.requested_provider = str(primary_target.provider)
        self.requested_model = str(primary_target.model)
        self.requested_fallback_provider = (
            str(fallback_target.provider) if fallback_target else ""
        )
        self.requested_fallback_model = (
            str(fallback_target.model) if fallback_target else ""
        )
        self.profile = str(profile)
        self.policy = str(policy)
        self.deadline_seconds = float(deadline_seconds)

    def record_success(
        self,
        *,
        target: ModelTarget,
        role: str,
        attempt_index: int,
        upstream_provider: str | None,
        outcome: str,
        attempts: int,
        pinned_attempts: int,
        alternate_attempts: int,
        elapsed_ms: int,
    ) -> None:
        """Note the attempt whose text is the one being returned."""
        self.recorded = True
        self.provider = str(target.provider)
        self.model = str(target.model)
        self.upstream_provider = str(upstream_provider or "")
        self.role = str(role)
        self.attempt_index = int(attempt_index)
        self.outcome = str(outcome)
        self.attempts = int(attempts)
        self.pinned_attempts = int(pinned_attempts)
        self.alternate_attempts = int(alternate_attempts)
        self.elapsed_ms = int(elapsed_ms)
        self.failure_reason = ""

    def record_failure(
        self,
        *,
        outcome: str,
        reason: str,
        attempts: int,
        pinned_attempts: int,
        alternate_attempts: int,
        elapsed_ms: int,
        deadline_exceeded: bool,
    ) -> None:
        """Note that the ladder ended with no usable text."""
        self.recorded = True
        self.outcome = str(outcome)
        self.failure_reason = str(reason)
        self.attempts = int(attempts)
        self.pinned_attempts = int(pinned_attempts)
        self.alternate_attempts = int(alternate_attempts)
        self.elapsed_ms = int(elapsed_ms)
        self.deadline_exceeded = bool(deadline_exceeded)
        self.provider = ""
        self.model = ""
        self.upstream_provider = ""
        self.role = ""
        self.attempt_index = 0

    def as_metadata(self) -> dict[str, Any]:
        """The compact form persisted next to a message.

        Keys are omitted when empty rather than written as nulls: this lands in
        a jsonb column on every creator message, and a record that says nothing
        should cost nothing. ``served_by_requested_model`` is stored even though
        it is derivable, because it is the field an operator filters on.
        """
        record: dict[str, Any] = {
            "requested": {
                "provider": self.requested_provider,
                "model": self.requested_model,
            },
            "attempts": self.attempts,
            "elapsed_ms": self.elapsed_ms,
        }
        if self.requested_fallback_model:
            record["requested"]["fallback_provider"] = self.requested_fallback_provider
            record["requested"]["fallback_model"] = self.requested_fallback_model
        if self.profile:
            record["profile"] = self.profile
        if self.policy:
            record["policy"] = self.policy
        if self.succeeded:
            record["actual"] = {
                "provider": self.provider,
                "model": self.model,
                "role": self.role,
                "attempt": self.attempt_index,
            }
            if self.upstream_provider:
                record["actual"]["upstream_provider"] = self.upstream_provider
            record["served_by_requested_model"] = self.served_by_requested_model
        if self.outcome:
            record["outcome"] = self.outcome
        if self.failure_reason:
            record["failure_reason"] = self.failure_reason
        if self.deadline_exceeded:
            record["deadline_exceeded"] = True
        if self.pinned_attempts:
            record["pinned_attempts"] = self.pinned_attempts
        if self.alternate_attempts:
            record["alternate_attempts"] = self.alternate_attempts
        return record

    def describe(self) -> str:
        """One log line naming the model that actually answered."""
        if not self.recorded:
            return "[WRITER ACTUAL] no generation attempted"
        if not self.succeeded:
            return (
                f"[WRITER ACTUAL] profile={self.profile or 'unknown'} "
                f"requested={self.requested_model or 'none'} served=none "
                f"attempts={self.attempts} reason={self.failure_reason or 'unknown'}"
            )
        upstream = self.upstream_provider or self.provider
        return (
            f"[WRITER ACTUAL] profile={self.profile or 'unknown'} "
            f"requested={self.requested_model} served={self.model} "
            f"upstream={upstream} role={self.role} attempt={self.attempt_index} "
            f"as_requested={str(self.served_by_requested_model).lower()} "
            f"elapsed_ms={self.elapsed_ms}"
        )
