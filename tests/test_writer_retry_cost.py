"""COST-001 — writer retry, backoff, and validation semantics.

The writer ran `[primary, primary, fallback or primary]` with no sleep between
attempts and advanced the loop on validation failure as readily as on a
transport error. Two consequences:

- a reply containing an ordinary word like "yourself" cost three full paid
  generations and then returned nothing;
- a 429 from the pinned Inceptron upstream produced three immediate requests
  into that same throttled provider, repeated under every durable retry.

Delays are asserted through a fake clock; no test actually waits.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from ai import generator
from ai.generator import (
    PARSE_ALL_REJECTED,
    PARSE_OK,
    PARSE_UNPARSEABLE,
    parse_reply_candidates,
    parse_reply_outcome,
)
from models.model_runtime import ModelResult, ModelTarget, ModelUsage
from models.schemas import Persona, SuggestionResponse


def _target(name: str, model: str) -> ModelTarget:
    return ModelTarget(
        name=name,
        provider="together",
        model=model,
        base_url="https://api.together.xyz/v1",
        api_key_env="TOGETHER_API_KEY",
    )


KIMI = _target("Kimi", "moonshotai/Kimi-K3")
COMPLEX_WRITER = _target("Qwen3.7 Plus", "Qwen/Qwen3.7-Plus")


class _FakeError(Exception):
    """A provider error shaped like the SDK ones, without importing them."""

    def __init__(self, status_code: int, retry_after: str | None = None):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code

        class _Response:
            def __init__(self, headers):
                self.headers = headers
                self.status_code = status_code

        self.response = _Response(
            {"retry-after": retry_after} if retry_after is not None else {}
        )


@pytest.fixture
def harness(monkeypatch):
    """Records attempts and sleeps; never actually waits."""
    state = {"calls": [], "sleeps": [], "responses": []}

    async def fake_sleep(seconds):
        state["sleeps"].append(seconds)

    async def fake_complete(target, **_kwargs):
        state["calls"].append(target.model)
        index = len(state["calls"]) - 1
        behaviour = state["responses"][min(index, len(state["responses"]) - 1)]
        if isinstance(behaviour, Exception):
            raise behaviour
        return ModelResult(
            text=behaviour,
            target=target,
            usage=ModelUsage(input_tokens=100, output_tokens=10),
            latency_ms=50,
        )

    async def noop(*_args, **_kwargs):
        return None

    monkeypatch.setattr(generator, "_sleep", fake_sleep)
    monkeypatch.setattr(generator, "complete", fake_complete)
    monkeypatch.setattr(generator, "record_model_result", noop)
    monkeypatch.setattr(generator, "record_model_failure", noop)
    # Deterministic jitter so delay assertions are exact.
    monkeypatch.setattr(generator.random, "uniform", lambda low, high: high)
    return state


def _run(harness, *, primary=KIMI, fallback=COMPLEX_WRITER):
    return asyncio.run(
        generator.generate_replies(
            [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "user"},
            ],
            Persona(avg_message_length="short"),
            telemetry_context={"feature": "auto_reply", "writer_route": "default"},
            target_override=primary,
            fallback_target_override=fallback,
        )
    )


# --- validation no longer rejects legitimate copy ---------------------------


@pytest.mark.parametrize(
    "reply",
    [
        "come touch yourself for me",
        "are you by yourself right now",
        "that's interesting, tell me more",
        "i noted every word you said",
        "you got it babe",
        "i like that you noticed",
    ],
)
def test_legitimate_copy_is_not_rejected(reply):
    """Each of these was killed by a bare-substring ban."""
    assert parse_reply_candidates(json.dumps([reply]), Persona()) == [reply]


@pytest.mark.parametrize("reply", ["noted", "Got it.", "understood!", "interesting"])
def test_standalone_filler_is_still_rejected(reply):
    """The quality check survives where the word IS the whole reply."""
    outcome = parse_reply_outcome(json.dumps([reply]), Persona())
    assert outcome.replies == []
    assert outcome.reason == PARSE_ALL_REJECTED


def test_obvious_bot_speak_is_still_rejected():
    outcome = parse_reply_outcome(
        json.dumps(["as an ai i cannot", "great question!", "i apologize for that"]),
        Persona(),
    )
    assert outcome.replies == []
    assert outcome.reason == PARSE_ALL_REJECTED


def test_one_valid_candidate_is_returned_rather_than_nothing():
    """The old rule needed three survivors or it returned []."""
    payload = json.dumps(["as an ai i cannot", "great question!", "come closer"])
    outcome = parse_reply_outcome(payload, Persona())
    assert outcome.replies == ["come closer"]
    assert outcome.reason == PARSE_OK


def test_a_single_suggestion_is_a_valid_response():
    """SuggestionResponse required exactly three, so one reply was a 500."""
    assert SuggestionResponse(suggestions=["come closer"]).suggestions == ["come closer"]
    with pytest.raises(ValueError):
        SuggestionResponse(suggestions=[])


def test_parse_reasons_distinguish_garbage_from_rejection():
    assert parse_reply_outcome("not json", Persona()).reason == PARSE_UNPARSEABLE
    assert parse_reply_outcome("", Persona()).reason == PARSE_UNPARSEABLE
    assert parse_reply_outcome(json.dumps([]), Persona()).reason == PARSE_UNPARSEABLE


# --- validation failure does not buy another identical generation -----------


def test_validation_failure_does_not_retry_the_same_model(harness):
    """The COST-001 headline: one rejection used to cost three generations."""
    harness["responses"] = [json.dumps(["noted"]), json.dumps(["come closer"])]

    replies = _run(harness)

    assert replies == ["come closer"]
    # Kimi once, then straight to the explicitly configured fallback model.
    assert harness["calls"] == [KIMI.model, COMPLEX_WRITER.model]
    assert harness["sleeps"] == [], "validation failure must not sleep"


def test_validation_failure_with_no_fallback_stops_after_one_generation(harness):
    harness["responses"] = [json.dumps(["noted"])]

    replies = _run(harness, fallback=None)

    assert replies == []
    assert harness["calls"] == [KIMI.model], "paid for a repeat of a rejected model"


def test_unparseable_output_still_retries_the_same_model_then_falls_back(harness):
    """Garbage output is a different failure from a rejected-but-valid reply."""
    harness["responses"] = ["not json", "not json", json.dumps(["come closer"])]

    replies = _run(harness)

    assert replies == ["come closer"]
    assert harness["calls"] == [KIMI.model, KIMI.model, COMPLEX_WRITER.model]
    assert harness["sleeps"] == [], "a parse failure is not a throttling signal"


# --- transport failures back off --------------------------------------------


def test_429_backs_off_before_retrying(harness):
    harness["responses"] = [
        _FakeError(429),
        _FakeError(429),
        json.dumps(["come closer"]),
    ]

    replies = _run(harness)

    assert replies == ["come closer"]
    assert harness["calls"] == [KIMI.model, KIMI.model, COMPLEX_WRITER.model]
    assert len(harness["sleeps"]) == 2, "a 429 must not be retried immediately"
    # Exponential from _BACKOFF_BASE_SECONDS, capped; jitter pinned to the top.
    assert harness["sleeps"] == [0.5, 1.0]


def test_5xx_causes_bounded_retry(harness):
    harness["responses"] = [_FakeError(503), _FakeError(503), _FakeError(503)]

    replies = _run(harness)

    assert replies == []
    assert len(harness["calls"]) == 3, "retries must stay bounded"
    # Two waits, not three: nothing is slept after the final attempt.
    assert harness["sleeps"] == [0.5, 1.0]


def test_retry_after_header_is_honoured(harness):
    harness["responses"] = [_FakeError(429, retry_after="4"), json.dumps(["come closer"])]

    replies = _run(harness)

    assert replies == ["come closer"]
    assert harness["sleeps"] == [4.0]


def test_absurd_retry_after_is_capped(harness):
    harness["responses"] = [
        _FakeError(429, retry_after="86400"),
        json.dumps(["come closer"]),
    ]

    _run(harness)

    assert harness["sleeps"] == [generator._RETRY_AFTER_CAP_SECONDS]


def test_malformed_retry_after_falls_back_to_exponential(harness):
    harness["responses"] = [
        _FakeError(429, retry_after="Wed, 21 Oct 2026 07:28:00 GMT"),
        json.dumps(["come closer"]),
    ]

    _run(harness)

    assert harness["sleeps"] == [0.5]


def test_backoff_is_jittered():
    """Full jitter, so concurrent writers do not re-converge on one instant."""
    delays = {generator._backoff_delay(1, _FakeError(429)) for _ in range(50)}
    assert len(delays) > 1
    assert all(0.0 <= delay <= 1.0 for delay in delays)


# --- routing is unchanged ---------------------------------------------------


def test_openrouter_random_provider_fallback_remains_disabled(monkeypatch):
    from ai.openrouter_routing import provider_preferences

    monkeypatch.delenv("OPENROUTER_ALLOW_FALLBACKS", raising=False)
    preferences = provider_preferences({"openrouter_providers": "Inceptron"})

    assert preferences["only"] == ["Inceptron"]
    assert preferences["allow_fallbacks"] is False


def test_only_the_explicit_fallback_target_is_ever_used(harness):
    """No target outside {primary, configured fallback} is contacted."""
    harness["responses"] = [_FakeError(500)]

    _run(harness)

    assert set(harness["calls"]) <= {KIMI.model, COMPLEX_WRITER.model}


def test_complex_route_without_a_fallback_never_switches_model(harness):
    """Complex-writer-only routes must stay on that model across every retry."""
    harness["responses"] = [_FakeError(500), _FakeError(500), _FakeError(500)]

    _run(harness, primary=COMPLEX_WRITER, fallback=None)

    assert set(harness["calls"]) == {COMPLEX_WRITER.model}
    assert len(harness["calls"]) == 3
