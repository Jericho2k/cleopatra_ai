"""Kimi is the real primary writer on cleo_v3; Qwen is a last resort.

The frozen plan fell back after ONE Kimi failure, so "primary: Kimi" described
the configuration and frequently not the output: a single 429, a truncated body
or one rejected candidate handed the turn to a different voice on a different
provider. This asserts the corrected plan — four Kimi attempts, spaced by the
configured schedule, before Qwen is reached at all — and that a failure no wait
can repair skips the waits entirely rather than sleeping ninety-five seconds to
rediscover a bad API key.

Every delay here is asserted through a fake clock. No test waits.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from ai import generator
from ai.generator import (
    CONTRACT_AUTO_MESSAGES,
    LEGACY_WRITER_RETRY_POLICY,
    PERSISTENT_PRIMARY_RETRY_POLICY,
    PRIMARY_RETRY_WAIT_SECONDS,
    is_permanent_failure,
)
from ai.writer_style import (
    MODE_ASSISTED,
    MODE_AUTO,
    WRITER_V1,
    WRITER_V2,
    WRITER_V3,
    persistent_primary_retries,
    uses_auto_messages_contract,
)
from models.model_runtime import ModelResult, ModelTarget, ModelUsage
from models.schemas import Persona
from services.suggestions import writer_retry_policy


def _target(provider: str, model: str) -> ModelTarget:
    return ModelTarget(
        name=model,
        provider=provider,
        model=model,
        base_url=f"https://{provider}.example/v1",
        api_key_env="TEST_API_KEY",
    )


KIMI = _target("openrouter", "moonshotai/kimi-k2.6")
QWEN = _target("together", "Qwen/Qwen3.7-Plus")


class _ProviderError(Exception):
    """Shaped like the SDK errors, without importing a client."""

    def __init__(self, status_code: int | None = None, message: str | None = None):
        super().__init__(message or f"HTTP {status_code}")
        if status_code is not None:
            self.status_code = status_code


@pytest.fixture
def harness(monkeypatch):
    state: dict = {"calls": [], "sleeps": [], "responses": [], "logs": []}

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
            usage=ModelUsage(input_tokens=10, output_tokens=5),
            latency_ms=1,
        )

    async def noop(*_args, **_kwargs):
        return None

    monkeypatch.setattr(generator, "_sleep", fake_sleep)
    monkeypatch.setattr(generator, "complete", fake_complete)
    monkeypatch.setattr(generator, "record_model_result", noop)
    monkeypatch.setattr(generator, "record_model_failure", noop)
    monkeypatch.setattr(generator.random, "uniform", lambda _low, high: high)
    monkeypatch.setitem(
        generator.__builtins__ if isinstance(generator.__builtins__, dict)
        else vars(generator.__builtins__),
        "print",
        lambda *args, **_k: state["logs"].append(" ".join(map(str, args))),
    )
    return state


def _run(
    harness,
    *,
    policy=PERSISTENT_PRIMARY_RETRY_POLICY,
    fallback=QWEN,
    contract=CONTRACT_AUTO_MESSAGES,
    profile_id="cleo_v3",
):
    return asyncio.run(
        generator.generate_replies(
            [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "user"},
            ],
            Persona(),
            telemetry_context={"feature": "auto_reply"},
            target_override=KIMI,
            fallback_target_override=fallback,
            output_contract=contract,
            retry_policy=policy,
            profile_id=profile_id,
        )
    )


def _messages(*bubbles: str) -> str:
    return json.dumps({"messages": list(bubbles)})


# --- the schedule -----------------------------------------------------------


def test_transient_kimi_failure_retries_kimi_and_never_reaches_qwen(harness):
    harness["responses"] = [
        _ProviderError(429),
        _ProviderError(503),
        _messages("hey you", "what are you up to"),
    ]

    replies = _run(harness)

    assert replies == ["hey you | what are you up to"]
    assert harness["calls"] == [KIMI.model, KIMI.model, KIMI.model]
    assert QWEN.model not in harness["calls"]
    # Waits before attempts 2 and 3 only; attempt 1 is immediate.
    assert harness["sleeps"] == [PRIMARY_RETRY_WAIT_SECONDS[0], PRIMARY_RETRY_WAIT_SECONDS[1]]


def test_qwen_is_only_reached_after_every_kimi_attempt_is_spent(harness):
    harness["responses"] = [
        _ProviderError(500),
        _ProviderError(500),
        _ProviderError(500),
        _ProviderError(500),
        _messages("qwen wrote this"),
    ]

    replies = _run(harness)

    assert replies == ["qwen wrote this"]
    assert harness["calls"] == [KIMI.model] * 4 + [QWEN.model]
    assert harness["sleeps"] == list(PRIMARY_RETRY_WAIT_SECONDS)


def test_the_whole_schedule_is_ninety_five_seconds_and_is_never_actually_waited():
    assert PRIMARY_RETRY_WAIT_SECONDS == (5.0, 30.0, 60.0)
    assert sum(PRIMARY_RETRY_WAIT_SECONDS) == 95.0
    assert PERSISTENT_PRIMARY_RETRY_POLICY.primary_attempts == 4


def test_invalid_structured_output_retries_kimi_rather_than_falling_back(harness):
    harness["responses"] = [
        "not json at all",
        json.dumps(["an array, which is the wrong shape"]),
        _messages("third time lucky"),
    ]

    replies = _run(harness)

    assert replies == ["third time lucky"]
    assert harness["calls"] == [KIMI.model] * 3


def test_rejected_candidate_output_retries_kimi_under_the_persistent_policy(harness):
    harness["responses"] = [_messages("noted"), _messages("actually saying something")]

    replies = _run(harness)

    assert replies == ["actually saying something"]
    assert harness["calls"] == [KIMI.model, KIMI.model]
    assert harness["sleeps"] == [PRIMARY_RETRY_WAIT_SECONDS[0]]


def test_a_provider_retry_after_longer_than_the_schedule_wins(harness):
    class _RateLimited(Exception):
        def __init__(self):
            super().__init__("429")
            self.status_code = 429

            class _Response:
                headers = {"retry-after": "12"}
                status_code = 429

            self.response = _Response()

    harness["responses"] = [_RateLimited(), _messages("ok now")]

    _run(harness)

    # 12s advertised beats the 5s floor for attempt 2.
    assert harness["sleeps"] == [12.0]


def test_every_retry_is_logged_with_profile_model_attempt_wait_and_reason(harness):
    harness["responses"] = [_ProviderError(503), _messages("fine")]

    _run(harness)

    retries = [line for line in harness["logs"] if line.startswith("[WRITER RETRY]")]
    assert retries, harness["logs"]
    assert "profile=cleo_v3" in retries[0]
    assert "model=moonshotai/kimi-k2.6" in retries[0]
    assert "attempt=2" in retries[0]
    assert "wait=5" in retries[0]
    assert "reason=transport:503" in retries[0]


# --- permanent failures skip the waits --------------------------------------


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_permanent_statuses_are_recognised(status):
    assert is_permanent_failure(_ProviderError(status))


@pytest.mark.parametrize("status", [408, 409, 425, 429, 500, 502, 503, 504])
def test_retryable_statuses_are_not_permanent(status):
    assert not is_permanent_failure(_ProviderError(status))


def test_a_missing_api_key_is_permanent_even_without_a_status():
    assert is_permanent_failure(_ProviderError(message="OPENROUTER_API_KEY not configured"))


def test_a_permanent_failure_goes_straight_to_qwen_without_sleeping(harness):
    harness["responses"] = [_ProviderError(401), _messages("qwen answered")]

    replies = _run(harness)

    assert replies == ["qwen answered"]
    assert harness["calls"] == [KIMI.model, QWEN.model]
    assert harness["sleeps"] == [], "no wait can fix a rejected key"


def test_a_permanent_failure_with_no_fallback_stops_immediately(harness):
    harness["responses"] = [_ProviderError(403)]

    replies = _run(harness, fallback=None)

    assert replies == []
    assert harness["calls"] == [KIMI.model]
    assert harness["sleeps"] == []


def test_the_persistent_policy_does_not_invent_a_fifth_attempt_without_a_fallback(
    harness,
):
    harness["responses"] = [_ProviderError(503)]

    replies = _run(harness, fallback=None)

    assert replies == []
    assert harness["calls"] == [KIMI.model] * 4


# --- the frozen profiles are untouched --------------------------------------


def test_v1_and_v2_keep_the_frozen_two_attempts_then_fallback_plan(harness):
    harness["responses"] = [
        _ProviderError(503),
        _ProviderError(503),
        json.dumps(["qwen wrote this"]),
    ]

    replies = _run(
        harness,
        policy=LEGACY_WRITER_RETRY_POLICY,
        contract="candidates",
        profile_id="cleo_v2",
    )

    assert replies == ["qwen wrote this"]
    assert harness["calls"] == [KIMI.model, KIMI.model, QWEN.model]


def test_the_legacy_policy_still_retires_a_model_whose_output_was_rejected(harness):
    harness["responses"] = [json.dumps(["noted"]), json.dumps(["come closer"])]

    replies = _run(
        harness,
        policy=LEGACY_WRITER_RETRY_POLICY,
        contract="candidates",
        profile_id="cleo_v2",
    )

    assert replies == ["come closer"]
    assert harness["calls"] == [KIMI.model, QWEN.model]
    assert harness["sleeps"] == []


@pytest.mark.parametrize("version", [WRITER_V1, WRITER_V2])
def test_only_v3_opts_into_the_persistent_plan(version):
    assert persistent_primary_retries(version) is False
    assert writer_retry_policy(version) is LEGACY_WRITER_RETRY_POLICY


def test_v3_selects_the_persistent_plan():
    assert persistent_primary_retries(WRITER_V3) is True
    assert writer_retry_policy(WRITER_V3) is PERSISTENT_PRIMARY_RETRY_POLICY


# --- the auto contract ------------------------------------------------------


def test_only_v3_auto_uses_the_messages_object():
    assert uses_auto_messages_contract(WRITER_V3, MODE_AUTO) is True
    assert uses_auto_messages_contract(WRITER_V3, MODE_ASSISTED) is False
    assert uses_auto_messages_contract(WRITER_V2, MODE_AUTO) is False
    assert uses_auto_messages_contract(WRITER_V1, MODE_AUTO) is False


def test_two_bubbles_are_one_reply_not_two_candidates(harness):
    harness["responses"] = [_messages("okay wait", "that's actually funny")]

    replies = _run(harness)

    assert replies == ["okay wait | that's actually funny"]
    assert len(replies) == 1, "a multi-bubble reply is still ONE reply"


def test_no_candidate_truncation_log_on_the_auto_path(harness):
    harness["responses"] = [_messages("one", "two", "three")]

    replies = _run(harness)

    assert replies == ["one | two | three"]
    assert not [line for line in harness["logs"] if "candidates for a" in line]
