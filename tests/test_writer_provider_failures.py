"""Regressions for the production writer outage of 2026-09.

One simulated fan turn ("hii") produced no creator message at all, and Railway
showed only a single line about the LAST model in the ladder. What actually
happened was two separate faults stacked on top of each other:

* the ordinary writer, Kimi K2.6 on OpenRouter, returned HTTP 200 with
  ``message.content`` empty. K2.6 reasons by default and spent the whole
  completion budget on hidden reasoning, so the writer saw unparseable output —
  twice, silently, because an unparseable attempt logged nothing;
* the fallback, ``deepseek-ai/DeepSeek-V4-Pro`` on Together, was not callable at
  all: Together answers 400 "Unable to access non-serverless model ... Please
  create and start a dedicated endpoint" for this account. A permanently dead
  fallback is not a fallback.

These tests hold each of those closed.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ai import model_providers
from ai.generator import generate_replies
from ai.model_providers import complete, find_catalog_target
from ai.writer_router import (
    COMPLEX_WRITER_MODEL,
    COMPLEX_WRITER_PROVIDER,
    DEFAULT_WRITER_MODEL,
    DEFAULT_WRITER_PROVIDER,
    WriterRoute,
    select_writer_route,
)
from models.model_runtime import ModelTarget
from models.schemas import Persona


@pytest.fixture(autouse=True)
def clean_writer_env(monkeypatch):
    for name in (
        "WRITER_DEFAULT_PROVIDER",
        "WRITER_DEFAULT_MODEL",
        "WRITER_COMPLEX_PROVIDER",
        "WRITER_COMPLEX_MODEL",
        "OPENROUTER_PROVIDERS",
        "OPENROUTER_ALLOW_FALLBACKS",
        "OPENROUTER_DATA_COLLECTION",
        "OPENROUTER_ZDR",
        "OPENROUTER_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("WRITER_ROUTING_ENABLED", "true")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("TOGETHER_API_KEY", "test-together-key")


class _RecordingCompletions:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        response = self._responses[min(len(self.calls) - 1, len(self._responses) - 1)]
        if isinstance(response, Exception):
            raise response
        return response


class _RecordingClient:
    def __init__(self, responses):
        self.chat = SimpleNamespace(completions=_RecordingCompletions(responses))


def _response(content='["one", "two"]', completion_tokens=20):
    return SimpleNamespace(
        id="gen-1",
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=completion_tokens,
            prompt_tokens_details=SimpleNamespace(cached_tokens=0),
        ),
    )


class _Stream:
    """An async-iterable stand-in for a streamed chat completion.

    The complex writer streams (``stream: true`` in the catalog), so the
    reasoning assertion has to survive the streaming branch of the transport,
    not just the buffered one.
    """

    def __init__(self, content: str) -> None:
        self._chunks = [
            SimpleNamespace(
                id="gen-1",
                choices=[SimpleNamespace(delta=SimpleNamespace(content=content))],
                usage=None,
            ),
            SimpleNamespace(
                id="gen-1",
                choices=[],
                usage=SimpleNamespace(
                    prompt_tokens=100,
                    completion_tokens=20,
                    prompt_tokens_details=SimpleNamespace(cached_tokens=0),
                ),
            ),
        ]

    def __aiter__(self):
        async def _iterate():
            for chunk in self._chunks:
                yield chunk

        return _iterate()


def _install(monkeypatch, *responses):
    client = _RecordingClient(responses)
    monkeypatch.setattr(
        model_providers,
        "_openai_compatible_client",
        lambda *_a, **_k: client,
    )
    return client


def _prompt():
    return [
        {"role": "system", "content": "stable prefix"},
        {"role": "user", "content": "hii"},
    ]


# --- 1: the Kimi target sends the intended reasoning configuration -----------


def test_kimi_catalog_entry_disables_reasoning():
    """The configuration fix itself, asserted at the catalog.

    K2.6 reasons by default. With reasoning left on, the model returns
    message.content=null and the writer has nothing to parse — which is the
    outage this file is named after, not a style preference.
    """
    target = find_catalog_target("openrouter", DEFAULT_WRITER_MODEL)

    assert target is not None, "the ordinary writer must be in the catalog"
    assert target.metadata.get("reasoning_enabled") is False


def test_complex_writer_catalog_entry_disables_reasoning():
    """The same trap, closed on the other provider before it can fire."""
    target = find_catalog_target(COMPLEX_WRITER_PROVIDER, COMPLEX_WRITER_MODEL)

    assert target is not None, "the complex writer must be in the catalog"
    assert target.metadata.get("reasoning_enabled") is False


def test_kimi_request_actually_carries_reasoning_disabled(monkeypatch):
    """Catalog metadata is only worth anything if it reaches the wire."""
    client = _install(monkeypatch, _response())
    target = find_catalog_target("openrouter", DEFAULT_WRITER_MODEL)

    asyncio.run(
        complete(
            target,
            system="stable prefix",
            messages=[{"role": "user", "content": "hii"}],
            max_tokens=1000,
            session_id="cleo-abc",
            end_user_id="fan-abc",
        )
    )

    body = client.chat.completions.calls[0]["extra_body"]
    assert body["reasoning"] == {"enabled": False}
    # The provider pin must survive alongside it, not be replaced by it.
    assert body["provider"]["only"] == ["Inceptron"]
    assert body["provider"]["allow_fallbacks"] is False


def test_complex_writer_request_carries_reasoning_disabled(monkeypatch):
    client = _install(monkeypatch, _Stream('["one", "two"]'))
    target = find_catalog_target(COMPLEX_WRITER_PROVIDER, COMPLEX_WRITER_MODEL)

    asyncio.run(
        complete(
            target,
            system="stable prefix",
            messages=[{"role": "user", "content": "hii"}],
            max_tokens=1000,
        )
    )

    body = client.chat.completions.calls[0]["extra_body"]
    assert body["reasoning"] == {"enabled": False}
    # Together is not OpenRouter: no provider pin, no affinity key.
    assert "provider" not in body
    assert "session_id" not in body


# --- 2: the configured complex writer is a reachable serverless model --------


def test_complex_writer_is_not_the_dead_deepseek_handle():
    """Together answers 400 for this handle on the production account.

    Keeping a model in the config that the account provably cannot call means
    every ordinary turn whose primary attempts fail ends with no reply.
    """
    assert COMPLEX_WRITER_PROVIDER == "together", "provider diversity is deliberate"
    assert "DeepSeek-V4-Pro" not in COMPLEX_WRITER_MODEL


def test_dead_deepseek_entry_is_disabled_in_the_catalog():
    """Still described, so the reason is discoverable — but never selected."""
    catalog = json.loads(
        (Path(__file__).resolve().parents[1] / "config" / "model_candidates.json")
        .read_text(encoding="utf-8")
    )
    rows = [
        row
        for row in catalog["models"]
        if row["model"] == "deepseek-ai/DeepSeek-V4-Pro"
    ]
    assert rows, "the entry is kept so the failure stays documented"
    assert rows[0]["enabled"] is False
    assert "non-serverless" in rows[0]["metadata"]["unavailable"]


def test_ordinary_route_pairs_openrouter_primary_with_together_fallback():
    decision = select_writer_route(
        SimpleNamespace(
            situation={},
            commercial_decision=None,
            conversation_stage="WARMING_UP",
            active_session=None,
            fan_profile=SimpleNamespace(
                total_spent=0, spend_tier="cold", needs_human_review=False
            ),
        )
    )

    assert decision.route == WriterRoute.DEFAULT
    assert decision.primary_target.provider == DEFAULT_WRITER_PROVIDER
    assert decision.primary_target.model == DEFAULT_WRITER_MODEL
    assert decision.fallback_target is not None
    assert decision.fallback_target.provider == COMPLEX_WRITER_PROVIDER
    assert decision.fallback_target.model == COMPLEX_WRITER_MODEL


# --- 3: a working primary never pays for the fallback ------------------------


def test_successful_kimi_primary_never_calls_the_fallback(monkeypatch):
    attempted: list[str] = []

    async def fake_complete(target, **_kwargs):
        attempted.append(f"{target.provider}:{target.model}")
        return SimpleNamespace(
            text='["hey", "what are you up to?"]',
            target=target,
            usage=SimpleNamespace(
                input_tokens=100, output_tokens=20,
                cache_read_tokens=0, cache_write_tokens=0,
            ),
            latency_ms=10,
            raw_response_id="gen-1",
            upstream_provider="Inceptron",
            reported_cost_usd=None,
            gate_wait_ms=0,
        )

    monkeypatch.setattr("ai.generator.complete", fake_complete)

    replies = asyncio.run(
        generate_replies(
            _prompt(),
            Persona(),
            telemetry_context={"creator_id": "c1", "fan_id": "f1"},
            target_override=find_catalog_target("openrouter", DEFAULT_WRITER_MODEL),
            fallback_target_override=find_catalog_target(
                COMPLEX_WRITER_PROVIDER, COMPLEX_WRITER_MODEL
            ),
        )
    )

    assert replies == ["hey", "what are you up to?"]
    assert attempted == [f"openrouter:{DEFAULT_WRITER_MODEL}"]


# --- 4: the production failure shape, now survivable -------------------------


def test_empty_kimi_content_falls_through_to_a_working_together_fallback(monkeypatch):
    """The exact production sequence, with a fallback that actually answers.

    Two HTTP-successful Kimi attempts with empty content, then the Together
    writer — which previously was a dead handle, so the turn ended in silence.
    """
    attempted: list[str] = []

    async def fake_complete(target, **_kwargs):
        attempted.append(f"{target.provider}:{target.model}")
        empty = target.provider == "openrouter"
        return SimpleNamespace(
            # An entire completion budget spent on hidden reasoning.
            text="" if empty else '["hey you", "what are you doing tonight?"]',
            target=target,
            usage=SimpleNamespace(
                input_tokens=100,
                output_tokens=1000 if empty else 24,
                cache_read_tokens=0,
                cache_write_tokens=0,
            ),
            latency_ms=10,
            raw_response_id="gen-1",
            upstream_provider="Inceptron" if empty else None,
            reported_cost_usd=None,
            gate_wait_ms=0,
        )

    monkeypatch.setattr("ai.generator.complete", fake_complete)

    replies = asyncio.run(
        generate_replies(
            _prompt(),
            Persona(),
            telemetry_context={"creator_id": "c1", "fan_id": "f1"},
            target_override=find_catalog_target("openrouter", DEFAULT_WRITER_MODEL),
            fallback_target_override=find_catalog_target(
                COMPLEX_WRITER_PROVIDER, COMPLEX_WRITER_MODEL
            ),
        )
    )

    assert replies == ["hey you", "what are you doing tonight?"]
    assert attempted == [
        f"openrouter:{DEFAULT_WRITER_MODEL}",
        f"openrouter:{DEFAULT_WRITER_MODEL}",
        f"{COMPLEX_WRITER_PROVIDER}:{COMPLEX_WRITER_MODEL}",
    ]


# --- 5: an unavailable model produces an actionable failure ------------------


class _NonServerless(RuntimeError):
    """Shaped like the Together 400 the account actually receives."""

    status_code = 400

    def __init__(self) -> None:
        super().__init__(
            "Error code: 400 - {'message': 'Unable to access non-serverless model "
            "deepseek-ai/DeepSeek-V4-Pro. Please create and start a dedicated "
            "endpoint.'}"
        )


def test_unavailable_fallback_model_is_reported_actionably(monkeypatch, capsys):
    """Fail closed, but never silently: the log must name model and status."""

    async def fake_complete(target, **_kwargs):
        if target.provider == "together":
            raise _NonServerless()
        return SimpleNamespace(
            text="",
            target=target,
            usage=SimpleNamespace(
                input_tokens=100, output_tokens=1000,
                cache_read_tokens=0, cache_write_tokens=0,
            ),
            latency_ms=10,
            raw_response_id="gen-1",
            upstream_provider="Inceptron",
            reported_cost_usd=None,
            gate_wait_ms=0,
        )

    monkeypatch.setattr("ai.generator.complete", fake_complete)
    monkeypatch.setattr("ai.generator._sleep", lambda _s: asyncio.sleep(0))

    replies = asyncio.run(
        generate_replies(
            _prompt(),
            Persona(),
            telemetry_context={"creator_id": "c1", "fan_id": "f1"},
            target_override=find_catalog_target("openrouter", DEFAULT_WRITER_MODEL),
            fallback_target_override=ModelTarget(
                name="together:deepseek-ai/DeepSeek-V4-Pro",
                provider="together",
                model="deepseek-ai/DeepSeek-V4-Pro",
                base_url="https://api.together.xyz/v1",
                api_key_env="TOGETHER_API_KEY",
            ),
        )
    )

    assert replies == [], "the writer must still fail closed"
    logged = capsys.readouterr().out
    assert "status=400" in logged
    assert "Unable to access non-serverless model" in logged
    assert "deepseek-ai/DeepSeek-V4-Pro" in logged


# --- 6: observability — a silent unparseable attempt is now impossible -------


def test_empty_content_attempts_are_logged_without_any_model_output(monkeypatch, capsys):
    """What would have made the outage obvious in Railway within a minute.

    Every HTTP-successful but unusable attempt must report itself, and must do
    so without ever printing a prompt, a fan message, or model output.
    """

    async def fake_complete(target, **_kwargs):
        return SimpleNamespace(
            text="",
            target=target,
            usage=SimpleNamespace(
                input_tokens=100, output_tokens=1000,
                cache_read_tokens=0, cache_write_tokens=0,
            ),
            latency_ms=10,
            raw_response_id="gen-1",
            upstream_provider="Inceptron",
            reported_cost_usd=None,
            gate_wait_ms=0,
        )

    monkeypatch.setattr("ai.generator.complete", fake_complete)

    asyncio.run(
        generate_replies(
            _prompt(),
            Persona(),
            telemetry_context={"creator_id": "c1", "fan_id": "f1"},
            target_override=find_catalog_target("openrouter", DEFAULT_WRITER_MODEL),
        )
    )

    logged = capsys.readouterr().out
    lines = [line for line in logged.splitlines() if line.startswith("[GENERATOR] attempt=")]
    assert len(lines) == 3, logged
    for index, line in enumerate(lines, start=1):
        assert f"attempt={index}" in line
        assert "provider=openrouter" in line
        assert f"model={DEFAULT_WRITER_MODEL}" in line
        assert "outcome=unparseable" in line
        assert "content_empty=true" in line
        assert "output_tokens=1000" in line
    # The fan's message and the prompt must never reach a log line.
    assert "hii" not in logged
    assert "stable prefix" not in logged


def test_rejected_candidates_are_logged_as_a_different_outcome(monkeypatch, capsys):
    """"The model said nothing" and "our filter refused it" are not the same
    event, and the log must not blur them."""

    async def fake_complete(target, **_kwargs):
        return SimpleNamespace(
            # Valid JSON the writer's own validator refuses as bot-speak.
            text='["as an ai, i am happy to help"]',
            target=target,
            usage=SimpleNamespace(
                input_tokens=100, output_tokens=12,
                cache_read_tokens=0, cache_write_tokens=0,
            ),
            latency_ms=10,
            raw_response_id="gen-1",
            upstream_provider="Inceptron",
            reported_cost_usd=None,
            gate_wait_ms=0,
        )

    monkeypatch.setattr("ai.generator.complete", fake_complete)

    asyncio.run(
        generate_replies(
            _prompt(),
            Persona(),
            telemetry_context={"creator_id": "c1", "fan_id": "f1"},
            target_override=find_catalog_target("openrouter", DEFAULT_WRITER_MODEL),
        )
    )

    logged = capsys.readouterr().out
    assert "outcome=all_rejected" in logged
    assert "content_empty=false" in logged
    assert "as an ai" not in logged, "model output must never be logged"
