"""Regression tests for the OpenRouter Kimi K2.6 ordinary writer route.

Covers routing, provider pinning, cache affinity, usage accounting, failure
visibility, and that none of it disturbed the DeepSeek commercial route.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from ai import model_providers, openrouter_routing
from ai.generator import flatten_message_content, generate_replies
from ai.model_migrations import resolve_supported_model
from ai.model_providers import complete, get_runtime_target
from ai.session_affinity import writer_end_user_id, writer_session_id
from ai.writer_router import WriterRoute, select_writer_route
from models.model_runtime import ModelTarget, ModelUsage, resolve_cost_usd
from models.schemas import Persona


@pytest.fixture(autouse=True)
def clean_writer_env(monkeypatch):
    """Start every test from the shipped defaults, not a leaked override."""
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


def _ctx(**overrides):
    values = {
        "situation": {},
        "commercial_decision": None,
        "conversation_stage": "WARMING_UP",
        "active_session": None,
        "fan_profile": SimpleNamespace(
            total_spent=0,
            spend_tier="cold",
            needs_human_review=False,
        ),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


# --- 1 & 2: the K2.6 -> K3 redirect stays scoped to Together -----------------


def test_together_kimi_k26_still_redirects_to_k3():
    assert (
        resolve_supported_model("together", "moonshotai/Kimi-K2.6")
        == "moonshotai/Kimi-K3"
    )


def test_openrouter_kimi_k26_is_not_redirected_to_k3():
    for identifier in ("moonshotai/kimi-k2.6", "moonshotai/Kimi-K2.6"):
        assert resolve_supported_model("openrouter", identifier) == identifier


def test_openrouter_writer_route_keeps_k26_end_to_end(monkeypatch):
    monkeypatch.setenv("WRITER_DEFAULT_PROVIDER", "openrouter")
    monkeypatch.setenv("WRITER_DEFAULT_MODEL", "moonshotai/kimi-k2.6")

    decision = select_writer_route(_ctx())

    assert decision.primary_target.model == "moonshotai/kimi-k2.6"


def test_together_writer_route_still_redirects_k26(monkeypatch):
    monkeypatch.setenv("WRITER_DEFAULT_PROVIDER", "together")
    monkeypatch.setenv("WRITER_DEFAULT_MODEL", "moonshotai/Kimi-K2.6")

    decision = select_writer_route(_ctx())

    assert decision.primary_target.model == "moonshotai/Kimi-K3"


# --- 3: the shipped default resolves to OpenRouter K2.6 ----------------------


def test_default_writer_resolves_to_openrouter_kimi_k26():
    decision = select_writer_route(_ctx())

    assert decision.route == WriterRoute.DEFAULT
    assert decision.primary_target.provider == "openrouter"
    assert decision.primary_target.model == "moonshotai/kimi-k2.6"
    assert decision.primary_target.base_url == "https://openrouter.ai/api/v1"
    assert decision.primary_target.api_key_env == "OPENROUTER_API_KEY"


def test_generic_chat_runtime_supports_openrouter(monkeypatch):
    monkeypatch.setenv("CHAT_PROVIDER", "openrouter")
    monkeypatch.setenv("CHAT_MODEL", "moonshotai/kimi-k2.6")

    target = get_runtime_target("CHAT")

    assert target.provider == "openrouter"
    assert target.model == "moonshotai/kimi-k2.6"
    assert target.base_url == "https://openrouter.ai/api/v1"
    assert target.api_key_env == "OPENROUTER_API_KEY"


# --- 9: DeepSeek commercial routing is untouched ------------------------------


def test_commercial_complex_route_still_uses_together_deepseek():
    decision = select_writer_route(
        _ctx(commercial_decision={"action": "PRESENT_SESSION_OPTIONS"})
    )

    assert decision.route == WriterRoute.COMMERCIAL_COMPLEX
    assert decision.primary_target.provider == "together"
    assert decision.primary_target.model == "deepseek-ai/DeepSeek-V4-Pro"
    assert decision.fallback_target is None


def test_safety_sensitive_route_still_uses_together_deepseek():
    decision = select_writer_route(_ctx(situation={"crisis_signal": "self_harm"}))

    assert decision.route == WriterRoute.SAFETY_SENSITIVE
    assert decision.primary_target.provider == "together"
    assert decision.primary_target.model == "deepseek-ai/DeepSeek-V4-Pro"


def test_ordinary_route_falls_back_to_deepseek_not_another_openrouter_provider():
    decision = select_writer_route(_ctx())

    assert decision.fallback_target is not None
    assert decision.fallback_target.provider == "together"
    assert decision.fallback_target.model == "deepseek-ai/DeepSeek-V4-Pro"


# --- 4: the Kimi route carries the intended pinned provider controls ---------


def test_provider_preferences_pin_inceptron_and_fail_closed():
    preferences = openrouter_routing.provider_preferences()

    assert preferences["only"] == ["Inceptron"]
    assert preferences["allow_fallbacks"] is False
    assert preferences["data_collection"] == "deny"
    # ZDR is not asserted unless a deployment explicitly opts in.
    assert "zdr" not in preferences


def test_provider_pin_is_overridable_without_a_deploy(monkeypatch):
    monkeypatch.setenv("OPENROUTER_PROVIDERS", "Inceptron, DeepInfra")

    assert openrouter_routing.provider_preferences()["only"] == [
        "Inceptron",
        "DeepInfra",
    ]


def test_zdr_is_only_requested_when_explicitly_enabled(monkeypatch):
    monkeypatch.setenv("OPENROUTER_ZDR", "true")

    assert openrouter_routing.provider_preferences()["zdr"] is True


def test_catalog_metadata_supplies_the_default_pin():
    target = model_providers.find_catalog_target("openrouter", "moonshotai/kimi-k2.6")

    assert target is not None
    assert openrouter_routing.pinned_providers(target.metadata) == ["Inceptron"]


# --- 5 & 6: conversation-stable session affinity ------------------------------


def test_same_creator_and_fan_produce_stable_affinity():
    first = writer_session_id("creator-1", "fan-1")
    second = writer_session_id("creator-1", "fan-1")

    assert first == second
    assert first


def test_different_fan_produces_different_affinity():
    assert writer_session_id("creator-1", "fan-1") != writer_session_id(
        "creator-1", "fan-2"
    )


def test_different_creator_produces_different_affinity():
    assert writer_session_id("creator-1", "fan-1") != writer_session_id(
        "creator-2", "fan-1"
    )


def test_affinity_key_never_leaks_raw_identifiers_and_fits_the_limit():
    key = writer_session_id("creator-1", "fan-1")

    assert "creator-1" not in key
    assert "fan-1" not in key
    assert len(key) <= 256


def test_affinity_is_absent_when_an_identifier_is_missing():
    assert writer_session_id("creator-1", None) is None
    assert writer_session_id("", "fan-1") is None
    assert writer_end_user_id(None, None) is None


def test_end_user_id_differs_from_the_session_key():
    # Same determinism, different namespace, so one value cannot be mistaken
    # for the other in provider-side logs.
    assert writer_end_user_id("creator-1", "fan-1") != writer_session_id(
        "creator-1", "fan-1"
    )


# --- request construction -----------------------------------------------------


class _RecordingCompletions:
    def __init__(self, response):
        self.response = response
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class _RecordingClient:
    def __init__(self, response):
        self.chat = SimpleNamespace(completions=_RecordingCompletions(response))


def _response(
    *,
    content="[\"a\", \"b\", \"c\"]",
    prompt_tokens=1000,
    completion_tokens=20,
    cached_tokens=0,
    cache_write_tokens=None,
    cost=None,
    provider=None,
):
    prompt_details = SimpleNamespace(cached_tokens=cached_tokens)
    if cache_write_tokens is not None:
        prompt_details.cache_write_tokens = cache_write_tokens
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        prompt_tokens_details=prompt_details,
    )
    if cost is not None:
        usage.cost = cost
    response = SimpleNamespace(
        id="gen-1",
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=usage,
    )
    if provider is not None:
        response.provider = provider
    return response


def _openrouter_target(**overrides):
    values = {
        "name": "openrouter:moonshotai/kimi-k2.6",
        "provider": "openrouter",
        "model": "moonshotai/kimi-k2.6",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
    }
    values.update(overrides)
    return ModelTarget(**values)


def _install_client(monkeypatch, response):
    client = _RecordingClient(response)
    monkeypatch.setattr(
        model_providers,
        "_openai_compatible_client",
        lambda *_args, **_kwargs: client,
    )
    return client


def test_openrouter_request_carries_pin_and_session_affinity(monkeypatch):
    client = _install_client(monkeypatch, _response(provider="Inceptron"))

    asyncio.run(
        complete(
            _openrouter_target(),
            system="stable prefix",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=64,
            session_id="cleo-abc",
            end_user_id="fan-abc",
        )
    )

    body = client.chat.completions.calls[0]["extra_body"]
    assert body["provider"] == {
        "only": ["Inceptron"],
        "allow_fallbacks": False,
        "data_collection": "deny",
    }
    assert body["session_id"] == "cleo-abc"
    assert body["user"] == "fan-abc"


def test_together_requests_carry_no_openrouter_fields(monkeypatch):
    client = _install_client(monkeypatch, _response())

    asyncio.run(
        complete(
            ModelTarget(
                name="together:deepseek-ai/DeepSeek-V4-Pro",
                provider="together",
                model="deepseek-ai/DeepSeek-V4-Pro",
                base_url="https://api.together.xyz/v1",
                api_key_env="TOGETHER_API_KEY",
            ),
            system="prefix",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=64,
            session_id="cleo-abc",
        )
    )

    body = client.chat.completions.calls[0].get("extra_body") or {}
    assert "provider" not in body
    assert "session_id" not in body


def test_session_affinity_is_omitted_when_the_conversation_is_unknown(monkeypatch):
    client = _install_client(monkeypatch, _response())

    asyncio.run(
        complete(
            _openrouter_target(),
            system="prefix",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=64,
        )
    )

    body = client.chat.completions.calls[0]["extra_body"]
    assert "session_id" not in body
    assert "user" not in body


# --- 7: cached token accounting ----------------------------------------------


def test_cached_tokens_are_not_double_counted_as_input(monkeypatch):
    _install_client(
        monkeypatch,
        _response(prompt_tokens=1000, cached_tokens=800, provider="Inceptron"),
    )

    result = asyncio.run(
        complete(
            _openrouter_target(),
            system="prefix",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=64,
        )
    )

    assert result.usage.cache_read_tokens == 800
    assert result.usage.input_tokens == 200
    assert result.usage.input_tokens + result.usage.cache_read_tokens == 1000


def test_cache_writes_inside_prompt_tokens_are_subtracted_once(monkeypatch):
    _install_client(
        monkeypatch,
        _response(prompt_tokens=1000, cached_tokens=600, cache_write_tokens=300),
    )

    result = asyncio.run(
        complete(
            _openrouter_target(),
            system="prefix",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=64,
        )
    )

    assert result.usage.cache_read_tokens == 600
    assert result.usage.cache_write_tokens == 300
    assert result.usage.input_tokens == 100


def test_cache_writes_reported_additively_do_not_understate_input(monkeypatch):
    # 900 cached + 900 written cannot both live inside 1000 prompt tokens, so
    # the writes are additive and must not be deducted from uncached input.
    _install_client(
        monkeypatch,
        _response(prompt_tokens=1000, cached_tokens=900, cache_write_tokens=900),
    )

    result = asyncio.run(
        complete(
            _openrouter_target(),
            system="prefix",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=64,
        )
    )

    assert result.usage.input_tokens == 100
    assert result.usage.cache_write_tokens == 900


def test_upstream_provider_and_reported_cost_are_captured(monkeypatch):
    _install_client(
        monkeypatch,
        _response(cost=0.00123, provider="Inceptron"),
    )

    result = asyncio.run(
        complete(
            _openrouter_target(),
            system="prefix",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=64,
        )
    )

    assert result.upstream_provider == "Inceptron"
    assert result.reported_cost_usd == 0.00123


def test_upstream_provider_is_read_from_router_metadata(monkeypatch):
    response = _response()
    response.openrouter_metadata = SimpleNamespace(
        attempts=[{"model": "moonshotai/kimi-k2.6", "provider": "Inceptron", "status": 200}]
    )
    _install_client(monkeypatch, response)

    result = asyncio.run(
        complete(
            _openrouter_target(),
            system="prefix",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=64,
        )
    )

    assert result.upstream_provider == "Inceptron"


def test_reported_cost_wins_over_the_catalog_estimate():
    target = _openrouter_target(input_per_million=100.0, output_per_million=100.0)
    usage = ModelUsage(input_tokens=1_000_000, output_tokens=1_000_000)

    assert resolve_cost_usd(target, usage) == 200.0
    assert resolve_cost_usd(target, usage, reported_cost_usd=0.42) == 0.42


def test_unusable_reported_cost_falls_back_to_the_estimate():
    target = _openrouter_target(input_per_million=1.0)
    usage = ModelUsage(input_tokens=1_000_000)

    assert resolve_cost_usd(target, usage, reported_cost_usd=None) == 1.0
    assert resolve_cost_usd(target, usage, reported_cost_usd=-5) == 1.0
    assert resolve_cost_usd(target, usage, reported_cost_usd="nope") == 1.0


# --- 8: OpenRouter failure stays visible --------------------------------------


def test_openrouter_failure_is_recorded_and_never_silently_reprovidered(monkeypatch):
    """A pinned-provider rejection must surface, then use Cleopatra's own fallback."""

    attempts: list[str] = []
    failures: list[tuple[str, str]] = []

    async def fake_complete(target, **kwargs):
        attempts.append(f"{target.provider}:{target.model}")
        if target.provider == "openrouter":
            raise RuntimeError(
                "No endpoints found matching your data policy (Inceptron)"
            )
        return SimpleNamespace(
            text='["one", "two", "three"]',
            target=target,
            usage=ModelUsage(input_tokens=10, output_tokens=5),
            latency_ms=12,
            raw_response_id="gen-2",
            upstream_provider=None,
            reported_cost_usd=None,
        )

    async def fake_record_failure(target, _context, *, error, retry_count=0):
        failures.append((target.provider, error))

    monkeypatch.setattr("ai.generator.complete", fake_complete)
    monkeypatch.setattr("ai.generator.record_model_failure", fake_record_failure)

    replies = asyncio.run(
        generate_replies(
            [
                {"role": "system", "content": "stable prefix"},
                {"role": "user", "content": "hi"},
            ],
            Persona(),
            telemetry_context={"creator_id": "creator-1", "fan_id": "fan-1"},
            target_override=_openrouter_target(),
            fallback_target_override=ModelTarget(
                name="together:deepseek-ai/DeepSeek-V4-Pro",
                provider="together",
                model="deepseek-ai/DeepSeek-V4-Pro",
                base_url="https://api.together.xyz/v1",
                api_key_env="TOGETHER_API_KEY",
            ),
        )
    )

    # Two Kimi attempts, then Cleopatra's explicit DeepSeek fallback. Never a
    # different OpenRouter upstream.
    assert attempts == [
        "openrouter:moonshotai/kimi-k2.6",
        "openrouter:moonshotai/kimi-k2.6",
        "together:deepseek-ai/DeepSeek-V4-Pro",
    ]
    assert [provider for provider, _ in failures] == ["openrouter", "openrouter"]
    assert all("data policy" in error for _, error in failures)
    assert replies == ["one", "two", "three"]


def test_total_openrouter_failure_fails_closed_with_no_suggestions(monkeypatch):
    async def always_fail(target, **kwargs):
        raise RuntimeError("upstream unavailable")

    monkeypatch.setattr("ai.generator.complete", always_fail)

    replies = asyncio.run(
        generate_replies(
            [
                {"role": "system", "content": "prefix"},
                {"role": "user", "content": "hi"},
            ],
            Persona(),
            target_override=_openrouter_target(),
        )
    )

    assert replies == []


# --- prompt transport ---------------------------------------------------------


def test_content_blocks_flatten_to_ordered_text_not_a_python_repr():
    flattened = flatten_message_content(
        [
            {"type": "text", "text": "STABLE PREFIX", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": " VOLATILE TAIL"},
        ]
    )

    assert flattened == "STABLE PREFIX VOLATILE TAIL"
    assert "cache_control" not in flattened
    assert "{" not in flattened


def test_plain_string_content_is_unchanged():
    assert flatten_message_content("already a string") == "already a string"


def test_generator_sends_a_stable_prefix_and_affinity_key(monkeypatch):
    seen: list[dict] = []

    async def fake_complete(target, **kwargs):
        seen.append(kwargs)
        return SimpleNamespace(
            text='["one", "two", "three"]',
            target=target,
            usage=ModelUsage(input_tokens=10, output_tokens=5),
            latency_ms=1,
            raw_response_id=None,
            upstream_provider="Inceptron",
            reported_cost_usd=None,
        )

    monkeypatch.setattr("ai.generator.complete", fake_complete)

    asyncio.run(
        generate_replies(
            [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "STABLE"},
                        {"type": "text", "text": "VOLATILE"},
                    ],
                },
                {"role": "user", "content": "hi"},
            ],
            Persona(),
            telemetry_context={"creator_id": "creator-1", "fan_id": "fan-1"},
            target_override=_openrouter_target(),
        )
    )

    # COST-002a — the generator hands the blocks down untouched. Flattening for
    # OpenAI-compatible providers now happens in the transport, which is the
    # only layer that knows whether the provider consumes blocks; asserting the
    # joined string here would have re-pinned the defect the audit found.
    assert seen[0]["system"] == [
        {"type": "text", "text": "STABLE"},
        {"type": "text", "text": "VOLATILE"},
    ]
    assert flatten_message_content(seen[0]["system"]) == "STABLEVOLATILE"
    assert seen[0]["session_id"] == writer_session_id("creator-1", "fan-1")
    assert seen[0]["end_user_id"] == writer_end_user_id("creator-1", "fan-1")
