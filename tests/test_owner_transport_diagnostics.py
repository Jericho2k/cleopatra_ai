"""What the transport must be able to say about a reasoning-model response.

The owner used to reduce an entire OpenAI-compatible response to
``response.choices[0].message.content or ""``. Every distinct failure — the
completion budget consumed by hidden reasoning, a content filter, a 200 carrying
an ``error`` object, an upstream that ignored ``response_format`` — arrived
identically as an empty string, and the only thing anyone could say afterwards
was "the response contained no JSON object".

These tests pin the structural record that replaced it, and pin that it stays
free of conversation text.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from ai import model_providers, openrouter_routing
from ai.model_providers import complete
from ai.stack_profiles import CLEO_V2, CLEO_V3, STAGE_CONVERSATIONAL_OWNER
from models.model_runtime import (
    FAILURE_CONTENT_FILTERED,
    FAILURE_EMPTY_TRUNCATED,
    FAILURE_EMPTY_UNEXPLAINED,
    FAILURE_PROVIDER_ERROR,
    FAILURE_REFUSAL,
    ModelResponseDiagnostics,
    ModelTarget,
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in (
        "OPENROUTER_PROVIDERS",
        "OPENROUTER_ALLOW_FALLBACKS",
        "OPENROUTER_DATA_COLLECTION",
        "OPENROUTER_ZDR",
        "OPENROUTER_REQUIRE_PARAMETERS",
        "CONVERSATIONAL_OWNER_MAX_TOKENS",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")


class _Client:
    def __init__(self, response):
        self.calls: list[dict] = []

        async def create(**kwargs):
            self.calls.append(kwargs)
            if isinstance(response, Exception):
                raise response
            return response

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))


def _install(monkeypatch, response):
    client = _Client(response)
    monkeypatch.setattr(
        model_providers,
        "_openai_compatible_client",
        lambda *_a, **_k: client,
    )
    return client


def _owner_target(**metadata):
    return ModelTarget(
        name="openrouter:z-ai/glm-5.3-flash",
        provider="openrouter",
        model="z-ai/glm-5.3-flash",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        metadata={
            "reasoning_enabled": True,
            "reasoning_effort": "low",
            **metadata,
        },
    )


def _response(
    *,
    content,
    finish_reason="stop",
    reasoning=None,
    refusal=None,
    reasoning_tokens=0,
    completion_tokens=40,
    error=None,
    native_finish_reason=None,
):
    message = SimpleNamespace(
        content=content,
        reasoning=reasoning,
        refusal=refusal,
        tool_calls=None,
    )
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    if native_finish_reason is not None:
        choice.native_finish_reason = native_finish_reason
    usage = SimpleNamespace(
        prompt_tokens=2_000,
        completion_tokens=completion_tokens,
        prompt_tokens_details=SimpleNamespace(cached_tokens=0),
        completion_tokens_details=SimpleNamespace(reasoning_tokens=reasoning_tokens),
        cost=0.0002,
    )
    response = SimpleNamespace(
        id="gen-abc", choices=[choice], usage=usage, provider="z-ai"
    )
    if error is not None:
        response.error = error
    return response


def _call(target, **kwargs):
    return asyncio.run(
        complete(
            target,
            system="prefix",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=kwargs.pop("max_tokens", 8_192),
            response_format={"type": "json_object"},
            **kwargs,
        )
    )


# ---------------------------------------------------------------------------
# The production failure, now explainable
# ---------------------------------------------------------------------------


def test_a_completion_budget_spent_on_reasoning_is_named_as_such(monkeypatch):
    _install(
        monkeypatch,
        _response(
            content=None,
            finish_reason="length",
            reasoning="x" * 24_000,
            reasoning_tokens=8_100,
            completion_tokens=8_192,
        ),
    )
    result = _call(_owner_target())
    diagnostics = result.diagnostics

    assert result.text == ""
    assert diagnostics.content_is_null is True
    assert diagnostics.content_chars == 0
    assert diagnostics.finish_reason == "length"
    assert diagnostics.truncated is True
    assert diagnostics.reasoning_present is True
    assert diagnostics.reasoning_chars == 24_000
    assert diagnostics.reasoning_tokens == 8_100
    assert diagnostics.completion_tokens == 8_192
    assert diagnostics.max_tokens_requested == 8_192
    assert diagnostics.response_format_requested == "json_object"
    assert diagnostics.reasoning_requested == "on,effort=low"
    assert diagnostics.response_id == "gen-abc"
    assert diagnostics.upstream_provider == "z-ai"
    assert diagnostics.message_fields == ("reasoning",)
    assert diagnostics.empty_content_category() == FAILURE_EMPTY_TRUNCATED


def test_an_empty_completion_with_no_reason_is_not_reported_as_truncation(monkeypatch):
    _install(
        monkeypatch,
        _response(content="", finish_reason="stop", completion_tokens=0),
    )
    diagnostics = _call(_owner_target()).diagnostics
    assert diagnostics.empty_content_category() == FAILURE_EMPTY_UNEXPLAINED


def test_a_content_filter_is_not_reported_as_a_parse_failure(monkeypatch):
    _install(monkeypatch, _response(content=None, finish_reason="content_filter"))
    diagnostics = _call(_owner_target()).diagnostics
    assert diagnostics.empty_content_category() == FAILURE_CONTENT_FILTERED


def test_a_refusal_is_visible_as_a_refusal(monkeypatch):
    _install(
        monkeypatch,
        _response(content=None, finish_reason="stop", refusal="I can't help with that"),
    )
    diagnostics = _call(_owner_target()).diagnostics
    assert diagnostics.refusal_present is True
    assert diagnostics.empty_content_category() == FAILURE_REFUSAL
    assert "refusal" in diagnostics.message_fields


def test_a_two_hundred_carrying_an_error_object_is_a_provider_error(monkeypatch):
    _install(
        monkeypatch,
        _response(
            content=None,
            error={"message": "No allowed providers are available for the selected model."},
        ),
    )
    diagnostics = _call(_owner_target()).diagnostics
    assert diagnostics.error_category == FAILURE_PROVIDER_ERROR
    assert "No allowed providers" in diagnostics.provider_error
    assert diagnostics.empty_content_category() == FAILURE_PROVIDER_ERROR


def test_a_native_finish_reason_also_counts_as_truncation(monkeypatch):
    _install(
        monkeypatch,
        _response(content="", finish_reason="", native_finish_reason="MAX_LENGTH"),
    )
    diagnostics = _call(_owner_target()).diagnostics
    assert diagnostics.truncated is True


def test_a_healthy_response_reports_no_failure(monkeypatch):
    _install(
        monkeypatch,
        _response(content='{"reply":"hello"}', reasoning="short", reasoning_tokens=40),
    )
    result = _call(_owner_target())
    assert result.text == '{"reply":"hello"}'
    assert result.diagnostics.empty_content_category() == ""
    assert result.diagnostics.content_chars == 17
    assert result.diagnostics.message_fields == ("content", "reasoning")
    assert result.diagnostics.latency_ms >= 0


def test_diagnostics_carry_no_conversation_text(monkeypatch):
    secret = "the fan said something private about their divorce"
    _install(
        monkeypatch,
        _response(
            content=json.dumps({"reply": secret}),
            reasoning=f"the fan is upset because {secret}",
            reasoning_tokens=120,
        ),
    )
    diagnostics = _call(_owner_target()).diagnostics
    rendered = json.dumps(diagnostics.as_dict()) + diagnostics.describe()
    assert "divorce" not in rendered
    assert secret not in rendered
    assert diagnostics.reasoning_chars > 0


# ---------------------------------------------------------------------------
# What the request now asks for
# ---------------------------------------------------------------------------


def test_reasoning_is_capped_so_the_answer_always_has_budget(monkeypatch):
    client = _install(monkeypatch, _response(content='{"reply":"x"}'))
    _call(_owner_target(reasoning_max_tokens=1_024), max_tokens=8_192)
    reasoning = client.calls[0]["extra_body"]["reasoning"]
    # effort and max_tokens are ALTERNATIVES in OpenRouter's reasoning object:
    # sending both risks a 400 on every owner call, which would be worse than
    # the failure this cap exists to prevent.
    assert reasoning == {"enabled": True, "max_tokens": 1_024}
    assert "effort" not in reasoning


def test_a_reasoning_cap_can_never_exceed_the_total_budget(monkeypatch):
    client = _install(monkeypatch, _response(content='{"reply":"x"}'))
    _call(_owner_target(reasoning_max_tokens=4_000), max_tokens=1_000)
    assert client.calls[0]["extra_body"]["reasoning"]["max_tokens"] == 744


def test_an_unparseable_reasoning_cap_falls_back_to_effort(monkeypatch):
    client = _install(monkeypatch, _response(content='{"reply":"x"}'))
    _call(_owner_target(reasoning_max_tokens="not a number"))
    assert client.calls[0]["extra_body"]["reasoning"] == {
        "enabled": True,
        "effort": "low",
    }


def test_an_uncapped_reasoning_target_keeps_the_previous_request_shape(monkeypatch):
    client = _install(monkeypatch, _response(content='{"reply":"x"}'))
    _call(_owner_target())
    assert client.calls[0]["extra_body"]["reasoning"] == {
        "enabled": True,
        "effort": "low",
    }


def test_the_owner_route_requires_upstreams_that_honour_its_parameters(monkeypatch):
    client = _install(monkeypatch, _response(content='{"reply":"x"}'))
    _call(_owner_target(openrouter_require_parameters=True))
    assert client.calls[0]["extra_body"]["provider"]["require_parameters"] is True


def test_require_parameters_is_off_for_routes_that_did_not_ask_for_it():
    preferences = openrouter_routing.provider_preferences(
        {"openrouter_providers": ["Inceptron"]}
    )
    assert "require_parameters" not in preferences


def test_require_parameters_can_be_widened_without_a_deploy(monkeypatch):
    monkeypatch.setenv("OPENROUTER_REQUIRE_PARAMETERS", "false")
    preferences = openrouter_routing.provider_preferences(
        {"openrouter_require_parameters": True}
    )
    assert "require_parameters" not in preferences


# ---------------------------------------------------------------------------
# The shipped owner configuration
# ---------------------------------------------------------------------------


def test_the_shipped_owner_budget_leaves_room_for_an_answer():
    for profile in (CLEO_V2, CLEO_V3):
        owner = profile.stage(STAGE_CONVERSATIONAL_OWNER)
        assert owner.reasoning is True, "this model requires reasoning"
        target = owner.primary_target()
        cap = int(target.metadata.get("reasoning_max_tokens") or 0)
        assert cap > 0, "an uncapped reasoning trace can consume the whole budget"
        assert owner.resolved_max_tokens() >= cap * 4, (
            "the visible answer needs materially more budget than the trace"
        )
        assert target.metadata.get("openrouter_require_parameters") is True


def test_the_owner_budget_is_re_sizable_without_a_deploy(monkeypatch):
    monkeypatch.setenv("CONVERSATIONAL_OWNER_MAX_TOKENS", "12000")
    assert CLEO_V3.stage(STAGE_CONVERSATIONAL_OWNER).resolved_max_tokens() == 12_000


def test_the_writer_routes_were_not_disturbed():
    from ai.stack_profiles import STAGE_WRITER_COMMERCIAL, STAGE_WRITER_DEFAULT

    for stage in (STAGE_WRITER_DEFAULT, STAGE_WRITER_COMMERCIAL):
        spec = CLEO_V3.stage(stage)
        assert spec.reasoning is False
        assert "reasoning_max_tokens" not in (spec.primary_target().metadata or {})
        assert not spec.primary_target().metadata.get(
            "openrouter_require_parameters"
        )


def test_the_anthropic_route_also_explains_an_empty_completion():
    diagnostics = ModelResponseDiagnostics(
        provider="anthropic",
        model="claude-haiku-4-5-20251001",
        finish_reason="max_tokens",
        content_chars=0,
        reasoning_present=True,
        reasoning_chars=900,
        completion_tokens=1_000,
    )
    assert diagnostics.truncated is True
    assert diagnostics.empty_content_category() == FAILURE_EMPTY_TRUNCATED
