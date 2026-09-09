"""COST-002a — a cache directive must survive all the way to the provider.

The audit proved that ``build_prompt`` produced ``cache_control`` blocks and
``generate_replies`` flattened them into a plain string before the transport, so
Anthropic prompt caching was inert for the entire deployment. The blocks now
travel intact to Anthropic and are joined only for OpenAI-compatible providers,
which use implicit prefix caching and would reject an unknown marker.

These tests assert both halves at the transport, because that is the layer the
defect lived one call above.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from ai import model_providers
from ai.prompt_blocks import (
    MIN_CACHEABLE_CHARS,
    cacheable_system_blocks,
    flatten_message_content,
    has_cache_control,
)
from ai.situation_analyzer import ANALYZER_SYSTEM, build_analyzer_prompt
from models.model_runtime import ModelTarget

from tests.test_prompt_cache_structure import _ctx


def _anthropic_target() -> ModelTarget:
    return ModelTarget(
        name="anthropic:claude-haiku-4-5-20251001",
        provider="anthropic",
        model="claude-haiku-4-5-20251001",
        api_key_env="ANTHROPIC_API_KEY",
    )


def _openrouter_target() -> ModelTarget:
    return ModelTarget(
        name="openrouter:moonshotai/kimi-k2.6",
        provider="openrouter",
        model="moonshotai/kimi-k2.6",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
    )


class _RecordingAnthropic:
    def __init__(self) -> None:
        self.kwargs: dict = {}

        outer = self

        class _Messages:
            async def create(self, **kwargs):
                outer.kwargs = kwargs
                return SimpleNamespace(
                    id="msg-1",
                    content=[SimpleNamespace(type="text", text="{}")],
                    usage=SimpleNamespace(
                        input_tokens=10,
                        output_tokens=2,
                        cache_read_input_tokens=1500,
                        cache_creation_input_tokens=0,
                    ),
                )

        self.messages = _Messages()


class _RecordingOpenAI:
    def __init__(self) -> None:
        self.kwargs: dict = {}

        outer = self

        class _Completions:
            async def create(self, **kwargs):
                outer.kwargs = kwargs
                return SimpleNamespace(
                    id="cmp-1",
                    choices=[
                        SimpleNamespace(message=SimpleNamespace(content="{}"))
                    ],
                    usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2),
                )

        self.chat = SimpleNamespace(completions=_Completions())


LONG_STABLE = "STABLE RULES. " * 400  # comfortably past MIN_CACHEABLE_CHARS


def test_cache_control_reaches_the_anthropic_transport(monkeypatch):
    client = _RecordingAnthropic()
    monkeypatch.setattr(model_providers, "_anthropic_client", lambda key: client)
    monkeypatch.setattr(model_providers, "_api_key", lambda target: "key")

    blocks = cacheable_system_blocks(LONG_STABLE, "VOLATILE TAIL")
    asyncio.run(
        model_providers.complete(
            _anthropic_target(),
            system=blocks,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=10,
        )
    )

    sent = client.kwargs["system"]
    assert isinstance(sent, list)
    assert sent[0]["cache_control"] == {"type": "ephemeral"}
    assert sent[0]["text"] == LONG_STABLE
    # The volatile tail stays behind the cached prefix so it cannot evict it.
    assert sent[1]["text"] == "VOLATILE TAIL"
    assert "cache_control" not in sent[1]


def test_no_cache_marker_reaches_an_openai_compatible_transport(monkeypatch):
    client = _RecordingOpenAI()
    monkeypatch.setattr(
        model_providers,
        "_openai_compatible_client",
        lambda base_url, key, timeout: client,
    )
    monkeypatch.setattr(model_providers, "_api_key", lambda target: "key")

    asyncio.run(
        model_providers.complete(
            _openrouter_target(),
            system=cacheable_system_blocks(LONG_STABLE, "VOLATILE TAIL"),
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=10,
        )
    )

    system_message = client.kwargs["messages"][0]
    assert system_message["role"] == "system"
    assert isinstance(system_message["content"], str)
    assert system_message["content"] == LONG_STABLE + "VOLATILE TAIL"
    assert "cache_control" not in system_message["content"]
    assert "ephemeral" not in system_message["content"]


def test_a_plain_string_system_still_works_on_both_transports(monkeypatch):
    anthropic_client = _RecordingAnthropic()
    openai_client = _RecordingOpenAI()
    monkeypatch.setattr(
        model_providers, "_anthropic_client", lambda key: anthropic_client
    )
    monkeypatch.setattr(
        model_providers,
        "_openai_compatible_client",
        lambda base_url, key, timeout: openai_client,
    )
    monkeypatch.setattr(model_providers, "_api_key", lambda target: "key")

    for target in (_anthropic_target(), _openrouter_target()):
        asyncio.run(
            model_providers.complete(
                target,
                system="plain instructions",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=10,
            )
        )

    assert anthropic_client.kwargs["system"] == "plain instructions"
    assert openai_client.kwargs["messages"][0]["content"] == "plain instructions"


def test_anthropic_cache_read_tokens_are_still_recorded(monkeypatch):
    client = _RecordingAnthropic()
    monkeypatch.setattr(model_providers, "_anthropic_client", lambda key: client)
    monkeypatch.setattr(model_providers, "_api_key", lambda target: "key")

    result = asyncio.run(
        model_providers.complete(
            _anthropic_target(),
            system=cacheable_system_blocks(LONG_STABLE),
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=10,
        )
    )

    assert result.usage.cache_read_tokens == 1500


def test_a_block_too_short_to_cache_is_not_marked():
    short = "tiny system prompt"
    assert len(short) < MIN_CACHEABLE_CHARS
    blocks = cacheable_system_blocks(short)

    assert not has_cache_control(blocks)
    assert flatten_message_content(blocks) == short


@pytest.mark.parametrize("volatile", ["", "TAIL"])
def test_flattening_preserves_block_order(volatile):
    blocks = cacheable_system_blocks(LONG_STABLE, volatile)
    assert flatten_message_content(blocks) == LONG_STABLE + volatile


# --- COST-002b: the analyzer's static half must be a prefix, not a suffix ----


def test_analyzer_static_rules_live_in_the_system_block():
    system, user = build_analyzer_prompt(
        _ctx(fan_message="can we do the $28 one", history=[])
    )

    for marker in (
        "COMMERCIAL INTERPRETATION RULES:",
        "SAFETY:",
        '"desired_experience"',
        "Return ONLY valid JSON",
    ):
        assert marker in system, marker
        assert marker not in user, marker


def test_analyzer_system_block_is_identical_across_unrelated_conversations():
    first, _ = build_analyzer_prompt(_ctx(fan_message="hey", history=[]))
    second, _ = build_analyzer_prompt(
        _ctx(fan_message="totally different question about pricing", history=[])
    )

    assert first == second == ANALYZER_SYSTEM


def test_analyzer_user_turn_carries_only_the_volatile_conversation():
    _, user = build_analyzer_prompt(
        _ctx(fan_message="what are you up to", history=[])
    )

    assert user.startswith("Conversation so far:")
    assert user.endswith('Latest fan message: "what are you up to"')
    assert len(user) < 200
