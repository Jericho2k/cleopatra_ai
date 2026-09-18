"""Assisted under cleo_v3: still a list of three, still nothing persisted.

Auto output cardinality and Assisted output cardinality are separate questions.
V3 answers them differently — one reply where one is sent, three where an
operator chooses — and the risk of that change is that "Full Auto makes one
candidate" quietly becomes "the writer makes one candidate", which would leave
the operator UI with a one-item list.

These drive the real ``get_suggestions`` with the world faked around it.
"""
from __future__ import annotations

import asyncio

import pytest

from models.schemas import Fan, Message, Persona
from services import suggestions
from services.ai_stack import clear_ai_stack_cache


def _run(coro):
    return asyncio.run(coro)


async def _value(value):
    return value


@pytest.fixture
def assisted_world(monkeypatch):
    """Everything ``get_suggestions`` reads, faked. Returns (configure, calls)."""
    calls: dict[str, list] = {"writer": [], "legend_writes": []}
    replies = ["first option", "second option", "third option"]

    from services.conversation_core import (
        CORE_LEGACY,
        SOURCE_BUILTIN,
        ConversationCoreResolution,
    )

    monkeypatch.setattr(
        "services.conversation_core.resolve_conversation_core",
        lambda **_kwargs: _value(
            ConversationCoreResolution(CORE_LEGACY, SOURCE_BUILTIN)
        ),
    )

    async def fake_generate(prompt, persona, **kwargs):
        calls["writer"].append({"prompt": prompt, **kwargs})
        return list(replies)

    async def fake_update_legend(creator_id, facts):
        calls["legend_writes"].append((creator_id, facts))
        return {}

    history = [
        Message(role="fan", content="hey"),
        Message(role="creator", content="hey you"),
    ]

    monkeypatch.setattr(suggestions, "generate_replies", fake_generate)
    monkeypatch.setattr(suggestions, "update_creator_legend", fake_update_legend)
    monkeypatch.setattr(suggestions, "get_conversation_history", lambda _f: _value(history))
    monkeypatch.setattr(
        suggestions,
        "get_fan_by_id",
        lambda _f: _value(Fan(id="fan-1", display_name="Marcus")),
    )
    monkeypatch.setattr(suggestions, "get_fan_intelligence_context", lambda _f: _value({}))
    monkeypatch.setattr(
        suggestions, "get_fan_lifecycle_context", lambda _f: _value({"stage": "PROSPECT"})
    )
    monkeypatch.setattr(suggestions, "get_affordability_context", lambda _f: _value({}))
    monkeypatch.setattr(
        suggestions, "get_price_learning_context", lambda _f: _value({"mode": "learning"})
    )
    monkeypatch.setattr(
        suggestions, "get_creator_persona", lambda _c: _value(Persona(character="Warm."))
    )
    monkeypatch.setattr(
        suggestions,
        "get_creator_legend",
        lambda _c: _value({"name": "Sophia", "other": ["favorite color: dark green"]}),
    )
    monkeypatch.setattr(suggestions, "get_ppv_offers", lambda _c: _value([]))
    monkeypatch.setattr(suggestions, "get_sent_ppv", lambda _f: _value([]))
    monkeypatch.setattr(suggestions, "get_fan_session", lambda _f: _value(None))
    monkeypatch.setattr(suggestions, "find_similar_exchanges", lambda *_a, **_k: _value([]))
    monkeypatch.setattr(suggestions, "get_approved_asset_types", lambda _c: _value(()))
    monkeypatch.setattr(
        suggestions,
        "analyze_situation",
        lambda *_a, **_k: _value(
            {
                "fan_mood": "curious",
                "purchase_signal": "none",
                "crisis_signal": "none",
                "strategic_move": "build connection",
            }
        ),
    )
    monkeypatch.setattr(
        suggestions, "refresh_affordability_from_situation", lambda **_k: _value({})
    )
    monkeypatch.setattr(
        suggestions, "refresh_fan_lifecycle", lambda **_k: _value({"stage": "PROSPECT"})
    )
    monkeypatch.setattr(
        suggestions, "refresh_price_learning", lambda **_k: _value({"mode": "learning"})
    )
    monkeypatch.setattr(
        suggestions,
        "direct_conversation",
        lambda **_k: _value({"phase": "OPENING", "action": "RESPOND_AND_OPEN"}),
    )
    monkeypatch.setattr(
        suggestions,
        "plan_next_action",
        lambda **_k: _value({"goal": "RAPPORT", "next_action": "CONTINUE_CHAT"}),
    )
    clear_ai_stack_cache()

    def configure(profile_id: str):
        monkeypatch.setenv("AI_STACK_PROFILE", profile_id)
        clear_ai_stack_cache()

    yield configure, calls
    clear_ai_stack_cache()


def ask() -> object:
    return _run(
        suggestions.get_suggestions(
            fan_id="fan-1",
            creator_id="creator-1",
            fan_message="what's your favorite color?",
            save_fan_message=False,
        )
    )


def writer_prompt(calls) -> str:
    prompt = calls["writer"][-1]["prompt"]
    system = prompt[0]["content"]
    if isinstance(system, list):
        system = "".join(str(block.get("text", "")) for block in system)
    return f"{system}\n{prompt[1]['content']}"


@pytest.mark.parametrize("profile_id", ("cleo_legacy_v1", "cleo_v2", "cleo_v3"))
def test_assisted_always_offers_the_operator_three_options(
    profile_id, assisted_world
):
    configure, calls = assisted_world
    configure(profile_id)

    result = ask()

    assert len(result.suggestions) == 3
    assert calls["writer"][-1]["max_candidates"] == 3
    prompt = writer_prompt(calls)
    assert "Write 3 reply options" in prompt
    assert "Return ONLY a JSON array of 3 strings" in prompt
    assert "Write ONE reply." not in prompt


def test_assisted_under_v3_uses_the_v3_voice(assisted_world):
    """Same profile, same voice — only the cardinality differs from Auto."""
    configure, calls = assisted_world
    configure("cleo_v3")
    ask()

    prompt = writer_prompt(calls)
    assert "Do not adapt to the mechanics of how he types" in prompt
    assert "favorite creator" not in prompt
    assert "improvise a plausible one that fits your persona" in prompt
    # And the canon it already has is in front of the writer.
    assert "favorite color: dark green" in prompt


def test_generating_suggestions_never_writes_to_the_creator_legend(assisted_world):
    """Two of these three candidates will never be sent."""
    configure, calls = assisted_world
    configure("cleo_v3")

    result = ask()

    assert len(result.suggestions) == 3
    assert calls["legend_writes"] == []
