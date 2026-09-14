"""Improvised creator facts: allowed under V3, persisted only once sent.

The audit these tests encode: ``creators.legend`` was already the canonical
store of creator self-facts, already loaded into every prompt, and already had a
first-established-wins merge (``db/queries.update_creator_legend``). What it did
not have was a path from Full Auto — ``_update_fan_memory`` runs on the Assisted
path only — and its extraction was aimed at identity rather than at "her
favourite colour is dark green".

``services/creator_canon.py`` closes exactly that gap, through the existing
store. So these tests check three things and no more: that an ordinary fact
survives, that a protected one cannot be established this way, and that nothing
is written for a message that was never sent.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from ai.stack_profiles import CLEO_V2, CLEO_V3
from ai.writer_style import (
    MODE_ASSISTED,
    MODE_AUTO,
    WRITER_V2,
    WRITER_V3,
    persists_improvised_facts,
)
from models.model_runtime import ModelResult, ModelUsage
from models.schemas import (
    ConversationContext,
    Fan,
    Message,
    Persona,
    StageType,
)
from services import creator_canon


def _run(coro):
    return asyncio.run(coro)


def context(**overrides) -> ConversationContext:
    base = dict(
        fan_message="what's your favorite color?",
        conversation_history=[Message(role="fan", content="what's your favorite color?")],
        fan_profile=Fan(id="fan-1", display_name="Marcus"),
        creator_persona=Persona(character="Warm, dry humour."),
        similar_exchanges=[],
        conversation_stage=StageType.WARMING_UP,
        creator_name="Sophia",
    )
    base.update(overrides)
    return ConversationContext(**base)


def build(version: str, mode: str = MODE_AUTO, **overrides) -> str:
    from ai.prompt_builder import build_prompt

    prompt = build_prompt(
        context(**overrides), prompt_version=version, reply_mode=mode
    )
    system = prompt[0]["content"]
    if isinstance(system, list):
        system = "\n".join(str(block.get("text", "")) for block in system)
    return f"{system}\n{prompt[1]['content']}"


class FakeLegendStore:
    """The creators.legend column, and nothing else."""

    def __init__(self, legend: dict | None = None) -> None:
        self.legend = dict(legend or {})
        self.writes: list[dict] = []

    async def get(self, _creator_id: str) -> dict:
        return dict(self.legend)

    async def update(self, creator_id: str, new_facts: dict) -> dict:
        # The real merge, so first-wins is exercised rather than restated.
        from db import queries

        merged = dict(self.legend)
        for key in queries._LEGEND_STABLE_KEYS:
            incoming = (new_facts.get(key) or "").strip()
            if incoming and not (merged.get(key) or "").strip():
                merged[key] = incoming
        other = list(merged.get("other") or [])
        seen = {str(item).strip().lower() for item in other}
        for item in new_facts.get("other") or []:
            text = str(item or "").strip()
            if text and text.lower() not in seen:
                other.append(text)
                seen.add(text.lower())
        merged["other"] = other[:20]
        self.legend = merged
        self.writes.append(dict(new_facts))
        return dict(merged)


@pytest.fixture
def canon_world(monkeypatch):
    """A legend store and a writer-facing extraction model that says what we tell it."""
    store = FakeLegendStore()
    extracted: dict[str, str] = {"text": '{"facts": []}'}
    calls: list[str] = []

    async def fake_complete(target, *, system, messages, **_kwargs):
        calls.append(messages[0]["content"])
        return ModelResult(
            text=extracted["text"],
            target=target,
            usage=ModelUsage(input_tokens=10, output_tokens=10),
            latency_ms=1,
        )

    async def fake_record(*_a, **_k):
        return None

    monkeypatch.setattr(creator_canon, "complete", fake_complete)
    monkeypatch.setattr(creator_canon, "record_model_result", fake_record)
    monkeypatch.setattr(creator_canon, "get_creator_legend", store.get)
    monkeypatch.setattr(creator_canon, "update_creator_legend", store.update)
    return store, extracted, calls


# --- 1. an unspecified harmless fact may be improvised ----------------------


def test_v3_lets_the_writer_improvise_an_unestablished_ordinary_detail():
    prompt = build(WRITER_V3, creator_legend={})
    assert "improvise a plausible one that fits your persona" in prompt
    assert "favourite colour" in prompt
    # And explicitly refuses the non-answer this exists to prevent.
    assert "Answering \"I don't have one\" merely because nobody wrote it down" in prompt


def test_v2_forbids_it_and_stays_that_way():
    prompt = build(WRITER_V2)
    assert "DO NOT INVENT YOUR CURRENT LIFE" in prompt
    assert "improvise" not in prompt.lower()


def test_v3_still_refuses_to_improvise_identity():
    prompt = build(WRITER_V3)
    assert (
        "Your name, age, where you are from, where you live, what you do, your "
        "background, and whether you can meet in person are identity facts"
        in prompt
    )


# --- 2. a fact that was actually sent is persisted --------------------------


def test_a_sent_improvised_fact_becomes_canon(canon_world):
    store, extracted, _calls = canon_world
    extracted["text"] = '{"facts": [{"topic": "favorite color", "value": "dark green"}]}'

    added = _run(
        creator_canon.persist_sent_creator_facts(
            creator_id="creator-1",
            sent_reply="probably dark green",
            fan_message="what's your favorite color?",
            conversation_history=[
                Message(role="fan", content="what's your favorite color?")
            ],
            profile_id=CLEO_V3.profile_id,
        )
    )

    assert added == ["favorite color: dark green"]
    assert store.legend["other"] == ["favorite color: dark green"]


def test_the_extractor_is_only_shown_the_message_that_was_sent(canon_world):
    _store, extracted, calls = canon_world
    extracted["text"] = '{"facts": []}'

    _run(
        creator_canon.persist_sent_creator_facts(
            creator_id="creator-1",
            sent_reply="probably dark green",
            fan_message="what's your favorite color?",
            profile_id=CLEO_V3.profile_id,
        )
    )

    assert len(calls) == 1
    assert "THE MESSAGE THE CREATOR JUST SENT (the only source of facts)" in calls[0]
    assert calls[0].strip().endswith("probably dark green")


def test_an_ordinary_turn_costs_no_extraction_call(canon_world):
    _store, _extracted, calls = canon_world

    added = _run(
        creator_canon.persist_sent_creator_facts(
            creator_id="creator-1",
            sent_reply="yeah I get that",
            fan_message="work was rough today",
            profile_id=CLEO_V3.profile_id,
        )
    )

    assert added == []
    assert calls == []


# --- 3. the next prompt carries it ------------------------------------------


def test_the_next_prompt_states_the_persisted_fact_as_canon():
    prompt = build(
        WRITER_V3,
        creator_legend={"name": "Sophia", "other": ["favorite color: dark green"]},
        fan_message="what was your favorite color again?",
    )
    assert "FACTS YOU'VE ALREADY ESTABLISHED ABOUT YOURSELF" in prompt
    assert "favorite color: dark green" in prompt
    assert "never contradict these" in prompt


# --- 4. a later reply cannot quietly replace it -----------------------------


def test_a_second_answer_on_the_same_topic_is_dropped(canon_world):
    store, extracted, _calls = canon_world
    store.legend = {"other": ["favorite color: dark green"]}
    extracted["text"] = '{"facts": [{"topic": "favorite color", "value": "blue"}]}'

    added = _run(
        creator_canon.persist_sent_creator_facts(
            creator_id="creator-1",
            sent_reply="i love blue",
            fan_message="whats your favorite color?",
            profile_id=CLEO_V3.profile_id,
        )
    )

    assert added == []
    assert store.legend["other"] == ["favorite color: dark green"]


def test_a_different_topic_is_still_added(canon_world):
    store, extracted, _calls = canon_world
    store.legend = {"other": ["favorite color: dark green"]}
    extracted["text"] = '{"facts": [{"topic": "coffee", "value": "oat flat white"}]}'

    added = _run(
        creator_canon.persist_sent_creator_facts(
            creator_id="creator-1",
            sent_reply="i love an oat flat white",
            fan_message="are you a coffee person?",
            profile_id=CLEO_V3.profile_id,
        )
    )

    assert added == ["coffee: oat flat white"]
    assert store.legend["other"] == [
        "favorite color: dark green",
        "coffee: oat flat white",
    ]


# --- 5. protected facts are not improvisable --------------------------------


@pytest.mark.parametrize(
    "topic,value",
    (
        ("name", "Ariana"),
        ("age", "22"),
        ("where she lives", "Berlin"),
        ("nationality", "Czech"),
        ("real job", "dental nurse"),
        ("meeting up", "she would fly out"),
        ("subscription price", "$15"),
    ),
)
def test_an_identity_or_platform_fact_is_never_established_this_way(
    topic, value, canon_world
):
    store, extracted, _calls = canon_world
    extracted["text"] = (
        '{"facts": [{"topic": "%s", "value": "%s"}]}' % (topic, value)
    )

    added = _run(
        creator_canon.persist_sent_creator_facts(
            creator_id="creator-1",
            sent_reply=f"i love that, {value}",
            fan_message="tell me about you?",
            profile_id=CLEO_V3.profile_id,
        )
    )

    assert added == []
    assert store.legend.get("other") in (None, [])


def test_a_configured_protected_fact_survives_an_attempt_to_change_it(canon_world):
    store, extracted, _calls = canon_world
    store.legend = {"name": "Sophia", "age": "24", "origin": "Prague"}
    extracted["text"] = (
        '{"facts": [{"topic": "name", "value": "Ariana"},'
        ' {"topic": "favorite food", "value": "ramen"}]}'
    )

    added = _run(
        creator_canon.persist_sent_creator_facts(
            creator_id="creator-1",
            sent_reply="i love ramen honestly",
            fan_message="whats your favorite food?",
            profile_id=CLEO_V3.profile_id,
        )
    )

    assert added == ["favorite food: ramen"]
    assert store.legend["name"] == "Sophia"
    assert store.legend["age"] == "24"
    assert store.legend["origin"] == "Prague"
    # The merge is never even offered a protected key.
    assert set(store.writes[-1]) == {"other"}


def test_the_protected_key_list_is_the_legend_s_own(canon_world):
    from db.queries import _LEGEND_STABLE_KEYS

    assert creator_canon.PROTECTED_LEGEND_KEYS == _LEGEND_STABLE_KEYS


# --- 6. an unused Assisted suggestion establishes nothing -------------------


def test_only_a_sent_message_can_establish_canon():
    """The capture point is the send path, never generation.

    ``get_suggestions`` produces candidates and returns them; nothing in it
    touches the legend. The write happens in ``/reply`` (assisted) and after
    delivery in ``_debounced_auto_reply`` (auto), so the two candidates the
    operator did not pick never reach it.
    """
    import inspect

    from services import suggestions

    source = inspect.getsource(suggestions.get_suggestions)
    assert "persist_sent_creator_facts" not in source

    auto_source = inspect.getsource(suggestions._debounced_auto_reply)
    # In the Auto path it sits after the delivery loop, so an aborted turn
    # cannot reach it.
    assert auto_source.index("persist_sent_creator_facts") > auto_source.index(
        "[AUTO REPLY] Sent part"
    )


def test_an_unsent_candidate_is_not_offered_to_the_canon_writer(canon_world):
    """Belt and braces: the function only ever sees one string, the sent one."""
    _store, extracted, calls = canon_world
    extracted["text"] = '{"facts": [{"topic": "favorite color", "value": "dark green"}]}'

    _run(
        creator_canon.persist_sent_creator_facts(
            creator_id="creator-1",
            sent_reply="probably dark green",
            fan_message="whats your favorite color?",
            profile_id=CLEO_V3.profile_id,
        )
    )

    assert "an alternative phrasing" not in calls[0]
    assert calls[0].count("probably dark green") == 1


# --- which profiles do this at all ------------------------------------------


def test_only_v3_writes_improvised_facts_back():
    assert persists_improvised_facts(CLEO_V3.writer_prompt_version()) is True
    assert persists_improvised_facts(CLEO_V2.writer_prompt_version()) is False


def test_a_failing_extraction_never_breaks_the_turn(monkeypatch):
    async def exploding(*_a, **_k):
        raise RuntimeError("provider down")

    monkeypatch.setattr(creator_canon, "complete", exploding)

    assert (
        _run(
            creator_canon.persist_sent_creator_facts(
                creator_id="creator-1",
                sent_reply="my favorite color is dark green",
                fan_message="favorite color?",
                profile_id=CLEO_V3.profile_id,
            )
        )
        == []
    )


def test_the_gate_recognises_both_a_personal_question_and_a_stated_preference():
    assert creator_canon.mentions_self_fact(
        reply="probably dark green", fan_message="whats your favorite color?"
    )
    assert creator_canon.mentions_self_fact(
        reply="i love thunderstorms", fan_message="weather is wild today"
    )
    # A question that is not about her tastes costs nothing.
    assert not creator_canon.mentions_self_fact(
        reply="yeah exactly", fan_message="did you see the game?"
    )
    assert not creator_canon.mentions_self_fact(
        reply="pretty good", fan_message="hows your day?"
    )
    assert not creator_canon.mentions_self_fact(reply="", fan_message="favorite color?")


def test_assisted_mode_is_unaffected_by_any_of_this():
    """Nothing here changes what the operator is shown."""
    prompt = build(WRITER_V3, MODE_ASSISTED)
    assert "Write 3 reply options" in prompt


def test_the_extraction_reuses_the_profile_s_own_extractor(monkeypatch):
    from ai.stack_profiles import STAGE_FAN_INTELLIGENCE

    seen: dict = {}

    async def fake_complete(target, *, system, messages, **kwargs):
        seen["target"] = target
        seen["max_tokens"] = kwargs.get("max_tokens")
        return ModelResult(
            text='{"facts": []}',
            target=target,
            usage=ModelUsage(input_tokens=1, output_tokens=1),
            latency_ms=1,
        )

    async def fake_record(*_a, **_k):
        return None

    monkeypatch.setattr(creator_canon, "complete", fake_complete)
    monkeypatch.setattr(creator_canon, "record_model_result", fake_record)
    monkeypatch.setattr(
        creator_canon, "get_creator_legend", lambda _c: _async_value({})
    )

    _run(
        creator_canon.persist_sent_creator_facts(
            creator_id="creator-1",
            sent_reply="my favorite color is dark green",
            fan_message="favorite color?",
            profile_id=CLEO_V3.profile_id,
        )
    )

    spec = CLEO_V3.stage(STAGE_FAN_INTELLIGENCE)
    assert (seen["target"].provider, seen["target"].model) == spec.resolved_primary()
    assert seen["max_tokens"] == spec.resolved_max_tokens()


async def _async_value(value):
    return value


def test_a_non_dict_extraction_payload_is_ignored(canon_world):
    store, extracted, _calls = canon_world
    extracted["text"] = "not json at all"

    assert (
        _run(
            creator_canon.persist_sent_creator_facts(
                creator_id="creator-1",
                sent_reply="my favorite color is dark green",
                fan_message="favorite color?",
                profile_id=CLEO_V3.profile_id,
            )
        )
        == []
    )
    assert store.writes == []


def test_a_fact_with_no_value_is_not_recorded(canon_world):
    store, extracted, _calls = canon_world
    extracted["text"] = '{"facts": [{"topic": "favorite color", "value": "   "}]}'

    assert (
        _run(
            creator_canon.persist_sent_creator_facts(
                creator_id="creator-1",
                sent_reply="my favorite color is hard to pick",
                fan_message="favorite color?",
                profile_id=CLEO_V3.profile_id,
            )
        )
        == []
    )
    assert store.writes == []


def test_unused_simple_namespace_history_is_tolerated(canon_world):
    """History arrives as ORM-ish objects on one path and dicts on another."""
    _store, extracted, calls = canon_world
    extracted["text"] = '{"facts": []}'

    _run(
        creator_canon.persist_sent_creator_facts(
            creator_id="creator-1",
            sent_reply="my favorite color is dark green",
            fan_message="favorite color?",
            conversation_history=[
                {"role": "fan", "content": "hey"},
                SimpleNamespace(role="creator", content="hi you"),
            ],
            profile_id=CLEO_V3.profile_id,
        )
    )

    assert "Fan: hey" in calls[0]
    assert "Creator: hi you" in calls[0]
