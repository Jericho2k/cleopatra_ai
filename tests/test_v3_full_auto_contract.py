"""Full Auto under cleo_v3: one reply asked for, one reply sent, no shape policy.

These drive the REAL Full Auto turn — ``services.suggestions.run_simulated_inbound``
awaits the same ``_debounced_auto_reply`` the durable AUTO_REPLY worker awaits —
with the world around it faked exactly as ``tests/test_full_auto_simulation.py``
fakes it. Writer ROUTING is deliberately NOT faked here: the point is that the
profile decides the prompt version, so a test that pins the version by hand
would be testing itself.

What each profile must do is asserted side by side, because the value of V3 is
entirely in the difference and a change that quietly moved V2 as well would make
the Simulator comparison meaningless.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from models.schemas import Fan, Persona
from services import suggestions
from services.ai_stack import clear_ai_stack_cache
from tests.test_full_auto_simulation import (  # reused world, not a second engine
    FakeDB,
    SpyTransport,
    _run,
    _value,
)

import httpx


def _creator_rows(db) -> list[dict]:
    return [row for row in db.tables["messages"] if row["role"] == "creator"]


@pytest.fixture
def spy_transport():
    transport = SpyTransport()
    client = httpx.AsyncClient(transport=transport)
    from services import apifansly

    apifansly.set_shared_client(client)
    yield transport
    apifansly.set_shared_client(None)


@pytest.fixture
def auto_world(monkeypatch):
    """A test fan on a chosen AI stack profile, with only the world faked.

    Returns ``(configure, calls)``. ``configure(profile_id, replies=...)`` sets
    the deployment profile for the turn and what the writer returns, then hands
    back the fake database.
    """
    db = FakeDB(
        {
            "messages": [
                {
                    "id": "msg-1",
                    "fan_id": "fan-test",
                    "creator_id": "creator-1",
                    "role": "fan",
                    "content": "hey there",
                    "sent_at": "2026-07-18T11:58:00+00:00",
                },
            ],
            "fans": [
                {
                    "id": "fan-test",
                    "creator_id": "creator-1",
                    "platform_fan_id": "test_jostar",
                    "display_name": "Jostar",
                    "fansly_group_id": "stale-group-99",
                    "pending_tip": None,
                    "pending_ppv_check": None,
                }
            ],
            "creators": [
                {
                    "id": "creator-1",
                    "name": "Sophia",
                    "apifansly_account_id": "acct-1",
                    "fansly_account_id": "plat-1",
                    "auto_mode": True,
                }
            ],
        }
    )

    calls: dict[str, list] = {"writer": [], "canon": []}
    replies: list[str] = ["sounds like a good day honestly"]

    async def fake_generate(prompt, persona, **kwargs):
        calls["writer"].append({"prompt": prompt, **kwargs})
        return list(replies)

    async def fake_canon(**kwargs):
        calls["canon"].append(kwargs)
        return []

    async def noop_async(*_a, **_k):
        return None

    async def fake_sleep(_seconds):
        return None

    async def fake_analyze(ctx, telemetry_context=None, **_kwargs):
        return {
            "purchase_signal": "none",
            "crisis_signal": "none",
            "resend_requested": "false",
            "strategic_move": "build_rapport",
        }

    async def fake_direct(**_kwargs):
        return {"phase": "rapport", "action": "chat", "transition_reason": "test"}

    async def fake_plan_next(**_kwargs):
        return {"goal": "rapport", "next_action": "chat"}

    monkeypatch.setattr(suggestions, "get_supabase", lambda: db)
    monkeypatch.setattr("db.queries.get_supabase", lambda: db)
    monkeypatch.setattr("services.ai_stack.get_supabase", lambda: db)
    monkeypatch.setattr(suggestions, "analyze_situation", fake_analyze)
    monkeypatch.setattr(suggestions, "generate_replies", fake_generate)
    monkeypatch.setattr(suggestions, "direct_conversation", fake_direct)
    monkeypatch.setattr(suggestions, "plan_next_action", fake_plan_next)
    monkeypatch.setattr(suggestions, "persist_sent_creator_facts", fake_canon)
    monkeypatch.setattr(suggestions, "get_creator_persona", lambda _c: _value(Persona()))
    monkeypatch.setattr(suggestions, "get_ppv_offers", lambda _c: _value([]))
    monkeypatch.setattr(suggestions, "get_sent_ppv", lambda _f: _value([]))
    monkeypatch.setattr(suggestions, "get_fan_session", lambda _f: _value(None))
    monkeypatch.setattr(suggestions, "find_similar_exchanges", lambda *_a, **_k: _value([]))
    monkeypatch.setattr(suggestions, "get_fan_intelligence_context", lambda _f: _value({}))
    monkeypatch.setattr(
        suggestions, "get_fan_lifecycle_context", lambda _f: _value({"stage": "new"})
    )
    monkeypatch.setattr(suggestions, "get_affordability_context", lambda _f: _value({}))
    monkeypatch.setattr(
        suggestions, "get_price_learning_context", lambda _f: _value({"mode": "learning"})
    )
    monkeypatch.setattr(
        suggestions, "refresh_affordability_from_situation", lambda **_k: _value({})
    )
    monkeypatch.setattr(
        suggestions, "refresh_fan_lifecycle", lambda **_k: _value({"stage": "new"})
    )
    monkeypatch.setattr(
        suggestions, "refresh_price_learning", lambda **_k: _value({"mode": "learning"})
    )
    monkeypatch.setattr(suggestions, "learn_from_fan_message", lambda **_k: _value(None))
    monkeypatch.setattr(suggestions, "_crisis_freezes_chat", lambda *_a, **_k: _value(False))
    monkeypatch.setattr(
        suggestions,
        "get_fan_by_id",
        lambda _f: _value(
            Fan(
                id="fan-test",
                display_name="Jostar",
                platform_fan_id="test_jostar",
                fansly_group_id="stale-group-99",
            )
        ),
    )
    monkeypatch.setattr(
        "services.inactivity_reengagement.schedule_inactivity_reengagement", noop_async
    )
    monkeypatch.setattr(suggestions.asyncio, "sleep", fake_sleep)
    suggestions._pending_auto_replies.clear()
    clear_ai_stack_cache()

    def configure(profile_id: str, writer_replies: list[str] | None = None):
        monkeypatch.setenv("AI_STACK_PROFILE", profile_id)
        clear_ai_stack_cache()
        if writer_replies is not None:
            replies[:] = writer_replies
        return db

    yield configure, calls
    clear_ai_stack_cache()


def run_turn(message: str = "how was your day?") -> dict:
    return _run(
        suggestions.run_simulated_inbound(
            fan_id="fan-test",
            creator_id="creator-1",
            message=message,
            fast=True,
        )
    )


def writer_prompt(calls) -> str:
    prompt = calls["writer"][-1]["prompt"]
    system = prompt[0]["content"]
    if isinstance(system, list):
        system = "\n".join(str(block.get("text", "")) for block in system)
    return f"{system}\n{prompt[1]['content']}"


# --- one reply, not three options -------------------------------------------


def test_v3_full_auto_asks_the_writer_for_exactly_one_reply(auto_world, spy_transport):
    configure, calls = auto_world
    configure("cleo_v3")
    run_turn()

    assert calls["writer"][-1]["max_candidates"] == 1
    assert calls["writer"][-1]["output_contract"] == "auto_messages"
    prompt = writer_prompt(calls)
    assert "Write ONE reply." in prompt
    # The Full Auto contract is one reply in its own bubbles, not an array of
    # alternatives with a cardinality of one.
    assert '{"messages": ["first message", "second message"]}' in prompt
    assert "These are not alternatives" in prompt
    assert "Write 3 reply options" not in prompt
    assert "JSON array of 3 strings" not in prompt
    assert "auto mode may send option 1" not in prompt


@pytest.mark.parametrize("profile_id", ("cleo_legacy_v1", "cleo_v2"))
def test_the_frozen_profiles_still_ask_for_three_options(
    profile_id, auto_world, spy_transport
):
    configure, calls = auto_world
    configure(profile_id)
    run_turn()

    assert calls["writer"][-1]["max_candidates"] == 3
    assert calls["writer"][-1]["output_contract"] == "candidates"
    prompt = writer_prompt(calls)
    assert "Write 3 reply options" in prompt
    assert "Return ONLY a JSON array of 3 strings" in prompt
    assert "Write ONE reply." not in prompt


def test_v3_sends_only_the_one_candidate_even_if_the_writer_returns_more(
    auto_world, spy_transport
):
    """Belt and braces on the parser cap: nothing downstream may pick another.

    ``generate_replies`` is faked here, so its own ``max_candidates`` truncation
    does not apply — which is the point. Even handed three, the Auto path must
    send the first and only the first.
    """
    configure, calls = auto_world
    db = configure(
        "cleo_v3",
        writer_replies=["the one that gets sent", "an alternative", "another"],
    )
    run_turn()

    sent = [row["content"] for row in _creator_rows(db)]
    assert sent == ["the one that gets sent"]


def test_v3_full_auto_delivers_the_single_candidate(auto_world, spy_transport):
    configure, calls = auto_world
    db = configure("cleo_v3", writer_replies=["pretty good honestly, yours?"])
    result = run_turn()

    assert result["outcome"] == suggestions.AUTO_OUTCOME_REPLIED
    assert [row["content"] for row in _creator_rows(db)] == [
        "pretty good honestly, yours?"
    ]


# --- the deterministic bubble-count policy ----------------------------------


def test_v3_does_not_impose_a_bubble_count_on_the_writer(auto_world, spy_transport):
    configure, calls = auto_world
    configure("cleo_v3")
    run_turn()

    prompt = writer_prompt(calls)
    assert "MESSAGE SHAPE FOR THIS TURN" not in prompt
    assert "Write this reply as ONE message" not in prompt
    assert "message bubbles separated by" not in prompt


@pytest.mark.parametrize("profile_id", ("cleo_legacy_v1", "cleo_v2"))
def test_the_frozen_profiles_still_get_a_bubble_count(
    profile_id, auto_world, spy_transport
):
    configure, calls = auto_world
    configure(profile_id)
    run_turn()

    assert "MESSAGE SHAPE FOR THIS TURN" in writer_prompt(calls)


def test_v3_keeps_a_multi_bubble_reply_the_writer_chose(auto_world, spy_transport):
    """No post-generation merge: three bubbles are delivered as three messages."""
    configure, calls = auto_world
    db = configure(
        "cleo_v3",
        writer_replies=["okay wait | that's actually funny | tell me more"],
    )
    run_turn()

    assert [row["content"] for row in _creator_rows(db)] == [
        "okay wait",
        "that's actually funny",
        "tell me more",
    ]


def test_the_frozen_profiles_still_merge_down_to_the_chosen_shape(
    auto_world, spy_transport, monkeypatch
):
    from services.message_shape import MessageShape

    monkeypatch.setattr(
        suggestions,
        "choose_message_shape",
        lambda **_kwargs: MessageShape(target_bubbles=1, reason="pinned_by_test"),
    )
    configure, calls = auto_world
    db = configure(
        "cleo_v2",
        writer_replies=["okay wait | that's actually funny | tell me more"],
    )
    run_turn()

    assert len(_creator_rows(db)) == 1


def test_v3_still_honours_a_commercial_cap_on_message_parts(
    auto_world, spy_transport, monkeypatch
):
    """Commercial authority is not shape micromanagement.

    A decision that says "at most one part" is a business constraint the writer
    expresses, so it survives the removal of the shape policy. Merging only —
    nothing is ever padded up to a count.
    """
    from models.commercial import ActionType

    async def fake_orchestrate(**_kwargs):
        return SimpleNamespace(
            action=ActionType.CONTINUE_NORMAL_CHAT,
            selected_package_set_ids=None,
            session_budget_cents=None,
            max_messages=1,
            model_dump=lambda mode=None: {
                "action": "CONTINUE_NORMAL_CHAT",
                "max_messages": 1,
            },
        )

    monkeypatch.setenv("COMMERCIAL_LAYER_ENABLED", "true")
    monkeypatch.setattr(suggestions, "orchestrate", fake_orchestrate)
    monkeypatch.setattr(
        suggestions, "_within_daily_caps", lambda *_a, **_k: _value((True, ""))
    )

    configure, calls = auto_world
    db = configure("cleo_v3", writer_replies=["first thought | second thought"])
    run_turn()

    rows = _creator_rows(db)
    assert len(rows) == 1
    assert "first thought" in rows[0]["content"]
    assert "second thought" in rows[0]["content"]


# --- what V3 must not have changed ------------------------------------------


def test_v3_still_states_the_commercial_and_inventory_constraints(
    auto_world, spy_transport
):
    configure, calls = auto_world
    configure("cleo_v3")
    run_turn()

    prompt = writer_prompt(calls)
    assert "The supplied commercial decision and active session are authoritative" in prompt
    assert "never independently change whether to sell" in prompt
    assert "decided outside this conversation and supplied to you separately" in prompt


def test_v3_uses_the_same_models_as_v2(auto_world, spy_transport):
    """The prompt is the variable under test, so routing must not move with it."""
    from ai.stack_profiles import (
        CLEO_V2,
        CLEO_V3,
        STAGE_ORDER,
    )

    for stage in STAGE_ORDER:
        v2 = CLEO_V2.stage(stage)
        v3 = CLEO_V3.stage(stage)
        assert v3.resolved_primary() == v2.resolved_primary(), stage
        assert v3.resolved_fallback() == v2.resolved_fallback(), stage
        assert v3.reasoning is False, stage
