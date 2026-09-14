"""The LLM does not control delivery. End to end, through the real pipeline.

The failure this pins: the creator said "sending it", "here's the first piece",
"here it is" — three times — and no media ever appeared, because delivery was a
``[PPV:media_id:price]`` tag the writer had to serialise correctly into free text
and the sender parsed back out. A model that wrote the sentence and not the tag
produced a message that claimed a delivery and was, to every layer below it, an
ordinary text message.

Six things are asserted here, in the order the request asked for them:

1. the commercial decision selects media X at price Y;
2. the writer returns ordinary text;
3. the sender receives X and Y separately, from the plan and not from the copy;
4. the message persists with the correct media metadata;
5. the Simulator renders it as a PPV card, from that same metadata;
6. a delivery that FAILED cannot leave a message behind claiming it succeeded.
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.schemas import Fan, Persona  # noqa: E402
from services import suggestions  # noqa: E402
from tests.test_full_auto_simulation import FakeDB  # noqa: E402
from tests.test_full_auto_simulation import spy as spy  # noqa: E402,F401


NOW = datetime(2026, 7, 18, 12, 0, tzinfo=timezone.utc)

# The plan is the ONLY authority for what is attached and what it costs.
PLANNED_MEDIA = ["media-x1", "media-x2"]
PLANNED_PRICE_CENTS = 3500


async def _value(value):
    return value


def _run(coro):
    return asyncio.run(coro)


def _session(**overrides) -> dict:
    step = {
        "step_number": 1,
        "step_count": 1,
        "media_ids": list(PLANNED_MEDIA),
        "media_id": PLANNED_MEDIA[0],
        "price": PLANNED_PRICE_CENTS / 100,
        "price_cents": PLANNED_PRICE_CENTS,
        "set_id": "set-x",
        "asset_type": "photo_set",
        "description": "beach day bundle (2 pcs)",
        "sent": False,
        "purchased": False,
    }
    step.update(overrides.pop("step", {}))
    session = {
        "status": "active",
        "current_index": 0,
        "awaiting_purchase_index": None,
        "plan": [step],
    }
    session.update(overrides)
    return session


@pytest.fixture
def delivery_world(monkeypatch):
    """The real Full Auto turn, with a writer that writes ORDINARY TEXT."""
    db = FakeDB(
        {
            "messages": [
                {
                    "id": "msg-1",
                    "fan_id": "fan-test",
                    "creator_id": "creator-1",
                    "role": "fan",
                    "content": "yes send it",
                    "sent_at": "2026-07-18T11:59:00+00:00",
                }
            ],
            "fans": [
                {
                    "id": "fan-test",
                    "creator_id": "creator-1",
                    "platform_fan_id": "test_jostar",
                    "display_name": "Jostar",
                    "fansly_group_id": "group-1",
                    "pending_tip": None,
                    "pending_ppv_check": None,
                }
            ],
            "creators": [
                {
                    "id": "creator-1",
                    "name": "Sophia",
                    "apifansly_account_id": "acct-1",
                    "auto_mode": True,
                }
            ],
        }
    )
    state: dict = {"writer_replies": ["here it is 😏"], "session": _session()}
    calls: dict[str, list] = {"writer": [], "receipts": [], "reconciliations": []}

    from models.commercial import ActionType, CreatorPolicy

    async def fake_analyze(_ctx, telemetry_context=None, **_kwargs):
        return {
            "purchase_signal": "ready_to_buy",
            "crisis_signal": "none",
            "resend_requested": "false",
            "strategic_move": "close",
        }

    async def fake_generate(prompt, persona, **kwargs):
        calls["writer"].append({"prompt": prompt, **kwargs})
        return list(state["writer_replies"])

    async def fake_orchestrate(**_kwargs):
        return SimpleNamespace(
            action=ActionType.SEND_NEXT_PPV_STEP,
            accepted_offer_set_id=None,
            session_budget_cents=None,
            max_messages=1,
            authorized_asset_types=["photo_set"],
            vault_asset_types=["photo_set"],
            unavailable_asset_type_requested=None,
            next_offer=None,
            model_dump=lambda mode=None: {"action": "SEND_NEXT_PPV_STEP"},
        )

    async def capture_receipt(**kwargs):
        calls["receipts"].append(kwargs)
        return "ppv-message-1"

    async def capture_reconciliation(**kwargs):
        calls["reconciliations"].append(kwargs)
        return kwargs.get("session"), NOW + timedelta(hours=1), True

    async def noop_async(*_a, **_k):
        return None

    async def fake_sleep(_seconds):
        return None

    def fake_route(ctx, **kwargs):
        return SimpleNamespace(
            route=SimpleNamespace(value="commercial_complex"),
            reason="test",
            primary_target=SimpleNamespace(model="test-writer", provider="test"),
            fallback_target=None,
            prompt_version="writer_v3",
            ai_stack_profile="cleo_v3",
            telemetry_metadata=lambda: {},
        )

    monkeypatch.setenv("COMMERCIAL_LAYER_ENABLED", "true")
    monkeypatch.setenv("APIFANSLY_ENABLED", "true")
    monkeypatch.setattr(suggestions, "get_supabase", lambda: db)
    monkeypatch.setattr("db.queries.get_supabase", lambda: db)
    monkeypatch.setattr(suggestions, "analyze_situation", fake_analyze)
    monkeypatch.setattr(suggestions, "generate_replies", fake_generate)
    monkeypatch.setattr(suggestions, "select_writer_route", fake_route)
    monkeypatch.setattr(suggestions, "orchestrate", fake_orchestrate)
    monkeypatch.setattr(suggestions, "save_ppv_message_receipt", capture_receipt)
    monkeypatch.setattr(suggestions, "persist_ppv_reconciliation", capture_reconciliation)
    monkeypatch.setattr(suggestions, "get_fan_session", lambda _f: _value(state["session"]))
    monkeypatch.setattr(suggestions, "_within_daily_caps", lambda *_a, **_k: _value((True, "")))
    monkeypatch.setattr(suggestions, "get_creator_policy", lambda _c: _value(CreatorPolicy()))
    monkeypatch.setattr(
        "db.commercial_queries.get_creator_policy", lambda _c: _value(CreatorPolicy())
    )
    monkeypatch.setattr(suggestions, "direct_conversation", lambda **_k: _value({}))
    monkeypatch.setattr(suggestions, "plan_next_action", lambda **_k: _value({}))
    monkeypatch.setattr(suggestions, "get_creator_persona", lambda _c: _value(Persona()))
    monkeypatch.setattr(suggestions, "get_ppv_offers", lambda _c: _value([]))
    monkeypatch.setattr(suggestions, "get_sent_ppv", lambda _f: _value([]))
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
    monkeypatch.setattr(suggestions, "persist_sent_creator_facts", noop_async)
    monkeypatch.setattr(
        suggestions,
        "get_fan_by_id",
        lambda _f: _value(
            Fan(
                id="fan-test",
                display_name="Jostar",
                platform_fan_id="test_jostar",
                fansly_group_id="group-1",
            )
        ),
    )
    monkeypatch.setattr(
        "services.inactivity_reengagement.schedule_inactivity_reengagement", noop_async
    )
    monkeypatch.setattr(suggestions.asyncio, "sleep", fake_sleep)
    suggestions._pending_auto_replies.clear()
    return db, state, calls


def _turn(message: str = "yes send it") -> dict:
    return _run(
        suggestions.run_simulated_inbound(
            fan_id="fan-test", creator_id="creator-1", message=message, fast=True
        )
    )


# --- 1-4: decision -> ordinary text -> separate media/price -> persistence ---


def test_the_planned_media_and_price_reach_the_sender_without_passing_through_copy(
    delivery_world, spy
):
    _db, _state, calls = delivery_world
    _turn()

    # 2. The writer wrote text. No tag, nowhere.
    writer_prompt = str(calls["writer"][-1]["prompt"])
    assert "[PPV:" not in writer_prompt
    assert "media-x1" not in writer_prompt, "the writer is never given a media id"
    assert str(PLANNED_PRICE_CENTS) not in writer_prompt

    # 3 + 4. The sender got the plan's media and price, and persisted them.
    assert len(calls["receipts"]) == 1
    receipt = calls["receipts"][0]
    ppv = receipt["media_context"]["ppv"]
    assert ppv["media_ids"] == PLANNED_MEDIA
    assert ppv["media_id"] == PLANNED_MEDIA[0]
    assert ppv["price_cents"] == PLANNED_PRICE_CENTS
    assert ppv["price"] == PLANNED_PRICE_CENTS / 100
    assert ppv["access_type"] == "ppv"
    assert ppv["set_id"] == "set-x"
    assert receipt["content"] == "here it is 😏"

    # And the local commercial state was attached for reconciliation.
    assert len(calls["reconciliations"]) == 1
    pending = calls["reconciliations"][0]["pending"]
    assert pending["media_ids"] == PLANNED_MEDIA
    assert pending["price_cents"] == PLANNED_PRICE_CENTS


def test_a_writer_that_still_emits_a_tag_is_not_obeyed(delivery_world, spy):
    _db, state, calls = delivery_world
    state["writer_replies"] = ["here it is [PPV:some-other-media:9999]"]

    _turn()

    receipt = calls["receipts"][0]
    assert "[PPV:" not in receipt["content"], "the tag never reaches the fan"
    assert "some-other-media" not in receipt["content"]
    ppv = receipt["media_context"]["ppv"]
    assert ppv["media_ids"] == PLANNED_MEDIA, "the plan decided, not the copy"
    assert ppv["price_cents"] == PLANNED_PRICE_CENTS


def test_paid_media_goes_out_as_exactly_one_message(delivery_world, spy):
    """Text and attachment travel together, so no text can outrun the send."""
    _db, state, calls = delivery_world
    state["writer_replies"] = ["okay | wait till you see this | here"]

    _turn()

    assert len(calls["receipts"]) == 1
    assert "|" not in calls["receipts"][0]["content"]


# --- 5: the Simulator renders it from that metadata -------------------------


def test_the_simulator_renders_the_persisted_metadata_as_a_ppv_card(
    delivery_world, spy
):
    _db, _state, calls = delivery_world
    _turn()

    persisted = calls["receipts"][0]["media_context"]
    # This is the exact shape lib/simulationWorkspace.ts reads to decide whether
    # a message is a locked PPV card, a purchased one, or ordinary text.
    ppv = persisted["ppv"]
    assert ppv.get("access_type") == "ppv"
    assert ppv.get("media_ids")
    assert isinstance(ppv.get("price"), (int, float))
    assert not ppv.get("purchased"), "a freshly sent PPV is locked, not unlocked"


# --- 6: a failed delivery leaves no claim behind ----------------------------


def test_a_failed_delivery_persists_no_message_claiming_it_succeeded(
    delivery_world, monkeypatch, spy
):
    db, _state, calls = delivery_world

    async def refuse(*_args, **_kwargs):
        raise RuntimeError("platform rejected the attachment")

    frozen: list[tuple] = []

    async def capture_freeze(fan_id, reason):
        frozen.append((fan_id, reason))

    # A REAL fan, so the live delivery branch runs and can actually fail.
    for row in db.tables["fans"]:
        row["platform_fan_id"] = "real_fan_99"
    monkeypatch.setattr(
        suggestions,
        "get_fan_by_id",
        lambda _f: _value(
            Fan(
                id="fan-test",
                display_name="Jostar",
                platform_fan_id="real_fan_99",
                fansly_group_id="group-1",
            )
        ),
    )
    monkeypatch.setattr(suggestions, "send_apifansly_message", refuse)
    monkeypatch.setattr(suggestions, "freeze_fan_for_review", capture_freeze)

    result = _turn()

    assert calls["receipts"] == [], "nothing was persisted for a delivery that failed"
    assert calls["reconciliations"] == []
    assert result["creator_messages"] == [], "the fan sees no 'here it is'"
    creator_rows = [row for row in db.tables["messages"] if row["role"] == "creator"]
    assert creator_rows == []
    assert frozen and frozen[0][1] == "ppv_send_failed"


def test_no_delivery_route_persists_no_message_either(
    delivery_world, monkeypatch, spy
):
    db, _state, calls = delivery_world
    frozen: list[tuple] = []

    async def capture_freeze(fan_id, reason):
        frozen.append((fan_id, reason))

    for row in db.tables["fans"]:
        row["fansly_group_id"] = None
        row["platform_fan_id"] = "real_fan_99"
    monkeypatch.setattr(suggestions, "freeze_fan_for_review", capture_freeze)
    monkeypatch.setattr(
        suggestions,
        "get_fan_by_id",
        lambda _f: _value(
            Fan(id="fan-test", display_name="Jostar", platform_fan_id="real_fan_99")
        ),
    )

    result = _turn()

    assert calls["receipts"] == []
    assert result["creator_messages"] == []
    assert frozen and frozen[0][1] == "ppv_delivery_route_missing"


# --- the simulated route is the same route ----------------------------------


def test_a_test_fan_gets_the_same_metadata_without_touching_the_platform(
    delivery_world, spy
):
    _db, _state, calls = delivery_world
    _turn()

    assert spy.requests == [], "a simulated delivery makes zero remote calls"
    ppv = calls["receipts"][0]["media_context"]["ppv"]
    assert ppv["media_ids"] == PLANNED_MEDIA
    assert calls["receipts"][0]["platform_message_id"].startswith("local-test:")
