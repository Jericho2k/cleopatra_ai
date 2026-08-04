from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from models.schemas import Message
from services import suggestions
from services.human_delivery import AvailabilityMode
from workers import scheduled_actions


def test_auto_reply_is_persisted_instead_of_sleeping_in_memory(monkeypatch):
    now = datetime.now(timezone.utc)
    history = [
        Message(role="creator", content="hey", sent_at=now - timedelta(minutes=2)),
        Message(role="fan", content="hi", sent_at=now),
    ]
    scheduled = {}
    cancelled = []

    async def fake_cancel(fan_id, action_type):
        cancelled.append((fan_id, action_type))

    async def fake_schedule(**kwargs):
        scheduled.update(kwargs)

    async def fake_session(_fan_id):
        return None

    async def fake_sleep_hours(_creator_id):
        return 0, 0

    monkeypatch.setattr(suggestions, "cancel_actions_for_fan", fake_cancel)
    monkeypatch.setattr(suggestions, "schedule_action", fake_schedule)
    monkeypatch.setattr(suggestions, "get_fan_session", fake_session)
    monkeypatch.setattr("db.queries.get_creator_sleep_hours", fake_sleep_hours)
    monkeypatch.setattr(
        suggestions,
        "build_availability_delay",
        lambda *_args, **_kwargs: (AvailabilityMode.LIVE, 30.0),
    )
    monkeypatch.setattr(suggestions.random, "uniform", lambda *_args: 10.0)

    asyncio.run(
        suggestions.schedule_auto_reply(
            "fan",
            "creator",
            conversation_history=history,
            source_message_id="message-1",
        )
    )

    assert cancelled == [("fan", "AUTO_REPLY")]
    assert scheduled["action_type"] == "AUTO_REPLY"
    assert scheduled["dedupe_key"] == "auto-reply:fan:message-1"
    assert scheduled["payload"]["trigger_sent_at"] == now.isoformat()
    assert scheduled["payload"]["availability_mode"] == "live"
    assert 39 <= (scheduled["execute_at"] - now).total_seconds() <= 42
    assert "fan" not in suggestions._pending_auto_replies


def test_newer_activity_invalidates_a_claimed_auto_reply(monkeypatch):
    trigger_at = datetime(2026, 8, 4, 12, 0, tzinfo=timezone.utc)

    async def fake_fan(_fan_id):
        return type("Fan", (), {"needs_human_review": False, "auto_mode": True})()

    async def fake_history(_fan_id, limit=10):
        return [
            Message(role="fan", content="first", sent_at=trigger_at),
            Message(role="fan", content="newer", sent_at=trigger_at + timedelta(seconds=2)),
        ]

    async def fake_availability(_creator_id):
        return {"auto_available": True}

    monkeypatch.setattr("db.queries.get_fan_by_id", fake_fan)
    monkeypatch.setattr("db.queries.get_conversation_history", fake_history)
    monkeypatch.setattr("main._creator_auto_availability", fake_availability)

    result = asyncio.run(
        scheduled_actions._should_still_send(
            {
                "fan_id": "fan",
                "creator_id": "creator",
                "action_type": "AUTO_REPLY",
                "payload": {"trigger_sent_at": trigger_at.isoformat()},
            }
        )
    )

    assert result.ok is False
    assert "newer conversation activity" in result.reason


def test_reclaimed_auto_reply_reconciles_platform_before_retry(monkeypatch):
    trigger_at = datetime(2026, 8, 4, 12, 0, tzinfo=timezone.utc)
    reconciled = []
    generated = []

    async def fake_sync(creator_id, fan_id):
        reconciled.append((creator_id, fan_id))
        return {"status": "ok"}

    async def fake_reply(*_args, **_kwargs):
        generated.append(True)

    async def fake_history(_fan_id, limit=10):
        return [
            Message(
                role="creator",
                content="already delivered",
                sent_at=trigger_at + timedelta(seconds=10),
            )
        ]

    monkeypatch.setattr("main.sync_recent_fan_messages", fake_sync)
    monkeypatch.setattr(suggestions, "_debounced_auto_reply", fake_reply)
    monkeypatch.setattr(suggestions, "get_conversation_history", fake_history)

    sent = asyncio.run(
        suggestions.deliver_scheduled_auto_reply(
            {
                "fan_id": "fan",
                "creator_id": "creator",
                "status": "PROCESSING",
                "payload": {"trigger_sent_at": trigger_at.isoformat()},
            }
        )
    )

    assert reconciled == [("creator", "fan")]
    assert generated == []
    assert sent is True


def test_worker_dispatches_and_completes_durable_auto_reply(monkeypatch):
    due = {
        "id": "action-1",
        "fan_id": "fan",
        "creator_id": "creator",
        "action_type": "AUTO_REPLY",
        "payload": {"trigger_sent_at": datetime.now(timezone.utc).isoformat()},
        "attempts": 0,
    }
    calls = []

    async def fake_handler(action):
        calls.append(("handled", action["id"]))
        return scheduled_actions.HandlerResult(sent_message=True, reason="sent")

    async def fake_complete(action_id):
        calls.append(("completed", action_id))

    monkeypatch.setattr(
        scheduled_actions, "repair_followup_obligations", lambda: _async_value(0)
    )
    monkeypatch.setattr(
        scheduled_actions, "claim_due_actions", lambda: _async_value([due])
    )
    monkeypatch.setattr(
        scheduled_actions,
        "_should_still_send",
        lambda _action: _async_value(scheduled_actions.ActionCheck(True)),
    )
    monkeypatch.setitem(scheduled_actions.HANDLERS, "AUTO_REPLY", fake_handler)
    monkeypatch.setattr(scheduled_actions, "complete_action", fake_complete)
    monkeypatch.setattr(
        scheduled_actions,
        "_record_message_action_resolution",
        lambda *_args, **_kwargs: _async_value(None),
    )

    assert asyncio.run(scheduled_actions.process_once()) == 1
    assert calls == [("handled", "action-1"), ("completed", "action-1")]


async def _async_value(value):
    return value
