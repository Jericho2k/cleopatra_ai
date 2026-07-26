from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

from models.commercial import CreatorPolicy, FanCommercialState, FanStatus
from services import personal_callbacks as callbacks
from workers import scheduled_actions as worker


NOW = datetime(2026, 7, 25, 10, 0, tzinfo=timezone.utc)


def run(coro):
    return asyncio.run(coro)


async def async_value(value):
    return value


def enabled_policy(**overrides):
    values = {
        "personal_event_callbacks_enabled": True,
        "personal_event_callback_send_hour_local": 18,
        "personal_event_callback_max_per_30_days": 3,
        "timezone": "UTC",
    }
    values.update(overrides)
    return CreatorPolicy(**values)


def test_only_explicit_dated_personal_events_resolve():
    event = callbacks.resolve_personal_event(
        "i have a job interview tomorrow",
        enabled_policy(),
        now=NOW,
    )
    assert event is not None
    assert event.summary == "job interview"
    assert event.execute_at == datetime(2026, 7, 26, 18, 0, tzinfo=timezone.utc)

    assert callbacks.resolve_personal_event(
        "i might look for a new job sometime",
        enabled_policy(),
        now=NOW,
    ) is None
    assert callbacks.resolve_personal_event(
        "my interview was last friday",
        enabled_policy(),
        now=NOW,
    ) is None
    assert callbacks.resolve_personal_event(
        "i get paid friday",
        enabled_policy(),
        now=NOW,
    ) is None


def test_schedule_uses_durable_queue_and_replaces_prior_callback(monkeypatch):
    calls = []
    monkeypatch.setattr(
        callbacks,
        "get_creator_policy",
        lambda _creator_id: async_value(enabled_policy()),
    )
    monkeypatch.setattr(
        callbacks,
        "resolve_auto_eligibility_for_fan",
        lambda _creator_id, _fan_id: async_value(SimpleNamespace(eligible=True, reason="all_fans")),
    )
    monkeypatch.setattr(
        callbacks,
        "get_fan_state",
        lambda _fan_id: async_value(FanCommercialState(status=FanStatus.IDLE)),
    )
    monkeypatch.setattr(callbacks, "_completed_count", lambda _fan_id, now: async_value(0))

    async def cancel(fan_id, action_type):
        calls.append(("cancel", fan_id, action_type))

    async def schedule(**kwargs):
        calls.append(("schedule", kwargs))

    monkeypatch.setattr(callbacks, "cancel_actions_for_fan", cancel)
    monkeypatch.setattr(callbacks, "schedule_action", schedule)

    created = run(callbacks.schedule_personal_event_callback(
        creator_id="creator-1",
        fan_id="fan-1",
        fan_message="my exam is tomorrow",
        source_message_id="message-1",
        now=NOW,
    ))

    assert created is True
    assert calls[0] == ("cancel", "fan-1", callbacks.ACTION_TYPE)
    scheduled = calls[1][1]
    assert scheduled["action_type"] == callbacks.ACTION_TYPE
    assert scheduled["dedupe_key"] == "personal-event:fan-1:message-1"
    assert scheduled["payload"]["event_summary"] == "exam"
    assert scheduled["payload"]["evidence"] == "my exam is tomorrow"


def test_commercial_obligation_blocks_personal_callback(monkeypatch):
    monkeypatch.setattr(
        callbacks,
        "get_creator_policy",
        lambda _creator_id: async_value(enabled_policy()),
    )
    monkeypatch.setattr(
        callbacks,
        "resolve_auto_eligibility_for_fan",
        lambda _creator_id, _fan_id: async_value(SimpleNamespace(eligible=True, reason="all_fans")),
    )
    monkeypatch.setattr(
        callbacks,
        "get_fan_state",
        lambda _fan_id: async_value(FanCommercialState(
            status=FanStatus.PAUSED_UNTIL_PAYDAY,
            next_followup_type="PAYDAY_REENGAGEMENT",
        )),
    )

    assert run(callbacks.schedule_personal_event_callback(
        creator_id="creator-1",
        fan_id="fan-1",
        fan_message="i have an interview tomorrow",
        source_message_id="message-1",
        now=NOW,
    )) is False


def test_worker_personal_callback_does_not_require_single_commercial_obligation(monkeypatch):
    import db.queries as queries

    action = {
        "id": "action-1",
        "creator_id": "creator-1",
        "fan_id": "fan-1",
        "action_type": callbacks.ACTION_TYPE,
        "payload": {
            "event_summary": "interview",
            "event_at": NOW.isoformat(),
            "callback_window_start_at": NOW.isoformat(),
        },
        "dedupe_key": "personal-event:fan-1:message-1",
    }
    monkeypatch.setattr(
        queries,
        "get_fan_by_id",
        lambda _fan_id: async_value(SimpleNamespace(needs_human_review=False, auto_mode=True)),
    )
    monkeypatch.setattr(
        queries,
        "get_conversation_history",
        lambda _fan_id, limit=5: async_value([]),
    )
    monkeypatch.setattr(
        queries,
        "get_creator_sleep_hours",
        lambda _creator_id: async_value((0, 0)),
    )
    monkeypatch.setattr(
        worker,
        "get_fan_state",
        lambda _fan_id: async_value(FanCommercialState(status=FanStatus.IDLE)),
    )
    monkeypatch.setattr(
        worker,
        "get_creator_policy",
        lambda _creator_id: async_value(enabled_policy()),
    )
    monkeypatch.setattr(
        callbacks,
        "validate_personal_event_action",
        lambda *_args, **_kwargs: async_value(
            callbacks.PersonalCallbackCheck(True, "eligible")
        ),
    )

    assert run(worker._should_still_send(action)).ok is True


def test_personal_callback_goal_is_non_commercial_and_treats_evidence_as_untrusted(monkeypatch):
    captured = []

    async def send(_action, goal):
        captured.append(goal)
        return worker.HandlerResult(sent_message=True)

    monkeypatch.setattr(worker, "_send_goal", send)
    result = run(worker._run_personal_event_callback({
        "payload": {
            "event_summary": "job interview",
            "evidence": "i have my interview tomorrow",
        }
    }))

    assert result.sent_message is True
    assert "untrusted conversation data" in captured[0]
    assert "Do not announce that you remembered" in captured[0]
    assert "sale" in captured[0]
