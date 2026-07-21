from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from models.commercial import CreatorPolicy, FanCommercialState, FanStatus
from services import inactivity_reengagement as inactivity


NOW = datetime(2026, 7, 21, 12, 0, tzinfo=timezone.utc)


def run(coro):
    return asyncio.run(coro)


async def async_value(value):
    return value


async def async_append(target, value):
    target.append(value)
    return None


def enabled_policy(**overrides):
    values = {
        "inactivity_reengagement_enabled": True,
        "inactivity_reengagement_delay_hours": 48,
        "inactivity_reengagement_cooldown_hours": 168,
        "inactivity_reengagement_max_per_30_days": 2,
    }
    values.update(overrides)
    return CreatorPolicy(**values)


def test_frequency_limits_apply_per_fan_and_reset_after_30_days():
    state = FanCommercialState(
        inactivity_reengagement_window_started_at=NOW - timedelta(days=10),
        inactivity_reengagement_count=2,
    )
    assert inactivity.frequency_check(state, enabled_policy(), now=NOW).ok is False

    state.inactivity_reengagement_window_started_at = NOW - timedelta(days=31)
    assert inactivity.frequency_check(state, enabled_policy(), now=NOW).ok is True
    assert state.inactivity_reengagement_count == 0

    state.last_inactivity_reengagement_at = NOW - timedelta(hours=24)
    assert inactivity.frequency_check(state, enabled_policy(), now=NOW).ok is False


def test_idle_eligible_auto_chat_schedules_one_restart_safe_obligation(monkeypatch):
    state = FanCommercialState(status=FanStatus.IDLE)
    saved = []
    scheduled = []
    monkeypatch.setattr(inactivity, "get_creator_policy", lambda _cid: async_value(enabled_policy()))
    monkeypatch.setattr(inactivity, "get_fan_state", lambda _fid: async_value(state))
    monkeypatch.setattr(
        inactivity,
        "resolve_auto_eligibility_for_fan",
        lambda _cid, _fid: async_value(SimpleNamespace(eligible=True, reason="all_fans")),
    )
    monkeypatch.setattr(
        inactivity,
        "_latest_message",
        lambda _fid: async_value({"id": "message-1", "role": "creator", "sent_at": NOW.isoformat()}),
    )
    monkeypatch.setattr(
        inactivity,
        "save_fan_state",
        lambda fan_id, creator_id, value: async_append(saved, (fan_id, creator_id, value.model_copy(deep=True))),
    )
    monkeypatch.setattr(
        inactivity,
        "cancel_actions_for_fan",
        lambda *_args: async_value(None),
    )
    monkeypatch.setattr(
        inactivity,
        "schedule_action",
        lambda **kwargs: async_append(scheduled, kwargs),
    )

    assert run(inactivity.schedule_inactivity_reengagement(
        creator_id="creator",
        fan_id="fan",
        source_message_id="message-1",
        now=NOW,
    )) is True
    assert saved[0][2].next_followup_type == inactivity.ACTION_TYPE
    assert saved[0][2].next_followup_at == NOW + timedelta(hours=48)
    assert scheduled[0]["dedupe_key"] == "inactivity:fan:message-1"


def test_specific_commercial_followup_always_beats_generic_inactivity(monkeypatch):
    state = FanCommercialState(
        status=FanStatus.IDLE,
        next_followup_type="POST_SESSION_FOLLOWUP",
        next_followup_at=NOW + timedelta(hours=2),
    )
    monkeypatch.setattr(inactivity, "get_creator_policy", lambda _cid: async_value(enabled_policy()))
    monkeypatch.setattr(inactivity, "get_fan_state", lambda _fid: async_value(state))
    assert run(inactivity.schedule_inactivity_reengagement(
        creator_id="creator",
        fan_id="fan",
        source_message_id="message-1",
        now=NOW,
    )) is False


def test_fan_return_or_newer_message_invalidates_scheduled_nudge(monkeypatch):
    state = FanCommercialState(
        status=FanStatus.IDLE,
        next_followup_type=inactivity.ACTION_TYPE,
        next_followup_dedupe_key="inactivity:fan:message-1",
    )
    action = {
        "creator_id": "creator",
        "fan_id": "fan",
        "payload": {"source_message_id": "message-1"},
    }
    monkeypatch.setattr(
        inactivity,
        "resolve_auto_eligibility_for_fan",
        lambda _cid, _fid: async_value(SimpleNamespace(eligible=True, reason="all_fans")),
    )
    monkeypatch.setattr(
        inactivity,
        "_latest_message",
        lambda _fid: async_value({"id": "message-2", "role": "fan", "sent_at": NOW.isoformat()}),
    )
    result = run(inactivity.validate_inactivity_action(action, state, enabled_policy(), now=NOW))
    assert result.ok is False
    assert "returned" in result.reason or "newer" in result.reason


def test_sent_nudge_increments_window_counter():
    state = FanCommercialState()
    inactivity.record_inactivity_sent(state, now=NOW)
    assert state.inactivity_reengagement_count == 1
    assert state.last_inactivity_reengagement_at == NOW
