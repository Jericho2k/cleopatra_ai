from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

from models.commercial import CreatorPolicy, FanCommercialState, FanStatus
from models.schemas import Message
from workers import scheduled_actions as worker


NOW = datetime.now(timezone.utc)


def run(coro):
    return asyncio.run(coro)


def action(action_type="PAYDAY_REENGAGEMENT"):
    return {
        "id": "action-1",
        "creator_id": "creator-1",
        "fan_id": "fan-1",
        "action_type": action_type,
        "payload": {},
        "dedupe_key": f"{action_type}:fan-1",
        "attempts": 0,
    }


def test_reconciliation_retry_reschedules_without_marking_complete(monkeypatch):
    due = action("PPV_RECONCILE")
    retry_at = datetime(2026, 7, 18, 13, 0, tzinfo=timezone.utc)
    calls = []

    monkeypatch.setattr(worker, "repair_followup_obligations", lambda: async_value(0))
    monkeypatch.setattr(worker, "claim_due_actions", lambda: async_value([due]))
    monkeypatch.setitem(
        worker.HANDLERS,
        "PPV_RECONCILE",
        lambda _action: async_value(worker.HandlerResult(retry_at=retry_at, reason="still pending")),
    )
    monkeypatch.setattr(
        worker,
        "reschedule_action",
        lambda action_id, when: async_append(calls, ("retry", action_id, when)),
    )
    monkeypatch.setattr(
        worker,
        "complete_action",
        lambda action_id: async_append(calls, ("complete", action_id)),
    )

    assert run(worker.process_once()) == 0
    assert calls == [("retry", "action-1", retry_at)]


def test_recent_fan_activity_postpones_valid_followup(monkeypatch):
    due = action()
    due["payload"] = {"payday_at": NOW.isoformat()}
    fan = SimpleNamespace(
        needs_human_review=False,
        auto_mode=True,
    )
    state = FanCommercialState(
        status=FanStatus.PAUSED_UNTIL_PAYDAY,
        payday_at=NOW,
        next_followup_at=NOW,
        next_followup_type="PAYDAY_REENGAGEMENT",
        next_followup_dedupe_key="PAYDAY_REENGAGEMENT:fan-1",
    )

    import db.queries as queries

    monkeypatch.setattr(queries, "get_fan_by_id", lambda _fan_id: async_value(fan))
    monkeypatch.setattr(
        queries,
        "get_conversation_history",
        lambda _fan_id, limit=5: async_value([
            Message(role="fan", content="hey", sent_at=NOW),
        ]),
    )
    monkeypatch.setattr(worker, "get_fan_state", lambda _fan_id: async_value(state))
    monkeypatch.setattr(
        worker,
        "get_creator_policy",
        lambda _creator_id: async_value(
            CreatorPolicy(followup_recent_activity_suppression_hours=6)
        ),
    )

    check = run(worker._should_still_send(due))
    assert check.ok is False
    assert check.retry_at is not None
    assert check.retry_at > NOW


def test_followup_obligation_recreates_missing_action(monkeypatch):
    calls = []
    obligation = {
        "creator_id": "creator-1",
        "fan_id": "fan-1",
        "next_followup_at": "2026-07-19T06:00:00+00:00",
        "next_followup_type": "POST_SESSION_FOLLOWUP",
        "next_followup_payload": {"experience": "shower"},
        "next_followup_dedupe_key": "post-session:fan-1:completed",
    }
    monkeypatch.setattr(
        worker,
        "get_followup_obligations",
        lambda: async_value([obligation]),
    )
    monkeypatch.setattr(
        worker,
        "ensure_action_pending",
        lambda **kwargs: async_append(calls, kwargs),
    )

    assert run(worker.repair_followup_obligations()) == 1
    assert calls[0]["action_type"] == "POST_SESSION_FOLLOWUP"
    assert calls[0]["payload"]["experience"] == "shower"


def test_post_session_goal_uses_buyer_stage_without_selling(monkeypatch):
    captured = []
    monkeypatch.setattr(
        worker,
        "_send_goal",
        lambda _action, goal: capture_handler_result(captured, goal),
    )
    due = action("POST_SESSION_FOLLOWUP")
    due["payload"] = {
        "experience": "shower",
        "buyer_stage": "FIRST_TIME_BUYER",
    }
    result = run(worker._run_post_session_followup(due))
    assert result.sent_message is True
    assert "first purchase" in captured[0]
    assert "do not mention" in captured[0].lower()
    assert "a new price" in captured[0].lower()


async def async_value(value):
    return value


async def async_append(target, value):
    target.append(value)


async def capture_handler_result(target, value):
    target.append(value)
    return worker.HandlerResult(sent_message=True)
