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


def test_followup_obligation_scan_paginates_past_first_page(monkeypatch):
    import db.commercial_queries as commercial_queries

    ranges = []

    class Query:
        current_range = (0, 1)

        def table(self, _name):
            return self

        def select(self, _columns):
            return self

        @property
        def not_(self):
            return self

        def is_(self, _column, _value):
            return self

        def order(self, _column):
            return self

        def range(self, start, end):
            self.current_range = (start, end)
            ranges.append(self.current_range)
            return self

        def execute(self):
            pages = {
                (0, 1): [{"fan_id": "1"}, {"fan_id": "2"}],
                (2, 3): [{"fan_id": "3"}],
            }
            return SimpleNamespace(data=pages[self.current_range])

    monkeypatch.setattr(commercial_queries, "get_supabase", lambda: Query())

    rows = run(commercial_queries.get_followup_obligations(page_size=2))

    assert [row["fan_id"] for row in rows] == ["1", "2", "3"]
    assert ranges == [(0, 1), (2, 3)]


def test_message_followups_get_extended_delivery_retries(monkeypatch):
    due = action("OFFER_EXPIRY")
    failures = []

    async def broken_handler(_action):
        raise RuntimeError("temporary platform failure")

    monkeypatch.setattr(worker, "repair_followup_obligations", lambda: async_value(0))
    monkeypatch.setattr(worker, "claim_due_actions", lambda: async_value([due]))
    monkeypatch.setitem(worker.HANDLERS, "OFFER_EXPIRY", broken_handler)

    async def capture_failure(*args, **kwargs):
        failures.append((args, kwargs))

    monkeypatch.setattr(worker, "fail_action", capture_failure)

    assert run(worker.process_once()) == 0
    assert failures[0][1]["max_attempts"] == 8


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


def test_creator_auto_mode_falls_back_when_old_queries_module_lacks_helper(monkeypatch):
    import core.supabase as supabase_module
    import db.queries as queries

    class Query:
        def table(self, _name):
            return self

        def select(self, _columns):
            return self

        def eq(self, _column, _value):
            return self

        def single(self):
            return self

        def execute(self):
            return SimpleNamespace(data={"auto_mode": True})

    monkeypatch.delattr(queries, "get_creator_auto_mode_default")
    monkeypatch.setattr(supabase_module, "get_supabase", lambda: Query())

    assert run(worker._creator_auto_mode_default("creator-1")) is True


def test_repair_requeues_the_known_payday_import_failure(monkeypatch):
    import db.commercial_queries as commercial_queries

    updates = []

    class Query:
        operation = "select"

        def table(self, _name):
            return self

        def select(self, _columns):
            self.operation = "select"
            return self

        def update(self, payload):
            self.operation = "update"
            updates.append(payload)
            return self

        def eq(self, _column, _value):
            return self

        def limit(self, _value):
            return self

        def execute(self):
            if self.operation == "select":
                return SimpleNamespace(data=[{
                    "id": "action-1",
                    "status": "FAILED",
                    "last_error": (
                        "cannot import name 'get_creator_auto_mode_default' "
                        "from 'db.queries'"
                    ),
                }])
            return SimpleNamespace(data=[{"id": "action-1"}])

    query = Query()
    monkeypatch.setattr(commercial_queries, "get_supabase", lambda: query)

    run(commercial_queries.ensure_action_pending(
        creator_id="creator-1",
        fan_id="fan-1",
        action_type="PAYDAY_REENGAGEMENT",
        execute_at=NOW,
        payload={"payday_at": NOW.isoformat()},
        dedupe_key="payday:fan-1",
    ))

    assert updates
    assert updates[0]["status"] == "PENDING"
    assert updates[0]["attempts"] == 0


def test_stale_same_type_action_cannot_clear_newer_offer_obligation(monkeypatch):
    state = FanCommercialState(
        status=FanStatus.OFFER_PENDING,
        next_followup_at=NOW,
        next_followup_type="OFFER_EXPIRY",
        next_followup_payload={"offer_reference": "new"},
        next_followup_dedupe_key="offer-expiry:fan-1:new",
    )
    saved = []
    monkeypatch.setattr(worker, "get_fan_state", lambda _fan_id: async_value(state))
    monkeypatch.setattr(
        worker,
        "save_fan_state",
        lambda fan_id, creator_id, value: async_append(saved, (fan_id, creator_id, value)),
    )
    stale = action("OFFER_EXPIRY")
    stale["dedupe_key"] = "offer-expiry:fan-1:old"

    run(worker._record_message_action_resolution(stale, sent=False))

    assert saved[0][2].next_followup_type == "OFFER_EXPIRY"
    assert saved[0][2].next_followup_dedupe_key == "offer-expiry:fan-1:new"


async def async_value(value):
    return value


async def async_append(target, value):
    target.append(value)


async def capture_handler_result(target, value):
    target.append(value)
    return worker.HandlerResult(sent_message=True)
