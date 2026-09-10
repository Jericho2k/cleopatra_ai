"""REL-005 — retry budget follows the reason for the failure.

Sprint 1 removed the writer's 3x generation retry amplification. The durable
layer underneath still retried everything eight times: an AUTO_REPLY for a
creator whose API Fansly account is disconnected ran the analyzer and the writer
eight times to reach an answer that one indexed read could have given before the
first one.

Three things are asserted here, and the first is the expensive one:

  * a permanently unsendable action never reaches the model pipeline at all;
  * it consumes ONE attempt, not eight, and lands FAILED with an operator-
    readable cause;
  * a transient failure keeps the full retry budget, because calling a
    recoverable failure permanent would silently drop a real message.
"""

from __future__ import annotations

import asyncio

import pytest

from core.action_failures import (
    PermanentActionFailure,
    WriterQualityFailure,
    is_terminal_error,
    terminal_code,
)
from tests.fake_supabase import FakeSupabase
from workers import scheduled_actions

CREATOR_ID = "creator-1"
FAN_ID = "fan-1"


def _action(**overrides) -> dict:
    action = {
        "id": "action-1",
        "action_type": "AUTO_REPLY",
        "fan_id": FAN_ID,
        "creator_id": CREATOR_ID,
        "attempts": 0,
        "status": "PROCESSING",
        "execute_at": "2026-01-01T00:00:00+00:00",
        "payload": {},
    }
    action.update(overrides)
    return action


@pytest.fixture
def worker(monkeypatch):
    """_resolve_action wired to recording transition functions."""
    state: dict = {
        "failed": [],
        "failed_terminal": [],
        "completed": [],
        "rescheduled": [],
        "pipeline_runs": 0,
    }

    async def fake_fail(action_id, error, attempts, max_attempts=3):
        state["failed"].append({
            "id": action_id,
            "error": error,
            "attempts": attempts,
            "max_attempts": max_attempts,
        })

    async def fake_fail_terminal(action_id, code, detail, attempts):
        state["failed_terminal"].append({
            "id": action_id,
            "code": code,
            "detail": detail,
            "attempts": attempts,
        })

    async def fake_complete(action_id):
        state["completed"].append(action_id)

    async def fake_reschedule(action_id, execute_at, **_kwargs):
        state["rescheduled"].append((action_id, execute_at))

    async def fake_resolution(action, *, sent):
        return None

    monkeypatch.setattr(scheduled_actions, "fail_action", fake_fail)
    monkeypatch.setattr(scheduled_actions, "fail_action_terminal", fake_fail_terminal)
    monkeypatch.setattr(scheduled_actions, "complete_action", fake_complete)
    monkeypatch.setattr(scheduled_actions, "reschedule_action", fake_reschedule)
    monkeypatch.setattr(
        scheduled_actions, "_record_message_action_resolution", fake_resolution
    )

    async def always_send(_action):
        return scheduled_actions.ActionCheck(ok=True)

    monkeypatch.setattr(scheduled_actions, "_should_still_send", always_send)
    return state


def _resolve(action):
    return asyncio.run(scheduled_actions._resolve_action(action, sent_counter=[0]))


# --- the classification itself ---------------------------------------------


def test_a_permanent_failure_fails_once_with_a_cause(worker, monkeypatch):
    async def handler(_action):
        raise PermanentActionFailure(
            "creator_not_connected", "creator has no API Fansly account"
        )

    monkeypatch.setitem(scheduled_actions.HANDLERS, "AUTO_REPLY", handler)

    outcome = _resolve(_action())

    assert outcome == "failed_terminal"
    assert worker["failed"] == [], "a permanent failure consumed a retry budget"
    assert len(worker["failed_terminal"]) == 1
    assert worker["failed_terminal"][0]["code"] == "creator_not_connected"


def test_a_transient_failure_keeps_the_full_retry_budget(worker, monkeypatch):
    async def handler(_action):
        raise RuntimeError("OpenRouter 503")

    monkeypatch.setitem(scheduled_actions.HANDLERS, "AUTO_REPLY", handler)

    outcome = _resolve(_action())

    assert outcome == "failed"
    assert worker["failed_terminal"] == []
    assert worker["failed"][0]["max_attempts"] == 8


def test_an_unrecognised_failure_is_treated_as_transient(worker, monkeypatch):
    """Guessing 'permanent' wrongly drops a real message, so the default has to
    be the recoverable one."""

    class SomethingNew(Exception):
        pass

    async def handler(_action):
        raise SomethingNew("never seen before")

    monkeypatch.setitem(scheduled_actions.HANDLERS, "AUTO_REPLY", handler)

    assert _resolve(_action()) == "failed"
    assert worker["failed"][0]["max_attempts"] == 8


def test_a_writer_quality_failure_gets_a_small_budget(worker, monkeypatch):
    async def handler(_action):
        raise WriterQualityFailure("no_confirmed_message", "nothing usable")

    monkeypatch.setitem(scheduled_actions.HANDLERS, "AUTO_REPLY", handler)

    outcome = _resolve(_action())

    assert outcome == "failed_writer_quality"
    budget = worker["failed"][0]["max_attempts"]
    assert budget == scheduled_actions.WRITER_QUALITY_MAX_ATTEMPTS
    assert budget < 8, "the writer budget is not smaller than the transient one"


def test_ppv_reconcile_keeps_its_long_budget(worker, monkeypatch):
    async def handler(_action):
        raise RuntimeError("still pending")

    monkeypatch.setitem(scheduled_actions.HANDLERS, "PPV_RECONCILE", handler)

    _resolve(_action(action_type="PPV_RECONCILE"))

    assert worker["failed"][0]["max_attempts"] == 50


# --- the cost the classification exists to avoid ---------------------------


def test_a_disconnected_creator_never_reaches_the_model_pipeline(
    worker, monkeypatch
):
    """The finding, measured. Previously: eight analyzer+writer runs per fan."""
    db = FakeSupabase({
        "creators": [{"id": CREATOR_ID, "apifansly_account_id": None}],
    })
    monkeypatch.setattr("core.supabase.get_supabase", lambda: db)

    pipeline_runs = []

    async def fake_deliver(action):
        pipeline_runs.append(action["id"])
        return False

    import services.suggestions as suggestions

    monkeypatch.setattr(suggestions, "deliver_scheduled_auto_reply", fake_deliver)

    outcome = _resolve(_action())

    assert outcome == "failed_terminal"
    assert pipeline_runs == [], "the analyzer/writer pipeline ran anyway"
    assert worker["failed_terminal"][0]["code"] == "creator_not_connected"


def test_a_connected_creator_still_runs_normally(worker, monkeypatch):
    """The guard must not stand in front of working creators."""
    db = FakeSupabase({
        "creators": [{"id": CREATOR_ID, "apifansly_account_id": "acct-1"}],
    })
    monkeypatch.setattr("core.supabase.get_supabase", lambda: db)

    import services.suggestions as suggestions

    async def fake_deliver(_action):
        return True

    monkeypatch.setattr(suggestions, "deliver_scheduled_auto_reply", fake_deliver)

    assert _resolve(_action()) == "sent"
    assert worker["failed"] == []
    assert worker["failed_terminal"] == []


def test_a_missing_creator_is_permanent(worker, monkeypatch):
    db = FakeSupabase({"creators": []})
    monkeypatch.setattr("core.supabase.get_supabase", lambda: db)

    assert _resolve(_action()) == "failed_terminal"
    assert worker["failed_terminal"][0]["code"] == "creator_missing"


# --- the marker an operator reads ------------------------------------------


def test_terminal_markers_round_trip() -> None:
    marker = PermanentActionFailure("creator_not_connected", "no account").marker()

    assert is_terminal_error(marker)
    assert terminal_code(marker) == "creator_not_connected"


def test_a_transient_error_is_not_mistaken_for_terminal() -> None:
    assert not is_terminal_error("OpenRouter 503")
    assert not is_terminal_error(None)
    assert terminal_code("OpenRouter 503") == ""


def test_health_separates_broken_bindings_from_provider_trouble() -> None:
    """The operator question REL-005 exists to answer."""
    from services.operational_health import evaluate

    result = evaluate(
        database={"reachable": True},
        queue={
            "available": True,
            "pending": 0,
            "oldest_pending_age_seconds": 0,
            "blocked_by_reason": {"creator_not_connected": 20},
        },
        scheduler={"poll_seconds": 5, "seconds_since_last_cycle": 1,
                   "cycles_completed": 3},
        model_gate={},
        model_availability={"status": "ok"},
    )

    assert result["status"] == "degraded"
    assert "actions_blocked_creator_not_connected:20" in result["degraded_reasons"]
    # And it is NOT reported as provider trouble.
    assert not any(
        reason.startswith("model_") for reason in result["degraded_reasons"]
    )


def test_provider_trouble_is_not_reported_as_a_broken_binding() -> None:
    from services.operational_health import evaluate

    result = evaluate(
        database={"reachable": True},
        queue={"available": True, "pending": 0, "oldest_pending_age_seconds": 0},
        scheduler={"poll_seconds": 5, "seconds_since_last_cycle": 1,
                   "cycles_completed": 3},
        model_gate={},
        model_availability={"status": "unavailable"},
    )

    assert "model_unavailable" in result["degraded_reasons"]
    assert not any(
        reason.startswith("actions_blocked_")
        for reason in result["degraded_reasons"]
    )
