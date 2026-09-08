"""The ten invariants that must survive concurrent action execution.

These are written against the real worker, real gate and real revalidation
predicate. The revalidation logic itself is unchanged by this sprint; what is
new is that it now runs inside a concurrent unit, so each guard is re-asserted
under that condition.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from ai import model_providers
from core.model_gate import MODEL_GATE
from models.commercial import CreatorPolicy, FanCommercialState, FanStatus
from models.model_runtime import ModelResult, ModelTarget, ModelUsage
from models.schemas import Message
from tests.worker_harness import (
    ConcurrencyProbe,
    FakeQueue,
    install,
    make_actions,
    stub_handler,
)
from workers import scheduled_actions as worker

NOW = datetime.now(timezone.utc)


def run(coro):
    return asyncio.run(coro)


def auto_reply_action(fan="fan-1", trigger_at=None, action_id="action-1") -> dict:
    return {
        "id": action_id,
        "creator_id": "creator-1",
        "fan_id": fan,
        "action_type": "AUTO_REPLY",
        "payload": {"trigger_sent_at": (trigger_at or NOW).isoformat()},
        "dedupe_key": f"auto-reply:{fan}:m1",
        "attempts": 0,
        "status": "PENDING",
        "execute_at": (NOW - timedelta(seconds=1)).isoformat(),
    }


def wire_revalidation(
    monkeypatch,
    *,
    fan=None,
    history=None,
    auto_available=True,
    sleep_hours=(0, 0),
):
    from db import queries

    monkeypatch.setattr(
        queries, "get_fan_by_id",
        lambda _id: _value(fan or SimpleNamespace(needs_human_review=False, auto_mode=True)),
    )
    monkeypatch.setattr(
        queries, "get_conversation_history",
        lambda _id, limit=10: _value(list(history or [])),
    )
    monkeypatch.setattr(
        queries, "get_creator_sleep_hours", lambda _id: _value(sleep_hours)
    )
    import main

    monkeypatch.setattr(
        main, "_creator_auto_availability",
        lambda _id: _value({"auto_available": auto_available}),
    )


async def _value(v):
    return v


# 1 --------------------------------------------------------------------------
def test_fan_reply_cancels_a_pending_auto_send(monkeypatch):
    trigger = NOW - timedelta(minutes=2)
    wire_revalidation(
        monkeypatch,
        history=[Message(role="fan", content="still there?", sent_at=NOW)],
    )
    check = run(worker._should_still_send(auto_reply_action(trigger_at=trigger)))
    assert check.ok is False
    assert "newer conversation activity" in check.reason


# 2 --------------------------------------------------------------------------
def test_operator_send_stops_a_pending_auto_send(monkeypatch):
    trigger = NOW - timedelta(minutes=2)
    wire_revalidation(
        monkeypatch,
        history=[Message(role="creator", content="handled by a human", sent_at=NOW)],
    )
    check = run(worker._should_still_send(auto_reply_action(trigger_at=trigger)))
    assert check.ok is False


# 3 --------------------------------------------------------------------------
def test_auto_turned_off_revalidates_to_no_send(monkeypatch):
    wire_revalidation(
        monkeypatch,
        fan=SimpleNamespace(needs_human_review=False, auto_mode=False),
    )
    check = run(worker._should_still_send(auto_reply_action()))
    assert check.ok is False
    assert check.reason == "auto mode off"


def test_no_approved_sets_revalidates_to_no_send(monkeypatch):
    wire_revalidation(monkeypatch, auto_available=False)
    check = run(worker._should_still_send(auto_reply_action()))
    assert check.ok is False
    assert "approved sets" in check.reason


def test_frozen_fan_is_never_auto_messaged(monkeypatch):
    wire_revalidation(
        monkeypatch,
        fan=SimpleNamespace(needs_human_review=True, auto_mode=True),
    )
    check = run(worker._should_still_send(auto_reply_action()))
    assert check.ok is False
    assert "human review" in check.reason


# 4 --------------------------------------------------------------------------
def test_purchase_makes_a_pending_followup_stale(monkeypatch):
    from db import queries

    action = {
        "id": "a1",
        "creator_id": "creator-1",
        "fan_id": "fan-1",
        "action_type": "PAYDAY_REENGAGEMENT",
        "payload": {"payday_at": NOW.isoformat()},
        "dedupe_key": "payday:fan-1",
        "attempts": 0,
    }
    monkeypatch.setattr(
        queries, "get_fan_by_id",
        lambda _id: _value(SimpleNamespace(needs_human_review=False, auto_mode=True)),
    )
    monkeypatch.setattr(
        queries, "get_conversation_history", lambda _id, limit=5: _value([])
    )
    monkeypatch.setattr(queries, "get_creator_sleep_hours", lambda _id: _value((0, 0)))
    # He paid: no longer paused, so the promised follow-up must not fire.
    monkeypatch.setattr(
        worker, "get_fan_state",
        lambda _id: _value(FanCommercialState(
            status=FanStatus.PAID_SESSION_ACTIVE,
            next_followup_type="PAYDAY_REENGAGEMENT",
            next_followup_dedupe_key="payday:fan-1",
        )),
    )
    monkeypatch.setattr(worker, "get_creator_policy", lambda _id: _value(CreatorPolicy()))

    check = run(worker._should_still_send(action))
    assert check.ok is False
    assert "no longer paused" in check.reason


# 5 --------------------------------------------------------------------------
def test_two_cycles_racing_for_one_action_only_let_one_own_it(monkeypatch):
    """The DB compare-and-swap is the owner, not process memory."""
    claimed_by = []
    lock = {"owner": None}

    class CASQueue(FakeQueue):
        async def claim(self, limit=20, stale_minutes=10):
            self.claim_calls += 1
            out = []
            for row in self.rows.values():
                # Exactly the CAS the real query performs: status must still be
                # what we read before the update lands.
                if row["status"] != "PENDING":
                    continue
                if lock["owner"] is not None:
                    continue
                lock["owner"] = self
                row["status"] = "PROCESSING"
                out.append(dict(row))
            return out

    queue = CASQueue(make_actions(1))

    async def handler(action):
        claimed_by.append(action["id"])
        await asyncio.sleep(0.01)
        return worker.HandlerResult(sent_message=True)

    install(monkeypatch, queue=queue, handler=handler)

    async def race():
        return await asyncio.gather(
            worker.process_cycle(concurrency=4, limit=10),
            worker.process_cycle(concurrency=4, limit=10),
        )

    first, second = run(race())
    assert claimed_by == ["action-0"], "one owner only"
    assert first.processed + second.processed == 1


# 6 --------------------------------------------------------------------------
def test_reclaimed_processing_action_reconciles_before_resending(monkeypatch):
    """Preserved from the previous sprint: no duplicate external send on retry."""
    from services import suggestions

    reconciled = []

    async def fake_sync(creator_id, fan_id):
        reconciled.append((creator_id, fan_id))
        return {"status": "ok"}

    import main

    monkeypatch.setattr(main, "sync_recent_fan_messages", fake_sync)
    monkeypatch.setattr(
        suggestions, "get_conversation_history",
        lambda _id, limit=10: _value([
            Message(role="creator", content="already sent", sent_at=NOW + timedelta(minutes=1))
        ]),
    )

    action = auto_reply_action()
    action["status"] = "PROCESSING"
    sent = run(suggestions.deliver_scheduled_auto_reply(action))

    assert reconciled == [("creator-1", "fan-1")]
    # The already-delivered creator message is recognised, so nothing is resent.
    assert sent is True


# 7 --------------------------------------------------------------------------
def test_two_queued_actions_for_one_fan_never_send_simultaneously(monkeypatch):
    queue = FakeQueue(
        [
            auto_reply_action(action_id="action-a"),
            auto_reply_action(action_id="action-b"),
        ]
    )
    probe = ConcurrencyProbe()
    install(monkeypatch, queue=queue, handler=stub_handler(probe, seconds=0.05))

    result = run(worker.process_cycle(concurrency=8, limit=24))

    assert result.processed == 2
    assert probe.same_fan_overlaps == 0
    assert probe.max_inflight == 1


# 8 --------------------------------------------------------------------------
def test_different_fans_process_concurrently(monkeypatch):
    queue = FakeQueue(
        [
            auto_reply_action(fan="fan-1", action_id="a1"),
            auto_reply_action(fan="fan-2", action_id="a2"),
            auto_reply_action(fan="fan-3", action_id="a3"),
        ]
    )
    probe = ConcurrencyProbe()
    install(monkeypatch, queue=queue, handler=stub_handler(probe, seconds=0.05))

    run(worker.process_cycle(concurrency=8, limit=24))
    assert probe.max_inflight == 3


# 9 --------------------------------------------------------------------------
def test_action_cancelled_while_waiting_for_a_model_slot_sends_nothing(monkeypatch):
    """The queued generation must die in the queue, not wake up and send."""
    monkeypatch.setenv("MODEL_MAX_CONCURRENCY", "1")
    MODEL_GATE.reset()
    provider_calls = []
    sends = []

    async def fake_provider(target_, **_kwargs):
        provider_calls.append(target_.model)
        await asyncio.sleep(0.15)
        return ModelResult(text="ok", target=target_, usage=ModelUsage(), latency_ms=0)

    monkeypatch.setattr(model_providers, "_complete_openai_compatible", fake_provider)

    target = ModelTarget(
        name="together:test", provider="together", model="m",
        base_url="https://example.test/v1", api_key_env="TOGETHER_API_KEY",
    )

    async def action_body(label):
        result = await model_providers.complete(
            target, system="s", messages=[{"role": "user", "content": "x"}],
            max_tokens=8,
        )
        sends.append(label)
        return result

    async def scenario():
        holder = asyncio.create_task(action_body("holder"))
        await asyncio.sleep(0.02)
        cancelled = asyncio.create_task(action_body("cancelled"))
        await asyncio.sleep(0.02)
        # The fan replied / the creator took over: the action is obsolete.
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        await holder

    run(asyncio.wait_for(scenario(), timeout=3))

    assert sends == ["holder"]
    assert len(provider_calls) == 1
    assert MODEL_GATE.snapshot()["inflight"] == 0
    MODEL_GATE.reset()


# 10 -------------------------------------------------------------------------
@pytest.mark.parametrize(
    "failure",
    ["success", "exception", "timeout", "cancellation"],
)
def test_model_slot_is_always_released(monkeypatch, failure):
    monkeypatch.setenv("MODEL_MAX_CONCURRENCY", "1")
    MODEL_GATE.reset()

    async def provider(target_, **_kwargs):
        if failure == "exception":
            raise RuntimeError("provider 500")
        if failure in {"timeout", "cancellation"}:
            await asyncio.sleep(5)
        return ModelResult(text="ok", target=target_, usage=ModelUsage(), latency_ms=0)

    monkeypatch.setattr(model_providers, "_complete_openai_compatible", provider)
    target = ModelTarget(
        name="together:test", provider="together", model="m",
        base_url="https://example.test/v1", api_key_env="TOGETHER_API_KEY",
    )

    async def call():
        return await model_providers.complete(
            target, system="s", messages=[{"role": "user", "content": "x"}],
            max_tokens=8,
        )

    async def scenario():
        if failure == "success":
            await call()
        elif failure == "exception":
            with pytest.raises(RuntimeError):
                await call()
        elif failure == "timeout":
            with pytest.raises((asyncio.TimeoutError, TimeoutError)):
                await asyncio.wait_for(call(), timeout=0.05)
        else:
            task = asyncio.create_task(call())
            await asyncio.sleep(0.01)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        # A leaked slot would make this hang instead of completing.
        await asyncio.wait_for(_acquire_and_release(), timeout=1)
        return MODEL_GATE.snapshot()

    snapshot = run(scenario())
    assert snapshot["inflight"] == 0
    MODEL_GATE.reset()


async def _acquire_and_release():
    async with MODEL_GATE.acquire():
        return True
