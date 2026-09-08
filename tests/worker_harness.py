"""Shared stubs for exercising the real scheduled-actions worker in tests.

The point is to run ``workers.scheduled_actions.process_cycle`` itself — its
claiming, its per-fan grouping, its semaphore, its status transitions — with
every external dependency replaced by an in-memory double. A test that drives a
toy coroutine instead would not have caught the sequential loop it replaces.
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone

from workers import scheduled_actions as worker


class FakeQueue:
    """An in-memory stand-in for the scheduled_actions table."""

    def __init__(self, actions: list[dict]):
        self.rows = {str(a["id"]): dict(a) for a in actions}
        self.completed: list[str] = []
        self.failed: list[tuple[str, str]] = []
        self.rescheduled: list[tuple[str, datetime]] = []
        self.claim_calls = 0

    async def claim(self, limit: int = 20, stale_minutes: int = 10) -> list[dict]:
        self.claim_calls += 1
        now = datetime.now(timezone.utc)
        claimed = []
        for row in self.rows.values():
            if len(claimed) >= limit:
                break
            if row.get("status") != "PENDING":
                continue
            execute_at = row.get("execute_at")
            if execute_at and _parse(execute_at) > now:
                continue
            row["status"] = "PROCESSING"
            claimed.append(dict(row))
        return claimed

    async def complete(self, action_id: str) -> None:
        self.completed.append(str(action_id))
        self.rows[str(action_id)]["status"] = "COMPLETED"

    async def fail(self, action_id, error, attempts, max_attempts=3) -> None:
        self.failed.append((str(action_id), str(error)))
        self.rows[str(action_id)]["status"] = "FAILED"

    async def reschedule(self, action_id, execute_at, payload=None) -> None:
        self.rescheduled.append((str(action_id), execute_at))
        self.rows[str(action_id)]["status"] = "PENDING"


def _parse(value) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


class ConcurrencyProbe:
    """Records true simultaneity, plus per-fan simultaneity separately."""

    def __init__(self) -> None:
        self.inflight = 0
        self.max_inflight = 0
        self.per_fan_inflight: dict[str, int] = {}
        self.same_fan_overlaps = 0
        self.completed: list[str] = []

    def enter(self, fan_id: str) -> None:
        self.inflight += 1
        self.max_inflight = max(self.max_inflight, self.inflight)
        count = self.per_fan_inflight.get(fan_id, 0) + 1
        self.per_fan_inflight[fan_id] = count
        if count > 1:
            self.same_fan_overlaps += 1

    def exit(self, fan_id: str) -> None:
        self.inflight -= 1
        self.per_fan_inflight[fan_id] = self.per_fan_inflight.get(fan_id, 1) - 1


def make_actions(count: int, *, action_type="AUTO_REPLY", fans: int | None = None) -> list[dict]:
    fan_count = count if fans is None else fans
    past = datetime.now(timezone.utc) - timedelta(seconds=5)
    return [
        {
            "id": f"action-{i}",
            "creator_id": "creator-1",
            "fan_id": f"fan-{i % fan_count}",
            "action_type": action_type,
            "payload": {},
            "dedupe_key": f"{action_type}:fan-{i % fan_count}:{i}",
            "attempts": 0,
            "status": "PENDING",
            "execute_at": past.isoformat(),
        }
        for i in range(count)
    ]


def install(
    monkeypatch,
    *,
    queue: FakeQueue,
    handler,
    action_type: str = "AUTO_REPLY",
    revalidate=None,
) -> None:
    """Point the real worker at in-memory doubles."""
    monkeypatch.setattr(worker, "repair_followup_obligations", _zero)
    monkeypatch.setattr(worker, "claim_due_actions", queue.claim)
    monkeypatch.setattr(worker, "complete_action", queue.complete)
    monkeypatch.setattr(worker, "fail_action", queue.fail)
    monkeypatch.setattr(worker, "reschedule_action", queue.reschedule)
    monkeypatch.setattr(worker, "_record_message_action_resolution", _noop2)
    monkeypatch.setattr(worker, "_record_followup_postponed", _noop2)
    monkeypatch.setattr(
        worker,
        "_should_still_send",
        revalidate or (lambda _action: _ok()),
    )
    monkeypatch.setitem(worker.HANDLERS, action_type, handler)


async def _zero(**_kwargs) -> int:
    return 0


async def _noop2(*_args, **_kwargs) -> None:
    return None


async def _ok():
    return worker.ActionCheck(True)


def stub_handler(probe: ConcurrencyProbe, *, seconds: float = 0.05, sends: bool = True):
    """A handler that occupies a slot for a measurable time, like a real send."""

    async def _handler(action: dict):
        fan_id = str(action["fan_id"])
        probe.enter(fan_id)
        try:
            await asyncio.sleep(seconds)
            probe.completed.append(str(action["id"]))
            return worker.HandlerResult(sent_message=sends, reason="stub")
        finally:
            probe.exit(fan_id)

    return _handler


def elapsed(fn):
    started = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - started
