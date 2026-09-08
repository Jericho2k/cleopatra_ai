"""One correlated timing record per scheduled action.

The question this exists to answer is "why did this Auto reply take 47
seconds?". Before concurrency the answer was almost always "it was behind
something else"; afterwards it can be queue wait, admission wait, provider
latency, or the deliberate human-like composition delay — and those have
completely different remedies.

Stages accumulate into a context-local record so deep call sites (the analyzer,
the writer, the typing simulation, the platform send) can contribute without
threading a parameter through every signature. Exactly one structured log line
is emitted per action; nothing here writes to the database.

No message content, prompt text, fan name, or credential ever enters this
record — only identifiers the operator already has and elapsed milliseconds.
"""

from __future__ import annotations

import contextlib
import json
import time
from contextvars import ContextVar
from dataclasses import dataclass, field

_CURRENT: ContextVar = ContextVar("cleopatra_action_timings", default=None)


@dataclass
class ActionTimings:
    action_id: str
    action_type: str
    fan_id: str
    creator_id: str
    queue_wait_ms: int = 0
    stages: dict[str, float] = field(default_factory=dict)
    started_at: float = field(default_factory=time.perf_counter)
    outcome: str = "unknown"

    def add(self, stage: str, elapsed_ms: float) -> None:
        self.stages[stage] = round(self.stages.get(stage, 0.0) + elapsed_ms, 1)

    def bump(self, stage: str, amount: int = 1) -> None:
        self.stages[stage] = self.stages.get(stage, 0) + amount

    def as_record(self) -> dict:
        total_ms = int((time.perf_counter() - self.started_at) * 1000)
        record = {
            "action_id": self.action_id,
            "action_type": self.action_type,
            "fan_id": self.fan_id,
            "creator_id": self.creator_id,
            "outcome": self.outcome,
            "queue_wait_ms": int(self.queue_wait_ms),
            "total_processing_ms": total_ms,
        }
        for stage, value in self.stages.items():
            record[stage] = int(value)
        return record


def current() -> ActionTimings | None:
    return _CURRENT.get()


def record_stage(stage: str, elapsed_ms: float) -> None:
    """Attribute elapsed time to a stage of the action currently in scope."""
    timings = _CURRENT.get()
    if timings is not None:
        timings.add(stage, elapsed_ms)


def record_count(stage: str, amount: int = 1) -> None:
    timings = _CURRENT.get()
    if timings is not None:
        timings.bump(stage, amount)


@contextlib.contextmanager
def stage(name: str):
    """Time a block and attribute it to ``name`` even if it raises."""
    started = time.perf_counter()
    try:
        yield
    finally:
        record_stage(name, (time.perf_counter() - started) * 1000.0)


@contextlib.contextmanager
def action_scope(timings: ActionTimings):
    """Bind one action's timing record for the duration of its processing."""
    token = _CURRENT.set(timings)
    try:
        yield timings
    finally:
        _CURRENT.reset(token)


def emit(timings: ActionTimings) -> None:
    """Print the single structured record for one completed action."""
    try:
        print(f"[ACTION TIMING] {json.dumps(timings.as_record(), sort_keys=True)}")
    except Exception:  # pragma: no cover - telemetry must never break delivery
        pass
