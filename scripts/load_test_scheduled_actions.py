"""Synthetic load harness for the scheduled-actions worker.

NON-PRODUCTION. Nothing here touches live Fansly, a live creator, a real
database, or a paid provider. External dependencies are replaced with stubs that
have configurable latency, and the code under test is the REAL worker —
``process_cycle``, its per-fan grouping, its semaphore, and the real global model
gate in ``core.model_gate``. A harness that drove a toy coroutine instead would
not have detected the sequential loop this sprint removed.

What each synthetic action does, in order:

    revalidate (DB latency)
      -> analyzer  (model gate + model latency)
      -> writer    (model gate + model latency)
      -> composition delay   <- deliberate human realism, kept
      -> platform send (API latency)
      -> persist (DB latency)

Run::

    python scripts/load_test_scheduled_actions.py
    python scripts/load_test_scheduled_actions.py --sizes 10 50 100 --model-latency 2.0
    python scripts/load_test_scheduled_actions.py --sequential   # the old shape
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

for _key, _value in {
    "SUPABASE_URL": "https://synthetic.invalid",
    "SUPABASE_SERVICE_KEY": "synthetic",
    "TOGETHER_API_KEY": "synthetic",
    "UPSTASH_REDIS_URL": "https://synthetic.invalid",
    "UPSTASH_REDIS_TOKEN": "synthetic",
    "OPENAI_API_KEY": "synthetic",
    "ANTHROPIC_API_KEY": "synthetic",
    "APIFANSLY_API_KEY": "synthetic",
    "FANSLY_SESSION_KEY": "synthetic",
    "DASHBOARD_API_SECRET": "synthetic",
    "WEBHOOK_SECRET": "synthetic",
    "APP_ENV": "test",
    "MODEL_TELEMETRY_ENABLED": "false",
}.items():
    os.environ.setdefault(_key, _value)

from ai import model_providers
from core.model_gate import MODEL_GATE
from models.model_runtime import ModelResult, ModelTarget, ModelUsage
from workers import scheduled_actions as worker

TARGET = ModelTarget(
    name="synthetic:writer",
    provider="together",
    model="synthetic-writer",
    base_url="https://synthetic.invalid/v1",
    api_key_env="TOGETHER_API_KEY",
)


@dataclass
class Scenario:
    """Everything the outside world does, and how slowly."""

    model_latency: float = 2.0
    db_latency: float = 0.004
    api_latency: float = 0.15
    composition_delay: float = 0.4
    model_failure_ids: set[str] = field(default_factory=set)
    rate_limited_ids: set[str] = field(default_factory=set)
    cancel_ids: set[str] = field(default_factory=set)
    superseded_fans: set[str] = field(default_factory=set)


@dataclass
class Metrics:
    db_calls: int = 0
    model_calls: int = 0
    model_failures: int = 0
    rate_limit_retries: int = 0
    sends: list[str] = field(default_factory=list)
    completions: list[float] = field(default_factory=list)
    errors: int = 0
    skipped: int = 0
    action_inflight: int = 0
    max_action_concurrency: int = 0
    per_fan_inflight: dict = field(default_factory=dict)
    duplicate_sends: int = 0
    same_fan_overlaps: int = 0

    def enter(self, fan_id: str) -> None:
        self.action_inflight += 1
        self.max_action_concurrency = max(
            self.max_action_concurrency, self.action_inflight
        )
        count = self.per_fan_inflight.get(fan_id, 0) + 1
        self.per_fan_inflight[fan_id] = count
        if count > 1:
            self.same_fan_overlaps += 1

    def exit(self, fan_id: str) -> None:
        self.action_inflight -= 1
        self.per_fan_inflight[fan_id] = self.per_fan_inflight.get(fan_id, 1) - 1


class ModelProbe:
    """Stands in for the provider, and reports true simultaneity at the wire."""

    def __init__(self, scenario: Scenario, metrics: Metrics):
        self.scenario = scenario
        self.metrics = metrics
        self.inflight = 0
        self.max_inflight = 0
        self.current_action: str = ""

    async def __call__(self, target, **_kwargs):
        action_id = self.current_action
        self.metrics.model_calls += 1
        self.inflight += 1
        self.max_inflight = max(self.max_inflight, self.inflight)
        try:
            await asyncio.sleep(self.scenario.model_latency)
            if action_id in self.scenario.model_failure_ids:
                self.metrics.model_failures += 1
                raise RuntimeError("synthetic provider 500")
            if action_id in self.scenario.rate_limited_ids:
                self.metrics.rate_limit_retries += 1
                self.scenario.rate_limited_ids.discard(action_id)
                raise RuntimeError("429 Too Many Requests")
            return ModelResult(
                text='["one","two","three"]',
                target=target,
                usage=ModelUsage(input_tokens=100, output_tokens=30),
                latency_ms=int(self.scenario.model_latency * 1000),
            )
        finally:
            self.inflight -= 1


class SyntheticQueue:
    def __init__(self, actions: list[dict]):
        self.rows = {str(a["id"]): dict(a) for a in actions}
        self.claim_calls = 0
        self.completed: list[str] = []
        self.failed: list[str] = []

    async def claim(self, limit: int = 20, stale_minutes: int = 10) -> list[dict]:
        self.claim_calls += 1
        now = datetime.now(timezone.utc)
        out = []
        for row in self.rows.values():
            if len(out) >= limit:
                break
            if row["status"] != "PENDING":
                continue
            if datetime.fromisoformat(row["execute_at"]) > now:
                continue
            row["status"] = "PROCESSING"
            out.append(dict(row))
        return out

    def remaining(self) -> int:
        return sum(1 for r in self.rows.values() if r["status"] == "PENDING")

    async def complete(self, action_id):
        self.completed.append(str(action_id))
        self.rows[str(action_id)]["status"] = "COMPLETED"

    async def fail(self, action_id, error, attempts, max_attempts=3):
        self.failed.append(str(action_id))
        self.rows[str(action_id)]["status"] = "FAILED"

    async def reschedule(self, action_id, execute_at, payload=None):
        self.rows[str(action_id)]["status"] = "PENDING"
        self.rows[str(action_id)]["execute_at"] = execute_at.isoformat()


def build_actions(count: int) -> list[dict]:
    past = datetime.now(timezone.utc) - timedelta(seconds=1)
    return [
        {
            "id": f"action-{i}",
            "creator_id": f"creator-{i % 10}",
            "fan_id": f"fan-{i}",
            "action_type": "AUTO_REPLY",
            "payload": {"trigger_sent_at": past.isoformat()},
            "dedupe_key": f"auto-reply:fan-{i}:m1",
            "attempts": 0,
            "status": "PENDING",
            "execute_at": past.isoformat(),
        }
        for i in range(count)
    ]


class Harness:
    def __init__(self, scenario: Scenario, metrics: Metrics):
        self.scenario = scenario
        self.metrics = metrics
        self.probe = ModelProbe(scenario, metrics)
        self.sent_fans: set[str] = set()

    async def db(self, calls: int = 1) -> None:
        self.metrics.db_calls += calls
        await asyncio.sleep(self.scenario.db_latency * calls)

    async def revalidate(self, action: dict):
        # Four reads, matching the audited _should_still_send shape.
        await self.db(4)
        if str(action["fan_id"]) in self.scenario.superseded_fans:
            # The fan sent a newer message while this action sat in the queue.
            return worker.ActionCheck(False, "newer conversation activity")
        return worker.ActionCheck(True)

    async def model_call(self, action_id: str):
        self.probe.current_action = action_id
        return await model_providers.complete(
            TARGET,
            system="synthetic system prompt",
            messages=[{"role": "user", "content": "synthetic"}],
            max_tokens=64,
        )

    async def handler(self, action: dict):
        action_id = str(action["id"])
        fan_id = str(action["fan_id"])
        started = time.perf_counter()
        self.metrics.enter(fan_id)
        try:
            if action_id in self.scenario.cancel_ids:
                # The action became obsolete while queued behind the gate.
                raise asyncio.CancelledError

            await self.db(20)                      # context refreshers
            await self.model_call(action_id)       # analyzer
            await self.model_call(action_id)       # writer
            await self.db(4)                       # post-generation re-check

            # Deliberate human realism. Kept, and measured separately.
            await asyncio.sleep(self.scenario.composition_delay)

            await asyncio.sleep(self.scenario.api_latency)   # platform send
            if fan_id in self.sent_fans:
                self.metrics.duplicate_sends += 1
            self.sent_fans.add(fan_id)
            self.metrics.sends.append(action_id)

            await self.db(2)                       # persist the sent message
            self.metrics.completions.append(time.perf_counter() - started)
            return worker.HandlerResult(sent_message=True, reason="synthetic")
        finally:
            self.metrics.exit(fan_id)


async def _repair_noop(**_kwargs) -> int:
    return 0


async def drain(
    *,
    size: int,
    concurrency: int,
    claim_limit: int,
    model_concurrency: int,
    scenario: Scenario,
) -> dict:
    """Drain a synthetic backlog the way the live loop would, and measure it."""
    os.environ["MODEL_MAX_CONCURRENCY"] = str(model_concurrency)
    MODEL_GATE.reset()

    metrics = Metrics()
    harness = Harness(scenario, metrics)
    queue = SyntheticQueue(build_actions(size))

    saved = {
        name: getattr(worker, name)
        for name in (
            "repair_followup_obligations",
            "claim_due_actions",
            "complete_action",
            "fail_action",
            "reschedule_action",
            "_record_message_action_resolution",
            "_record_followup_postponed",
            "_should_still_send",
        )
    }
    saved_handler = worker.HANDLERS.get("AUTO_REPLY")
    saved_provider = model_providers._complete_openai_compatible

    worker.repair_followup_obligations = _repair_noop
    worker.claim_due_actions = queue.claim
    worker.complete_action = queue.complete
    worker.fail_action = queue.fail
    worker.reschedule_action = queue.reschedule
    worker._record_message_action_resolution = _noop
    worker._record_followup_postponed = _noop
    worker._should_still_send = harness.revalidate
    worker.HANDLERS["AUTO_REPLY"] = harness.handler
    model_providers._complete_openai_compatible = harness.probe

    started = time.perf_counter()
    cycles = 0
    try:
        while queue.remaining() and cycles < 200:
            cycles += 1
            await worker.process_cycle(
                run_repair=False, concurrency=concurrency, limit=claim_limit
            )
            if queue.remaining():
                # Exactly what the live loop does after a full claim.
                await asyncio.sleep(worker.BUSY_POLL_SECONDS)
        drain_seconds = time.perf_counter() - started
    finally:
        for name, value in saved.items():
            setattr(worker, name, value)
        if saved_handler is not None:
            worker.HANDLERS["AUTO_REPLY"] = saved_handler
        model_providers._complete_openai_compatible = saved_provider

    completions = sorted(metrics.completions)
    return {
        "size": size,
        "action_concurrency_limit": concurrency,
        "claim_limit": claim_limit,
        "model_concurrency_limit": model_concurrency,
        "cycles": cycles,
        "claim_calls": queue.claim_calls,
        "drain_seconds": round(drain_seconds, 2),
        "actions_per_minute": round(size / drain_seconds * 60, 1) if drain_seconds else 0,
        "max_action_concurrency": metrics.max_action_concurrency,
        "max_model_concurrency": harness.probe.max_inflight,
        "p50_completion_seconds": round(_pct(completions, 0.50), 2),
        "p95_completion_seconds": round(_pct(completions, 0.95), 2),
        "sends": len(metrics.sends),
        "duplicate_sends": metrics.duplicate_sends,
        "same_fan_overlaps": metrics.same_fan_overlaps,
        "errors": len(queue.failed),
        "skipped": len([r for r in queue.rows.values() if r["status"] == "COMPLETED"]) - len(metrics.sends),
        "db_calls": metrics.db_calls,
        "model_calls": metrics.model_calls,
        "model_failures": metrics.model_failures,
        "rate_limit_retries": metrics.rate_limit_retries,
        "composition_delay_seconds_per_action": scenario.composition_delay,
    }


def _pct(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    index = max(0, min(len(values) - 1, round(fraction * len(values)) - 1))
    return values[index]


async def _noop(*_args, **_kwargs):
    return None


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[10, 50, 100])
    parser.add_argument("--concurrency", type=int, default=worker.DEFAULT_CONCURRENCY)
    parser.add_argument("--claim-limit", type=int, default=worker.DEFAULT_CLAIM_LIMIT)
    parser.add_argument("--model-concurrency", type=int, default=8)
    parser.add_argument("--model-latency", type=float, default=2.0)
    parser.add_argument("--composition-delay", type=float, default=0.4)
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Reproduce the pre-sprint shape: concurrency 1, claim 20.",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    concurrency = 1 if args.sequential else args.concurrency
    claim_limit = 20 if args.sequential else args.claim_limit
    model_concurrency = 1 if args.sequential else args.model_concurrency

    rows = []
    for size in args.sizes:
        scenario = Scenario(
            model_latency=args.model_latency,
            composition_delay=args.composition_delay,
        )
        rows.append(
            await drain(
                size=size,
                concurrency=concurrency,
                claim_limit=claim_limit,
                model_concurrency=model_concurrency,
                scenario=scenario,
            )
        )

    if args.json:
        print(json.dumps(rows, indent=2))
        return

    label = "SEQUENTIAL (before)" if args.sequential else "BOUNDED CONCURRENT (after)"
    print(f"\n=== {label} ===")
    print(
        f"{'n':>5} {'drain s':>9} {'act/min':>9} {'p50 s':>7} {'p95 s':>7} "
        f"{'maxAct':>7} {'maxLLM':>7} {'dupes':>6} {'errs':>5} {'dbCalls':>8}"
    )
    for row in rows:
        print(
            f"{row['size']:>5} {row['drain_seconds']:>9} {row['actions_per_minute']:>9} "
            f"{row['p50_completion_seconds']:>7} {row['p95_completion_seconds']:>7} "
            f"{row['max_action_concurrency']:>7} {row['max_model_concurrency']:>7} "
            f"{row['duplicate_sends']:>6} {row['errors']:>5} {row['db_calls']:>8}"
        )
    print(
        "\nSynthetic numbers with stubbed providers. Not a production SLA claim."
    )


if __name__ == "__main__":
    asyncio.run(main())
