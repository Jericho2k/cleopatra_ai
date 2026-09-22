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
from types import SimpleNamespace

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
            "_acquire_fan_slot",
            "_release_fan_slot",
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
    # The durable per-fan lease is a database round trip; this harness has no
    # database. The in-memory equivalent asserts the same invariant, and the
    # real claim is measured against a real PostgreSQL in
    # tests/test_conversation_supersession_schema.py.
    held: dict = {}

    async def _acquire(action: dict):
        fan_id = str(action.get("fan_id") or "")
        if not fan_id:
            return True, ""
        token = f"synthetic:{action.get('id')}"
        owner = held.get(fan_id)
        if owner is not None and owner != token:
            return False, ""
        held[fan_id] = token
        return True, token

    async def _release(action: dict, token: str) -> None:
        fan_id = str(action.get("fan_id") or "")
        if token and held.get(fan_id) == token:
            held.pop(fan_id, None)

    worker._acquire_fan_slot = _acquire
    worker._release_fan_slot = _release
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
    parser.add_argument(
        "--scale",
        action="store_true",
        help="run the conversation-scale harness instead of the action sweep",
    )
    parser.add_argument("--creators", type=int, default=100)
    parser.add_argument("--fans-per-creator", type=int, default=12)
    parser.add_argument("--interrupt-fraction", type=float, default=0.25)
    args = parser.parse_args()

    if args.scale:
        result = await drain_conversations(
            ConversationScenario(
                creators=args.creators,
                fans_per_creator=args.fans_per_creator,
                interrupt_fraction=args.interrupt_fraction,
                model_latency=args.model_latency,
            )
        )
        print(json.dumps(result, indent=2))
        return

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



# ---------------------------------------------------------------------------
# Conversation-scale harness
#
# The sweep above measures ONE action type draining through the worker. It was
# the right question while a reply was one indivisible unit of work. It is no
# longer the whole question, because a reply is now a planned outbound sequence
# whose bubbles are separate durable actions with their own due times, and
# because the property we most need to hold at scale is not throughput but
# "no stale bubble ever leaves".
#
# So this second harness drives, for many creators and many fans at once:
#
#   AUTO_REPLY  ->  two gated model calls
#               ->  the REAL services/human_delivery.py schedule
#               ->  the REAL services/outbound_delivery.py planner
#               ->  N durable DELIVER_OUTBOUND_PART actions
#               ->  the REAL send gate on each one
#
# and interrupts a configurable share of those conversations mid-sequence by
# advancing the fan's conversation generation — which is exactly what a fan
# message handled by ANOTHER process does. Nothing about the interruption is
# delivered through this process's memory, which is the point.
#
# Time is compressed by a scale factor so a 22-second composition pause costs
# milliseconds. The PLAN is the real one; only the clock is small.
# ---------------------------------------------------------------------------


@dataclass
class ConversationScenario:
    creators: int = 100
    fans_per_creator: int = 12
    max_bubbles: int = 3
    burst_messages: int = 2
    # Share of conversations whose fan replies while bubbles are still queued.
    interrupt_fraction: float = 0.25
    model_latency: float = 0.02
    db_latency: float = 0.0002
    api_latency: float = 0.002
    # Human-like seconds are multiplied by this before becoming due times.
    time_scale: float = 0.01
    claim_limit: int = 24
    seed: int = 20260922


@dataclass
class ConversationMetrics:
    planned_sequences: int = 0
    bubbles_sent: int = 0
    duplicate_sends: int = 0
    stale_sends: int = 0
    concurrent_generation_bumps: int = 0
    superseded_sequences: int = 0
    same_fan_overlaps: int = 0
    model_calls: int = 0
    max_model_concurrency: int = 0
    max_action_concurrency: int = 0
    queue_waits: list = field(default_factory=list)
    time_to_first_bubble: list = field(default_factory=list)
    queue_depth_samples: list = field(default_factory=list)
    pending_delay_actions_peak: int = 0
    model_busy_seconds: float = 0.0
    worker_busy_seconds: float = 0.0


class ScaleQueue:
    """A durable queue double that accepts work created DURING a cycle.

    The original SyntheticQueue was built from a fixed list, which cannot model
    the shape under test: planning a reply enqueues its own bubbles.
    """

    def __init__(self) -> None:
        self.rows: dict = {}
        self.by_dedupe: dict = {}
        self.claim_calls = 0
        self.completed: list = []
        self.failed: list = []
        self._next_id = 0

    def add(
        self,
        *,
        creator_id: str,
        fan_id: str,
        action_type: str,
        execute_at: datetime,
        payload: dict,
        dedupe_key: str,
        replace_existing: bool = True,
    ) -> None:
        existing = self.by_dedupe.get(dedupe_key)
        if existing is not None and not replace_existing:
            return
        self._next_id += 1
        action_id = f"a{self._next_id}"
        row = {
            "id": action_id,
            "creator_id": creator_id,
            "fan_id": fan_id,
            "action_type": action_type,
            "payload": payload,
            "dedupe_key": dedupe_key,
            "attempts": 0,
            "status": "PENDING",
            "execute_at": execute_at.isoformat(),
        }
        if existing is not None:
            self.rows.pop(existing, None)
        self.rows[action_id] = row
        self.by_dedupe[dedupe_key] = action_id

    async def claim(self, limit: int = 20, stale_minutes: int = 10) -> list:
        self.claim_calls += 1
        now = datetime.now(timezone.utc)
        out = []
        for row in sorted(self.rows.values(), key=lambda r: r["execute_at"]):
            if len(out) >= limit:
                break
            if row["status"] != "PENDING":
                continue
            if _parse_iso(row["execute_at"]) > now:
                continue
            row["status"] = "PROCESSING"
            out.append(dict(row))
        return out

    def pending(self) -> int:
        return sum(1 for r in self.rows.values() if r["status"] == "PENDING")

    def pending_of_type(self, action_type: str) -> int:
        return sum(
            1
            for r in self.rows.values()
            if r["status"] == "PENDING" and r["action_type"] == action_type
        )

    def due_now(self) -> int:
        now = datetime.now(timezone.utc)
        return sum(
            1
            for r in self.rows.values()
            if r["status"] == "PENDING" and _parse_iso(r["execute_at"]) <= now
        )

    def seconds_until_next_due(self) -> float:
        """The same adaptive hint the real dispatcher asks the database for."""
        now = datetime.now(timezone.utc)
        soonest = None
        for row in self.rows.values():
            if row["status"] != "PENDING":
                continue
            due = _parse_iso(row["execute_at"])
            if soonest is None or due < soonest:
                soonest = due
        if soonest is None:
            return 0.05
        return max(0.0, (soonest - now).total_seconds())

    async def complete(self, action_id):
        self.completed.append(str(action_id))
        self.rows[str(action_id)]["status"] = "COMPLETED"

    async def fail(self, action_id, error, attempts, max_attempts=3):
        self.failed.append(f"{action_id}: {error}")
        self.rows[str(action_id)]["status"] = "FAILED"

    async def reschedule(self, action_id, execute_at, payload=None):
        row = self.rows[str(action_id)]
        row["status"] = "PENDING"
        row["execute_at"] = execute_at.isoformat()

    async def cancel_by_dedupe(self, dedupe_key: str) -> None:
        action_id = self.by_dedupe.get(dedupe_key)
        if action_id and self.rows.get(action_id, {}).get("status") == "PENDING":
            self.rows[action_id]["status"] = "CANCELLED"


def _parse_iso(value) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


class SequenceStore:
    """In-memory stand-in for outbound_sequences / outbound_sequence_parts."""

    def __init__(self) -> None:
        self.sequences: dict = {}
        self.parts: dict = {}
        self._next = 0

    async def create_sequence(
        self,
        *,
        creator_id,
        fan_id,
        conversation_generation,
        trigger_identity,
        turn_id,
        parts,
        planned_timing,
        metadata=None,
    ):
        from db import outbound_queries as store

        key = (str(fan_id), str(trigger_identity))
        if key in self.sequences:
            return await self.get_sequence(self.sequences[key])
        self._next += 1
        sequence_id = f"seq{self._next}"
        self.sequences[key] = sequence_id
        self.parts[sequence_id] = {
            "header": {
                "id": sequence_id,
                "creator_id": str(creator_id),
                "fan_id": str(fan_id),
                "conversation_generation": int(conversation_generation),
                "trigger_identity": str(trigger_identity),
                "turn_id": str(turn_id),
                "status": store.STATUS_PLANNED,
                "cancel_reason": "",
                "planned_timing": planned_timing,
                "metadata": dict(metadata or {}),
            },
            "parts": [
                {
                    "id": f"{sequence_id}-{part['part_index']}",
                    "part_index": int(part["part_index"]),
                    "body": part["body"],
                    "due_at": part["due_at"],
                    "planned_delay_seconds": part.get("planned_delay_seconds", 0.0),
                    "status": store.PART_PENDING,
                    "platform_message_id": "",
                    "sent_at": None,
                }
                for part in parts
            ],
        }
        return await self.get_sequence(sequence_id)

    async def get_sequence(self, sequence_id):
        from db import outbound_queries as store

        record = self.parts.get(str(sequence_id))
        if record is None:
            return None
        return store._sequence_from_rows(record["header"], record["parts"])

    async def get_sequence_by_trigger(self, fan_id, trigger_identity):
        sequence_id = self.sequences.get((str(fan_id), str(trigger_identity)))
        return await self.get_sequence(sequence_id) if sequence_id else None

    async def active_sequences_for_fan(self, fan_id):
        from db import outbound_queries as store

        out = []
        for sequence_id, record in self.parts.items():
            header = record["header"]
            if header["fan_id"] == str(fan_id) and header["status"] in (
                store.ACTIVE_STATUSES
            ):
                sequence = await self.get_sequence(sequence_id)
                if sequence is not None:
                    out.append(sequence)
        return out

    async def set_sequence_status(self, sequence_id, status, *, reason=""):
        record = self.parts.get(str(sequence_id))
        if record is not None:
            record["header"]["status"] = status
            record["header"]["cancel_reason"] = reason

    async def supersede_pending_parts(self, sequence_id, *, reason):
        from db import outbound_queries as store

        record = self.parts.get(str(sequence_id))
        if record is None:
            return
        for part in record["parts"]:
            if part["status"] == store.PART_PENDING:
                part["status"] = store.PART_SUPERSEDED
        await self.set_sequence_status(
            sequence_id, store.STATUS_SUPERSEDED, reason=reason
        )

    async def record_part_sent(
        self, *, sequence_id, part_index, platform_message_id, message_id
    ):
        from db import outbound_queries as store

        record = self.parts.get(str(sequence_id))
        if record is None:
            return
        for part in record["parts"]:
            if part["part_index"] == int(part_index):
                part["status"] = store.PART_SENT
                part["platform_message_id"] = platform_message_id
                part["sent_at"] = datetime.now(timezone.utc)


async def drain_conversations(scenario: ConversationScenario) -> dict:
    """Run many creators' conversations through the real delivery machinery."""
    import random as _random

    from db import outbound_queries as store
    from db import queries as db_queries
    from services import outbound_delivery
    from services.human_delivery import DeliverySchedule, build_delivery_schedule

    rng = _random.Random(scenario.seed)
    claim_limit = max(24, scenario.claim_limit)
    metrics = ConversationMetrics()
    queue = ScaleQueue()
    sequences = SequenceStore()
    generations: dict = {}
    # What the send gate last OBSERVED for each fan. A bubble is stale if it
    # left after the gate could have known the conversation had moved on; a fan
    # message that lands during the send itself is concurrent, not stale, and is
    # counted separately rather than hidden.
    gate_generations: dict = {}
    sent_bubbles: set = set()
    turn_started: dict = {}
    per_fan_inflight: dict = {}

    os.environ["MODEL_MAX_CONCURRENCY"] = "8"
    MODEL_GATE.reset()

    fans = []
    for creator in range(scenario.creators):
        count = scenario.fans_per_creator + rng.randint(0, 8)
        for index in range(count):
            fan_id = f"c{creator}-f{index}"
            fans.append((f"creator-{creator}", fan_id))
            generations[fan_id] = 1
    interrupted = {
        fan_id
        for _creator, fan_id in fans
        if rng.random() < scenario.interrupt_fraction
    }

    async def db(calls: int = 1) -> None:
        await asyncio.sleep(scenario.db_latency * calls)

    model_inflight = [0]

    async def model_call() -> None:
        async with MODEL_GATE.acquire(feature="synthetic_conversation"):
            metrics.model_calls += 1
            model_inflight[0] += 1
            metrics.max_model_concurrency = max(
                metrics.max_model_concurrency, model_inflight[0]
            )
            started = time.perf_counter()
            try:
                await asyncio.sleep(scenario.model_latency)
            finally:
                metrics.model_busy_seconds += time.perf_counter() - started
                model_inflight[0] -= 1

    def scaled(schedule: DeliverySchedule) -> DeliverySchedule:
        factor = scenario.time_scale
        return DeliverySchedule(
            availability_delay_seconds=schedule.availability_delay_seconds * factor,
            composition_delay_seconds=schedule.composition_delay_seconds * factor,
            inter_part_delays_seconds=tuple(
                value * factor for value in schedule.inter_part_delays_seconds
            ),
            availability_mode=schedule.availability_mode,
        )

    async def auto_reply(action: dict):
        fan_id = str(action["fan_id"])
        turn_started[fan_id] = time.perf_counter()
        await db(20)
        await model_call()      # GLM semantic decision
        await model_call()      # Kimi fan-facing writer
        bubbles = [
            "x" * rng.randint(20, 120)
            for _ in range(rng.randint(1, scenario.max_bubbles))
        ]
        schedule = scaled(
            build_delivery_schedule(
                "hey are you around",
                bubbles,
                conversation_history=[],
                active_session=None,
            )
        )
        sequence = await outbound_delivery.schedule_outbound_sequence(
            creator_id=str(action["creator_id"]),
            fan_id=fan_id,
            trigger_identity=str(action["dedupe_key"]),
            turn_id=str(action["id"]),
            parts=bubbles,
            schedule=schedule,
            conversation_generation=generations[fan_id],
            metadata=outbound_delivery.sequence_metadata(
                message_metadata={"synthetic": True}
            ),
        )
        if sequence is not None:
            metrics.planned_sequences += 1
        return worker.HandlerResult(sent_message=False, reason="reply planned")

    async def deliver_part(action: dict):
        outcome = await outbound_delivery.deliver_due_part(action)
        if outcome.retry_at is not None:
            return worker.HandlerResult(retry_at=outcome.retry_at, reason=outcome.reason)
        return worker.HandlerResult(sent_message=outcome.sent, reason=outcome.reason)

    async def fake_save_message(
        fan_id, creator_id, role, content, **kwargs
    ):
        platform_id = str(kwargs.get("fansly_message_id") or "")
        context = kwargs.get("media_context") or {}
        if platform_id in sent_bubbles:
            metrics.duplicate_sends += 1
        sent_bubbles.add(platform_id)
        planned = int(context.get("conversation_generation") or 0)
        if planned != gate_generations.get(str(fan_id), planned):
            metrics.stale_sends += 1
        elif planned != generations.get(str(fan_id), planned):
            metrics.concurrent_generation_bumps += 1
        metrics.bubbles_sent += 1
        if int(context.get("part") or 0) == 0 and fan_id in turn_started:
            metrics.time_to_first_bubble.append(
                time.perf_counter() - turn_started[str(fan_id)]
            )
        await db(2)
        return f"m-{platform_id}"

    async def fake_route(creator_id, fan_id):
        await asyncio.sleep(scenario.api_latency)
        return "", "", True

    async def fake_generation(fan_id):
        value = generations.get(str(fan_id), 0)
        gate_generations[str(fan_id)] = value
        return value

    async def fake_get_fan(fan_id):
        return SimpleNamespace(
            id=str(fan_id),
            needs_human_review=False,
            auto_mode=True,
            platform_fan_id=f"test_{fan_id}",
            fansly_group_id="",
        )

    async def fake_schedule_action(**kwargs):
        queue.add(
            creator_id=str(kwargs["creator_id"]),
            fan_id=str(kwargs["fan_id"]),
            action_type=str(kwargs["action_type"]),
            execute_at=kwargs["execute_at"],
            payload=dict(kwargs.get("payload") or {}),
            dedupe_key=str(kwargs["dedupe_key"]),
            replace_existing=bool(kwargs.get("replace_existing", True)),
        )

    async def revalidate(action: dict):
        await db(4)
        return worker.ActionCheck(True)

    async def lease(action: dict):
        fan_id = str(action.get("fan_id") or "")
        count = per_fan_inflight.get(fan_id, 0) + 1
        per_fan_inflight[fan_id] = count
        if count > 1:
            metrics.same_fan_overlaps += 1
        metrics.max_action_concurrency = max(
            metrics.max_action_concurrency, sum(per_fan_inflight.values())
        )
        return True, f"token:{action.get('id')}"

    async def release(action, token):
        fan_id = str(action.get("fan_id") or "")
        per_fan_inflight[fan_id] = max(0, per_fan_inflight.get(fan_id, 1) - 1)

    saved_worker = {
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
            "_acquire_fan_slot",
            "_release_fan_slot",
        )
    }
    saved_handlers = dict(worker.HANDLERS)
    saved_store = {
        name: getattr(store, name)
        for name in (
            "create_sequence",
            "get_sequence",
            "get_sequence_by_trigger",
            "active_sequences_for_fan",
            "set_sequence_status",
            "supersede_pending_parts",
            "record_part_sent",
        )
    }
    saved_delivery = {
        name: getattr(outbound_delivery, name)
        for name in ("current_generation", "save_message", "_delivery_route",
                     "schedule_action", "cancel_action_by_dedupe_key")
    }
    saved_get_fan = db_queries.get_fan_by_id

    worker.repair_followup_obligations = _repair_noop
    worker.claim_due_actions = queue.claim
    worker.complete_action = queue.complete
    worker.fail_action = queue.fail
    worker.reschedule_action = queue.reschedule
    worker._record_message_action_resolution = _noop
    worker._record_followup_postponed = _noop
    worker._should_still_send = revalidate
    worker._acquire_fan_slot = lease
    worker._release_fan_slot = release
    worker.HANDLERS["AUTO_REPLY"] = auto_reply
    worker.HANDLERS["DELIVER_OUTBOUND_PART"] = deliver_part

    store.create_sequence = sequences.create_sequence
    store.get_sequence = sequences.get_sequence
    store.get_sequence_by_trigger = sequences.get_sequence_by_trigger
    store.active_sequences_for_fan = sequences.active_sequences_for_fan
    store.set_sequence_status = sequences.set_sequence_status
    store.supersede_pending_parts = sequences.supersede_pending_parts
    store.record_part_sent = sequences.record_part_sent

    outbound_delivery.current_generation = fake_generation
    outbound_delivery.save_message = fake_save_message
    outbound_delivery._delivery_route = fake_route
    outbound_delivery.schedule_action = fake_schedule_action
    outbound_delivery.cancel_action_by_dedupe_key = queue.cancel_by_dedupe
    db_queries.get_fan_by_id = fake_get_fan

    started = time.perf_counter()
    try:
        past = datetime.now(timezone.utc) - timedelta(seconds=1)
        for creator_id, fan_id in fans:
            # A burst: several fan messages collapse into ONE reply obligation,
            # which is the dedupe key doing its job.
            for message in range(scenario.burst_messages):
                queue.add(
                    creator_id=creator_id,
                    fan_id=fan_id,
                    action_type="AUTO_REPLY",
                    execute_at=past,
                    payload={"burst": message},
                    dedupe_key=f"auto-reply:{fan_id}",
                    replace_existing=True,
                )

        stop = asyncio.Event()

        async def interrupt_mid_sequence() -> None:
            """Fan messages landing between bubbles, from outside this process.

            Deliberately targeted rather than timed: it waits for a conversation
            to have delivered at least one bubble with more still queued, then
            advances that fan's conversation generation. Nothing is delivered
            through this process's memory — the generation is the only signal —
            which is exactly the cross-process case the design has to survive.
            """
            done: set = set()
            while not stop.is_set():
                for (fan_id, _trigger), sequence_id in list(sequences.sequences.items()):
                    if fan_id not in interrupted or fan_id in done:
                        continue
                    record = sequences.parts.get(sequence_id)
                    if record is None:
                        continue
                    statuses = [part["status"] for part in record["parts"]]
                    if store.PART_SENT in statuses and store.PART_PENDING in statuses:
                        generations[fan_id] = generations[fan_id] + 1
                        done.add(fan_id)
                await asyncio.sleep(0.002)

        interrupter = asyncio.create_task(interrupt_mid_sequence())
        cycles = 0
        idle_cycles = 0
        while cycles < 20000:
            cycles += 1
            metrics.queue_depth_samples.append(queue.pending())
            metrics.pending_delay_actions_peak = max(
                metrics.pending_delay_actions_peak,
                queue.pending_of_type("DELIVER_OUTBOUND_PART"),
            )
            cycle_started = time.perf_counter()
            result = await worker.process_cycle(
                run_repair=False, concurrency=8, limit=claim_limit
            )
            metrics.worker_busy_seconds += time.perf_counter() - cycle_started
            if result.claimed:
                idle_cycles = 0
                continue
            if not queue.pending():
                idle_cycles += 1
                if idle_cycles > 2:
                    break
                await asyncio.sleep(0.002)
                continue
            idle_cycles = 0
            # The same shape as the live dispatcher: sleep until the next thing
            # is actually due rather than polling at a fixed rate.
            await asyncio.sleep(min(max(queue.seconds_until_next_due(), 0.001), 0.05))
        stop.set()
        await interrupter
        drain_seconds = time.perf_counter() - started
    finally:
        for name, value in saved_worker.items():
            setattr(worker, name, value)
        worker.HANDLERS.clear()
        worker.HANDLERS.update(saved_handlers)
        for name, value in saved_store.items():
            setattr(store, name, value)
        for name, value in saved_delivery.items():
            setattr(outbound_delivery, name, value)
        db_queries.get_fan_by_id = saved_get_fan

    superseded = sum(
        1
        for record in sequences.parts.values()
        if record["header"]["status"] == store.STATUS_SUPERSEDED
    )
    first_bubbles = sorted(metrics.time_to_first_bubble)
    depths = sorted(metrics.queue_depth_samples)
    return {
        "creators": scenario.creators,
        "conversations": len(fans),
        "interrupted_conversations": len(interrupted),
        "planned_sequences": metrics.planned_sequences,
        "bubbles_sent": metrics.bubbles_sent,
        "duplicate_sends": metrics.duplicate_sends,
        "stale_sends": metrics.stale_sends,
        "concurrent_generation_bumps": metrics.concurrent_generation_bumps,
        "same_fan_overlaps": metrics.same_fan_overlaps,
        "superseded_sequences": superseded,
        "max_model_concurrency": metrics.max_model_concurrency,
        "max_action_concurrency": metrics.max_action_concurrency,
        "model_calls": metrics.model_calls,
        "peak_queue_depth": max(depths) if depths else 0,
        "p50_queue_depth": _pct(depths, 0.50),
        "p95_queue_depth": _pct(depths, 0.95),
        "peak_pending_human_delay_actions": metrics.pending_delay_actions_peak,
        "p50_time_to_first_bubble_seconds": round(_pct(first_bubbles, 0.50), 4),
        "p95_time_to_first_bubble_seconds": round(_pct(first_bubbles, 0.95), 4),
        "drain_seconds": round(drain_seconds, 2),
        "model_utilisation": round(
            metrics.model_busy_seconds / max(drain_seconds * 8, 1e-9), 3
        ),
        "worker_utilisation": round(
            metrics.worker_busy_seconds / max(drain_seconds, 1e-9), 3
        ),
        "queue_errors": len(queue.failed),
        "time_scale": scenario.time_scale,
    }


if __name__ == "__main__":
    asyncio.run(main())
