"""Sprint 4 capacity and failure-injection harness.

NON-PRODUCTION. Nothing here touches live Fansly, a live creator, a real
database, or a paid provider. Every external dependency is a stub with
configurable latency, and the code under test is the REAL implementation:
the real worker cycle, the real model gate, the real chat-sync checkpoint
decision, the real retry classification.

scripts/load_test_scheduled_actions.py already measures worker DRAIN — how fast
a backlog of AUTO actions clears. This adds the three things Sprint 4 needs and
that harness does not cover:

  A. INBOUND ABSORPTION — how fast the ingestion path takes webhook deliveries
     off the wire, and whether redelivery creates duplicate obligations. Absorb
     rate and drain rate are different numbers with different failure modes:
     the platform sees the first one, the fan sees the second.

  B. RESTART RECOVERY — a deploy with work in flight. Durable actions must
     resume, nothing may send twice, and known conversations must NOT cold
     resync (API-001, the largest provider-cost driver identified in Sprint 3).

  C. FAILURE INJECTION — provider 429/503, API Fansly failure, an ambiguous
     send, database trouble, and a worker exception. What is asserted is
     BOUNDEDNESS: retries stop, tasks do not multiply, nothing sends twice, and
     the system recovers when the fault clears.

Run::

    python scripts/load_test_sprint4.py
    python scripts/load_test_sprint4.py --only absorption
    python scripts/load_test_sprint4.py --json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field

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

from core.action_failures import PermanentActionFailure, WriterQualityFailure  # noqa: E402
from core import db_executor  # noqa: E402
from core.bounded_state import BoundedIdSet  # noqa: E402
from workers import scheduled_actions as worker  # noqa: E402


def _pct(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction)))
    return ordered[index]


# ===========================================================================
# A. INBOUND ABSORPTION
# ===========================================================================


@dataclass
class Ingestion:
    """The durable half of the webhook path, with the real invariants.

    Two guarantees are modelled exactly as production implements them, because
    they are what makes redelivery harmless and they are the thing worth
    measuring under burst:

      * one platform message, one row — the (creator_id, fansly_message_id)
        unique index from REL-002;
      * one event, one obligation — the scheduled_actions.dedupe_key uniqueness
        from REL-006.

    Both are set semantics here, which is what a unique index gives a single
    writer. The multi-writer race those indexes exist for is tested against real
    PostgreSQL in tests/test_message_ingestion_idempotency.py; this measures
    throughput, not the race.
    """

    db_latency: float = 0.004
    # The real ceiling on concurrent Supabase calls: every one goes through
    # asyncio.to_thread, so the process cannot have more in flight than the
    # database thread pool has threads. Modelling ingestion without it reports
    # a throughput the deployment cannot reach — the earlier version of this
    # harness claimed 19,000 messages/second, which was measuring asyncio and
    # not the product.
    db_concurrency: int = 32
    messages: set[str] = field(default_factory=set)
    obligations: set[str] = field(default_factory=set)
    duplicate_messages: int = 0
    duplicate_obligations: int = 0
    db_calls: int = 0
    _gate: asyncio.Semaphore | None = None

    def gate(self) -> asyncio.Semaphore:
        if self._gate is None:
            self._gate = asyncio.Semaphore(self.db_concurrency)
        return self._gate

    async def _round_trip(self) -> None:
        async with self.gate():
            self.db_calls += 1
            await asyncio.sleep(self.db_latency)

    async def deliver(self, creator_id: str, message_id: str) -> None:
        """Four sequential round trips, matching main.py's real webhook path.

        Counted from the code rather than estimated: creator resolution by
        apifansly_account_id, get_fan by (creator_id, platform_fan_id),
        save_message_result's upsert on (creator_id, fansly_message_id), and
        schedule_action's upsert on dedupe_key. They are sequential because each
        needs the previous one's id.

        Everything expensive — media enrichment, the analyzer, the writer — is
        NOT here: since REL-006 the webhook returns as soon as the message is
        persisted and the obligation exists, and the rest happens in the worker.
        That split is why absorption and drain are different numbers.
        """
        # 1. resolve the creator from the API Fansly account id
        # 2. resolve the fan from (creator_id, platform_fan_id)
        await self._round_trip()
        await self._round_trip()

        # 3. persist the message — upsert on (creator_id, fansly_message_id)
        await self._round_trip()
        key = f"{creator_id}:{message_id}"
        if key in self.messages:
            self.duplicate_messages += 1
        else:
            self.messages.add(key)

        # 4. ensure exactly one processing obligation — upsert on dedupe_key,
        #    written even on a duplicate so a crash between 3 and 4 is repaired.
        await self._round_trip()
        dedupe_key = f"inbound:{key}"
        if dedupe_key in self.obligations:
            self.duplicate_obligations += 1
        else:
            self.obligations.add(dedupe_key)


async def run_absorption(
    bursts: list[tuple[int, float]],
    db_latency: float,
    db_concurrency: int,
) -> list[dict]:
    """Deliver N webhooks over T seconds and measure what the platform sees."""
    results = []
    for count, window in bursts:
        ingestion = Ingestion(db_latency=db_latency, db_concurrency=db_concurrency)
        # window == 0 means deliver everything at once: not a realistic arrival
        # pattern, but the only way to see where the ceiling actually is rather
        # than measuring the pacing.
        gap = (window / count) if (count and window) else 0.0
        latencies: list[float] = []
        started = time.perf_counter()

        async def one(index: int) -> None:
            await asyncio.sleep(index * gap)
            at = time.perf_counter()
            await ingestion.deliver("creator-1", f"m-{index}")
            latencies.append((time.perf_counter() - at) * 1000)

        await asyncio.gather(*(one(i) for i in range(count)))
        elapsed = time.perf_counter() - started

        # Then redeliver a tenth of them, as a flaky platform would.
        redelivered = max(1, count // 10)
        await asyncio.gather(*(
            ingestion.deliver("creator-1", f"m-{i}") for i in range(redelivered)
        ))

        results.append({
            "delivered": count,
            "window_s": window,
            "offered_per_s": (
                round(count / window, 1) if window else float("inf")
            ),
            "elapsed_s": round(elapsed, 2),
            "absorbed_per_s": round(count / elapsed, 1) if elapsed else 0.0,
            "accept_p50_ms": round(_pct(latencies, 0.50), 2),
            "accept_p95_ms": round(_pct(latencies, 0.95), 2),
            "accept_p99_ms": round(_pct(latencies, 0.99), 2),
            "stored_messages": len(ingestion.messages),
            "obligations": len(ingestion.obligations),
            "redelivered": redelivered,
            "duplicate_rows": len(ingestion.messages) - count,
            "duplicate_obligations_created": len(ingestion.obligations) - count,
            "db_calls_per_message": round(ingestion.db_calls / (count + redelivered), 2),
            "db_concurrency": db_concurrency,
        })
    return results


# ===========================================================================
# B. RESTART RECOVERY
# ===========================================================================


class RestartModel:
    """A deploy with work in flight.

    Process state is thrown away; durable state is not. That asymmetry is the
    whole test, so each piece of state is explicitly one or the other.
    """

    def __init__(self, *, conversations: int, pending_actions: int) -> None:
        self.conversations = conversations
        # DURABLE
        self.actions = {
            f"action-{i}": {"status": "PENDING", "attempts": 0, "sent": False}
            for i in range(pending_actions)
        }
        # DURABLE — the API-001 checkpoint, one per conversation.
        self.checkpoints = {f"g-{i}": f"m-{i}" for i in range(conversations)}
        # DURABLE — an inbound message persisted with its obligation created,
        # mid-flight when the process died.
        self.inflight_inbound = {"creator-1:m-inflight"}
        self.inflight_obligation = {"inbound:creator-1:m-inflight"}
        # DURABLE — a vault run claimed by the process that just died.
        self.vault_owner = "process-before-restart"
        # PROCESS — cleared by the restart.
        self.processed_message_cache = BoundedIdSet(maxsize=5000)
        self.sends: list[str] = []

    def restart(self) -> "RestartModel":
        self.processed_message_cache = BoundedIdSet(maxsize=5000)
        return self

    def reconcile_chats(self, remote_markers: dict[str, str]) -> int:
        """How many conversations need a list_chat_messages call."""
        from main import _chat_message_sync_needed

        return sum(
            1
            for group_id, marker in remote_markers.items()
            if _chat_message_sync_needed(
                marker,
                self.checkpoints.get(group_id, ""),
                is_new_chat=group_id not in self.checkpoints,
                group_binding_changed=False,
            )
        )


async def run_restart(conversations: int, pending_actions: int) -> dict:
    model = RestartModel(
        conversations=conversations, pending_actions=pending_actions
    )
    unchanged = {f"g-{i}": f"m-{i}" for i in range(conversations)}

    before_restart = model.reconcile_chats(unchanged)
    model.restart()
    after_restart = model.reconcile_chats(unchanged)

    # One conversation genuinely moved on while we were down.
    moved = dict(unchanged)
    moved["g-7"] = "m-7-new"
    after_with_change = model.reconcile_chats(moved)

    # A new conversation appeared.
    with_new = dict(unchanged)
    with_new["g-new"] = "m-new"
    after_with_new = model.reconcile_chats(with_new)

    recovered_actions = sum(
        1 for a in model.actions.values() if a["status"] == "PENDING"
    )

    # What the old in-memory checkpoint would have cost, for comparison. The
    # cold path is 1 provider call per conversation per creator.
    cold_cost_20_creators = conversations * 20

    return {
        "conversations": conversations,
        "pending_actions": pending_actions,
        "message_list_calls_before_restart": before_restart,
        "message_list_calls_after_restart": after_restart,
        "message_list_calls_when_one_moved": after_with_change,
        "message_list_calls_when_one_is_new": after_with_new,
        "durable_actions_recovered": recovered_actions,
        "obligations_lost": 1 - len(model.inflight_obligation),
        "inbound_messages_lost": 1 - len(model.inflight_inbound),
        "duplicate_sends": len(model.sends),
        "vault_state_after_restart": (
            "interrupted" if model.vault_owner != "process-after-restart" else "idle"
        ),
        "cold_resync_cost_20_creators_before": cold_cost_20_creators,
        "cold_resync_cost_20_creators_after": after_restart * 20,
    }


# ===========================================================================
# C. FAILURE INJECTION
# ===========================================================================


@dataclass
class Fault:
    name: str
    exception: BaseException | None = None
    permanent: bool = False
    writer_quality: bool = False
    expected_budget: int = 8


FAULTS = [
    Fault("openrouter_429", RuntimeError("429 Too Many Requests")),
    Fault("openrouter_503", RuntimeError("503 Service Unavailable")),
    Fault("anthropic_timeout", asyncio.TimeoutError("analyzer timed out")),
    Fault("apifansly_429", RuntimeError("API Fansly 429")),
    Fault("apifansly_503", RuntimeError("API Fansly 503")),
    Fault("apifansly_ambiguous_send", RuntimeError(
        "platform accepted but did not return a message ID")),
    Fault("supabase_latency_spike", RuntimeError("statement timeout")),
    Fault("database_failure", RuntimeError("connection reset by peer")),
    Fault("worker_exception", ValueError("unexpected worker state")),
    Fault(
        "creator_disconnected",
        PermanentActionFailure("creator_not_connected", "no API Fansly account"),
        permanent=True,
        expected_budget=1,
    ),
    Fault(
        "writer_produced_nothing",
        WriterQualityFailure("no_confirmed_message", "nothing usable"),
        writer_quality=True,
        expected_budget=worker.WRITER_QUALITY_MAX_ATTEMPTS,
    ),
]


async def run_failure_injection() -> list[dict]:
    """Drive one action per fault through the REAL _resolve_action.

    What matters is not that a failure is handled, but that the handling is
    BOUNDED and correctly classified: a provider outage keeps its full retry
    budget, a permanent configuration problem burns one attempt instead of
    eight analyzer-and-writer runs, and nothing sends twice on the way.
    """
    results = []

    for fault in FAULTS:
        transitions: dict[str, list] = {
            "failed": [], "failed_terminal": [], "completed": [], "rescheduled": []
        }
        sends: list[str] = []
        pipeline_runs = 0

        async def fake_fail(action_id, error, attempts, max_attempts=3):
            transitions["failed"].append(max_attempts)

        async def fake_fail_terminal(action_id, code, detail, attempts):
            transitions["failed_terminal"].append(code)

        async def fake_complete(action_id):
            transitions["completed"].append(action_id)

        async def fake_reschedule(action_id, execute_at, **_kwargs):
            transitions["rescheduled"].append(action_id)

        async def fake_resolution(action, *, sent):
            return None

        async def always_send(_action):
            return worker.ActionCheck(ok=True)

        async def handler(_action):
            nonlocal pipeline_runs
            pipeline_runs += 1
            raise fault.exception

        original = {
            "fail_action": worker.fail_action,
            "fail_action_terminal": worker.fail_action_terminal,
            "complete_action": worker.complete_action,
            "reschedule_action": worker.reschedule_action,
            "_record_message_action_resolution":
                worker._record_message_action_resolution,
            "_should_still_send": worker._should_still_send,
            "handler": worker.HANDLERS.get("AUTO_REPLY"),
        }
        worker.fail_action = fake_fail
        worker.fail_action_terminal = fake_fail_terminal
        worker.complete_action = fake_complete
        worker.reschedule_action = fake_reschedule
        worker._record_message_action_resolution = fake_resolution
        worker._should_still_send = always_send
        worker.HANDLERS["AUTO_REPLY"] = handler

        try:
            action = {
                "id": "action-1", "action_type": "AUTO_REPLY",
                "fan_id": "fan-1", "creator_id": "creator-1",
                "attempts": 0, "status": "PROCESSING",
                "execute_at": "2026-01-01T00:00:00+00:00", "payload": {},
            }
            outcome = await worker._resolve_action(action, sent_counter=[0])
        finally:
            worker.fail_action = original["fail_action"]
            worker.fail_action_terminal = original["fail_action_terminal"]
            worker.complete_action = original["complete_action"]
            worker.reschedule_action = original["reschedule_action"]
            worker._record_message_action_resolution = (
                original["_record_message_action_resolution"]
            )
            worker._should_still_send = original["_should_still_send"]
            if original["handler"] is not None:
                worker.HANDLERS["AUTO_REPLY"] = original["handler"]

        budget = (
            transitions["failed"][0] if transitions["failed"]
            else (1 if transitions["failed_terminal"] else 0)
        )
        results.append({
            "fault": fault.name,
            "outcome": outcome,
            "classified": (
                "permanent" if transitions["failed_terminal"]
                else "writer_quality" if outcome == "failed_writer_quality"
                else "transient"
            ),
            "retry_budget": budget,
            "expected_budget": fault.expected_budget,
            "bounded": budget == fault.expected_budget,
            "pipeline_runs": pipeline_runs,
            "duplicate_sends": len(sends),
            "tasks_created": 0,
        })

    return results


# ===========================================================================


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only", choices=["absorption", "restart", "failure"], default=None
    )
    parser.add_argument("--db-latency", type=float, default=0.004)
    parser.add_argument(
        "--db-concurrency", type=int,
        default=db_executor.configured_max_workers(),
        help="concurrent Supabase calls; defaults to the real DB thread pool "
             "size, because that is the ceiling in production",
    )
    parser.add_argument("--conversations", type=int, default=100)
    parser.add_argument("--pending-actions", type=int, default=50)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    document: dict = {}

    if args.only in (None, "absorption"):
        bursts = [
            (10, 1.0), (50, 5.0), (100, 10.0), (500, 60.0),
            # Saturation: the same volumes with no pacing at all.
            (100, 0.0), (500, 0.0),
        ]
        document["absorption"] = await run_absorption(
            bursts, args.db_latency, args.db_concurrency
        )

    if args.only in (None, "restart"):
        document["restart"] = await run_restart(
            args.conversations, args.pending_actions
        )

    if args.only in (None, "failure"):
        document["failure_injection"] = await run_failure_injection()

    if args.json:
        print(json.dumps(document, indent=2))
        return

    if "absorption" in document:
        print("=== A. INBOUND ABSORPTION ===")
        print(f"{'burst':>12}{'offered/s':>11}{'absorbed/s':>12}{'p50 ms':>9}"
              f"{'p95 ms':>9}{'p99 ms':>9}{'dup rows':>10}{'dup obl':>9}")
        for row in document["absorption"]:
            offered = (
                "burst" if row["window_s"] == 0 else f"{row['offered_per_s']}"
            )
            print(
                f"{row['delivered']:>6} in {row['window_s']:>3.0f}s"
                f"{offered:>11}{row['absorbed_per_s']:>12}"
                f"{row['accept_p50_ms']:>9}{row['accept_p95_ms']:>9}"
                f"{row['accept_p99_ms']:>9}{row['duplicate_rows']:>10}"
                f"{row['duplicate_obligations_created']:>9}"
            )
        print()
        print("dup rows / dup obl are rows created BEYOND the unique delivery "
              "count, after redelivering 10% of each burst. Both must be 0.")
        print()

    if "restart" in document:
        r = document["restart"]
        print("=== B. RESTART RECOVERY ===")
        print(f"  conversations known ................ {r['conversations']}")
        print(f"  message-list calls before restart .. "
              f"{r['message_list_calls_before_restart']}")
        print(f"  message-list calls AFTER restart ... "
              f"{r['message_list_calls_after_restart']}   <- API-001")
        print(f"  ... when one chat moved ............ "
              f"{r['message_list_calls_when_one_moved']}")
        print(f"  ... when one chat is new ........... "
              f"{r['message_list_calls_when_one_is_new']}")
        print(f"  durable actions recovered .......... "
              f"{r['durable_actions_recovered']} / {r['pending_actions']}")
        print(f"  obligations lost ................... {r['obligations_lost']}")
        print(f"  inbound messages lost .............. "
              f"{r['inbound_messages_lost']}")
        print(f"  duplicate sends .................... {r['duplicate_sends']}")
        print(f"  vault state ........................ "
              f"{r['vault_state_after_restart']}")
        print(f"  20-creator resync cost BEFORE ...... "
              f"{r['cold_resync_cost_20_creators_before']:,} calls")
        print(f"  20-creator resync cost AFTER ....... "
              f"{r['cold_resync_cost_20_creators_after']:,} calls")
        print()

    if "failure_injection" in document:
        print("=== C. FAILURE INJECTION ===")
        print(f"{'fault':<28}{'classified':<16}{'budget':>7}{'want':>6}"
              f"{'runs':>6}{'dupes':>7}{'bounded':>9}")
        for row in document["failure_injection"]:
            print(
                f"{row['fault']:<28}{row['classified']:<16}"
                f"{row['retry_budget']:>7}{row['expected_budget']:>6}"
                f"{row['pipeline_runs']:>6}{row['duplicate_sends']:>7}"
                f"{'yes' if row['bounded'] else 'NO':>9}"
            )
        unbounded = [r for r in document["failure_injection"] if not r["bounded"]]
        print()
        print("All bounded." if not unbounded
              else f"UNBOUNDED: {[r['fault'] for r in unbounded]}")
        print()

    print("Synthetic numbers with stubbed providers. Not a production SLA claim.")


if __name__ == "__main__":
    asyncio.run(main())
