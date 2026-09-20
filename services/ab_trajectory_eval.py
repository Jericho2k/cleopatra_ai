"""Run the same scripted fan trajectory through two conversational runtimes.

WHAT THIS IS FOR
----------------
Conversational Core v1 is a change to the application architecture that owns a
turn, not a change of model. The question it has to answer is whether the
resulting conversations are better, and the only way to answer that fairly is to
put the *same* fan in front of both runtimes under the *same* conditions and
compare what came back.

So this is an A/B harness over ``services/trajectory_eval.py``. That module
already knows how to drive one whole conversation through the real Full Auto
turn, classify each turn by evidence, and refuse to invent a score. This one
adds the second arm, the conditions that have to be held constant for the
comparison to mean anything, and the machine-readable record of both runs.

It produces no verdict. ``metrics.json`` is arithmetic, the blind review is
where judgement happens, and neither of them knows which runtime is expected to
win.

THE RUNTIME IS A STRING
-----------------------
Nothing here imports a runtime, names ``conversational_v1`` in code, or knows
what a core does. A core id is a parameter, validated against whatever
``services.conversation_core.CORE_IDS`` contains *at call time* — so the day the
runtime branch registers a new id, this harness accepts it with no change. Until
then, asking for it fails with :class:`CoreNotAvailable`, which says exactly
what is missing rather than failing somewhere deeper as a ``ValueError``.

Baseline-only runs are first-class for the same reason: the harness has to be
usable before the candidate runtime exists, or it cannot be trusted the day it
does.

FAIRNESS, AND WHAT IT COSTS
---------------------------
Held constant, and checked rather than assumed:

*The fan's script.* Both arms get the same messages in the same order, and
``paired.json`` carries a digest of each arm's inputs so a run where they
diverged is visible instead of being averaged. Adaptive trajectories are refused
outright: a customer that reacts to the reply is a different customer per arm,
which is a legitimate thing to test and is not this.

*The creator, the stack profile and the starting state.* One creator, one
resolved AI stack profile, and each scenario starts from cleared seeded state
with the same seed applied.

*Persistent state, which must NOT be shared.* Each arm runs against its own
``test_`` fan. A conversation is persistent by design in this product — history,
commercial state, price learning, lifecycle — so two runtimes writing into one
fan would not be two runs, it would be one conversation with two authors. The
harness refuses to start when both arms name the same fan.

*Arm order.* Whichever arm runs second sees a slightly warmer system. The order
is therefore shuffled per scenario from the run seed and recorded, so the bias
is randomised and reproducible rather than systematic and invisible.

SAFETY
------
Every arm's fan is re-read from the database and must be a simulator test fan
(``platform_fan_id`` starting ``test_``). The harness checks this itself rather
than trusting the caller or the backend, because the caller is a CLI argument
and the cost of getting it wrong is writing an evaluation into a real
customer's conversation.
"""

from __future__ import annotations

import hashlib
import json
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

from core.simulation import is_simulatable_fan
from services.conversation_metrics import scenario_metrics
from services.reply_provenance import CORE_STATE_KEY
from services.trajectory_eval import (
    Trajectory,
    TrajectoryReport,
    TurnRecord,
    run_trajectory,
)

#: Bumped when the artifact shape changes, and written into every metadata.json
#: so a result directory can be read years later by something that knows which
#: shape it is.
HARNESS_VERSION = "conversational-core-ab/1"

ROLE_BASELINE = "baseline"
ROLE_CANDIDATE = "candidate"

#: Where a run's artifacts land. ``eval/results/`` is already gitignored, which
#: is deliberate: generated conversations are large, contain generated intimate
#: text, and are evidence for one run rather than source.
RESULTS_ROOT = Path(__file__).resolve().parents[1] / "eval" / "results"

METADATA_FILE = "metadata.json"
PAIRED_FILE = "paired.json"
METRICS_FILE = "metrics.json"

#: Optional Core v1 provenance. A runtime that keeps working state across a turn
#: records it through ``ReplyProvenance.record_core_state``; every field is
#: optional and a runtime that records nothing is not a failure. The key is
#: imported rather than restated so the producer and this reader cannot drift.
#: See ``docs/conversational_core_v1_evaluation.md``.
CORE_STATE_FIELDS: tuple[str, ...] = (
    "state_before",
    "proposed_delta",
    "accepted_fields",
    "rejected_fields",
    "state_after",
)


class EvaluationRefused(RuntimeError):
    """The run would not have measured what it claims to measure."""


class UnfairComparison(EvaluationRefused):
    """Two arms would not have received equivalent conditions."""


class UnsafeEvaluationTarget(EvaluationRefused):
    """A run would have written evaluation state somewhere it must not."""


class CoreNotAvailable(EvaluationRefused):
    """A requested conversational runtime is not registered in this build."""


def known_core_ids() -> tuple[str, ...]:
    """Every runtime this build can select, read at call time.

    Deliberately a function and deliberately not cached. The candidate runtime
    is being built on another branch; when it registers its id, this returns it
    without anything here being edited, which is the whole contract between the
    two branches.
    """
    from services.conversation_core import CORE_IDS

    return tuple(str(core_id) for core_id in CORE_IDS)


def require_known_core(core_id: str) -> str:
    cleaned = str(core_id or "").strip().lower()
    if not cleaned:
        raise CoreNotAvailable("a conversation core id is required")
    available = known_core_ids()
    if cleaned not in available:
        raise CoreNotAvailable(
            f"conversation core {cleaned!r} is not registered in this build "
            f"(services/conversation_core.py knows {', '.join(available)}). "
            "The runtime branch registers its own id; until it lands, run the "
            "baseline arm on its own."
        )
    return cleaned


# ---------------------------------------------------------------------------
# What a run is
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArmSpec:
    """One side of the comparison: a runtime, and the fan it runs against."""

    role: str
    core_id: str
    fan_id: str
    #: Whether this harness created the fan for this run. A provisioned fan
    #: starts blank, which is the strongest form of equivalent initial
    #: conditions available here.
    provisioned: bool = False
    platform_fan_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "conversation_core": self.core_id,
            "fan_id": self.fan_id,
            "platform_fan_id": self.platform_fan_id,
            "provisioned_for_this_run": self.provisioned,
        }


@dataclass(frozen=True)
class RunSpec:
    """Everything that decides what a run does, in one reproducible object."""

    run_id: str
    creator_id: str
    seed: int
    arms: tuple[ArmSpec, ...]
    scenario_ids: tuple[str, ...] = ()
    simulate_time: bool = False
    stack_profile: str = ""
    notes: str = ""

    @property
    def baseline(self) -> ArmSpec:
        return self.arms[0]

    @property
    def candidate(self) -> ArmSpec | None:
        return self.arms[1] if len(self.arms) > 1 else None


def new_run_id(prefix: str = "ccv1") -> str:
    """A run id that sorts by time and cannot collide with another run."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{stamp}-{uuid.uuid4().hex[:6]}"


def fan_input_digest(trajectory: Trajectory) -> str:
    """A stable identifier for exactly what the fan says, in order.

    This is what makes "both arms got the same script" checkable after the fact
    rather than asserted in a docstring. Empty turns are included as such,
    because a due-work turn is part of the script.
    """
    parts = [
        f"{index}:{disturbance.message}:{disturbance.days_since_previous:g}"
        for index, disturbance in enumerate(trajectory.disturbances)
    ]
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:16]


def assert_scripted(trajectory: Trajectory) -> None:
    """Refuse an adaptive customer in an A/B comparison.

    An adaptive customer answers what it was told, so each arm would face a
    different conversation. That is a useful experiment and it is not this one:
    the brief is the same scripted fan trajectory through both runtimes.
    """
    if any(
        disturbance.responds_to is not None for disturbance in trajectory.disturbances
    ):
        raise UnfairComparison(
            f"trajectory {trajectory.name!r} is adaptive; an adaptive customer "
            "produces different inputs per arm, so it cannot be used for an A/B "
            "comparison of two runtimes"
        )


def assert_arm_isolation(arms: Sequence[ArmSpec]) -> None:
    """Refuse a configuration where the arms would share persistent state.

    A fan in this product accumulates conversation history, commercial state,
    price learning and lifecycle rows. Two runtimes writing into one fan is one
    conversation with two authors, and every number computed from it would be
    describing something that never happened.
    """
    if not arms:
        raise EvaluationRefused("a run needs at least one arm")
    fans = [arm.fan_id for arm in arms]
    if len(set(fans)) != len(fans):
        raise UnfairComparison(
            "baseline and candidate must run against different test fans; "
            "sharing one fan would let each arm read and overwrite the other's "
            "conversation state"
        )
    for arm in arms:
        if not arm.fan_id:
            raise EvaluationRefused(f"{arm.role} arm has no fan id")


def assert_safe_targets(arms: Sequence[ArmSpec], fan_rows: dict[str, dict]) -> None:
    """Every arm runs against a simulator test fan of the right creator.

    Checked here, against rows read from the database, rather than trusted from
    the caller. The fan id is a command-line argument.
    """
    for arm in arms:
        row = fan_rows.get(arm.fan_id) or {}
        if not row:
            raise UnsafeEvaluationTarget(f"fan {arm.fan_id} was not found")
        if not is_simulatable_fan(row.get("platform_fan_id")):
            raise UnsafeEvaluationTarget(
                f"fan {arm.fan_id} is not a simulator test fan. A trajectory "
                "evaluation writes conversation and commercial state; it is "
                "allowed only against test_ fans."
            )


# ---------------------------------------------------------------------------
# The seam to the simulator
# ---------------------------------------------------------------------------


class SimulatorBackend(Protocol):
    """Everything the harness needs from the running system.

    A protocol rather than direct imports so the harness can be tested without a
    database, and so every database-touching call is in one place that can be
    read for safety in one sitting.
    """

    async def describe_fan(self, creator_id: str, fan_id: str) -> dict[str, Any]:
        """The fan row, at least ``platform_fan_id`` and ``creator_id``."""

    async def select_core(self, fan_id: str, core_id: str | None) -> str | None:
        """Pin this fan to a runtime; return the previous selection."""

    async def reset_state(self, creator_id: str, fan_id: str) -> None:
        """Clear whatever a previous scenario seeded."""

    async def apply_seed(
        self, creator_id: str, fan_id: str, seed: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Establish authoritative fixture state for one scenario."""

    async def send_turn(self, creator_id: str, fan_id: str, message: str) -> dict:
        """Run one real Full Auto turn for a fan message."""

    async def run_due_work(self, creator_id: str, fan_id: str) -> dict:
        """Run one due-work cycle and report what the fan received."""


class LiveSimulatorBackend:
    """The real thing: the simulator, the fixtures and the core selection.

    Every import is local to the method that needs it, so importing this module
    never requires a reachable database — ``--describe`` and the unit tests both
    depend on that.
    """

    async def describe_fan(self, creator_id: str, fan_id: str) -> dict[str, Any]:
        import asyncio

        from core.supabase import get_supabase

        def _load() -> dict[str, Any]:
            result = (
                get_supabase()
                .table("fans")
                .select("id, creator_id, platform_fan_id, display_name")
                .eq("id", str(fan_id))
                .limit(1)
                .execute()
            )
            rows = result.data or []
            return dict(rows[0]) if rows else {}

        row = await asyncio.to_thread(_load)
        if row and str(row.get("creator_id")) != str(creator_id):
            raise UnsafeEvaluationTarget(
                f"fan {fan_id} does not belong to creator {creator_id}"
            )
        return row

    async def provision_fan(self, creator_id: str, label: str) -> dict[str, Any]:
        from services.simulation_workspace import create_test_fan

        return await create_test_fan(creator_id=creator_id, display_name=label)

    async def select_core(self, fan_id: str, core_id: str | None) -> str | None:
        from services.conversation_core import (
            set_simulation_fan_core_override,
            simulation_fan_core_override,
        )

        previous = await simulation_fan_core_override(fan_id)
        await set_simulation_fan_core_override(fan_id, core_id)
        return previous

    async def reset_state(self, creator_id: str, fan_id: str) -> None:
        from services.trajectory_fixtures import clear_seeded

        await clear_seeded(creator_id, fan_id)

    async def apply_seed(
        self, creator_id: str, fan_id: str, seed: dict[str, Any]
    ) -> list[dict[str, Any]]:
        from services.trajectory_fixtures import apply_seed as _apply

        if not seed:
            return []
        return await _apply(creator_id=creator_id, fan_id=fan_id, seed=seed)

    async def send_turn(self, creator_id: str, fan_id: str, message: str) -> dict:
        from services.suggestions import run_simulated_inbound

        return await run_simulated_inbound(
            fan_id=fan_id, creator_id=creator_id, message=message, fast=True
        )

    async def run_due_work(self, creator_id: str, fan_id: str) -> dict:
        """Identical in meaning to scripts/run_trajectory_eval.py's version.

        A turn with no fan message is the moment queued work becomes due, which
        is the only way the post-event and delayed-return scenarios can observe
        an unprompted message at all.
        """
        from services.suggestions import _recent_creator_message_rows
        from workers.scheduled_actions import process_cycle

        before = {
            str(row.get("id")) for row in await _recent_creator_message_rows(fan_id)
        }
        result = await process_cycle(limit=20)
        fresh = [
            row
            for row in await _recent_creator_message_rows(fan_id)
            if str(row.get("id")) not in before
        ]
        return {
            "outcome": "due_work_ran" if fresh else "due_work_sent_nothing",
            "creator_messages": fresh,
            "due_worker_ran": True,
            "due_work": {
                "claimed": getattr(result, "claimed", 0),
                "processed": getattr(result, "processed", 0),
                "errors": getattr(result, "errors", 0),
            },
        }


# ---------------------------------------------------------------------------
# Turning a run into a record
# ---------------------------------------------------------------------------


def _first(records: Sequence[dict[str, Any]], *path: str) -> Any:
    for record in records:
        cursor: Any = record
        for key in path:
            if not isinstance(cursor, dict):
                cursor = None
                break
            cursor = cursor.get(key)
        if cursor not in (None, "", {}, []):
            return cursor
    return None


def _optional_core_state(records: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    """Core v1's working state for this turn, if the runtime recorded any.

    Every field is optional and the whole block is optional. ``semantic_v2``
    records none of it and that is not a gap in the run — it is a runtime that
    does not have the concept. A missing block is therefore absent from the
    turn record rather than present and empty, so nothing downstream can average
    a zero into a comparison.
    """
    for record in records:
        block = record.get(CORE_STATE_KEY)
        if isinstance(block, dict) and block:
            return {
                field_name: block[field_name]
                for field_name in CORE_STATE_FIELDS
                if field_name in block
            } or None
    return None


def _tokens(records: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    """Token usage, if anything on the path recorded it.

    Usage is recorded by model telemetry rather than by reply provenance in this
    build, so most runs will have none. Reported as absent rather than as zero,
    which is what ``tokens.turns_reporting`` in the metrics exists to expose.
    """
    usage = _first(records, "writer", "usage") or _first(records, "usage")
    if not isinstance(usage, dict):
        return None
    total = usage.get("total_tokens")
    if total is None:
        try:
            total = int(usage.get("input_tokens") or 0) + int(
                usage.get("output_tokens") or 0
            )
        except (TypeError, ValueError):
            total = None
    record = {key: value for key, value in usage.items() if value is not None}
    if total is not None:
        record["total"] = total
    return record or None


def observe_turn(turn: TurnRecord, *, core_id: str) -> dict[str, Any]:
    """One turn, as the machine-readable record the brief asks for.

    Reads the provenance the pipeline already writes. Everything optional is
    optional: a runtime that records no operation, no context fingerprint and no
    Core v1 state produces a turn record with those keys absent, and nothing
    downstream treats absence as a value.
    """
    records = list(turn.provenance or [])
    decision = _first(records, "decision") or {}
    if not isinstance(decision, dict):
        decision = {}
    delivery_records = [
        record.get("delivery") or {}
        for record in records
        if isinstance(record.get("delivery"), dict)
    ]
    outcome = str(turn.outcome or "")
    classified = turn.classify().value

    operation_kind = str(
        decision.get("semantic_operation") or decision.get("action") or ""
    )
    approved = decision.get("validator_approved")
    observation: dict[str, Any] = {
        "turn_index": turn.index,
        "fan_input": turn.customer_message,
        "creator_output": list(turn.replies),
        "bubbles": len(turn.replies),
        # What the pipeline itself recorded, when it did; the requested core
        # otherwise. Recorded rather than assumed, because "the fan was pinned
        # to X" and "X answered this turn" are different claims.
        "conversation_core": str(decision.get("conversation_core") or core_id),
        "conversation_core_requested": core_id,
        "conversation_core_recorded": bool(decision.get("conversation_core")),
        "latency_ms": int(turn.latency_ms or 0),
        "outcome": outcome,
        "classified_outcome": classified,
        "control": {
            "outcome": outcome,
            "classified": classified,
            "disposition": str(decision.get("disposition") or ""),
            "response_intent": str(decision.get("response_intent") or ""),
            "hold": str(decision.get("hold") or ""),
            "handoff": outcome == "human_review"
            or str(decision.get("disposition") or "") == "handoff",
            "freeze": "freez" in str(decision.get("reason") or "").lower()
            or "frozen" in str(decision.get("reason") or "").lower(),
            "error": turn.error or "",
        },
        "model_requested": str(_first(records, "writer", "requested", "model") or ""),
        "model_served": str(_first(records, "writer", "actual", "model") or ""),
        "provider_served": str(_first(records, "writer", "actual", "provider") or ""),
        "context_fingerprint": str(
            _first(records, "context", "packet", "fingerprint") or ""
        ),
        "turn_id": str(_first(records, "turn_id") or ""),
        "provenance_turn_ids": sorted(
            {str(record.get("turn_id")) for record in records if record.get("turn_id")}
        ),
        "transforms": list(_first(records, "transforms") or []),
    }
    if operation_kind and operation_kind != "none":
        observation["operation_proposal"] = {
            "kind": operation_kind,
            "because": str(decision.get("reason") or ""),
            "offer_id": str(decision.get("semantic_offer_id") or ""),
            "set_id": str(decision.get("semantic_set_id") or ""),
        }
        observation["operation_result"] = {
            "approved": bool(approved) if approved is not None else None,
            "executed": bool(delivery_records) or None,
            "approval_required": bool(decision.get("semantic_approval_required"))
            if decision.get("semantic_approval_required") is not None
            else None,
            "validator_reasons": str(decision.get("validator_reasons") or ""),
            "deliveries": [
                {
                    "kind": record.get("kind"),
                    "reference": record.get("reference"),
                    "price_cents": record.get("price_cents"),
                    "accepted_by_platform": bool(record.get("accepted_by_platform")),
                }
                for record in delivery_records
            ],
        }
    elif delivery_records:
        observation["operation_result"] = {
            "deliveries": [
                {
                    "kind": record.get("kind"),
                    "reference": record.get("reference"),
                    "accepted_by_platform": bool(record.get("accepted_by_platform")),
                }
                for record in delivery_records
            ]
        }

    tokens = _tokens(records)
    if tokens:
        observation["tokens"] = tokens
    cost = _first(records, "writer", "estimated_cost_usd") or _first(
        records, "estimated_cost_usd"
    )
    if isinstance(cost, (int, float)) and not isinstance(cost, bool):
        observation["cost_usd"] = float(cost)
    core_state = _optional_core_state(records)
    if core_state:
        observation[CORE_STATE_KEY] = core_state
    return observation


def observe_report(
    report: TrajectoryReport,
    *,
    core_id: str,
    scenario_id: str,
    fan_id: str,
    input_digest: str,
) -> dict[str, Any]:
    """One finished conversation, as a record that stands on its own."""
    return {
        "scenario_id": scenario_id,
        "name": report.trajectory,
        "covers": report.covers,
        "conversation_core": core_id,
        "fan_id": fan_id,
        "fan_input_digest": input_digest,
        "summary": report.summary(),
        "coverage_gaps": [
            {"claim": gap.claim, "required": gap.required, "actual": gap.actual}
            for gap in report.coverage_gaps
        ],
        "findings": [
            {
                "severity": finding.severity.value,
                "kind": finding.kind,
                "detail": finding.detail,
                "turn": finding.turn,
            }
            for finding in report.findings
        ],
        "seeded_purchases": [
            {
                "reference": row.get("reference"),
                "price_cents": row.get("price_cents"),
                "media_ids": row.get("media_ids"),
            }
            for row in report.seeded_purchases
        ],
        "turns": [observe_turn(turn, core_id=core_id) for turn in report.turns],
    }


# ---------------------------------------------------------------------------
# Running both arms
# ---------------------------------------------------------------------------


def arm_order(arms: Sequence[ArmSpec], *, seed: int, scenario_index: int) -> list[ArmSpec]:
    """Which arm runs first for one scenario, decided by the run seed.

    Whichever arm runs second faces a slightly different system: warmer caches,
    a later moment in whatever the providers are doing. Fixing the order would
    hand that difference to the same arm every time. Shuffling it from the seed
    randomises it and keeps the run reproducible, and ``metadata.json`` records
    the order that was used.
    """
    order = list(arms)
    random.Random(f"{seed}:{scenario_index}").shuffle(order)
    return order


async def run_scenario_arm(
    *,
    trajectory: Trajectory,
    arm: ArmSpec,
    creator_id: str,
    backend: SimulatorBackend,
    advance_clock: Callable[[float], None] | None,
    scenario_id: str,
) -> dict[str, Any]:
    """One scenario, one runtime, against that runtime's own fan."""
    assert_scripted(trajectory)
    digest = fan_input_digest(trajectory)

    async def send_turn(message: str) -> dict:
        if not str(message).strip():
            return await backend.run_due_work(creator_id, arm.fan_id)
        return await backend.send_turn(creator_id, arm.fan_id, message)

    await backend.reset_state(creator_id, arm.fan_id)
    seeded: list[dict[str, Any]] = []
    seed_error = ""
    if trajectory.seed:
        try:
            seeded = await backend.apply_seed(creator_id, arm.fan_id, trajectory.seed)
        except Exception as exc:
            # Refusing to seed is never a reason to fabricate the state anyway.
            # The run continues and the coverage check reports the claim as
            # uncovered, which is the honest outcome.
            seed_error = f"{type(exc).__name__}: {exc}"

    report = await run_trajectory(
        trajectory, send_turn=send_turn, advance_clock=advance_clock
    )
    report.seeded_purchases = seeded
    from services.trajectory_eval import coverage_gaps

    report.coverage_gaps = coverage_gaps(trajectory, report)
    record = observe_report(
        report,
        core_id=arm.core_id,
        scenario_id=scenario_id,
        fan_id=arm.fan_id,
        input_digest=digest,
    )
    if seed_error:
        record["seed_error"] = seed_error
    return record


@dataclass
class ArmRun:
    """Everything one runtime produced across the whole suite."""

    arm: ArmSpec
    conversations: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm.to_dict(),
            "conversations": self.conversations,
        }


async def run_suite(
    *,
    spec: RunSpec,
    trajectories: Sequence[Trajectory],
    backend: SimulatorBackend,
    advance_clock: Callable[[float], None] | None = None,
    reset_clock: Callable[[], None] | None = None,
    scenario_ids: Sequence[str] | None = None,
) -> dict[str, ArmRun]:
    """Run every scenario through every arm, restoring each fan's selection.

    Scenario by scenario rather than arm by arm, so the two runtimes face the
    same moment in time as closely as this can arrange. Each arm's previous core
    selection is restored afterwards even when the run raises, because a test
    fan left pinned to an evaluation runtime is a surprise waiting for whoever
    opens it next.
    """
    for trajectory in trajectories:
        assert_scripted(trajectory)

    ids = list(scenario_ids or [trajectory.name for trajectory in trajectories])
    runs = {arm.role: ArmRun(arm=arm) for arm in spec.arms}
    previous: dict[str, str | None] = {}

    try:
        for arm in spec.arms:
            previous[arm.fan_id] = await backend.select_core(arm.fan_id, arm.core_id)
        for index, trajectory in enumerate(trajectories):
            scenario_id = ids[index] if index < len(ids) else trajectory.name
            for arm in arm_order(spec.arms, seed=spec.seed, scenario_index=index):
                if reset_clock is not None:
                    # Fresh state between independent scenarios, and between the
                    # two arms of one scenario: a clock carried over is state.
                    reset_clock()
                runs[arm.role].conversations.append(
                    await run_scenario_arm(
                        trajectory=trajectory,
                        arm=arm,
                        creator_id=spec.creator_id,
                        backend=backend,
                        advance_clock=advance_clock,
                        scenario_id=scenario_id,
                    )
                )
    finally:
        for arm in spec.arms:
            if arm.fan_id in previous:
                try:
                    await backend.select_core(arm.fan_id, previous[arm.fan_id])
                except Exception as exc:  # pragma: no cover - restoration is best effort
                    print(
                        f"[AB EVAL] could not restore core selection for fan "
                        f"{arm.fan_id}: {exc}"
                    )
        if reset_clock is not None:
            reset_clock()
    return runs


# ---------------------------------------------------------------------------
# Pairing and artifacts
# ---------------------------------------------------------------------------


def pair_conversations(runs: dict[str, ArmRun]) -> dict[str, Any]:
    """Line the two arms' conversations up by scenario, and check fairness.

    The fairness check is the reason this exists rather than a zip(): a pair
    whose input digests differ is not a comparison, and saying so in the
    artifact is the only thing that stops it being read as one. A baseline-only
    run pairs cleanly with nothing and says so.
    """
    baseline = runs.get(ROLE_BASELINE)
    candidate = runs.get(ROLE_CANDIDATE)
    if baseline is None:
        raise EvaluationRefused("a run must have a baseline arm")

    pairs: list[dict[str, Any]] = []
    by_scenario = {
        str(row.get("scenario_id")): row for row in (candidate.conversations if candidate else [])
    }
    mismatches: list[str] = []
    for row in baseline.conversations:
        scenario_id = str(row.get("scenario_id"))
        other = by_scenario.get(scenario_id)
        identical = bool(other) and row.get("fan_input_digest") == other.get(
            "fan_input_digest"
        )
        if other and not identical:
            mismatches.append(scenario_id)
        pairs.append(
            {
                "scenario_id": scenario_id,
                "name": row.get("name"),
                "covers": row.get("covers"),
                "fan_inputs_identical": identical if other else None,
                "fan_input_digest": row.get("fan_input_digest"),
                "arms": {
                    ROLE_BASELINE: {
                        "conversation_core": row.get("conversation_core"),
                        "fan_id": row.get("fan_id"),
                        "turns": row.get("turns"),
                        "summary": row.get("summary"),
                    },
                    **(
                        {
                            ROLE_CANDIDATE: {
                                "conversation_core": other.get("conversation_core"),
                                "fan_id": other.get("fan_id"),
                                "turns": other.get("turns"),
                                "summary": other.get("summary"),
                            }
                        }
                        if other
                        else {}
                    ),
                },
            }
        )
    return {
        "paired": bool(candidate),
        "scenarios": len(pairs),
        # Named rather than counted: a mismatch means that scenario's pair is
        # not evidence about the runtimes, and a reader has to know which one.
        "input_mismatches": mismatches,
        "fan_inputs_identical": not mismatches,
        "pairs": pairs,
    }


def build_metrics(runs: dict[str, ArmRun]) -> dict[str, Any]:
    """Objective metrics per arm, with the unmeasured dimensions named."""
    return {
        role: scenario_metrics(run.conversations) for role, run in runs.items()
    }


def git_sha() -> str:
    """The commit this run was produced from, or an explicit unknown."""
    try:
        from core.build_info import build_snapshot

        snapshot = build_snapshot(include_flags=False)
        sha = str(snapshot.get("sha") or "")
        if sha:
            return sha
    except Exception:  # pragma: no cover - build info is never load-bearing
        pass
    return "unknown"


def build_metadata(
    spec: RunSpec,
    runs: dict[str, ArmRun],
    *,
    trajectories: Sequence[Trajectory] = (),
) -> dict[str, Any]:
    """The run-level record: what ran, against what, under which configuration."""
    served: dict[str, list[str]] = {}
    requested: dict[str, list[str]] = {}
    for role, run in runs.items():
        served[role] = sorted(
            {
                str(turn.get("model_served"))
                for conversation in run.conversations
                for turn in conversation.get("turns") or []
                if turn.get("model_served")
            }
        )
        requested[role] = sorted(
            {
                str(turn.get("model_requested"))
                for conversation in run.conversations
                for turn in conversation.get("turns") or []
                if turn.get("model_requested")
            }
        )
    return {
        "harness_version": HARNESS_VERSION,
        "run_id": spec.run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "creator_id": spec.creator_id,
        "seed": spec.seed,
        "simulate_time": spec.simulate_time,
        "ai_stack_profile": spec.stack_profile,
        "scenario_ids": list(spec.scenario_ids)
        or [trajectory.name for trajectory in trajectories],
        "baseline_core": spec.baseline.core_id,
        "candidate_core": spec.candidate.core_id if spec.candidate else None,
        "arms": [arm.to_dict() for arm in spec.arms],
        "arm_order_per_scenario": [
            [arm.role for arm in arm_order(spec.arms, seed=spec.seed, scenario_index=index)]
            for index in range(len(spec.scenario_ids) or len(trajectories))
        ],
        "models_requested": requested,
        "models_served": served,
        "isolation": {
            "strategy": "separate simulator test fans, one per arm",
            "fans": {arm.role: arm.fan_id for arm in spec.arms},
            "provisioned_for_this_run": {
                arm.role: arm.provisioned for arm in spec.arms
            },
            "seeded_state_cleared_per_scenario": True,
            "clock_reset_per_scenario": True,
        },
        "notes": spec.notes,
        "no_quality_score": (
            "This directory contains measurements and transcripts. It contains "
            "no overall score, and the blind review is where the rubric "
            "dimensions are judged."
        ),
    }


def write_artifacts(
    *,
    spec: RunSpec,
    runs: dict[str, ArmRun],
    trajectories: Sequence[Trajectory] = (),
    results_root: Path | None = None,
) -> Path:
    """Write one run's directory and return it.

    Arm files are named after the runtime that produced them — ``semantic_v2.json``,
    ``conversational_v1.json`` — because these are the unblinded artifacts. The
    blind review is built from them separately and never carries those names.
    """
    root = Path(results_root or RESULTS_ROOT) / spec.run_id
    root.mkdir(parents=True, exist_ok=True)

    def _write(name: str, payload: Any) -> None:
        (root / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    _write(METADATA_FILE, build_metadata(spec, runs, trajectories=trajectories))
    for run in runs.values():
        _write(f"{run.arm.core_id}.json", run.to_dict())
    _write(PAIRED_FILE, pair_conversations(runs))
    _write(METRICS_FILE, build_metrics(runs))
    return root


# ---------------------------------------------------------------------------
# Loading the scenario suite
# ---------------------------------------------------------------------------


def load_scenario_file(path: Path) -> dict[str, Any]:
    """Read a scenario file, tolerating both shapes the repo already uses."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return {"trajectories": payload, "suites": {}}
    return payload if isinstance(payload, dict) else {"trajectories": [], "suites": {}}


def suite_ids(payload: dict[str, Any], suite: str) -> list[str]:
    """The scenario ids a named suite selects.

    Suites live in the data file rather than in code so adding one is a fixture
    change. ``_about`` inside the suites block is documentation and is not a
    suite.
    """
    suites = payload.get("suites") or {}
    if suite not in suites or suite == "_about":
        available = sorted(name for name in suites if name != "_about")
        raise EvaluationRefused(
            f"unknown suite {suite!r}; this file defines {', '.join(available) or 'none'}"
        )
    return [str(value) for value in suites[suite]]


def select_trajectories(
    payload: dict[str, Any],
    *,
    suite: str = "",
    scenario_ids: Sequence[str] = (),
) -> tuple[list[Trajectory], list[str]]:
    """Trajectories to run, and their stable ids, in the requested order.

    Ids are returned alongside rather than attached, because ``Trajectory`` is
    owned by ``services/trajectory_eval.py`` and an A/B-only field does not
    belong on it. Selecting a scenario that is not in the file is an error
    rather than an empty run: a suite that silently ran nothing is how an
    evaluation reports that everything passed.
    """
    rows = [row for row in (payload.get("trajectories") or []) if isinstance(row, dict)]
    by_id = {str(row.get("id") or row.get("name") or ""): row for row in rows}

    wanted: list[str]
    if scenario_ids:
        wanted = [str(value) for value in scenario_ids]
    elif suite:
        wanted = suite_ids(payload, suite)
    else:
        wanted = list(by_id)

    missing = [value for value in wanted if value not in by_id]
    if missing:
        raise EvaluationRefused(
            f"no such scenario(s): {', '.join(missing)}. This file has "
            f"{', '.join(by_id)}"
        )
    selected = [by_id[value] for value in wanted]
    from services.trajectory_eval import load_trajectories

    return load_trajectories(selected), wanted


__all__ = [
    "ArmRun",
    "ArmSpec",
    "CORE_STATE_FIELDS",
    "CORE_STATE_KEY",
    "CoreNotAvailable",
    "EvaluationRefused",
    "HARNESS_VERSION",
    "LiveSimulatorBackend",
    "METADATA_FILE",
    "METRICS_FILE",
    "PAIRED_FILE",
    "RESULTS_ROOT",
    "ROLE_BASELINE",
    "ROLE_CANDIDATE",
    "RunSpec",
    "SimulatorBackend",
    "UnfairComparison",
    "UnsafeEvaluationTarget",
    "arm_order",
    "assert_arm_isolation",
    "assert_safe_targets",
    "assert_scripted",
    "build_metadata",
    "build_metrics",
    "fan_input_digest",
    "known_core_ids",
    "load_scenario_file",
    "new_run_id",
    "observe_report",
    "observe_turn",
    "pair_conversations",
    "require_known_core",
    "select_trajectories",
    "suite_ids",
    "run_scenario_arm",
    "run_suite",
    "write_artifacts",
]
