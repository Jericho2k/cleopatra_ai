"""The repeatable stability gate, and what it is allowed to claim.

The point of this suite is not that a number is good. It is that the harness
drives the REAL pipeline and that its arithmetic is honest — a reliability
report that measures its own stub is worse than no report.
"""

from __future__ import annotations

import io
import contextlib

import pytest

from services import owner_stability_eval as gate


def quiet(coro):
    """Run a harness coroutine without its per-call production log lines."""
    import asyncio

    with contextlib.redirect_stdout(io.StringIO()):
        return asyncio.run(coro)


@pytest.mark.asyncio
async def test_every_named_response_shape_is_generated_and_survives_the_pipeline():
    """No shape may crash the runtime, whatever it does to the turn."""
    from models.conversational_core import ConversationalWorkingState
    from models.model_runtime import ModelTarget

    target = ModelTarget(name="t", provider="synthetic", model="synthetic-owner")
    for index, shape in enumerate(gate.RESPONSE_SHAPES):
        owner = gate.SyntheticOwner(
            seed=index, shape_weights={shape: 1.0}, repair_success_rate=1.0
        )
        loaded = gate.synthetic_loaded(
            gate.synthetic_snapshot(index, "hi"), target=target
        )
        with contextlib.redirect_stdout(io.StringIO()):
            observation, _state = await gate.run_turn(
                turn_index=index,
                loaded=loaded,
                working_state=ConversationalWorkingState(),
                owner_complete=owner,
            )
        assert observation.outcome in {
            gate.OUTCOME_REPLIED,
            gate.OUTCOME_NO_SEND,
            gate.OUTCOME_HANDOFF,
            gate.OUTCOME_OWNER_FAILED,
        }, shape
        assert observation.owner_calls in (1, 2), shape


@pytest.mark.asyncio
async def test_a_shape_that_only_loses_its_operation_still_replies():
    from models.conversational_core import ConversationalWorkingState
    from models.model_runtime import ModelTarget

    target = ModelTarget(name="t", provider="synthetic", model="synthetic-owner")
    owner = gate.SyntheticOwner(seed=1, shape_weights={"invalid_operation_refs": 1.0})
    loaded = gate.synthetic_loaded(gate.synthetic_snapshot(0, "hi"), target=target)
    with contextlib.redirect_stdout(io.StringIO()):
        observation, _state = await gate.run_turn(
            turn_index=0,
            loaded=loaded,
            working_state=ConversationalWorkingState(),
            owner_complete=owner,
        )
    assert observation.outcome == gate.OUTCOME_REPLIED
    assert observation.operation_rejected is True
    assert observation.operation_executed == "none"
    assert observation.owner_calls == 1


@pytest.mark.asyncio
async def test_a_shape_that_only_loses_its_delta_still_replies():
    from models.conversational_core import ConversationalWorkingState
    from models.model_runtime import ModelTarget

    target = ModelTarget(name="t", provider="synthetic", model="synthetic-owner")
    owner = gate.SyntheticOwner(seed=2, shape_weights={"malformed_state_delta": 1.0})
    loaded = gate.synthetic_loaded(gate.synthetic_snapshot(0, "hi"), target=target)
    with contextlib.redirect_stdout(io.StringIO()):
        observation, _state = await gate.run_turn(
            turn_index=0,
            loaded=loaded,
            working_state=ConversationalWorkingState(),
            owner_complete=owner,
        )
    assert observation.outcome == gate.OUTCOME_REPLIED
    assert observation.state_delta_rejected is True


def test_a_run_of_many_turns_reports_every_required_rate():
    report = quiet(gate.run_synthetic_stability(turns=120, seed=5))
    metrics = report.metrics()
    for key in (
        "owner_failed_rate",
        "malformed_first_response_rate",
        "repair_attempt_rate",
        "repair_success_rate",
        "no_send_rate",
        "operation_rejection_rate",
        "state_delta_rejection_rate",
        "latency_ms_p50",
        "latency_ms_p95",
    ):
        assert key in metrics, key
    assert metrics["turns"] == 120
    assert len(report.turns) == 120


def test_the_run_is_deterministic_for_one_seed():
    first = quiet(gate.run_synthetic_stability(turns=60, seed=13)).metrics()
    second = quiet(gate.run_synthetic_stability(turns=60, seed=13)).metrics()
    assert first == second


def test_a_healthy_run_almost_never_ends_in_owner_failed():
    """The gate's whole reason for existing, stated as an assertion.

    Roughly one first response in twenty is unusable in this mix, and a repair
    answers three quarters of those. If a recoverable shape starts killing turns
    again, this is where it shows up.
    """
    report = quiet(gate.run_synthetic_stability(turns=400, seed=21))
    metrics = report.metrics()
    assert metrics["malformed_first_response_rate"] > 0, "the mix must exercise failure"
    assert metrics["owner_failed_rate"] <= 0.02
    assert metrics["reply_rate"] >= 0.95
    assert metrics["owner_calls_per_turn"] < 1.3, "ordinary turns stay at one call"


def test_a_run_where_no_repair_ever_works_still_bounds_the_call_budget():
    report = quiet(
        gate.run_synthetic_stability(turns=150, seed=4, repair_success_rate=0.0)
    )
    metrics = report.metrics()
    assert metrics["repair_success_rate"] == 0.0
    assert metrics["owner_failed_rate"] > 0, "a failed repair must still fail the turn"
    assert all(turn.owner_calls <= 2 for turn in report.turns)


def test_percentiles_are_arithmetic_over_the_recorded_turns():
    assert gate._percentile([], 0.5) == 0
    assert gate._percentile([5], 0.95) == 5
    assert gate._percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.5) == 5
    assert gate._percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.95) == 10


def test_the_harness_never_executes_a_commercial_operation():
    report = quiet(gate.run_synthetic_stability(turns=80, seed=9))
    assert any(turn.operation_proposed != "none" for turn in report.turns)
    assert all(turn.operation_executed == "none" for turn in report.turns), (
        "the synthetic world has no sellable offer, so authority must refuse every one"
    )


def test_the_cli_reports_metrics_and_can_gate_on_them():
    from scripts import run_owner_stability_eval as cli

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = cli.main(["--turns", "40", "--seed", "2"])
    assert code == 0
    import json

    metrics = json.loads(buffer.getvalue())
    assert metrics["turns"] == 40
    assert metrics["version"] == gate.REPORT_VERSION

    with contextlib.redirect_stdout(io.StringIO()):
        failing = cli.main(
            [
                "--turns",
                "40",
                "--seed",
                "2",
                "--repair-success-rate",
                "0",
                "--fail-over-owner-failed",
                "0.0",
            ]
        )
    assert failing == 1
