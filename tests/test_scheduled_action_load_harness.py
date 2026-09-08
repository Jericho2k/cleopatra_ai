"""Part 7: the synthetic load harness, run small enough to live in CI.

The full sweep (10 / 50 / 100 actions at production-like latencies) lives in
``scripts/load_test_scheduled_actions.py``. These tests run the same harness at
compressed latencies so the properties it measures — bounded concurrency, no
duplicate sends, backlog drain, failure and cancellation handling — are
regression-tested on every commit rather than only when someone remembers.
"""
from __future__ import annotations

import asyncio

from scripts.load_test_scheduled_actions import Scenario, drain

FAST = {
    "model_latency": 0.01,
    "db_latency": 0.0002,
    "api_latency": 0.005,
    "composition_delay": 0.02,
}


def run(coro):
    return asyncio.run(coro)


def test_burst_of_ten_drains_with_real_parallelism():
    result = run(drain(
        size=10, concurrency=8, claim_limit=24, model_concurrency=8,
        scenario=Scenario(**FAST),
    ))
    assert result["sends"] == 10
    assert result["duplicate_sends"] == 0
    assert result["same_fan_overlaps"] == 0
    assert result["max_action_concurrency"] == 8
    assert result["errors"] == 0


def test_burst_of_fifty_respects_both_ceilings():
    result = run(drain(
        size=50, concurrency=8, claim_limit=24, model_concurrency=6,
        scenario=Scenario(**FAST),
    ))
    assert result["sends"] == 50
    assert result["duplicate_sends"] == 0
    assert result["max_action_concurrency"] <= 8
    assert result["max_model_concurrency"] <= 6, "model fan-out must stay bounded"
    # Backlog keeps draining rather than one batch per idle poll.
    assert result["cycles"] >= 2
    assert result["claim_calls"] >= 2


def test_burst_of_one_hundred_drains_completely():
    result = run(drain(
        size=100, concurrency=8, claim_limit=24, model_concurrency=8,
        scenario=Scenario(**FAST),
    ))
    assert result["sends"] == 100
    assert result["duplicate_sends"] == 0
    assert result["errors"] == 0
    assert result["max_action_concurrency"] <= 8


def test_bounded_concurrency_beats_the_sequential_shape():
    """The load harness must be able to tell the two apart. That is the point."""
    sequential = run(drain(
        size=24, concurrency=1, claim_limit=20, model_concurrency=1,
        scenario=Scenario(model_latency=0.02, db_latency=0.0001,
                          api_latency=0.005, composition_delay=0.02),
    ))
    concurrent = run(drain(
        size=24, concurrency=8, claim_limit=24, model_concurrency=8,
        scenario=Scenario(model_latency=0.02, db_latency=0.0001,
                          api_latency=0.005, composition_delay=0.02),
    ))

    assert sequential["max_action_concurrency"] == 1
    assert concurrent["max_action_concurrency"] == 8
    assert concurrent["drain_seconds"] < sequential["drain_seconds"]
    assert concurrent["actions_per_minute"] > sequential["actions_per_minute"] * 2


def test_slower_model_does_not_change_the_concurrency_bound():
    slow = run(drain(
        size=16, concurrency=8, claim_limit=24, model_concurrency=4,
        scenario=Scenario(**{**FAST, "model_latency": 0.05}),
    ))
    assert slow["max_model_concurrency"] <= 4
    assert slow["sends"] == 16
    assert slow["duplicate_sends"] == 0


def test_one_model_exception_fails_only_its_own_action():
    scenario = Scenario(**FAST)
    scenario.model_failure_ids = {"action-3"}
    result = run(drain(
        size=10, concurrency=8, claim_limit=24, model_concurrency=8,
        scenario=scenario,
    ))
    assert result["sends"] == 9
    assert result["errors"] == 1
    assert result["model_failures"] >= 1
    assert result["duplicate_sends"] == 0


def test_provider_rate_limit_fails_the_action_for_the_queue_to_retry():
    """A 429 is a queue retry with backoff, not an in-line hammering loop."""
    scenario = Scenario(**FAST)
    scenario.rate_limited_ids = {"action-2"}
    result = run(drain(
        size=6, concurrency=6, claim_limit=24, model_concurrency=6,
        scenario=scenario,
    ))
    assert result["rate_limit_retries"] == 1
    assert result["errors"] == 1
    assert result["sends"] == 5


def test_cancelled_action_while_queued_sends_nothing():
    scenario = Scenario(**FAST)
    scenario.cancel_ids = {"action-1"}
    result = run(drain(
        size=6, concurrency=2, claim_limit=24, model_concurrency=2,
        scenario=scenario,
    ))
    assert result["sends"] == 5
    assert "action-1" not in result
    assert result["duplicate_sends"] == 0


def test_newer_fan_message_stops_a_queued_action_from_sending():
    scenario = Scenario(**FAST)
    scenario.superseded_fans = {"fan-4"}
    result = run(drain(
        size=8, concurrency=8, claim_limit=24, model_concurrency=8,
        scenario=scenario,
    ))
    assert result["sends"] == 7
    assert result["errors"] == 0
    assert result["skipped"] == 1
    # Crucially, the superseded action consumed no model budget.
    assert result["model_calls"] == 14


def test_composition_delay_is_reported_separately_from_throughput():
    """Human realism is not optimised away to inflate the benchmark."""
    result = run(drain(
        size=8, concurrency=8, claim_limit=24, model_concurrency=8,
        scenario=Scenario(**{**FAST, "composition_delay": 0.05}),
    ))
    assert result["composition_delay_seconds_per_action"] == 0.05
    # Every action still paid its full deliberate delay.
    assert result["p50_completion_seconds"] >= 0.05
