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


# ---------------------------------------------------------------------------
# Conversation scale
#
# The sweep above measures one action type draining. These measure the property
# that matters once a reply is a planned outbound sequence: many creators, many
# fans, multi-bubble replies, and fan messages landing BETWEEN bubbles from
# outside the process. The two hard requirements are zero stale sends and zero
# same-fan overlaps; the rest is reported so a regression in shape is visible.
#
# Run at compressed latencies so it belongs in CI. The production-like sweep is
# `python scripts/load_test_scheduled_actions.py --scale`.
# ---------------------------------------------------------------------------

import contextlib  # noqa: E402
import io  # noqa: E402

from scripts.load_test_scheduled_actions import (  # noqa: E402
    ConversationScenario,
    drain_conversations,
)


def drain_quietly(scenario: ConversationScenario) -> dict:
    """The worker narrates every action. A thousand of them is not test output."""
    with contextlib.redirect_stdout(io.StringIO()):
        return run(drain_conversations(scenario))


SCALE = {
    "model_latency": 0.002,
    "db_latency": 0.00005,
    "api_latency": 0.0005,
    "time_scale": 0.002,
}


def test_many_parallel_conversations_never_send_a_stale_bubble():
    result = drain_quietly(
        ConversationScenario(
            creators=25, fans_per_creator=4, interrupt_fraction=0.4, **SCALE
        )
    )

    assert result["planned_sequences"] == result["conversations"]
    assert result["stale_sends"] == 0
    assert result["duplicate_sends"] == 0
    assert result["same_fan_overlaps"] == 0
    assert result["queue_errors"] == 0


def test_mid_sequence_interruptions_actually_happen_and_are_absorbed():
    """A test where nothing was interrupted would prove nothing."""
    result = drain_quietly(
        ConversationScenario(
            creators=20, fans_per_creator=5, interrupt_fraction=0.6, **SCALE
        )
    )

    assert result["superseded_sequences"] > 0
    assert result["stale_sends"] == 0
    assert result["bubbles_sent"] > 0


def test_a_burst_collapses_into_one_reply_obligation():
    result = drain_quietly(
        ConversationScenario(
            creators=10,
            fans_per_creator=3,
            burst_messages=4,
            interrupt_fraction=0.0,
            **SCALE,
        )
    )

    # Four inbound messages, one planned reply per conversation.
    assert result["planned_sequences"] == result["conversations"]
    assert result["model_calls"] == result["conversations"] * 2


def test_human_delay_occupies_queue_rows_rather_than_worker_slots():
    """The sprint's central capacity claim, stated as an assertion."""
    result = drain_quietly(
        ConversationScenario(
            creators=25, fans_per_creator=4, interrupt_fraction=0.0, **SCALE
        )
    )

    # Far more bubbles are waiting out a deliberate pause at once than there
    # are worker slots. Under the previous design each of those would have been
    # a coroutine holding one of eight slots.
    assert result["peak_pending_human_delay_actions"] > result["max_action_concurrency"]
    assert result["max_action_concurrency"] <= 8
    assert result["max_model_concurrency"] <= 8
