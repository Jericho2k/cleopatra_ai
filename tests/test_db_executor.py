"""The database thread pool is a capacity limit, so it must be visible.

Every Supabase call in this codebase is synchronous and reaches the client
through asyncio.to_thread, which submits to the event loop's DEFAULT executor.
Nothing ever configured one, so the ceiling on concurrent database work was
whatever ThreadPoolExecutor() picks on its own: min(32, cpu_count + 4). On a
4-vCPU container that is EIGHT concurrent database calls for the entire
process — shared between the worker's action slots, the schedulers, and every
inbound webhook.

That is a real limit and it appeared in no configuration, no log line and no
health document, and it moved silently when the container size changed. These
tests are about it being explicit, bounded and observable.
"""

from __future__ import annotations

import asyncio

import pytest

from core import db_executor


@pytest.fixture(autouse=True)
def _clean_executor():
    db_executor.shutdown()
    yield
    db_executor.shutdown()


def test_the_default_is_higher_than_a_small_container_would_pick():
    """The whole point: on a 4-vCPU box Python would choose 8."""
    assert db_executor.configured_max_workers() == db_executor.DEFAULT_MAX_WORKERS
    assert db_executor.DEFAULT_MAX_WORKERS > 8


def test_it_is_tunable(monkeypatch):
    monkeypatch.setenv("DB_EXECUTOR_MAX_WORKERS", "64")
    assert db_executor.configured_max_workers() == 64


def test_a_nonsense_value_falls_back_to_the_default(monkeypatch):
    monkeypatch.setenv("DB_EXECUTOR_MAX_WORKERS", "not-a-number")
    assert db_executor.configured_max_workers() == db_executor.DEFAULT_MAX_WORKERS


def test_it_cannot_be_set_below_the_workers_own_concurrency(monkeypatch):
    """Below the worker's action concurrency this would serialise the product
    rather than protect anything."""
    monkeypatch.setenv("DB_EXECUTOR_MAX_WORKERS", "1")
    assert db_executor.configured_max_workers() == 4


def test_it_cannot_be_set_absurdly_high(monkeypatch):
    monkeypatch.setenv("DB_EXECUTOR_MAX_WORKERS", "100000")
    assert db_executor.configured_max_workers() == 128


def test_installing_replaces_the_default_executor():
    async def scenario():
        loop = asyncio.get_running_loop()
        executor = db_executor.install(loop)
        assert executor is not None
        # to_thread now runs on the pool we chose, not the one Python guessed.
        result = await asyncio.to_thread(lambda: "ran")
        assert result == "ran"
        return executor

    executor = asyncio.run(scenario())
    assert executor._max_workers == db_executor.DEFAULT_MAX_WORKERS


def test_installing_twice_does_not_leak_a_second_pool():
    async def scenario():
        loop = asyncio.get_running_loop()
        return db_executor.install(loop), db_executor.install(loop)

    first, second = asyncio.run(scenario())
    assert first is second


def test_the_limit_actually_bounds_concurrency(monkeypatch):
    """A limit that does not limit is worse than no limit, because it is
    reported in the health document as though it does."""
    monkeypatch.setenv("DB_EXECUTOR_MAX_WORKERS", "4")

    peak = 0
    inflight = 0

    async def scenario():
        nonlocal peak, inflight
        db_executor.install(asyncio.get_running_loop())
        import threading

        counter_lock = threading.Lock()

        def blocking():
            nonlocal peak, inflight
            with counter_lock:
                inflight += 1
                peak = max(peak, inflight)
            import time

            time.sleep(0.05)
            with counter_lock:
                inflight -= 1

        await asyncio.gather(*(asyncio.to_thread(blocking) for _ in range(40)))

    asyncio.run(scenario())
    assert peak <= 4, f"the pool ran {peak} concurrently with a limit of 4"
    assert peak > 1, "nothing ran concurrently; the test proved nothing"


def test_the_snapshot_reports_the_limit_before_installation():
    snapshot = db_executor.snapshot()

    assert snapshot["configured"] is False
    assert snapshot["max_workers"] == db_executor.DEFAULT_MAX_WORKERS


def test_the_snapshot_reports_queue_depth_after_installation():
    async def scenario():
        db_executor.install(asyncio.get_running_loop())
        await asyncio.to_thread(lambda: None)
        return db_executor.snapshot()

    snapshot = asyncio.run(scenario())

    assert snapshot["configured"] is True
    assert snapshot["max_workers"] == db_executor.DEFAULT_MAX_WORKERS
    assert snapshot["queued"] == 0
    assert snapshot["threads"] >= 1


def test_shutdown_is_idempotent():
    async def scenario():
        db_executor.install(asyncio.get_running_loop())

    asyncio.run(scenario())
    db_executor.shutdown()
    db_executor.shutdown()
    assert db_executor.snapshot()["configured"] is False


def test_the_health_document_reports_the_database_pool():
    """An operator has to be able to tell 'the pool is the bottleneck' from
    'the database is slow'. They look identical in latency alone."""
    from services import operational_health

    async def scenario():
        db_executor.install(asyncio.get_running_loop())
        operational_health.reset_cache()

        async def fake_db():
            return {"reachable": True, "latency_ms": 1.0, "error": None}

        async def fake_queue():
            return {"available": True, "pending": 0, "processing": 0,
                    "failed": 0, "oldest_pending_age_seconds": 0.0,
                    "pending_inbound_messages": 0, "error": None}

        original = (operational_health.probe_database, operational_health.probe_queue)
        operational_health.probe_database = fake_db
        operational_health.probe_queue = fake_queue
        try:
            return await operational_health.collect(use_cache=False)
        finally:
            operational_health.probe_database, operational_health.probe_queue = original

    document = asyncio.run(scenario())

    assert "db_executor" in document
    assert document["db_executor"]["max_workers"] == db_executor.DEFAULT_MAX_WORKERS
    assert document["db_executor"]["configured"] is True
