"""SCALE-004: global model admission control.

The gate lives at ``ai.model_providers.complete``, so these tests drive that
function with a fake provider rather than the semaphore in isolation — the
failure mode being guarded against is "someone wrapped generate_replies and left
the analyzer unbounded", which only a test at the shared layer can catch.
"""
from __future__ import annotations

import asyncio

import pytest

from ai import model_providers
from core.model_gate import MODEL_GATE, ModelGate, configured_model_concurrency
from models.model_runtime import ModelResult, ModelTarget, ModelUsage


def run(coro):
    return asyncio.run(coro)


def target(provider: str = "together") -> ModelTarget:
    return ModelTarget(
        name=f"{provider}:test",
        provider=provider,
        model="test-model",
        base_url="https://example.test/v1",
        api_key_env="TOGETHER_API_KEY",
    )


class Probe:
    def __init__(self, delay: float = 0.02):
        self.delay = delay
        self.inflight = 0
        self.max_inflight = 0
        self.calls = 0

    async def __call__(self, target_, **_kwargs):
        self.calls += 1
        self.inflight += 1
        self.max_inflight = max(self.max_inflight, self.inflight)
        try:
            await asyncio.sleep(self.delay)
            return ModelResult(
                text="ok", target=target_, usage=ModelUsage(), latency_ms=0
            )
        finally:
            self.inflight -= 1


@pytest.fixture(autouse=True)
def clean_gate():
    MODEL_GATE.reset()
    yield
    MODEL_GATE.reset()


def _call(provider="together"):
    return model_providers.complete(
        target(provider), system="s", messages=[{"role": "user", "content": "hi"}],
        max_tokens=10,
    )


def test_fifty_generations_queue_behind_ten_slots(monkeypatch):
    """Nothing is dropped, nothing is retried, and only N run at once."""
    monkeypatch.setenv("MODEL_MAX_CONCURRENCY", "10")
    probe = Probe(delay=0.02)
    monkeypatch.setattr(model_providers, "_complete_openai_compatible", probe)

    async def main():
        return await asyncio.gather(*[_call() for _ in range(50)])

    results = run(main())

    assert len(results) == 50
    assert probe.calls == 50, "queued work must run, not be shed"
    assert probe.max_inflight == 10, "the limit is the limit"


def test_the_gate_covers_every_provider_route(monkeypatch):
    """Kimi/OpenRouter, DeepSeek/Together and the analyzer share one budget."""
    monkeypatch.setenv("MODEL_MAX_CONCURRENCY", "3")
    probe = Probe(delay=0.02)
    anthropic_probe = Probe(delay=0.02)
    # Both provider families count against the same ceiling.
    anthropic_probe.inflight = 0
    monkeypatch.setattr(model_providers, "_complete_openai_compatible", probe)

    async def shared_anthropic(target_, **kwargs):
        probe.inflight += 1
        probe.max_inflight = max(probe.max_inflight, probe.inflight)
        try:
            await asyncio.sleep(0.02)
            return ModelResult(
                text="ok", target=target_, usage=ModelUsage(), latency_ms=0
            )
        finally:
            probe.inflight -= 1

    monkeypatch.setattr(model_providers, "_complete_anthropic", shared_anthropic)

    async def main():
        return await asyncio.gather(
            *[_call("openrouter") for _ in range(6)],
            *[_call("together") for _ in range(6)],
            *[_call("anthropic") for _ in range(6)],
        )

    run(main())
    assert probe.max_inflight == 3


def test_gate_wait_is_reported_separately_from_provider_latency(monkeypatch):
    monkeypatch.setenv("MODEL_MAX_CONCURRENCY", "1")
    probe = Probe(delay=0.05)
    monkeypatch.setattr(model_providers, "_complete_openai_compatible", probe)

    async def main():
        return await asyncio.gather(_call(), _call())

    first, second = run(main())
    waited = max(first.gate_wait_ms, second.gate_wait_ms)
    ran_first = min(first.gate_wait_ms, second.gate_wait_ms)

    # One went straight through; the other waited roughly a provider call.
    assert ran_first < 20
    assert waited >= 40
    # And provider latency stayed provider latency for both.
    assert first.latency_ms < 200 and second.latency_ms < 200


def test_slot_is_released_on_exception(monkeypatch):
    monkeypatch.setenv("MODEL_MAX_CONCURRENCY", "1")

    async def blowing_up(target_, **_kwargs):
        raise RuntimeError("provider exploded")

    monkeypatch.setattr(model_providers, "_complete_openai_compatible", blowing_up)

    async def main():
        for _ in range(3):
            with pytest.raises(RuntimeError):
                await _call()
        # If the slot leaked, this would hang forever rather than return.
        return MODEL_GATE.snapshot()

    snapshot = run(asyncio.wait_for(main(), timeout=2))
    assert snapshot["inflight"] == 0


def test_slot_is_released_on_timeout(monkeypatch):
    monkeypatch.setenv("MODEL_MAX_CONCURRENCY", "1")

    async def slow(target_, **_kwargs):
        await asyncio.sleep(5)

    monkeypatch.setattr(model_providers, "_complete_openai_compatible", slow)

    async def main():
        with pytest.raises((asyncio.TimeoutError, TimeoutError)):
            await asyncio.wait_for(_call(), timeout=0.05)
        return MODEL_GATE.snapshot()

    snapshot = run(asyncio.wait_for(main(), timeout=2))
    assert snapshot["inflight"] == 0


def test_slot_is_released_on_cancellation(monkeypatch):
    monkeypatch.setenv("MODEL_MAX_CONCURRENCY", "1")

    async def slow(target_, **_kwargs):
        await asyncio.sleep(5)

    monkeypatch.setattr(model_providers, "_complete_openai_compatible", slow)

    async def main():
        task = asyncio.create_task(_call())
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return MODEL_GATE.snapshot()

    snapshot = run(asyncio.wait_for(main(), timeout=2))
    assert snapshot["inflight"] == 0


def test_cancelling_while_waiting_for_a_slot_runs_no_provider_call(monkeypatch):
    """An action cancelled in the queue must never reach the provider."""
    monkeypatch.setenv("MODEL_MAX_CONCURRENCY", "1")
    probe = Probe(delay=0.2)
    monkeypatch.setattr(model_providers, "_complete_openai_compatible", probe)

    async def main():
        holder = asyncio.create_task(_call())
        await asyncio.sleep(0.02)
        waiter = asyncio.create_task(_call())
        await asyncio.sleep(0.02)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        await holder
        return probe.calls

    calls = run(asyncio.wait_for(main(), timeout=3))
    assert calls == 1, "the cancelled request must not become a stale provider call"


def test_limit_is_configuration_not_the_http_pool(monkeypatch):
    monkeypatch.delenv("MODEL_MAX_CONCURRENCY", raising=False)
    assert configured_model_concurrency() == 8
    monkeypatch.setenv("MODEL_MAX_CONCURRENCY", "16")
    assert configured_model_concurrency() == 16
    monkeypatch.setenv("MODEL_MAX_CONCURRENCY", "0")
    assert configured_model_concurrency() == 1
    monkeypatch.setenv("MODEL_MAX_CONCURRENCY", "not-a-number")
    assert configured_model_concurrency() == 8


def test_snapshot_reports_saturation_without_leaking_content(monkeypatch):
    monkeypatch.setenv("MODEL_MAX_CONCURRENCY", "2")
    probe = Probe(delay=0.05)
    monkeypatch.setattr(model_providers, "_complete_openai_compatible", probe)
    seen = {}

    async def main():
        tasks = [asyncio.create_task(_call()) for _ in range(6)]
        await asyncio.sleep(0.02)
        seen.update(MODEL_GATE.snapshot())
        await asyncio.gather(*tasks)

    run(main())
    assert seen["limit"] == 2
    assert seen["inflight"] == 2
    assert seen["waiting"] == 4
    assert set(seen) == {
        "limit",
        "inflight",
        "waiting",
        "max_inflight",
        "acquired_total",
        "avg_wait_ms",
        "max_wait_ms",
        "recent_wait_ms",
    }


def test_gate_survives_more_than_one_event_loop():
    """asyncio primitives bind to a loop; the gate must not break across runs."""
    gate = ModelGate()

    async def once():
        async with gate.acquire():
            return True

    assert run(once()) is True
    assert run(once()) is True
