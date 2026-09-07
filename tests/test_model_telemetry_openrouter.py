"""Writer telemetry must answer 'did the pin hold and did caching work?'."""

from __future__ import annotations

import asyncio

import pytest

from models.model_runtime import (
    ModelResult,
    ModelTarget,
    ModelTelemetryContext,
    ModelUsage,
)
from services import model_telemetry


@pytest.fixture
def enqueued(monkeypatch):
    """Capture what record_model_result hands to the queued writer."""
    calls: list[dict] = []
    monkeypatch.setattr(model_telemetry, "telemetry_enabled", lambda: True)
    monkeypatch.setattr(
        model_telemetry,
        "_enqueue_record",
        lambda **kwargs: calls.append(kwargs),
    )
    return calls


TARGET = ModelTarget(
    name="openrouter:moonshotai/kimi-k2.6",
    provider="openrouter",
    model="moonshotai/kimi-k2.6",
    base_url="https://openrouter.ai/api/v1",
    api_key_env="OPENROUTER_API_KEY",
    input_per_million=0.80,
    output_per_million=3.50,
    cache_read_per_million=0.16,
)


def _result(**overrides):
    values = {
        "text": "ok",
        "target": TARGET,
        "usage": ModelUsage(
            input_tokens=200,
            output_tokens=50,
            cache_read_tokens=800,
        ),
        "latency_ms": 900,
        "raw_response_id": "gen-1",
        "upstream_provider": "Inceptron",
        "reported_cost_usd": 0.00042,
    }
    values.update(overrides)
    return ModelResult(**values)


def _context():
    return ModelTelemetryContext(
        feature="assisted_reply",
        creator_id="creator-1",
        fan_id="fan-1",
        metadata={"writer_route": "default"},
    )


def test_upstream_provider_and_reported_cost_reach_the_writer(enqueued):
    asyncio.run(
        model_telemetry.record_model_result(_result(), _context(), success=True)
    )

    row = enqueued[0]
    assert row["target"].provider == "openrouter"
    assert row["upstream_provider"] == "Inceptron"
    assert row["reported_cost_usd"] == 0.00042
    assert row["usage"].cache_read_tokens == 800


def test_recorded_row_keeps_logical_provider_and_normalized_cache_fields(monkeypatch):
    rows: list[dict] = []
    monkeypatch.setattr(model_telemetry, "telemetry_enabled", lambda: True)

    class _Table:
        def table(self, _name):
            return self

        def insert(self, row):
            rows.append(row)
            return self

        def execute(self):
            return None

    monkeypatch.setitem(
        __import__("sys").modules,
        "core.supabase",
        type("_Module", (), {"get_supabase": staticmethod(lambda: _Table())}),
    )

    asyncio.run(
        model_telemetry._record(
            target=TARGET,
            usage=ModelUsage(input_tokens=200, output_tokens=50, cache_read_tokens=800),
            latency_ms=900,
            context=_context(),
            success=True,
            retry_count=0,
            parse_valid=True,
            error=None,
            raw_response_id="gen-1",
            upstream_provider="Inceptron",
            reported_cost_usd=0.00042,
        )
    )

    row = rows[0]
    # The logical route stays in the provider column; the upstream that served
    # it is additive metadata, not a replacement.
    assert row["provider"] == "openrouter"
    assert row["model"] == "moonshotai/kimi-k2.6"
    assert row["metadata"]["upstream_provider"] == "Inceptron"
    assert row["metadata"]["writer_route"] == "default"

    # Cached input is accounted once, and the hit ratio is derived from the
    # same numbers rather than a second counter.
    assert row["input_tokens"] == 200
    assert row["cache_read_tokens"] == 800
    assert row["metadata"]["cached_input_tokens"] == 800
    assert row["metadata"]["cache_hit_ratio"] == 0.8

    # Reported cost wins over the catalog estimate and says so.
    assert row["estimated_cost_usd"] == 0.00042
    assert row["metadata"]["cost_source"] == "provider_reported"

    # No prompt or conversation content is ever written.
    assert set(row) == {
        "creator_id",
        "fan_id",
        "feature",
        "provider",
        "model",
        "input_tokens",
        "output_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "latency_ms",
        "retry_count",
        "success",
        "parse_valid",
        "estimated_cost_usd",
        "error",
        "raw_response_id",
        "evaluation_run_id",
        "scenario_id",
        "metadata",
    }


def test_cost_source_reports_an_estimate_when_the_provider_does_not(monkeypatch):
    rows: list[dict] = []
    monkeypatch.setattr(model_telemetry, "telemetry_enabled", lambda: True)

    class _Table:
        def table(self, _name):
            return self

        def insert(self, row):
            rows.append(row)
            return self

        def execute(self):
            return None

    monkeypatch.setitem(
        __import__("sys").modules,
        "core.supabase",
        type("_Module", (), {"get_supabase": staticmethod(lambda: _Table())}),
    )

    asyncio.run(
        model_telemetry._record(
            target=TARGET,
            usage=ModelUsage(input_tokens=1_000_000, output_tokens=0),
            latency_ms=10,
            context=_context(),
            success=True,
            retry_count=0,
            parse_valid=True,
            error=None,
            raw_response_id=None,
        )
    )

    row = rows[0]
    assert row["metadata"]["cost_source"] == "catalog_estimate"
    assert row["estimated_cost_usd"] == 0.80
    assert row["metadata"]["cache_hit_ratio"] == 0.0
