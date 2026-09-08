"""Best-effort model usage telemetry. Telemetry must never block fan replies."""

from __future__ import annotations

import asyncio
import os
from typing import Any

from core.tasks import spawn
from models.model_runtime import (
    ModelResult,
    ModelTarget,
    ModelTelemetryContext,
    ModelUsage,
    resolve_cost_usd,
)

_MAX_PENDING_WRITES = 500
_pending_writes = 0


def telemetry_enabled() -> bool:
    return os.getenv("MODEL_TELEMETRY_ENABLED", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


async def record_model_result(
    result: ModelResult,
    context: ModelTelemetryContext,
    *,
    success: bool,
    retry_count: int = 0,
    parse_valid: bool | None = None,
    error: str | None = None,
) -> None:
    _enqueue_record(
        target=result.target,
        usage=result.usage,
        latency_ms=result.latency_ms,
        context=context,
        success=success,
        retry_count=retry_count,
        parse_valid=parse_valid,
        error=error,
        raw_response_id=result.raw_response_id,
        upstream_provider=result.upstream_provider,
        reported_cost_usd=result.reported_cost_usd,
        # getattr keeps a rolling deploy (or a test double) that predates the
        # model gate from turning telemetry into a generation failure.
        gate_wait_ms=getattr(result, "gate_wait_ms", 0) or 0,
    )


async def record_model_failure(
    target: ModelTarget,
    context: ModelTelemetryContext,
    *,
    error: str,
    retry_count: int = 0,
) -> None:
    _enqueue_record(
        target=target,
        usage=ModelUsage(),
        latency_ms=None,
        context=context,
        success=False,
        retry_count=retry_count,
        parse_valid=False,
        error=error,
        raw_response_id=None,
    )


def _enqueue_record(**kwargs: Any) -> None:
    """Queue telemetry off the reply path with a hard memory bound."""
    global _pending_writes
    if not telemetry_enabled():
        return
    if _pending_writes >= _MAX_PENDING_WRITES:
        print("[MODEL TELEMETRY] queue full; dropping event")
        return
    _pending_writes += 1

    async def _run() -> None:
        global _pending_writes
        try:
            await _record(**kwargs)
        finally:
            _pending_writes -= 1

    spawn(_run(), name="model_telemetry")


async def _record(
    *,
    target: ModelTarget,
    usage: ModelUsage,
    latency_ms: int | None,
    context: ModelTelemetryContext,
    success: bool,
    retry_count: int,
    parse_valid: bool | None,
    error: str | None,
    raw_response_id: str | None,
    upstream_provider: str | None = None,
    reported_cost_usd: float | None = None,
    gate_wait_ms: int = 0,
) -> None:
    if not telemetry_enabled():
        return

    # provider stays the logical route (for example "openrouter"). The upstream
    # that actually served the request, plus whether the cost is reported or
    # estimated, travel in metadata so no schema change is needed to answer
    # "did the pin hold and did caching work?".
    metadata = dict(context.metadata or {})
    metadata["upstream_provider"] = upstream_provider
    metadata["cost_source"] = (
        "provider_reported" if reported_cost_usd is not None else "catalog_estimate"
    )
    metadata["cached_input_tokens"] = usage.cache_read_tokens
    # Admission latency, not provider latency. Without it a saturated gate is
    # indistinguishable from a slow upstream in the telemetry table.
    metadata["model_gate_wait_ms"] = int(gate_wait_ms)
    prompt_tokens = usage.input_tokens + usage.cache_read_tokens
    metadata["cache_hit_ratio"] = (
        round(usage.cache_read_tokens / prompt_tokens, 4) if prompt_tokens else None
    )

    row: dict[str, Any] = {
        "creator_id": context.creator_id,
        "fan_id": context.fan_id,
        "feature": context.feature,
        "provider": target.provider,
        "model": target.model,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_read_tokens": usage.cache_read_tokens,
        "cache_write_tokens": usage.cache_write_tokens,
        "latency_ms": latency_ms,
        "retry_count": retry_count,
        "success": success,
        "parse_valid": parse_valid,
        "estimated_cost_usd": resolve_cost_usd(
            target,
            usage,
            reported_cost_usd=reported_cost_usd,
        ),
        "error": (error[:1000] if error else None),
        "raw_response_id": raw_response_id,
        "evaluation_run_id": context.evaluation_run_id,
        "scenario_id": context.scenario_id,
        "metadata": metadata,
    }

    try:
        from core.supabase import get_supabase

        await asyncio.to_thread(
            lambda: get_supabase().table("model_usage_events").insert(row).execute()
        )
    except Exception as exc:
        print(f"[MODEL TELEMETRY] write failed: {exc}")
