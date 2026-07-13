"""Best-effort model usage telemetry. Telemetry must never block fan replies."""

from __future__ import annotations

import asyncio
import os
from typing import Any

from models.model_runtime import (
    ModelResult,
    ModelTarget,
    ModelTelemetryContext,
    ModelUsage,
    estimate_cost_usd,
)


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
    await _record(
        target=result.target,
        usage=result.usage,
        latency_ms=result.latency_ms,
        context=context,
        success=success,
        retry_count=retry_count,
        parse_valid=parse_valid,
        error=error,
        raw_response_id=result.raw_response_id,
    )


async def record_model_failure(
    target: ModelTarget,
    context: ModelTelemetryContext,
    *,
    error: str,
    retry_count: int = 0,
) -> None:
    await _record(
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
) -> None:
    if not telemetry_enabled():
        return

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
        "estimated_cost_usd": estimate_cost_usd(target, usage),
        "error": (error[:1000] if error else None),
        "raw_response_id": raw_response_id,
        "evaluation_run_id": context.evaluation_run_id,
        "scenario_id": context.scenario_id,
        "metadata": context.metadata,
    }

    try:
        from core.supabase import get_supabase

        await asyncio.to_thread(
            lambda: get_supabase().table("model_usage_events").insert(row).execute()
        )
    except Exception as exc:
        print(f"[MODEL TELEMETRY] write failed: {exc}")
