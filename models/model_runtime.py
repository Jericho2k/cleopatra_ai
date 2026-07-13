"""Provider-neutral runtime types and cost estimation for Cleopatra models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModelTarget:
    """A concrete model endpoint plus its current pricing metadata."""

    name: str
    provider: str
    model: str
    base_url: str | None = None
    api_key_env: str | None = None

    input_per_million: float = 0.0
    output_per_million: float = 0.0
    cache_read_per_million: float = 0.0
    cache_write_per_million: float = 0.0

    adult_policy: str = "unverified"
    enabled: bool = True

    stream: bool = False
    timeout_seconds: float = 45.0

    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "ModelTarget":
        return cls(
            name=str(data.get("name") or f"{data.get('provider')}:{data.get('model')}"),
            provider=str(data["provider"]).strip().lower(),
            model=str(data["model"]).strip(),
            base_url=(str(data["base_url"]).strip() if data.get("base_url") else None),
            api_key_env=(str(data["api_key_env"]).strip() if data.get("api_key_env") else None),
            input_per_million=float(data.get("input_per_million") or 0.0),
            output_per_million=float(data.get("output_per_million") or 0.0),
            cache_read_per_million=float(data.get("cache_read_per_million") or 0.0),
            cache_write_per_million=float(data.get("cache_write_per_million") or 0.0),
            adult_policy=str(data.get("adult_policy") or "unverified"),
            enabled=bool(data.get("enabled", True)),
            metadata=dict(data.get("metadata") or {}),
            stream=bool(data.get("stream", False)),
            timeout_seconds=float(data.get("timeout_seconds") or 45.0),
        )


@dataclass(frozen=True)
class ModelUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


@dataclass(frozen=True)
class ModelResult:
    text: str
    target: ModelTarget
    usage: ModelUsage
    latency_ms: int
    raw_response_id: str | None = None


@dataclass(frozen=True)
class ModelTelemetryContext:
    feature: str
    creator_id: str | None = None
    fan_id: str | None = None
    evaluation_run_id: str | None = None
    scenario_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def estimate_cost_usd(target: ModelTarget, usage: ModelUsage) -> float:
    """Estimate provider cost without double-counting cached input."""

    uncached_input = max(int(usage.input_tokens), 0)
    output = max(int(usage.output_tokens), 0)
    cache_read = max(int(usage.cache_read_tokens), 0)
    cache_write = max(int(usage.cache_write_tokens), 0)

    cost = (
        uncached_input * target.input_per_million
        + output * target.output_per_million
        + cache_read * target.cache_read_per_million
        + cache_write * target.cache_write_per_million
    ) / 1_000_000
    return round(cost, 8)
