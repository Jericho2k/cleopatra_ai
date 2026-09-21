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


#: Message attributes an OpenAI-compatible response may carry. Only the NAMES
#: of the ones actually populated are recorded; no value ever is.
INSPECTED_MESSAGE_FIELDS: tuple[str, ...] = (
    "content",
    "reasoning",
    "reasoning_content",
    "reasoning_details",
    "refusal",
    "tool_calls",
    "function_call",
    "annotations",
)

# Why one owner call produced no usable structured output. These are the
# categories an operator can act on, and they are deliberately distinguishable:
# "the model thought until the budget ran out" and "the model wrote prose
# instead of JSON" have different fixes and used to share one message.
FAILURE_EMPTY_TRUNCATED = "empty_content_truncated"
FAILURE_EMPTY_UNEXPLAINED = "empty_content_unexplained"
FAILURE_CONTENT_FILTERED = "content_filtered"
FAILURE_REFUSAL = "provider_refusal"
FAILURE_PROVIDER_ERROR = "provider_error"
FAILURE_TIMEOUT = "provider_timeout"
FAILURE_TRANSPORT = "transport_error"
FAILURE_NO_JSON = "no_json_object"
FAILURE_INVALID_JSON = "invalid_json"
FAILURE_NOT_AN_OBJECT = "json_not_an_object"
FAILURE_MISSING_REPLY = "missing_reply"
FAILURE_EMPTY_REPLY = "empty_reply"


@dataclass(frozen=True)
class ModelResponseDiagnostics:
    """Structural facts about one provider response, safe to log anywhere.

    WHY THIS EXISTS
    ---------------
    The transport used to reduce an entire OpenAI-compatible response to
    ``response.choices[0].message.content or ""``. Every distinct way a
    reasoning model can fail to return usable text — the completion budget
    consumed by hidden reasoning, a content filter, a refusal, an upstream that
    ignored ``response_format``, a 200 carrying an ``error`` object — arrived at
    the caller as the same empty string, and the caller could only say "the
    response contained no JSON object".

    Everything here is a shape, a length, a count, an enum or a provider-authored
    status. No fan or creator conversation text is recorded, and none can be
    reconstructed from a character count.
    """

    provider: str = ""
    model: str = ""
    upstream_provider: str = ""
    response_id: str = ""
    latency_ms: int = 0
    streamed: bool = False

    #: What the request asked the provider for, so a mismatch is visible.
    response_format_requested: str = ""
    reasoning_requested: str = ""
    max_tokens_requested: int = 0

    #: What came back.
    choice_count: int = 0
    finish_reason: str = ""
    native_finish_reason: str = ""
    message_fields: tuple[str, ...] = ()
    content_is_null: bool = False
    content_chars: int = 0
    reasoning_present: bool = False
    reasoning_chars: int = 0
    refusal_present: bool = False
    tool_call_count: int = 0

    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0

    #: Provider-authored failure text, bounded. Providers describe routing,
    #: quota and policy here, never the prompt.
    provider_error: str = ""
    error_category: str = ""

    @property
    def truncated(self) -> bool:
        """Whether the provider stopped because the token budget ran out.

        OpenAI-compatible routes say ``length``; Anthropic says ``max_tokens``.
        Both mean the same thing and both must be distinguishable from a model
        that simply wrote nothing.
        """
        stopped = f"{self.finish_reason or ''} {self.native_finish_reason or ''}".lower()
        return "length" in stopped or "max_tokens" in stopped

    @property
    def content_empty(self) -> bool:
        return self.content_chars <= 0

    def empty_content_category(self) -> str:
        """Name WHY content is empty, rather than only that no JSON was found."""
        if self.error_category:
            return self.error_category
        if not self.content_empty:
            return ""
        reason = (self.finish_reason or "").lower()
        if "content_filter" in reason:
            return FAILURE_CONTENT_FILTERED
        if self.refusal_present:
            return FAILURE_REFUSAL
        if self.truncated:
            return FAILURE_EMPTY_TRUNCATED
        if self.reasoning_present and self.completion_tokens:
            # Everything the provider was willing to spend went into reasoning.
            return FAILURE_EMPTY_TRUNCATED
        return FAILURE_EMPTY_UNEXPLAINED

    def as_dict(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "provider": self.provider,
            "model": self.model,
            "latency_ms": self.latency_ms,
            "finish_reason": self.finish_reason,
            "content_chars": self.content_chars,
            "content_is_null": self.content_is_null,
            "completion_tokens": self.completion_tokens,
        }
        if self.upstream_provider:
            record["upstream_provider"] = self.upstream_provider
        if self.response_id:
            record["response_id"] = self.response_id
        if self.native_finish_reason and self.native_finish_reason != self.finish_reason:
            record["native_finish_reason"] = self.native_finish_reason
        if self.message_fields:
            record["message_fields"] = list(self.message_fields)
        if self.reasoning_present:
            record["reasoning_chars"] = self.reasoning_chars
        if self.reasoning_tokens:
            record["reasoning_tokens"] = self.reasoning_tokens
        if self.prompt_tokens:
            record["prompt_tokens"] = self.prompt_tokens
        if self.refusal_present:
            record["refusal"] = True
        if self.tool_call_count:
            record["tool_calls"] = self.tool_call_count
        if self.response_format_requested:
            record["response_format_requested"] = self.response_format_requested
        if self.reasoning_requested:
            record["reasoning_requested"] = self.reasoning_requested
        if self.max_tokens_requested:
            record["max_tokens_requested"] = self.max_tokens_requested
        if self.choice_count != 1:
            record["choice_count"] = self.choice_count
        if self.streamed:
            record["streamed"] = True
        if self.provider_error:
            record["provider_error"] = self.provider_error
        if self.error_category:
            record["error_category"] = self.error_category
        if self.truncated:
            record["truncated"] = True
        return record

    def describe(self) -> str:
        """One log line. Safe to print next to a failure reason."""
        parts = [
            f"model={self.model or 'unknown'}",
            f"upstream={self.upstream_provider or self.provider or 'unknown'}",
            f"latency_ms={self.latency_ms}",
            f"finish={self.finish_reason or 'none'}",
            f"content_chars={self.content_chars}",
            f"content_null={str(self.content_is_null).lower()}",
            f"completion_tokens={self.completion_tokens}",
            f"reasoning_tokens={self.reasoning_tokens}",
            f"reasoning_chars={self.reasoning_chars}",
            f"fields={'/'.join(self.message_fields) or 'none'}",
            f"format={self.response_format_requested or 'none'}",
            f"reasoning_req={self.reasoning_requested or 'none'}",
            f"max_tokens={self.max_tokens_requested}",
        ]
        if self.response_id:
            parts.append(f"response_id={self.response_id}")
        if self.error_category:
            parts.append(f"error={self.error_category}")
        if self.provider_error:
            parts.append(f"provider_error={self.provider_error}")
        return " ".join(parts)


@dataclass(frozen=True)
class ModelResult:
    text: str
    target: ModelTarget
    usage: ModelUsage
    latency_ms: int
    raw_response_id: str | None = None

    # Aggregator routes (OpenRouter) name the upstream provider that actually
    # served the request and report the real charged cost. Both stay None for
    # direct providers, where the logical provider is the upstream provider and
    # cost is derived from catalog pricing.
    upstream_provider: str | None = None
    reported_cost_usd: float | None = None

    # Milliseconds spent waiting for a slot in the global model gate before the
    # provider was called at all. Kept separate from ``latency_ms`` so a
    # saturated deployment is never misread as a slow provider.
    gate_wait_ms: int = 0

    # The structural record of what the provider actually returned. Always
    # present, including when ``text`` is empty — that is the case it exists for.
    diagnostics: ModelResponseDiagnostics = field(
        default_factory=ModelResponseDiagnostics
    )


@dataclass(frozen=True)
class ModelTelemetryContext:
    feature: str
    creator_id: str | None = None
    fan_id: str | None = None
    evaluation_run_id: str | None = None
    scenario_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def resolve_cost_usd(
    target: ModelTarget,
    usage: ModelUsage,
    *,
    reported_cost_usd: float | None = None,
) -> float:
    """Prefer the provider's reported cost, falling back to catalog pricing.

    OpenRouter returns the amount actually charged for a generation. That is
    strictly better than a catalog snapshot, and using it keeps one accounting
    path rather than a second cost system alongside estimate_cost_usd.
    """

    if reported_cost_usd is not None:
        try:
            value = float(reported_cost_usd)
        except (TypeError, ValueError):
            value = None
        if value is not None and value >= 0:
            return round(value, 8)
    return estimate_cost_usd(target, usage)


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
