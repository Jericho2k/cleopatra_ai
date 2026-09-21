"""Provider-neutral model access for Anthropic, OpenRouter, Together, and local endpoints."""

from __future__ import annotations

import json
import os
import time
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Any

from ai import openrouter_routing
from ai.model_migrations import resolve_supported_model
from ai.prompt_blocks import flatten_message_content
from core.action_telemetry import record_count, record_stage
from core.model_gate import MODEL_GATE
from models.model_runtime import (
    FAILURE_PROVIDER_ERROR,
    FAILURE_TIMEOUT,
    FAILURE_TRANSPORT,
    INSPECTED_MESSAGE_FIELDS,
    ModelResponseDiagnostics,
    ModelResult,
    ModelTarget,
    ModelUsage,
)

# Providers that speak the OpenAI chat-completions wire format.
OPENAI_COMPATIBLE_PROVIDERS = frozenset(
    {"together", "openrouter", "self_hosted", "openai_compatible"}
)

_DEFAULT_CATALOG = Path(__file__).resolve().parents[1] / "config" / "model_candidates.json"


@lru_cache(maxsize=4)
def load_model_catalog(path: str | None = None) -> list[ModelTarget]:
    catalog_path = Path(path or os.getenv("CLEOPATRA_MODEL_CATALOG") or _DEFAULT_CATALOG)
    if not catalog_path.exists():
        return []
    payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    rows = payload.get("models", payload) if isinstance(payload, dict) else payload
    return [ModelTarget.from_mapping(row) for row in rows]


def find_catalog_target(provider: str, model: str) -> ModelTarget | None:
    provider = provider.strip().lower()
    for target in load_model_catalog():
        if target.provider == provider and target.model == model:
            return target
    return None


def provider_transport_defaults(
    provider: str,
    *,
    base_url: str | None = None,
    api_key_env: str | None = None,
) -> tuple[str | None, str | None]:
    """Return the conventional base URL and key variable for one provider.

    Both the generic CHAT_*/ANALYZER_* runtime and the production writer router
    resolve targets that are absent from the catalog. They must agree on how a
    provider is reached, so the defaults live here rather than being duplicated.
    """

    provider = provider.strip().lower()
    if provider == "anthropic":
        api_key_env = api_key_env or "ANTHROPIC_API_KEY"
    elif provider == "together":
        base_url = base_url or "https://api.together.xyz/v1"
        api_key_env = api_key_env or "TOGETHER_API_KEY"
    elif provider == "openrouter":
        base_url = base_url or openrouter_routing.base_url()
        api_key_env = api_key_env or openrouter_routing.DEFAULT_API_KEY_ENV
    elif provider in {"self_hosted", "openai_compatible"}:
        base_url = base_url or os.getenv("SELF_HOSTED_BASE_URL")
        api_key_env = api_key_env or "SELF_HOSTED_API_KEY"
    return base_url, api_key_env


def get_runtime_target(prefix: str) -> ModelTarget:
    """Resolve CHAT_* or ANALYZER_* environment variables into one target."""

    prefix = prefix.strip().upper()
    defaults = {
        "CHAT": ("anthropic", "claude-sonnet-4-6"),
        "ANALYZER": ("anthropic", "claude-haiku-4-5-20251001"),
        "EXTRACTOR": ("together", "openai/gpt-oss-120b"),
    }
    default_provider, default_model = defaults.get(prefix, ("together", ""))

    provider = os.getenv(f"{prefix}_PROVIDER", default_provider).strip().lower()
    model = resolve_supported_model(
        provider,
        os.getenv(f"{prefix}_MODEL", default_model),
    )
    base_url = os.getenv(f"{prefix}_BASE_URL") or None
    api_key_env = os.getenv(f"{prefix}_API_KEY_ENV") or None

    catalog_target = find_catalog_target(provider, model)
    if catalog_target:
        return ModelTarget(
            **{
                **catalog_target.__dict__,
                "base_url": base_url or catalog_target.base_url,
                "api_key_env": api_key_env or catalog_target.api_key_env,
            }
        )

    base_url, api_key_env = provider_transport_defaults(
        provider,
        base_url=base_url,
        api_key_env=api_key_env,
    )

    return ModelTarget(
        name=f"{provider}:{model}",
        provider=provider,
        model=model,
        base_url=base_url,
        api_key_env=api_key_env,
    )


def _api_key(target: ModelTarget) -> str:
    env_name = target.api_key_env
    if env_name and os.getenv(env_name):
        return os.environ[env_name]
    if target.provider in {"self_hosted", "openai_compatible"}:
        return "not-required"
    raise RuntimeError(
        f"Missing API key for {target.name}. Expected environment variable {env_name!r}."
    )


@lru_cache(maxsize=8)
def _anthropic_client(api_key: str):
    from anthropic import AsyncAnthropic

    return AsyncAnthropic(api_key=api_key)


@lru_cache(maxsize=16)
def _openai_compatible_client(base_url: str, api_key: str, timeout_seconds: float):
    from openai import AsyncOpenAI

    return AsyncOpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=timeout_seconds,
    )


async def complete(
    target: ModelTarget,
    *,
    system: str | list[dict[str, Any]],
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float | None = None,
    session_id: str | None = None,
    end_user_id: str | None = None,
    response_format: dict[str, Any] | None = None,
) -> ModelResult:
    """Call a configured model endpoint and normalize text, usage, and latency.

    ``system`` may be a plain string or the ordered content blocks
    ``ai.prompt_blocks`` produces. Blocks reach Anthropic intact, so a
    ``cache_control`` marker actually arrives at the provider (COST-002a);
    OpenAI-compatible transports, which use implicit prefix caching, receive the
    same prose joined into one string. Deciding that here rather than in the
    caller is what stopped the marker being discarded before transport.

    ``session_id`` is the stable per-conversation affinity key. Providers that
    support sticky routing use it to keep consecutive turns on one upstream so
    prefix caching survives; providers that do not simply ignore it.

    Every call passes through the global model gate (see ``core.model_gate``).
    This is the single admission-control point for paid inference: writer,
    analyzer, extractor, and every future caller of this function are bounded by
    one limit rather than by whatever concurrency their caller happens to have.
    ``latency_ms`` remains pure provider time — the wait for a slot is reported
    separately so a slow provider is never confused with a saturated gate.
    """

    if target.provider not in OPENAI_COMPATIBLE_PROVIDERS and target.provider != "anthropic":
        raise ValueError(f"Unsupported model provider: {target.provider}")

    async with MODEL_GATE.acquire(feature=target.provider) as gate_wait_ms:
        record_stage("model_gate_wait_ms", gate_wait_ms)
        record_count("model_calls")
        started = time.perf_counter()
        if target.provider == "anthropic":
            result = await _complete_anthropic(
                target,
                system=system,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        else:
            result = await _complete_openai_compatible(
                target,
                system=system,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                session_id=session_id,
                end_user_id=end_user_id,
                response_format=response_format,
            )
        elapsed_ms = int((time.perf_counter() - started) * 1000)

    record_stage("model_provider_ms", float(elapsed_ms))
    return ModelResult(
        text=result.text,
        target=result.target,
        usage=result.usage,
        latency_ms=elapsed_ms,
        raw_response_id=result.raw_response_id,
        upstream_provider=result.upstream_provider,
        reported_cost_usd=result.reported_cost_usd,
        served_model=result.served_model,
        gate_wait_ms=int(gate_wait_ms),
        # Only ``complete`` knows the real provider time, so it is the one that
        # can stamp it onto the structural record.
        diagnostics=replace(result.diagnostics, latency_ms=elapsed_ms),
    )


async def _complete_anthropic(
    target: ModelTarget,
    *,
    system: str | list[dict[str, Any]],
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float | None,
) -> ModelResult:
    client = _anthropic_client(_api_key(target))
    kwargs: dict[str, Any] = {
        "model": target.model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": messages,
    }
    if temperature is not None:
        kwargs["temperature"] = temperature

    response = await client.messages.create(**kwargs)
    usage = response.usage
    blocks = list(response.content or [])
    text = "".join(
        getattr(block, "text", "")
        for block in blocks
        if getattr(block, "type", "") == "text"
    )
    thinking_chars = sum(
        len(getattr(block, "thinking", "") or "")
        for block in blocks
        if getattr(block, "type", "") in {"thinking", "redacted_thinking"}
    )
    block_types = tuple(
        dict.fromkeys(str(getattr(block, "type", "") or "") for block in blocks)
    )
    output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
    return ModelResult(
        text=text,
        target=target,
        usage=ModelUsage(
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=output_tokens,
            cache_read_tokens=int(getattr(usage, "cache_read_input_tokens", 0) or 0),
            cache_write_tokens=int(getattr(usage, "cache_creation_input_tokens", 0) or 0),
        ),
        latency_ms=0,
        raw_response_id=getattr(response, "id", None),
        served_model=str(getattr(response, "model", "") or "") or None,
        # Anthropic names things differently, but an empty completion has to be
        # explainable on every route, not only on the one that broke first.
        diagnostics=ModelResponseDiagnostics(
            provider=target.provider,
            model=target.model,
            served_model=str(getattr(response, "model", "") or ""),
            response_id=str(getattr(response, "id", "") or ""),
            max_tokens_requested=int(max_tokens),
            choice_count=1,
            finish_reason=str(getattr(response, "stop_reason", "") or ""),
            message_fields=block_types,
            content_chars=len(text),
            reasoning_present=bool(thinking_chars),
            reasoning_chars=thinking_chars,
            prompt_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            completion_tokens=output_tokens,
        ),
    )


def _int_field(source: Any, name: str) -> int:
    """Read one integer usage field from a mapping or a response model."""

    if source is None:
        return 0
    value = (
        source.get(name)
        if isinstance(source, dict)
        else getattr(source, name, None)
    )
    try:
        return max(int(value or 0), 0)
    except (TypeError, ValueError):
        return 0


def _float_field(source: Any, name: str) -> float | None:
    if source is None:
        return None
    value = (
        source.get(name)
        if isinstance(source, dict)
        else getattr(source, name, None)
    )
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _text_of(value: Any) -> str:
    """Flatten a content field that may be a string or a list of parts."""

    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
                continue
            text = (
                item.get("text")
                if isinstance(item, dict)
                else getattr(item, "text", None)
            )
            if isinstance(text, str):
                parts.append(text)
        return "".join(parts)
    return ""


def _attr(source: Any, name: str) -> Any:
    if source is None:
        return None
    if isinstance(source, dict):
        return source.get(name)
    return getattr(source, name, None)


def _populated_message_fields(message: Any) -> tuple[str, ...]:
    """Which known message attributes carry something. NAMES ONLY."""

    present: list[str] = []
    for name in INSPECTED_MESSAGE_FIELDS:
        value = _attr(message, name)
        if value in (None, "", [], {}):
            continue
        present.append(name)
    return tuple(present)


def _reasoning_text(message: Any) -> str:
    """Whatever the provider called the hidden reasoning, as one string.

    Never returned to a caller and never logged — only its LENGTH is recorded,
    because "the model produced 6k characters of reasoning and no content" is
    the single most useful fact about an empty completion.
    """

    for name in ("reasoning", "reasoning_content"):
        text = _text_of(_attr(message, name))
        if text:
            return text
    details = _attr(message, "reasoning_details")
    if isinstance(details, list):
        parts: list[str] = []
        for item in details:
            for name in ("text", "summary", "data"):
                value = _attr(item, name)
                if isinstance(value, str):
                    parts.append(value)
        return "".join(parts)
    return ""


def _bounded_provider_error(value: Any) -> str:
    """One provider-authored status line, bounded.

    Providers describe routing, quota, moderation policy and model availability
    here. It is deliberately truncated so that an unusual provider which echoed
    part of a request cannot write an unbounded amount into a log.
    """

    if value in (None, "", [], {}):
        return ""
    if isinstance(value, dict):
        for name in ("message", "detail", "code", "type"):
            text = value.get(name)
            if isinstance(text, str) and text.strip():
                return " ".join(text.split())[:200]
        return " ".join(str(value).split())[:200]
    text = _attr(value, "message")
    if isinstance(text, str) and text.strip():
        return " ".join(text.split())[:200]
    return " ".join(str(value).split())[:200]


def _response_error(response: Any, choice: Any) -> str:
    """OpenRouter can answer 200 with an error object instead of a completion."""

    return _bounded_provider_error(_attr(response, "error")) or _bounded_provider_error(
        _attr(choice, "error")
    )


def _requested_reasoning_label(reasoning: dict[str, Any] | None) -> str:
    if not reasoning:
        return "off"
    if not reasoning.get("enabled", True):
        return "off"
    parts = ["on"]
    if reasoning.get("effort"):
        parts.append(f"effort={reasoning['effort']}")
    if reasoning.get("max_tokens"):
        parts.append(f"max_tokens={reasoning['max_tokens']}")
    return ",".join(parts)


def classify_transport_error(error: BaseException) -> str:
    """Name the category of a transport failure without leaking the request.

    A timeout and a refused route are different operational problems and used
    to arrive as the same "the conversational owner could not be reached".
    """

    name = type(error).__name__.lower()
    text = str(error).lower()
    if "timeout" in name or "timed out" in text or "timeout" in text:
        return FAILURE_TIMEOUT
    if any(
        token in name
        for token in ("apistatus", "ratelimit", "badrequest", "notfound", "permission")
    ):
        return FAILURE_PROVIDER_ERROR
    if "status_code" in text or "error code" in text:
        return FAILURE_PROVIDER_ERROR
    return FAILURE_TRANSPORT


async def _complete_openai_compatible(
    target: ModelTarget,
    *,
    system: str | list[dict[str, Any]],
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float | None,
    session_id: str | None = None,
    end_user_id: str | None = None,
    response_format: dict[str, Any] | None = None,
) -> ModelResult:
    if not target.base_url:
        raise RuntimeError(f"No base URL configured for {target.name}")

    client = _openai_compatible_client(
        target.base_url,
        _api_key(target),
        float(target.timeout_seconds),
    )

    payload_messages = [
        {"role": "system", "content": flatten_message_content(system)},
        *messages,
    ]

    kwargs: dict[str, Any] = {
        "model": target.model,
        "messages": payload_messages,
        "max_tokens": max_tokens,
    }

    if temperature is not None:
        kwargs["temperature"] = temperature
    if response_format is not None:
        kwargs["response_format"] = response_format

    extra_body: dict[str, Any] = {}

    reasoning: dict[str, Any] | None = None
    reasoning_enabled = target.metadata.get("reasoning_enabled")
    if reasoning_enabled is not None:
        reasoning = {
            "enabled": bool(reasoning_enabled),
        }
        # A hard ceiling on hidden reasoning. The request's ``max_tokens`` is
        # ONE budget for reasoning plus content on every OpenAI-compatible
        # reasoning route, so a model that thinks until the budget is gone
        # returns ``message.content = null`` with ``finish_reason = "length"``
        # — which is exactly how the conversational owner produced 97-second
        # dead turns. Bounding the trace guarantees the answer has budget left.
        #
        # ``effort`` and ``max_tokens`` are ALTERNATIVES in OpenRouter's
        # reasoning object, so only one is ever sent. A cap wins where both are
        # configured: OpenRouter converts it to an effort level for upstreams
        # that only accept effort, so the cap is the safer of the two to send.
        bounded = 0
        reasoning_max_tokens = target.metadata.get("reasoning_max_tokens")
        if reasoning_max_tokens:
            try:
                bounded = int(reasoning_max_tokens)
            except (TypeError, ValueError):
                bounded = 0
        reasoning_effort = target.metadata.get("reasoning_effort")
        if bounded > 0:
            reasoning["max_tokens"] = min(bounded, max(max_tokens - 256, 1))
        elif reasoning_effort:
            reasoning["effort"] = str(reasoning_effort)
        extra_body["reasoning"] = reasoning

    if target.provider == "openrouter":
        # Provider pinning, privacy controls, and the sticky-routing key that
        # keeps one fan conversation on one upstream so its prefix cache stays
        # warm. These are OpenRouter body fields, not OpenAI ones, so they
        # travel in extra_body.
        extra_body.update(
            openrouter_routing.request_options(
                target_metadata=target.metadata,
                session_id=session_id,
                end_user_id=end_user_id,
            )
        )

    if extra_body:
        kwargs["extra_body"] = extra_body

    raw_response_id: str | None = None
    usage = None
    upstream_provider: str | None = None
    served_model: str | None = None
    reported_cost_usd: float | None = None
    finish_reason = ""
    native_finish_reason = ""
    message_fields: tuple[str, ...] = ()
    content_is_null = False
    reasoning_chars = 0
    refusal_present = False
    tool_call_count = 0
    choice_count = 0
    provider_error = ""

    if target.stream:
        stream = await client.chat.completions.create(
            **kwargs,
            stream=True,
            stream_options={"include_usage": True},
        )

        content_parts: list[str] = []

        reasoning_parts: list[str] = []

        async for chunk in stream:
            if raw_response_id is None:
                raw_response_id = getattr(chunk, "id", None)
            if served_model is None:
                served_model = str(getattr(chunk, "model", "") or "") or None

            if upstream_provider is None:
                upstream_provider = openrouter_routing.upstream_provider(chunk)

            provider_error = provider_error or _bounded_provider_error(
                _attr(chunk, "error")
            )

            choices = getattr(chunk, "choices", None) or []
            choice_count = max(choice_count, len(choices))

            for choice in choices:
                delta = getattr(choice, "delta", None)
                text = getattr(delta, "content", None)

                if isinstance(text, str) and text:
                    content_parts.append(text)

                reasoning_delta = _reasoning_text(delta)
                if reasoning_delta:
                    reasoning_parts.append(reasoning_delta)

                stop = _attr(choice, "finish_reason")
                if isinstance(stop, str) and stop:
                    finish_reason = stop
                native = _attr(choice, "native_finish_reason")
                if isinstance(native, str) and native:
                    native_finish_reason = native

            chunk_usage = getattr(chunk, "usage", None)

            if chunk_usage is not None:
                usage = chunk_usage

        content = "".join(content_parts)
        reasoning_chars = len("".join(reasoning_parts))
        message_fields = tuple(
            name
            for name, present in (
                ("content", bool(content)),
                ("reasoning", bool(reasoning_chars)),
            )
            if present
        )

    else:
        response = await client.chat.completions.create(**kwargs)
        served_model = str(getattr(response, "model", "") or "") or None

        choices = list(getattr(response, "choices", None) or [])
        choice_count = len(choices)
        choice = choices[0] if choices else None
        message = _attr(choice, "message")

        raw_content = _attr(message, "content")
        content_is_null = raw_content is None
        content = _text_of(raw_content)
        reasoning_chars = len(_reasoning_text(message))
        message_fields = _populated_message_fields(message)
        refusal_present = bool(_attr(message, "refusal"))
        tool_calls = _attr(message, "tool_calls")
        tool_call_count = len(tool_calls) if isinstance(tool_calls, list) else 0
        finish_reason = str(_attr(choice, "finish_reason") or "")
        native_finish_reason = str(_attr(choice, "native_finish_reason") or "")
        provider_error = _response_error(response, choice)

        usage = response.usage
        raw_response_id = getattr(response, "id", None)
        upstream_provider = openrouter_routing.upstream_provider(response)

    prompt_tokens = (
        int(getattr(usage, "prompt_tokens", 0) or 0)
        if usage
        else 0
    )
    completion_tokens = (
        int(getattr(usage, "completion_tokens", 0) or 0)
        if usage
        else 0
    )

    prompt_details = (
        getattr(usage, "prompt_tokens_details", None)
        if usage
        else None
    )
    nested_cached_tokens = _int_field(prompt_details, "cached_tokens")

    flat_cached_tokens = _int_field(usage, "cached_tokens")

    cached_tokens = max(
        nested_cached_tokens,
        flat_cached_tokens,
    )

    # OpenRouter reports cache writes separately from cache reads. Providers
    # disagree about whether written tokens are also inside prompt_tokens, so
    # they are only subtracted from uncached input when the arithmetic shows
    # they were included. Subtracting unconditionally would understate input on
    # providers that report them additively.
    cache_write_tokens = max(
        _int_field(prompt_details, "cache_write_tokens"),
        _int_field(usage, "cache_write_tokens"),
    )
    billed_cache_write_tokens = (
        cache_write_tokens
        if cached_tokens + cache_write_tokens <= prompt_tokens
        else 0
    )

    if usage is not None:
        reported_cost_usd = _float_field(usage, "cost")

    completion_details = (
        getattr(usage, "completion_tokens_details", None) if usage else None
    )
    reasoning_tokens = max(
        _int_field(completion_details, "reasoning_tokens"),
        _int_field(usage, "reasoning_tokens"),
    )

    diagnostics = ModelResponseDiagnostics(
        provider=target.provider,
        model=target.model,
        served_model=str(served_model or ""),
        upstream_provider=str(upstream_provider or ""),
        response_id=str(raw_response_id or ""),
        streamed=bool(target.stream),
        response_format_requested=str((response_format or {}).get("type") or ""),
        reasoning_requested=_requested_reasoning_label(reasoning),
        max_tokens_requested=int(max_tokens),
        choice_count=choice_count,
        finish_reason=finish_reason,
        native_finish_reason=native_finish_reason,
        message_fields=message_fields,
        content_is_null=content_is_null,
        content_chars=len(content),
        reasoning_present=bool(reasoning_chars),
        reasoning_chars=reasoning_chars,
        refusal_present=refusal_present,
        tool_call_count=tool_call_count,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        reasoning_tokens=reasoning_tokens,
        provider_error=provider_error,
        error_category=FAILURE_PROVIDER_ERROR if provider_error else "",
    )

    return ModelResult(
        text=content,
        target=target,
        usage=ModelUsage(
            input_tokens=max(
                prompt_tokens - cached_tokens - billed_cache_write_tokens,
                0,
            ),
            output_tokens=completion_tokens,
            cache_read_tokens=cached_tokens,
            cache_write_tokens=cache_write_tokens,
        ),
        latency_ms=0,
        raw_response_id=raw_response_id,
        upstream_provider=upstream_provider,
        reported_cost_usd=reported_cost_usd,
        served_model=served_model,
        diagnostics=diagnostics,
    )
