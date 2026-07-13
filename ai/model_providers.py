"""Provider-neutral model access for Anthropic, Together, and local endpoints."""

from __future__ import annotations

import json
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from models.model_runtime import ModelResult, ModelTarget, ModelUsage

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


def get_runtime_target(prefix: str) -> ModelTarget:
    """Resolve CHAT_* or ANALYZER_* environment variables into one target."""

    prefix = prefix.strip().upper()
    defaults = {
        "CHAT": ("anthropic", "claude-sonnet-4-6"),
        "ANALYZER": ("anthropic", "claude-haiku-4-5-20251001"),
    }
    default_provider, default_model = defaults.get(prefix, ("together", ""))

    provider = os.getenv(f"{prefix}_PROVIDER", default_provider).strip().lower()
    model = os.getenv(f"{prefix}_MODEL", default_model).strip()
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

    if provider == "anthropic":
        api_key_env = api_key_env or "ANTHROPIC_API_KEY"
    elif provider == "together":
        base_url = base_url or "https://api.together.xyz/v1"
        api_key_env = api_key_env or "TOGETHER_API_KEY"
    elif provider in {"self_hosted", "openai_compatible"}:
        base_url = base_url or os.getenv("SELF_HOSTED_BASE_URL")
        api_key_env = api_key_env or "SELF_HOSTED_API_KEY"

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


async def complete(
    target: ModelTarget,
    *,
    system: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float | None = None,
) -> ModelResult:
    """Call a configured model endpoint and normalize text, usage, and latency."""

    started = time.perf_counter()
    if target.provider == "anthropic":
        result = await _complete_anthropic(
            target,
            system=system,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    elif target.provider in {"together", "self_hosted", "openai_compatible"}:
        result = await _complete_openai_compatible(
            target,
            system=system,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    else:
        raise ValueError(f"Unsupported model provider: {target.provider}")

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    return ModelResult(
        text=result.text,
        target=result.target,
        usage=result.usage,
        latency_ms=elapsed_ms,
        raw_response_id=result.raw_response_id,
    )


async def _complete_anthropic(
    target: ModelTarget,
    *,
    system: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float | None,
) -> ModelResult:
    from anthropic import AsyncAnthropic

    client = AsyncAnthropic(api_key=_api_key(target))
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
    text = "".join(
        getattr(block, "text", "")
        for block in response.content
        if getattr(block, "type", "") == "text"
    )
    return ModelResult(
        text=text,
        target=target,
        usage=ModelUsage(
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            cache_read_tokens=int(getattr(usage, "cache_read_input_tokens", 0) or 0),
            cache_write_tokens=int(getattr(usage, "cache_creation_input_tokens", 0) or 0),
        ),
        latency_ms=0,
        raw_response_id=getattr(response, "id", None),
    )


async def _complete_openai_compatible(
    target: ModelTarget,
    *,
    system: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float | None,
) -> ModelResult:
    from openai import AsyncOpenAI

    if not target.base_url:
        raise RuntimeError(f"No base URL configured for {target.name}")

    client = AsyncOpenAI(base_url=target.base_url, api_key=_api_key(target))
    payload_messages = [{"role": "system", "content": system}, *messages]
    kwargs: dict[str, Any] = {
        "model": target.model,
        "messages": payload_messages,
        "max_tokens": max_tokens,
    }
    if temperature is not None:
        kwargs["temperature"] = temperature

    response = await client.chat.completions.create(**kwargs)
    content = response.choices[0].message.content or ""
    usage = response.usage
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
    prompt_details = getattr(usage, "prompt_tokens_details", None) if usage else None
    cached_tokens = int(getattr(prompt_details, "cached_tokens", 0) or 0)

    return ModelResult(
        text=content,
        target=target,
        usage=ModelUsage(
            input_tokens=max(prompt_tokens - cached_tokens, 0),
            output_tokens=completion_tokens,
            cache_read_tokens=cached_tokens,
        ),
        latency_ms=0,
        raw_response_id=getattr(response, "id", None),
    )
