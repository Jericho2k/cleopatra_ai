"""Provider-neutral model access for Anthropic, Together, and local endpoints."""

from __future__ import annotations

import json
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

from models.model_runtime import ModelResult, ModelTarget, ModelUsage, VisionImage

_DEFAULT_CATALOG = Path(__file__).resolve().parents[1] / "config" / "model_candidates.json"

# Providers whose terms do not permit adult workloads regardless of what a
# catalog row claims. Anthropic is listed because ``model_candidates.json``
# already marks it ``adult_policy: ineligible`` and nothing enforced it — the
# vault categorizer called it directly on explicit imagery.
ADULT_INELIGIBLE_PROVIDERS = frozenset({"anthropic"})


class AdultPolicyError(RuntimeError):
    """Raised before an adult workload can reach an ineligible provider."""


def adult_eligibility(target: ModelTarget) -> str:
    """Effective adult policy: the provider rule outranks the catalog row."""
    if target.provider in ADULT_INELIGIBLE_PROVIDERS:
        return "ineligible"
    return target.adult_policy


def assert_adult_eligible(target: ModelTarget) -> None:
    if adult_eligibility(target) == "ineligible":
        raise AdultPolicyError(
            f"{target.name} is ineligible for adult workloads. "
            "Point VISION_PROVIDER/VISION_MODEL at a provider whose terms "
            "permit this content."
        )


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
        "EXTRACTOR": ("together", "openai/gpt-oss-120b"),
        # Vault imagery is adult content, so the default target is one the
        # operator hosts. See ADULT_INELIGIBLE_PROVIDERS.
        "VISION": ("self_hosted", "Qwen/Qwen2.5-VL-72B-Instruct"),
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
    messages: list[dict[str, Any]],
    max_tokens: int,
    temperature: float | None = None,
    images: Sequence[VisionImage] | None = None,
) -> ModelResult:
    """Call a configured model endpoint and normalize text, usage, and latency.

    ``images`` are attached to the final user message in whichever content
    shape the provider expects, so callers stay provider-neutral.
    """

    started = time.perf_counter()
    if target.provider == "anthropic":
        result = await _complete_anthropic(
            target,
            system=system,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            images=images,
        )
    elif target.provider in {"together", "self_hosted", "openai_compatible"}:
        result = await _complete_openai_compatible(
            target,
            system=system,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            images=images,
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


def _attach_images(
    messages: list[dict[str, Any]],
    images: Sequence[VisionImage] | None,
    block_builder,
) -> list[dict[str, Any]]:
    """Return ``messages`` with image blocks prepended to the last user turn.

    Images lead the content list because both providers read a prompt that
    follows its images more reliably than one that precedes them.
    """
    if not images:
        return list(messages)

    prepared = [dict(message) for message in messages]
    index = next(
        (
            position
            for position in range(len(prepared) - 1, -1, -1)
            if prepared[position].get("role") == "user"
        ),
        None,
    )
    if index is None:
        prepared.append({"role": "user", "content": ""})
        index = len(prepared) - 1

    content = prepared[index].get("content")
    text_blocks: list[dict[str, Any]]
    if isinstance(content, str):
        text_blocks = [{"type": "text", "text": content}] if content else []
    elif isinstance(content, list):
        text_blocks = list(content)
    else:
        text_blocks = []

    prepared[index]["content"] = [
        *(block_builder(image) for image in images),
        *text_blocks,
    ]
    return prepared


def _anthropic_image_block(image: VisionImage) -> dict[str, Any]:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": image.media_type,
            "data": image.as_base64(),
        },
    }


def _openai_image_block(image: VisionImage) -> dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": image.as_data_uri()}}


async def _complete_anthropic(
    target: ModelTarget,
    *,
    system: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    temperature: float | None,
    images: Sequence[VisionImage] | None = None,
) -> ModelResult:
    from anthropic import AsyncAnthropic

    client = AsyncAnthropic(api_key=_api_key(target))
    kwargs: dict[str, Any] = {
        "model": target.model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": _attach_images(messages, images, _anthropic_image_block),
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
    messages: list[dict[str, Any]],
    max_tokens: int,
    temperature: float | None,
    images: Sequence[VisionImage] | None = None,
) -> ModelResult:
    from openai import AsyncOpenAI

    if not target.base_url:
        raise RuntimeError(f"No base URL configured for {target.name}")

    client = AsyncOpenAI(
        base_url=target.base_url,
        api_key=_api_key(target),
        timeout=target.timeout_seconds,
    )

    payload_messages = [
        {"role": "system", "content": system},
        *_attach_images(messages, images, _openai_image_block),
    ]

    kwargs: dict[str, Any] = {
        "model": target.model,
        "messages": payload_messages,
        "max_tokens": max_tokens,
    }

    if temperature is not None:
        kwargs["temperature"] = temperature

    reasoning_enabled = target.metadata.get("reasoning_enabled")
    if reasoning_enabled is not None:
        extra_body = dict(kwargs.get("extra_body") or {})
        extra_body["reasoning"] = {
            "enabled": bool(reasoning_enabled),
        }
        kwargs["extra_body"] = extra_body

    raw_response_id: str | None = None
    usage = None

    if target.stream:
        stream = await client.chat.completions.create(
            **kwargs,
            stream=True,
            stream_options={"include_usage": True},
        )

        content_parts: list[str] = []

        async for chunk in stream:
            if raw_response_id is None:
                raw_response_id = getattr(chunk, "id", None)

            choices = getattr(chunk, "choices", None) or []

            for choice in choices:
                delta = getattr(choice, "delta", None)
                text = getattr(delta, "content", None)

                if isinstance(text, str) and text:
                    content_parts.append(text)

            chunk_usage = getattr(chunk, "usage", None)

            if chunk_usage is not None:
                usage = chunk_usage

        content = "".join(content_parts)

    else:
        response = await client.chat.completions.create(**kwargs)

        content = response.choices[0].message.content or ""
        usage = response.usage
        raw_response_id = getattr(response, "id", None)

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
    nested_cached_tokens = int(
        getattr(prompt_details, "cached_tokens", 0) or 0
    )

    flat_cached_tokens = int(
        getattr(usage, "cached_tokens", 0) or 0
        if usage
        else 0
    )

    cached_tokens = max(
        nested_cached_tokens,
        flat_cached_tokens,
    )

    return ModelResult(
        text=content,
        target=target,
        usage=ModelUsage(
            input_tokens=max(prompt_tokens - cached_tokens, 0),
            output_tokens=completion_tokens,
            cache_read_tokens=cached_tokens,
        ),
        latency_ms=0,
        raw_response_id=raw_response_id,
    )
