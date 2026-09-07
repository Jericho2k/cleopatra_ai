"""Background availability checks for the configured writer models.

Each provider is checked against its own catalog. The check that used to be
Together-only would report an OpenRouter-hosted writer as missing simply
because it was not in Together's list, so the provider is now part of the
lookup rather than an assumption.

The catalog endpoints do not generate text and therefore consume no inference
tokens. Their result is kept in memory for the operator dashboard; actual reply
generation still owns the primary-to-fallback behavior.
"""

from __future__ import annotations

import asyncio
import copy
import os
from datetime import datetime, timezone
from typing import Any

import httpx

from ai import openrouter_routing
from ai.model_migrations import resolve_supported_model
from ai.writer_router import (
    COMPLEX_WRITER_MODEL,
    COMPLEX_WRITER_PROVIDER,
    DEFAULT_WRITER_MODEL,
    DEFAULT_WRITER_PROVIDER,
)

_DEFAULT_CHECK_SECONDS = 6 * 60 * 60
_TOGETHER_MODEL_LIST_URL = "https://api.together.xyz/v1/models"

# Providers this check knows how to verify. Anything else is reported as
# "unknown" rather than silently failing the creator's writer.
_CHECKED_PROVIDERS = ("together", "openrouter")

_PROVIDER_KEY_ENV = {
    "together": "TOGETHER_API_KEY",
    "openrouter": openrouter_routing.DEFAULT_API_KEY_ENV,
}

_state: dict[str, Any] = {
    "status": "unknown",
    "checked_at": None,
    "detail": "Model availability has not been checked yet.",
    "models": [],
}
_runtime_attempts: dict[str, dict[str, Any]] = {}


def configured_writer_models() -> list[dict[str, str]]:
    """Return the ordinary writer and its complex-turn/fallback target."""
    default_provider = os.getenv(
        "WRITER_DEFAULT_PROVIDER", DEFAULT_WRITER_PROVIDER
    ).strip().lower()
    complex_provider = os.getenv(
        "WRITER_COMPLEX_PROVIDER", COMPLEX_WRITER_PROVIDER
    ).strip().lower()
    configured = [
        {
            "role": "ordinary_writer",
            "provider": default_provider,
            "model": resolve_supported_model(
                default_provider,
                os.getenv("WRITER_DEFAULT_MODEL", DEFAULT_WRITER_MODEL),
            ),
        },
        {
            "role": "complex_writer_and_fallback",
            "provider": complex_provider,
            "model": resolve_supported_model(
                complex_provider,
                os.getenv("WRITER_COMPLEX_MODEL", COMPLEX_WRITER_MODEL),
            ),
        },
    ]
    return [row for row in configured if row["model"]]


def current_model_availability() -> dict[str, Any]:
    state = copy.deepcopy(_state)
    runtime_degraded: list[str] = []
    for model in state.get("models", []):
        runtime = copy.deepcopy(_runtime_attempts.get(model["model"], {}))
        model["runtime"] = runtime or None
        if int(runtime.get("consecutive_failures") or 0) >= 2:
            runtime_degraded.append(model["model"])
    if runtime_degraded and state.get("status") == "healthy":
        state["status"] = "degraded"
        state["detail"] = (
            "Configured model is failing live inference: "
            + ", ".join(runtime_degraded)
        )
    return state


def record_model_transport_success(model: str) -> None:
    _runtime_attempts[model] = {
        "consecutive_failures": 0,
        "last_success_at": datetime.now(timezone.utc).isoformat(),
        "last_failure_at": _runtime_attempts.get(model, {}).get("last_failure_at"),
        "last_error": None,
    }


def record_model_transport_failure(model: str, error: object) -> None:
    previous = _runtime_attempts.get(model, {})
    _runtime_attempts[model] = {
        "consecutive_failures": int(previous.get("consecutive_failures") or 0) + 1,
        "last_success_at": previous.get("last_success_at"),
        "last_failure_at": datetime.now(timezone.utc).isoformat(),
        "last_error": str(error)[:500],
    }


def _model_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        return [row for row in payload["data"] if isinstance(row, dict)]
    return []


def _catalog_url(provider: str) -> str:
    if provider == "openrouter":
        return f"{openrouter_routing.base_url()}/models"
    return _TOGETHER_MODEL_LIST_URL


async def _provider_model_ids(
    provider: str,
    *,
    client: httpx.AsyncClient,
) -> set[str]:
    """Return every model id the provider currently serves."""
    api_key = os.getenv(_PROVIDER_KEY_ENV[provider], "").strip()
    if not api_key:
        raise ApiKeyMissing(provider)
    response = await client.get(
        _catalog_url(provider),
        headers={"Authorization": f"Bearer {api_key}"},
    )
    response.raise_for_status()
    return {
        str(row.get("id") or "").strip()
        for row in _model_rows(response.json())
        if row.get("id")
    }


class ApiKeyMissing(RuntimeError):
    def __init__(self, provider: str) -> None:
        super().__init__(f"{_PROVIDER_KEY_ENV[provider]} is missing")
        self.provider = provider
        self.env_name = _PROVIDER_KEY_ENV[provider]


def _annotate(row: dict[str, str]) -> dict[str, Any]:
    """Attach the pinned upstream providers to an OpenRouter row."""
    if row["provider"] != "openrouter":
        return dict(row)
    return {**row, "pinned_providers": openrouter_routing.pinned_providers()}


async def refresh_model_availability(
    *,
    client: httpx.AsyncClient | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Check each configured writer model against its own provider's catalog."""
    global _state
    checked_at = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    configured = [_annotate(row) for row in configured_writer_models()]
    checkable = [row for row in configured if row["provider"] in _CHECKED_PROVIDERS]

    if not checkable:
        _state = {
            "status": "healthy",
            "checked_at": checked_at.isoformat(),
            "detail": "No writer model uses a provider with a catalog check.",
            "models": [{**row, "available": None} for row in configured],
        }
        return current_model_availability()

    owns_client = client is None
    active_client = client or httpx.AsyncClient(timeout=20)
    try:
        available_by_provider: dict[str, set[str]] = {}
        errors: dict[str, str] = {}
        misconfigured: set[str] = set()
        for provider in sorted({row["provider"] for row in checkable}):
            try:
                available_by_provider[provider] = await _provider_model_ids(
                    provider,
                    client=active_client,
                )
            except ApiKeyMissing as exc:
                misconfigured.add(provider)
                errors[provider] = (
                    f"{exc.env_name} is missing; AI replies cannot use {provider}."
                )
            except (httpx.HTTPError, TypeError, ValueError) as exc:
                errors[provider] = f"{provider} availability check failed: {exc}"

        models: list[dict[str, Any]] = []
        for row in configured:
            provider = row["provider"]
            if provider not in _CHECKED_PROVIDERS:
                models.append({**row, "available": None})
            elif provider in errors:
                # A provider whose catalog could not be read is unknown, not
                # absent: only a successful catalog read proves a model gone.
                # A missing API key is different — the route cannot work at all.
                models.append(
                    {
                        **row,
                        "available": False if provider in misconfigured else None,
                        "detail": errors[provider],
                    }
                )
            else:
                models.append(
                    {
                        **row,
                        "available": row["model"] in available_by_provider[provider],
                    }
                )

        missing = [row for row in models if row.get("available") is False]
        if misconfigured:
            status = "misconfigured"
            detail = " ".join(sorted(errors.values()))
        elif errors and not missing:
            status = "check_failed"
            detail = " ".join(sorted(errors.values()))
        elif not missing:
            status = "healthy"
            detail = "Configured writer models are available."
        elif len(missing) == len(checkable):
            status = "unavailable"
            detail = "No configured writer model is currently available: " + ", ".join(
                f"{row['provider']}:{row['model']}" for row in missing
            )
        else:
            status = "degraded"
            missing_names = ", ".join(
                f"{row['provider']}:{row['model']}" for row in missing
            )
            detail = f"Configured model unavailable: {missing_names}."
        if errors and status in {"degraded", "unavailable"}:
            # Some providers answered and some did not. Keep both facts.
            detail = f"{detail} {' '.join(sorted(errors.values()))}"

        _state = {
            "status": status,
            "checked_at": checked_at.isoformat(),
            "detail": detail,
            "models": models,
        }
    finally:
        if owns_client:
            await active_client.aclose()
    return current_model_availability()


async def model_availability_scheduler() -> None:
    try:
        check_seconds = max(
            300,
            int(os.getenv("MODEL_AVAILABILITY_CHECK_SECONDS", _DEFAULT_CHECK_SECONDS)),
        )
    except (TypeError, ValueError):
        check_seconds = _DEFAULT_CHECK_SECONDS

    while True:
        state = await refresh_model_availability()
        print(
            "[MODEL AVAILABILITY] "
            f"status={state['status']} detail={state['detail']}"
        )
        await asyncio.sleep(check_seconds)
