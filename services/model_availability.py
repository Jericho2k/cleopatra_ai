"""Background availability checks for configured Together writer models.

The catalog endpoint does not generate text and therefore consumes no inference
tokens. Its result is kept in memory for the operator dashboard; actual reply
generation still owns the primary-to-fallback behavior.
"""

from __future__ import annotations

import asyncio
import copy
import os
from datetime import datetime, timezone
from typing import Any

import httpx

_DEFAULT_CHECK_SECONDS = 6 * 60 * 60
_MODEL_LIST_URL = "https://api.together.xyz/v1/models"

_state: dict[str, Any] = {
    "status": "unknown",
    "checked_at": None,
    "detail": "Model availability has not been checked yet.",
    "models": [],
}
_runtime_attempts: dict[str, dict[str, Any]] = {}


def configured_writer_models() -> list[dict[str, str]]:
    """Return the ordinary writer and its complex-turn/fallback target."""
    configured = [
        {
            "role": "ordinary_writer",
            "provider": os.getenv("WRITER_DEFAULT_PROVIDER", "together").strip().lower(),
            "model": os.getenv(
                "WRITER_DEFAULT_MODEL", "moonshotai/Kimi-K2.6"
            ).strip(),
        },
        {
            "role": "complex_writer_and_fallback",
            "provider": os.getenv("WRITER_COMPLEX_PROVIDER", "together").strip().lower(),
            "model": os.getenv(
                "WRITER_COMPLEX_MODEL", "deepseek-ai/DeepSeek-V4-Pro"
            ).strip(),
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


async def refresh_model_availability(
    *,
    client: httpx.AsyncClient | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Check configured Together models against the provider's live catalog."""
    global _state
    checked_at = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    configured = configured_writer_models()
    together_models = [row for row in configured if row["provider"] == "together"]

    if not together_models:
        _state = {
            "status": "healthy",
            "checked_at": checked_at.isoformat(),
            "detail": "No Together writer models are configured.",
            "models": [{**row, "available": None} for row in configured],
        }
        return current_model_availability()

    api_key = os.getenv("TOGETHER_API_KEY", "").strip()
    if not api_key:
        _state = {
            "status": "misconfigured",
            "checked_at": checked_at.isoformat(),
            "detail": "TOGETHER_API_KEY is missing; AI replies cannot use Together.",
            "models": [{**row, "available": False} for row in configured],
        }
        return current_model_availability()

    owns_client = client is None
    active_client = client or httpx.AsyncClient(timeout=20)
    try:
        response = await active_client.get(
            _MODEL_LIST_URL,
            headers={"Authorization": f"Bearer {api_key}"},
        )
        response.raise_for_status()
        available_ids = {
            str(row.get("id") or "").strip()
            for row in _model_rows(response.json())
            if row.get("id")
        }
        models = [
            {
                **row,
                "available": (
                    row["model"] in available_ids
                    if row["provider"] == "together"
                    else None
                ),
            }
            for row in configured
        ]
        missing = [row for row in models if row.get("available") is False]
        if not missing:
            status = "healthy"
            detail = "Configured Together writer models are available."
        elif len(missing) == len(together_models):
            status = "unavailable"
            detail = "No configured Together writer model is currently available."
        else:
            status = "degraded"
            missing_names = ", ".join(row["model"] for row in missing)
            detail = f"Configured model unavailable: {missing_names}."
        _state = {
            "status": status,
            "checked_at": checked_at.isoformat(),
            "detail": detail,
            "models": models,
        }
    except (httpx.HTTPError, TypeError, ValueError) as exc:
        _state = {
            "status": "check_failed",
            "checked_at": checked_at.isoformat(),
            "detail": f"Together model availability check failed: {exc}",
            "models": [{**row, "available": None} for row in configured],
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
