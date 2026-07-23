"""Shared API Fansly configuration and error handling."""
from __future__ import annotations

import os
from typing import Any

import httpx


DEFAULT_BASE_URL = "https://v1.apifansly.com/api/fansly"


class ApiFanslyConfigurationError(RuntimeError):
    """The deployment does not have a usable API Fansly configuration."""


class ApiFanslyAccountAccessError(RuntimeError):
    """The configured key cannot access the creator's stored account connection."""


def api_key() -> str:
    value = str(os.environ.get("APIFANSLY_API_KEY") or "").strip()
    if not value:
        raise ApiFanslyConfigurationError("APIFANSLY_API_KEY is not configured")
    return value


def base_url() -> str:
    value = str(os.environ.get("APIFANSLY_BASE_URL") or DEFAULT_BASE_URL).strip()
    value = value.rstrip("/")
    if not value.startswith("https://") or not value.endswith("/api/fansly"):
        raise ApiFanslyConfigurationError(
            "APIFANSLY_BASE_URL must end with /api/fansly"
        )
    return value


def url(path: str = "") -> str:
    suffix = str(path or "").strip("/")
    return f"{base_url()}/{suffix}" if suffix else base_url()


def headers(*, json_content: bool = False) -> dict[str, str]:
    result = {"x-api-key": api_key()}
    if json_content:
        result["Content-Type"] = "application/json"
    return result


def response_message(response: httpx.Response) -> str:
    try:
        payload: Any = response.json()
    except Exception:
        payload = None
    if isinstance(payload, dict):
        for key in ("error", "message", "detail"):
            value = payload.get(key)
            if value:
                return str(value)
    return response.text[:200] or f"HTTP {response.status_code}"


def raise_for_response(
    response: httpx.Response,
    *,
    operation: str,
    account_id: str | None = None,
) -> None:
    if response.is_success:
        return
    message = response_message(response)
    if response.status_code in {401, 403}:
        target = f" for account {account_id}" if account_id else ""
        raise ApiFanslyAccountAccessError(
            f"API Fansly access denied{target} during {operation}: {message}. "
            "Reconnect this creator under the current APIFANSLY_API_KEY."
        )
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise httpx.HTTPStatusError(
            f"API Fansly {operation} failed: {message}",
            request=exc.request,
            response=exc.response,
        ) from exc
