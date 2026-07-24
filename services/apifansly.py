"""Shared API Fansly client contract.

Every call to the managed Fansly API goes through this module so payload shapes,
response unwrapping, authentication, and account-access errors remain identical
across chat sync, vault sync, PPV delivery, and purchase reconciliation.
"""
from __future__ import annotations

import os
from typing import Any, Iterable

import httpx


DEFAULT_BASE_URL = "https://v1.apifansly.com/api/fansly"


class ApiFanslyConfigurationError(RuntimeError):
    """The deployment does not have a usable API Fansly configuration."""


class ApiFanslyAccountAccessError(RuntimeError):
    """The configured key cannot access the creator's stored account connection."""


class ApiFanslyProtocolError(RuntimeError):
    """The upstream response was successful HTTP but not the documented shape."""


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


def response_data(payload: Any) -> Any:
    """Return the documented ``data.data.response`` value."""
    if not isinstance(payload, dict):
        return None
    outer = payload.get("data")
    if not isinstance(outer, dict):
        return None
    inner = outer.get("data")
    if not isinstance(inner, dict):
        return None
    return inner.get("response")


def response_cursor(payload: Any, *, response: Any | None = None) -> str | None:
    """Normalize cursors used by chat, message, follower, and vault endpoints."""
    if isinstance(response, dict):
        for key in ("cursor", "nextCursor"):
            if response.get(key) is not None and response.get(key) != "":
                return str(response[key])
    if not isinstance(payload, dict):
        return None
    outer = payload.get("data")
    if not isinstance(outer, dict):
        return None
    value = outer.get("nextCursor")
    return str(value) if value is not None and value != "" else None


def media_references(
    media_ids: Iterable[str],
    *,
    preview_ids: dict[str, str | None] | None = None,
) -> list[dict[str, str | None]]:
    """Build the documented media attachment objects for message sends."""
    seen: set[str] = set()
    references: list[dict[str, str | None]] = []
    previews = preview_ids or {}
    for raw in media_ids:
        media_id = str(raw or "").strip()
        if not media_id or media_id in seen:
            continue
        seen.add(media_id)
        references.append({
            "mediaId": media_id,
            "previewId": (
                str(previews[media_id])
                if previews.get(media_id)
                else None
            ),
        })
    return references


def message_payload(
    *,
    content: str,
    media_ids: Iterable[str] = (),
    preview_ids: dict[str, str | None] | None = None,
    price_dollars: float | None = None,
) -> dict[str, Any]:
    """Build a free-text, free-media, or locked-PPV message payload."""
    payload: dict[str, Any] = {"content": str(content or "")}
    media = media_references(media_ids, preview_ids=preview_ids)
    if media:
        payload["mediaIds"] = media
    if price_dollars is not None:
        price = round(float(price_dollars), 2)
        if not media:
            raise ValueError("PPV messages require at least one media item")
        if price < 1 or price > 500:
            raise ValueError("PPV price must be between $1 and $500")
        payload["access_type"] = ["ppv"]
        payload["price"] = price
    return payload


def sent_message_id(payload: Any) -> str | None:
    """Extract the message ID from the documented send-message response."""
    response = response_data(payload)
    if not isinstance(response, dict):
        return None
    value = response.get("id")
    return str(value) if value is not None and value != "" else None


async def request(
    method: str,
    path: str,
    *,
    operation: str,
    account_id: str | None = None,
    params: dict[str, Any] | None = None,
    json: dict[str, Any] | None = None,
    files: Any = None,
    timeout: float = 30,
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """Execute one API Fansly request with uniform errors and JSON validation."""
    owns_client = client is None
    active_client = client or httpx.AsyncClient()
    try:
        response = await active_client.request(
            method,
            url(path),
            headers=headers(json_content=json is not None),
            params=params,
            json=json,
            files=files,
            timeout=timeout,
        )
        raise_for_response(
            response,
            operation=operation,
            account_id=account_id,
        )
        try:
            payload = response.json()
        except Exception as exc:
            raise ApiFanslyProtocolError(
                f"API Fansly {operation} returned invalid JSON"
            ) from exc
        if not isinstance(payload, dict):
            raise ApiFanslyProtocolError(
                f"API Fansly {operation} returned a non-object response"
            )
        return payload
    finally:
        if owns_client:
            await active_client.aclose()


async def send_message(
    account_id: str,
    chat_id: str,
    *,
    content: str,
    media_ids: Iterable[str] = (),
    preview_ids: dict[str, str | None] | None = None,
    price_dollars: float | None = None,
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """Send one text/media/PPV message using the documented request contract."""
    return await request(
        "POST",
        f"{account_id}/chats/{chat_id}/messages",
        operation="message delivery",
        account_id=account_id,
        json=message_payload(
            content=content,
            media_ids=media_ids,
            preview_ids=preview_ids,
            price_dollars=price_dollars,
        ),
        timeout=15,
        client=client,
    )


async def list_chats(
    account_id: str,
    *,
    cursor: str | None = None,
    filter: str = "all",
    sort: str = "newest",
    search: str | None = None,
    subscription_tier_id: str | None = None,
    client: httpx.AsyncClient | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str | None]:
    params: dict[str, Any] = {"filter": filter, "sort": sort}
    if cursor:
        params["cursor"] = cursor
    if search:
        params["search"] = search
    if subscription_tier_id:
        params["subscriptionTierId"] = subscription_tier_id
    payload = await request(
        "GET",
        f"{account_id}/chats",
        operation="chat listing",
        account_id=account_id,
        params=params,
        client=client,
    )
    response = response_data(payload)
    if not isinstance(response, dict):
        raise ApiFanslyProtocolError("API Fansly chat listing response is invalid")
    chats = response.get("data")
    aggregation = response.get("aggregationData")
    return (
        chats if isinstance(chats, list) else [],
        aggregation.get("accounts", [])
        if isinstance(aggregation, dict)
        and isinstance(aggregation.get("accounts"), list)
        else [],
        response_cursor(payload, response=response),
    )


async def list_chat_messages(
    account_id: str,
    chat_id: str,
    *,
    cursor: str | None = None,
    limit: int = 10,
    client: httpx.AsyncClient | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str | None]:
    params: dict[str, Any] = {"limit": max(1, min(10, int(limit)))}
    if cursor:
        params["cursor"] = cursor
    payload = await request(
        "GET",
        f"{account_id}/chats/{chat_id}/messages",
        operation="chat message listing",
        account_id=account_id,
        params=params,
        client=client,
    )
    response = response_data(payload)
    if not isinstance(response, dict):
        raise ApiFanslyProtocolError(
            "API Fansly chat message listing response is invalid"
        )
    messages = response.get("messages")
    account_media = response.get("accountMedia")
    return (
        messages if isinstance(messages, list) else [],
        account_media if isinstance(account_media, list) else [],
        response_cursor(payload, response=response),
    )


async def list_vault_albums(
    account_id: str,
    *,
    client: httpx.AsyncClient | None = None,
) -> list[dict[str, Any]]:
    payload = await request(
        "GET",
        f"{account_id}/vault/albums",
        operation="vault album listing",
        account_id=account_id,
        client=client,
    )
    response = response_data(payload)
    if not isinstance(response, dict):
        raise ApiFanslyProtocolError(
            "API Fansly vault album listing response is invalid"
        )
    albums = response.get("albums")
    return albums if isinstance(albums, list) else []


async def list_vault_album_media(
    account_id: str,
    album_id: str,
    *,
    cursor: str | None = None,
    limit: int = 50,
    client: httpx.AsyncClient | None = None,
) -> tuple[list[dict[str, Any]], str | None]:
    params: dict[str, Any] = {"limit": max(1, int(limit))}
    if cursor:
        params["cursor"] = cursor
    payload = await request(
        "GET",
        f"{account_id}/vault/albums/{album_id}/media",
        operation="vault album media listing",
        account_id=account_id,
        params=params,
        client=client,
    )
    response = response_data(payload)
    if isinstance(response, list):
        items = response
    elif isinstance(response, dict):
        raw = response.get("data") or response.get("media") or []
        items = raw if isinstance(raw, list) else []
    else:
        raise ApiFanslyProtocolError(
            "API Fansly vault album media response is invalid"
        )
    return items, response_cursor(payload, response=response)


async def list_followers(
    account_id: str,
    *,
    cursor: str | None = None,
    limit: int = 100,
    client: httpx.AsyncClient | None = None,
) -> tuple[Any, str | None]:
    params: dict[str, Any] = {"limit": max(1, int(limit))}
    if cursor:
        params["cursor"] = cursor
    payload = await request(
        "GET",
        f"{account_id}/followers",
        operation="follower listing",
        account_id=account_id,
        params=params,
        client=client,
    )
    response = response_data(payload)
    return response, response_cursor(payload, response=response)


async def list_subscribers(
    account_id: str,
    *,
    status: str = "all",
    cursor: str | None = None,
    limit: int = 100,
    search: str | None = None,
    subscription_tier_ids: str | None = None,
    client: httpx.AsyncClient | None = None,
) -> tuple[Any, str | None]:
    if status not in {"all", "active", "expired"}:
        raise ValueError("subscriber status must be all, active, or expired")
    params: dict[str, Any] = {
        "status": status,
        "limit": max(1, int(limit)),
    }
    if cursor:
        params["cursor"] = cursor
    if search:
        params["search"] = search
    if subscription_tier_ids:
        params["subscriptionTierIds"] = subscription_tier_ids
    payload = await request(
        "GET",
        f"{account_id}/subscribers",
        operation="subscriber listing",
        account_id=account_id,
        params=params,
        client=client,
    )
    response = response_data(payload)
    return response, response_cursor(payload, response=response)


async def top_supporters(
    account_id: str,
    *,
    before_ms: int | None = None,
    after_ms: int | None = None,
    client: httpx.AsyncClient | None = None,
) -> Any:
    params: dict[str, Any] = {}
    if before_ms is not None:
        params["before"] = int(before_ms)
    if after_ms is not None:
        params["after"] = int(after_ms)
    payload = await request(
        "GET",
        f"{account_id}/top-supporters",
        operation="top supporter listing",
        account_id=account_id,
        params=params,
        client=client,
    )
    return response_data(payload)


async def current_account(
    account_id: str,
    *,
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    payload = await request(
        "GET",
        f"{account_id}/me",
        operation="account profile load",
        account_id=account_id,
        client=client,
    )
    response = response_data(payload)
    if not isinstance(response, dict):
        raise ApiFanslyProtocolError("API Fansly account response is invalid")
    return response
