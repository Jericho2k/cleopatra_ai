import asyncio
import json
from pathlib import Path

import httpx
import pytest

from services.apifansly import (
    ApiFanslyAccountAccessError,
    ApiFanslyConfigurationError,
    DEFAULT_BASE_URL,
    delete_message,
    download_media,
    headers,
    is_fansly_cdn_url,
    media_references,
    message_payload,
    ppv_delivery_evidence,
    raise_for_response,
    response_cursor,
    sent_message_id,
    url,
)


def test_default_api_fansly_configuration(monkeypatch):
    monkeypatch.setenv("APIFANSLY_API_KEY", "test-key")
    monkeypatch.delenv("APIFANSLY_BASE_URL", raising=False)

    assert url("account/chats") == f"{DEFAULT_BASE_URL}/account/chats"
    assert headers() == {"x-api-key": "test-key"}
    assert headers(json_content=True)["Content-Type"] == "application/json"


def test_invalid_api_fansly_base_url_fails_closed(monkeypatch):
    monkeypatch.setenv("APIFANSLY_API_KEY", "test-key")
    monkeypatch.setenv("APIFANSLY_BASE_URL", "https://app.apifansly.com/api")

    with pytest.raises(ApiFanslyConfigurationError, match="/api/fansly"):
        url("account/chats")


def test_missing_api_fansly_key_fails_closed(monkeypatch):
    monkeypatch.delenv("APIFANSLY_API_KEY", raising=False)

    with pytest.raises(ApiFanslyConfigurationError, match="APIFANSLY_API_KEY"):
        headers()


def test_account_access_error_requires_reconnect():
    request = httpx.Request("GET", f"{DEFAULT_BASE_URL}/old-account/chats")
    response = httpx.Response(
        403,
        request=request,
        json={"error": "Account not found or access denied"},
    )

    with pytest.raises(ApiFanslyAccountAccessError, match="Reconnect this creator"):
        raise_for_response(
            response,
            operation="chat synchronization",
            account_id="old-account",
        )


def test_other_upstream_errors_preserve_http_status():
    request = httpx.Request("GET", f"{DEFAULT_BASE_URL}/account/chats")
    response = httpx.Response(
        503,
        request=request,
        json={"message": "temporarily unavailable"},
    )

    with pytest.raises(httpx.HTTPStatusError, match="chat synchronization"):
        raise_for_response(response, operation="chat synchronization")


def test_all_api_fansly_calls_use_shared_configuration():
    root = Path(__file__).resolve().parents[1]
    sources = [
        root / "main.py",
        root / "services" / "ppv_delivery.py",
        root / "services" / "ppv_reconciliation.py",
        root / "services" / "suggestions.py",
    ]
    for source in sources:
        assert "https://v1.apifansly.com/api/fansly" not in source.read_text()

    main_source = (root / "main.py").read_text()
    shared_source = (root / "services" / "apifansly.py").read_text()
    assert 'operation="chat listing"' in shared_source
    assert "HTTPException(status_code=409" in main_source
    assert "creator_id: str | None = None" in main_source
    assert '"reconnected": bool(req.creator_id)' in main_source
    assert 'print(f"[CONNECT] req={req.dict()}")' not in main_source
    env_example = (root / ".env.example").read_text()
    assert "APIFANSLY_API_KEY=" in env_example
    assert f"APIFANSLY_BASE_URL={DEFAULT_BASE_URL}" in env_example
    assert "FANSLY_SESSION_KEY=" in env_example


def test_locked_ppv_payload_matches_current_api_contract():
    payload = message_payload(
        content="just for you",
        media_ids=["media-1", "media-2", "media-1"],
        price_dollars=30,
    )

    assert payload == {
        "content": "just for you",
        "mediaIds": [
            {"mediaId": "media-1", "previewId": None},
            {"mediaId": "media-2", "previewId": None},
        ],
        "access_type": ["ppv"],
        "price": 30.0,
    }
    assert "mediaId" not in payload


def test_ppv_delivery_evidence_rejects_silently_free_media():
    evidence = ppv_delivery_evidence(
        [{
            "id": "message-1",
            "attachments": [{"contentId": "account-media-1"}],
        }],
        [{
            "id": "account-media-1",
            "mediaId": "media-1",
            "price": 0,
        }],
        message_id="message-1",
        expected_media_ids=["media-1"],
        expected_price_cents=2500,
    )

    assert evidence == {
        "verified": False,
        "reason": "media_is_not_payment_gated",
        "raw_prices": [0.0],
    }


@pytest.mark.parametrize("upstream_price", [25, 2500])
def test_ppv_delivery_evidence_accepts_documented_dollars_or_cents(
    upstream_price,
):
    evidence = ppv_delivery_evidence(
        [{
            "id": "message-1",
            "attachments": [{"contentId": "account-media-1"}],
        }],
        [{
            "id": "account-media-1",
            "mediaId": "media-1",
            "price": upstream_price,
        }],
        message_id="message-1",
        expected_media_ids=["media-1"],
        expected_price_cents=2500,
    )

    assert evidence["verified"] is True
    assert evidence["actual_media_ids"] == ["media-1"]


def test_media_references_preserve_per_item_preview():
    assert media_references(
        ["media-1"],
        preview_ids={"media-1": "preview-1"},
    ) == [{"mediaId": "media-1", "previewId": "preview-1"}]


def test_send_response_id_and_endpoint_cursors_use_documented_nesting():
    send_payload = {
        "data": {"data": {"response": {"id": "message-123"}}}
    }
    message_page = {
        "data": {
            "data": {
                "response": {
                    "messages": [],
                    "cursor": "older-message-cursor",
                }
            }
        }
    }
    chat_page = {"data": {"nextCursor": "next-chat-cursor"}}

    assert sent_message_id(send_payload) == "message-123"
    assert response_cursor(
        message_page,
        response=message_page["data"]["data"]["response"],
    ) == "older-message-cursor"
    assert response_cursor(chat_page) == "next-chat-cursor"


def test_delete_message_uses_documented_compensation_endpoint(monkeypatch):
    monkeypatch.setenv("APIFANSLY_API_KEY", "test-key")
    captured = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        return httpx.Response(
            200,
            request=request,
            json={"data": {"data": {"response": {"success": True}}}},
        )

    async def run():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            return await delete_message(
                "account-1",
                "message-1",
                client=client,
            )

    assert asyncio.run(run()) is True
    assert captured == {
        "method": "DELETE",
        "path": "/api/fansly/account-1/messages/message-1",
    }


def test_protected_media_download_uses_documented_binary_endpoint(monkeypatch):
    monkeypatch.setenv("APIFANSLY_API_KEY", "test-key")
    image_bytes = b"\xff\xd8" + (b"x" * 1200)
    captured = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["key"] = request.headers.get("x-api-key")
        captured["json"] = json.loads(request.content)
        return httpx.Response(
            200,
            request=request,
            headers={"content-type": "image/jpeg"},
            content=image_bytes,
        )

    async def run():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            return await download_media(
                "https://cdn3.fansly.com/account/media.jpeg?Policy=signed",
                client=client,
            )

    assert asyncio.run(run()) == image_bytes
    assert captured == {
        "path": "/api/fansly/media/download",
        "key": "test-key",
        "json": {
            "cdnUrl": "https://cdn3.fansly.com/account/media.jpeg?Policy=signed"
        },
    }


def test_protected_media_download_rejects_non_fansly_urls(monkeypatch):
    monkeypatch.setenv("APIFANSLY_API_KEY", "test-key")
    assert is_fansly_cdn_url("https://cdn3.fansly.com/account/media.jpeg")
    assert not is_fansly_cdn_url("https://fansly.example/media.jpeg")

    with pytest.raises(ValueError, match="Fansly CDN"):
        asyncio.run(download_media("https://example.com/private.jpeg"))
