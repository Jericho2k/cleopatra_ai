from pathlib import Path

import httpx
import pytest

from services.apifansly import (
    ApiFanslyAccountAccessError,
    ApiFanslyConfigurationError,
    DEFAULT_BASE_URL,
    headers,
    raise_for_response,
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
    assert 'operation="chat synchronization"' in main_source
    assert "HTTPException(status_code=409" in main_source
    assert "creator_id: str | None = None" in main_source
    assert '"reconnected": bool(req.creator_id)' in main_source
    assert 'print(f"[CONNECT] req={req.dict()}")' not in main_source
    env_example = (root / ".env.example").read_text()
    assert "APIFANSLY_API_KEY=" in env_example
    assert f"APIFANSLY_BASE_URL={DEFAULT_BASE_URL}" in env_example
    assert "FANSLY_SESSION_KEY=" in env_example
