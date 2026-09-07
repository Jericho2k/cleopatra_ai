"""SEC-004 — APP_ENV must fail closed when it is missing or unrecognised.

The historical rule was ``os.environ.get("APP_ENV", "development") ==
"development"``, so the absence of one Railway variable disabled dashboard auth,
tenancy checks, and webhook signature verification at once. These tests pin the
inverted polarity: relaxed behaviour requires an explicit development value.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from core import environment
from core.auth import (
    _is_dev,
    authenticated_dashboard_user,
    require_dashboard,
    require_webhook,
)
from main import app


@pytest.fixture(autouse=True)
def _reset_warning_cache():
    environment._warned_values.clear()
    yield
    environment._warned_values.clear()


# --- resolution table -------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("development", environment.DEVELOPMENT),
        ("dev", environment.DEVELOPMENT),
        ("local", environment.DEVELOPMENT),
        ("DEVELOPMENT", environment.DEVELOPMENT),
        ("  development  ", environment.DEVELOPMENT),
        ("test", environment.TEST),
        ("testing", environment.TEST),
        ("production", environment.PRODUCTION),
        ("prod", environment.PRODUCTION),
    ],
)
def test_recognised_values_resolve(monkeypatch, raw, expected):
    monkeypatch.setenv("APP_ENV", raw)
    assert environment.app_env() == expected


@pytest.mark.parametrize(
    "raw",
    ["", "   ", "staging", "developement", "prod uction", "0", "none", "Development-2"],
)
def test_unrecognised_values_resolve_to_production(monkeypatch, raw):
    """SEC-004: an unrecognised value must never widen access."""
    monkeypatch.setenv("APP_ENV", raw)
    assert environment.app_env() == environment.PRODUCTION
    assert environment.is_production() is True
    assert environment.is_development() is False
    assert environment.local_test_endpoints_enabled() is False


def test_missing_app_env_resolves_to_production(monkeypatch):
    monkeypatch.delenv("APP_ENV", raising=False)
    assert environment.app_env() == environment.PRODUCTION
    assert environment.is_development() is False
    assert _is_dev() is False


def test_test_env_is_not_development(monkeypatch):
    """APP_ENV=test keeps production auth enforcement; it only unlocks /test/*."""
    monkeypatch.setenv("APP_ENV", "test")
    assert environment.is_development() is False
    assert _is_dev() is False
    assert environment.local_test_endpoints_enabled() is True


# --- auth guards with APP_ENV missing --------------------------------------


@pytest.mark.asyncio
async def test_missing_app_env_rejects_unconfigured_dashboard_secret(monkeypatch):
    monkeypatch.delenv("APP_ENV", raising=False)
    monkeypatch.delenv("DASHBOARD_API_SECRET", raising=False)

    with pytest.raises(HTTPException) as excinfo:
        await require_dashboard(x_api_key=None)
    assert excinfo.value.status_code == 500


@pytest.mark.asyncio
async def test_missing_app_env_rejects_request_without_bearer_token(monkeypatch):
    monkeypatch.delenv("APP_ENV", raising=False)

    with pytest.raises(HTTPException) as excinfo:
        await authenticated_dashboard_user(None)
    assert excinfo.value.status_code == 401


@pytest.mark.asyncio
async def test_missing_app_env_rejects_unconfigured_webhook_secret(monkeypatch):
    """A missing signing secret must not silently accept a production webhook."""
    monkeypatch.delenv("APP_ENV", raising=False)
    monkeypatch.delenv("WEBHOOK_SECRET", raising=False)

    with pytest.raises(HTTPException) as excinfo:
        await require_webhook(x_webhook_secret=None)
    assert excinfo.value.status_code == 500


def test_missing_app_env_rejects_unauthenticated_dashboard_request(monkeypatch):
    monkeypatch.delenv("APP_ENV", raising=False)
    monkeypatch.setenv("DASHBOARD_API_SECRET", "expected-dashboard-key")

    response = TestClient(app).get(
        "/fan/test-fan/operator-ppv-options?creator_id=test-creator"
    )

    assert response.status_code == 401


def test_missing_app_env_does_not_bypass_tenancy(monkeypatch):
    """require_creator_access must raise on a None operator when APP_ENV is unset."""
    import asyncio

    from core.tenancy import require_creator_access

    monkeypatch.delenv("APP_ENV", raising=False)

    class _Request:
        # Mimics starlette's request.state container.
        class state:
            pass

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(require_creator_access(_Request(), "some-creator"))
    assert excinfo.value.status_code == 404


# --- development still works ------------------------------------------------


@pytest.mark.asyncio
async def test_development_keeps_local_open_behaviour(monkeypatch):
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.delenv("DASHBOARD_API_SECRET", raising=False)
    monkeypatch.delenv("WEBHOOK_SECRET", raising=False)

    assert await require_dashboard(x_api_key=None) is None
    assert await require_webhook(x_webhook_secret=None) is None
    assert await authenticated_dashboard_user(None) is None


# --- production unchanged ---------------------------------------------------


def test_production_rejects_wrong_api_key(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("DASHBOARD_API_SECRET", "expected-dashboard-key")

    response = TestClient(app).get(
        "/fan/test-fan/operator-ppv-options?creator_id=test-creator",
        headers={"X-API-Key": "wrong-key"},
    )

    assert response.status_code == 401
