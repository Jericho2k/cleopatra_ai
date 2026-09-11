import asyncio
from datetime import datetime, timezone

import pytest

from services import model_availability
from services.model_availability import (
    current_model_availability,
    record_model_transport_failure,
    record_model_transport_success,
    refresh_model_availability,
)

NOW = datetime(2026, 8, 16, 12, 0, tzinfo=timezone.utc)

TOGETHER_URL = "https://api.together.xyz/v1/models"
OPENROUTER_URL = "https://openrouter.ai/api/v1/models"


@pytest.fixture(autouse=True)
def reset_runtime_attempts(monkeypatch):
    monkeypatch.setattr(model_availability, "_runtime_attempts", {})
    for name in (
        "WRITER_DEFAULT_PROVIDER",
        "WRITER_DEFAULT_MODEL",
        "WRITER_COMPLEX_PROVIDER",
        "WRITER_COMPLEX_MODEL",
        "OPENROUTER_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("TOGETHER_API_KEY", "test-key")


class Response:
    def __init__(self, model_ids):
        self.model_ids = model_ids

    def raise_for_status(self):
        return None

    def json(self):
        return [{"id": model_id} for model_id in self.model_ids]


class Client:
    """A catalog stub that answers per provider URL, like the real APIs do."""

    def __init__(self, catalogs: dict[str, list[str]]):
        self.catalogs = catalogs
        self.calls = []

    async def get(self, url, *, headers):
        self.calls.append((url, headers))
        return Response(self.catalogs.get(url, []))


def _default_client(**overrides):
    catalogs = {
        OPENROUTER_URL: ["moonshotai/kimi-k2.6"],
        TOGETHER_URL: ["Qwen/Qwen3.7-Plus"],
    }
    catalogs.update(overrides)
    return Client(catalogs)


def test_writer_models_are_checked_without_generation_tokens():
    client = _default_client()

    result = asyncio.run(refresh_model_availability(client=client, now=NOW))

    assert result["status"] == "healthy"
    assert all(model["available"] for model in result["models"])
    # One zero-token catalog read per distinct provider, never a generation.
    assert sorted(url for url, _ in client.calls) == sorted(
        [OPENROUTER_URL, TOGETHER_URL]
    )


def test_openrouter_writer_is_not_judged_against_the_together_catalog():
    """The regression this check exists to prevent.

    Together does not list moonshotai/kimi-k2.6, so a Together-only check would
    report the healthy ordinary writer as unavailable.
    """
    client = _default_client()

    result = asyncio.run(refresh_model_availability(client=client, now=NOW))
    ordinary = result["models"][0]

    assert ordinary["role"] == "ordinary_writer"
    assert ordinary["provider"] == "openrouter"
    assert ordinary["model"] == "moonshotai/kimi-k2.6"
    assert ordinary["available"] is True
    assert result["status"] == "healthy"


def test_each_provider_is_queried_with_its_own_credential():
    client = _default_client()

    asyncio.run(refresh_model_availability(client=client, now=NOW))
    by_url = {url: headers for url, headers in client.calls}

    assert by_url[OPENROUTER_URL]["Authorization"] == "Bearer test-openrouter-key"
    assert by_url[TOGETHER_URL]["Authorization"] == "Bearer test-key"


def test_pinned_upstream_providers_are_reported_for_openrouter():
    result = asyncio.run(refresh_model_availability(client=_default_client(), now=NOW))

    assert result["models"][0]["pinned_providers"] == ["Inceptron"]
    assert "pinned_providers" not in result["models"][1]


def test_missing_primary_model_surfaces_degraded_fallback():
    client = _default_client(**{OPENROUTER_URL: []})

    result = asyncio.run(refresh_model_availability(client=client, now=NOW))

    assert result["status"] == "degraded"
    assert result["models"][0]["role"] == "ordinary_writer"
    assert result["models"][0]["available"] is False
    assert result["models"][1]["available"] is True
    assert "openrouter:moonshotai/kimi-k2.6" in result["detail"]


def test_missing_openrouter_key_is_visible_without_network_call(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    client = _default_client()

    result = asyncio.run(refresh_model_availability(client=client, now=NOW))

    assert result["status"] == "misconfigured"
    assert "OPENROUTER_API_KEY" in result["detail"]
    assert result["models"][0]["available"] is False
    # Together is still reachable and must still be checked.
    assert [url for url, _ in client.calls] == [TOGETHER_URL]


def test_missing_together_key_is_visible_without_network_call(monkeypatch):
    monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
    client = _default_client()

    result = asyncio.run(refresh_model_availability(client=client, now=NOW))

    assert result["status"] == "misconfigured"
    assert "TOGETHER_API_KEY" in result["detail"]
    assert [url for url, _ in client.calls] == [OPENROUTER_URL]


def test_together_only_configuration_still_works(monkeypatch):
    monkeypatch.setenv("WRITER_DEFAULT_PROVIDER", "together")
    monkeypatch.setenv("WRITER_DEFAULT_MODEL", "moonshotai/Kimi-K3")
    client = Client(
        {TOGETHER_URL: ["moonshotai/Kimi-K3", "Qwen/Qwen3.7-Plus"]}
    )

    result = asyncio.run(refresh_model_availability(client=client, now=NOW))

    assert result["status"] == "healthy"
    assert [url for url, _ in client.calls] == [TOGETHER_URL]


def test_unknown_provider_is_reported_as_unknown_not_missing(monkeypatch):
    monkeypatch.setenv("WRITER_DEFAULT_PROVIDER", "anthropic")
    monkeypatch.setenv("WRITER_DEFAULT_MODEL", "claude-sonnet-4-6")

    result = asyncio.run(refresh_model_availability(client=_default_client(), now=NOW))

    assert result["models"][0]["available"] is None
    assert result["status"] == "healthy"


def test_repeated_live_primary_failures_surface_even_when_catalog_is_healthy():
    asyncio.run(refresh_model_availability(client=_default_client(), now=NOW))

    record_model_transport_failure("moonshotai/kimi-k2.6", "502 upstream error")
    record_model_transport_failure("moonshotai/kimi-k2.6", "502 upstream error")
    record_model_transport_success("Qwen/Qwen3.7-Plus")

    result = current_model_availability()

    assert result["status"] == "degraded"
    assert "moonshotai/kimi-k2.6" in result["detail"]
    assert result["models"][0]["runtime"]["consecutive_failures"] == 2
    assert result["models"][1]["runtime"]["consecutive_failures"] == 0


def test_availability_redirects_deprecated_kimi_environment_setting(monkeypatch):
    monkeypatch.setenv("WRITER_DEFAULT_PROVIDER", "together")
    monkeypatch.setenv("WRITER_DEFAULT_MODEL", "moonshotai/Kimi-K2.6")

    configured = model_availability.configured_writer_models()

    assert configured[0]["model"] == "moonshotai/Kimi-K3"


def test_availability_keeps_kimi_k26_on_openrouter(monkeypatch):
    monkeypatch.setenv("WRITER_DEFAULT_PROVIDER", "openrouter")
    monkeypatch.setenv("WRITER_DEFAULT_MODEL", "moonshotai/kimi-k2.6")

    configured = model_availability.configured_writer_models()

    assert configured[0]["provider"] == "openrouter"
    assert configured[0]["model"] == "moonshotai/kimi-k2.6"
