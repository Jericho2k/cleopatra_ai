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


@pytest.fixture(autouse=True)
def reset_runtime_attempts(monkeypatch):
    monkeypatch.setattr(model_availability, "_runtime_attempts", {})


class Response:
    def __init__(self, model_ids):
        self.model_ids = model_ids

    def raise_for_status(self):
        return None

    def json(self):
        return [{"id": model_id} for model_id in self.model_ids]


class Client:
    def __init__(self, model_ids):
        self.model_ids = model_ids
        self.calls = []

    async def get(self, url, *, headers):
        self.calls.append((url, headers))
        return Response(self.model_ids)


def test_writer_models_are_checked_without_generation_tokens(monkeypatch):
    monkeypatch.setenv("TOGETHER_API_KEY", "test-key")
    client = Client(
        ["moonshotai/Kimi-K3", "deepseek-ai/DeepSeek-V4-Pro"]
    )

    result = asyncio.run(refresh_model_availability(client=client, now=NOW))

    assert result["status"] == "healthy"
    assert all(model["available"] for model in result["models"])
    assert len(client.calls) == 1
    assert client.calls[0][1]["Authorization"] == "Bearer test-key"


def test_missing_primary_model_surfaces_degraded_fallback(monkeypatch):
    monkeypatch.setenv("TOGETHER_API_KEY", "test-key")
    client = Client(["deepseek-ai/DeepSeek-V4-Pro"])

    result = asyncio.run(refresh_model_availability(client=client, now=NOW))

    assert result["status"] == "degraded"
    assert result["models"][0]["role"] == "ordinary_writer"
    assert result["models"][0]["provider"] == "together"
    assert result["models"][0]["model"] == "moonshotai/Kimi-K3"
    assert result["models"][0]["available"] is False
    assert result["models"][1]["available"] is True


def test_missing_together_key_is_visible_without_network_call(monkeypatch):
    monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
    client = Client([])

    result = asyncio.run(refresh_model_availability(client=client, now=NOW))

    assert result["status"] == "misconfigured"
    assert "TOGETHER_API_KEY" in result["detail"]
    assert client.calls == []


def test_repeated_live_primary_failures_surface_even_when_catalog_is_healthy(
    monkeypatch,
):
    monkeypatch.setenv("TOGETHER_API_KEY", "test-key")
    client = Client(
        ["moonshotai/Kimi-K3", "deepseek-ai/DeepSeek-V4-Pro"]
    )
    asyncio.run(refresh_model_availability(client=client, now=NOW))

    record_model_transport_failure("moonshotai/Kimi-K3", "404 model removed")
    record_model_transport_failure("moonshotai/Kimi-K3", "404 model removed")
    record_model_transport_success("deepseek-ai/DeepSeek-V4-Pro")

    result = current_model_availability()

    assert result["status"] == "degraded"
    assert "moonshotai/Kimi-K3" in result["detail"]
    assert result["models"][0]["runtime"]["consecutive_failures"] == 2
    assert result["models"][1]["runtime"]["consecutive_failures"] == 0


def test_availability_redirects_deprecated_kimi_environment_setting(monkeypatch):
    monkeypatch.setenv("WRITER_DEFAULT_PROVIDER", "together")
    monkeypatch.setenv("WRITER_DEFAULT_MODEL", "moonshotai/Kimi-K2.6")

    configured = model_availability.configured_writer_models()

    assert configured[0]["model"] == "moonshotai/Kimi-K3"
