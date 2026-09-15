"""The historical-memory HTTP surface.

Two things are asserted here that unit tests cannot: that the routes exist with
the tenancy they claim, and that /load-history keeps the response shape the
dashboard already renders while gaining the telemetry an operator needs.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import main
from core import tenancy
from services import fan_history


ASSIGNMENTS = {"operator-1": {"creator-1"}}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("DASHBOARD_API_SECRET", "test-dashboard-secret")

    async def fake_user(authorization):
        token = str(authorization or "").split(" ")[-1]
        return token if token in ASSIGNMENTS else None

    async def fake_creator_ids(user_id):
        return ASSIGNMENTS.get(user_id, set())

    async def owning_creator(_fan_id):
        return "creator-1"

    monkeypatch.setattr(main, "authenticated_dashboard_user", fake_user, raising=False)
    monkeypatch.setattr("core.auth.authenticated_dashboard_user", fake_user)
    monkeypatch.setattr(tenancy, "_creator_ids_for_user", fake_creator_ids)
    monkeypatch.setattr(tenancy, "_fan_creator_id", owning_creator)
    return TestClient(app=main.app)


def _headers(operator: str = "operator-1") -> dict[str, str]:
    return {
        "X-API-Key": "test-dashboard-secret",
        "Authorization": f"Bearer {operator}",
    }


@pytest.fixture
def stubbed_history(monkeypatch):
    calls: dict[str, dict] = {}

    async def _warm(*, creator_id, fan_id, client=None):
        calls["warm"] = {"creator_id": creator_id, "fan_id": fan_id}
        return {"status": "resumed", "pages": 3, "imported": 30, "estimated_credits": 3.0}

    async def _advance(*, creator_id, fan_id, max_pages=None, client=None, respect_live_priority=True):
        calls["advance"] = {
            "max_pages": max_pages,
            "respect_live_priority": respect_live_priority,
        }
        return {"status": "paused", "pages": 20, "imported": 200, "estimated_credits": 24.0}

    async def _status(fan_id):
        return {
            "history": {
                "status": "paused",
                "pages_fetched": 23,
                "messages_imported": 230,
                "estimated_credits": 27.0,
                "fully_paged": False,
            }
        }

    # The profile refresh /load-history has always triggered. Stubbed rather
    # than removed, because asserting it still fires is the point of
    # test_load_history_still_refreshes_the_profile_panel below.
    spawned: list[str] = []

    async def _history(_fan_id, *args, **kwargs):
        return [SimpleNamespace(role="fan", content="hi") for _ in range(12)]

    async def _profile(_fan_id):
        return SimpleNamespace(total_spent=0)

    def _spawn(coro, name=""):
        coro.close()
        spawned.append(name)
        return None

    calls["spawned"] = spawned

    monkeypatch.setattr(fan_history, "warm_resume", _warm)
    monkeypatch.setattr(fan_history, "advance_backfill", _advance)
    monkeypatch.setattr(fan_history, "fan_history_status", _status)
    monkeypatch.setattr(main, "get_conversation_history", _history)
    monkeypatch.setattr(main, "get_fan_by_id", _profile)
    monkeypatch.setattr(main, "spawn", _spawn)
    return calls


def test_load_history_keeps_the_field_the_dashboard_renders(client, stubbed_history):
    """components/FanPanel.tsx shows `Imported ${data.imported} messages`."""
    response = client.post("/load-history/creator-1/fan-1", headers=_headers())
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "ok"
    assert body["imported"] == 230  # warm resume plus the deep run


def test_load_history_reports_cost_and_progress(client, stubbed_history):
    body = client.post("/load-history/creator-1/fan-1", headers=_headers()).json()
    assert body["estimated_credits"] == 27.0
    assert body["history"]["pages_fetched"] == 23
    assert body["history"]["fully_paged"] is False


def test_load_history_does_not_yield_to_the_operator_who_pressed_it(
    client, stubbed_history
):
    """The button must do something when pressed.

    An operator opening a conversation IS live activity, so a deep run that
    deferred to "a live call happened seconds ago" would make Load history a
    no-op exactly when it is used.
    """
    client.post("/load-history/creator-1/fan-1", headers=_headers())
    assert stubbed_history["advance"]["respect_live_priority"] is False


def test_load_history_honours_an_explicit_page_budget(client, stubbed_history):
    client.post("/load-history/creator-1/fan-1?pages=5", headers=_headers())
    assert stubbed_history["advance"]["max_pages"] == 5


def test_load_history_can_warm_resume_without_a_deep_run(client, stubbed_history):
    body = client.post("/load-history/creator-1/fan-1?deep=false", headers=_headers()).json()
    assert body["imported"] == 30
    assert "advance" not in stubbed_history
    assert body["deep"] == {"status": "skipped"}


def test_load_history_still_refreshes_the_profile_panel(client, stubbed_history):
    """Behaviour the dashboard already depends on, kept deliberately.

    The old route spawned the AI summary and fan-memory refresh after an import,
    and components/FanPanel.tsx renders both. Historical compaction adds
    evidence-backed facts alongside those documents; it does not replace them.
    """
    client.post("/load-history/creator-1/fan-1", headers=_headers())
    assert sorted(stubbed_history["spawned"]) == [
        "update_fan_ai_summary",
        "update_fan_memory",
    ]


def test_load_history_skips_the_profile_refresh_when_nothing_was_imported(
    client, stubbed_history, monkeypatch
):
    """A second press imports nothing, so it must not pay for two model calls."""

    async def _nothing(*args, **kwargs):
        return {"status": "sufficient_context", "imported": 0, "estimated_credits": 0.0}

    async def _nothing_deep(*args, **kwargs):
        return {"status": "complete", "imported": 0, "estimated_credits": 0.0}

    monkeypatch.setattr(fan_history, "warm_resume", _nothing)
    monkeypatch.setattr(fan_history, "advance_backfill", _nothing_deep)
    client.post("/load-history/creator-1/fan-1", headers=_headers())
    assert stubbed_history["spawned"] == []


def test_progress_needs_no_provider_call_and_no_connector(client, stubbed_history, monkeypatch):
    """An operator must be able to read progress with the connector off."""
    monkeypatch.setenv("APIFANSLY_ENABLED", "false")
    response = client.get("/fan-history/creator-1/fan-1", headers=_headers())
    assert response.status_code == 200, response.text
    assert response.json()["history"]["pages_fetched"] == 23


def test_history_routes_refuse_another_operator_s_creator(client, stubbed_history):
    """404, not 403: core.tenancy deliberately does not reveal existence."""
    response = client.get("/fan-history/creator-2/fan-1", headers=_headers())
    assert response.status_code == 404, response.text
    assert response.json()["detail"] == "Resource not found"


def test_history_routes_refuse_an_unauthenticated_caller(client, stubbed_history):
    """The new routes widen nobody's access; they reuse the same guards."""
    response = client.get(
        "/fan-history/creator-1/fan-1", headers={"X-API-Key": "test-dashboard-secret"}
    )
    assert response.status_code in (401, 403, 404), response.text


def test_creator_history_usage_reports_the_budget(client, monkeypatch):
    async def _totals(creator_id):
        return {
            "fans": 4,
            "complete": 1,
            "in_progress": 3,
            "errored": 0,
            "pages_fetched": 812,
            "messages_imported": 8_100,
            "messages_extracted": 3_000,
            "api_calls": 812,
            "response_bytes": 90_000_000,
            "estimated_credits": 1_412.5,
        }

    monkeypatch.setattr("db.fan_history_queries.creator_history_totals", _totals)
    monkeypatch.setenv("APIFANSLY_HISTORY_CREDIT_BUDGET_24H", "2000")

    body = client.get("/creator-history-usage/creator-1", headers=_headers()).json()
    assert body["history"]["estimated_credits"] == 1_412.5
    assert body["budget"]["budget_credits"] == 2000.0
    assert body["messages_per_page_max"] == 10
    assert "authoritative" in body["note"]


def test_usage_snapshot_endpoint_exposes_the_credit_view(client):
    body = client.get("/apifansly-usage", headers=_headers()).json()
    for key in (
        "estimated_credits",
        "estimated_monthly_credits",
        "credits_by_category",
        "credits_by_operation",
        "credits_by_account",
        "webhook_events",
        "background_history_budget",
        "credit_model",
    ):
        assert key in body, key
    # And the fields that shipped before it.
    assert body["by_operation"] == {}
    assert body["window_hours"] == 24
