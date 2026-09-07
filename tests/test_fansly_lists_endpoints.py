"""Tenancy and behaviour of the Fansly list read and refresh endpoints.

creator_id from the client is never trusted on its own: the route dependency
must confirm the caller is assigned to that creator before any API Fansly call,
and the account ID must come from the creator row rather than the request.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import main
from core import tenancy
from services.apifansly import ApiFanslyAccountAccessError


ASSIGNMENTS = {
    "operator-1": {"creator-1"},
    "operator-2": {"creator-2"},
}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("DASHBOARD_API_SECRET", "test-dashboard-secret")
    monkeypatch.setenv("FANSLY_LISTS_SYNC_ENABLED", "true")

    async def fake_user(authorization):
        token = str(authorization or "").split(" ")[-1]
        return token if token in ASSIGNMENTS else None

    async def fake_creator_ids(user_id):
        return ASSIGNMENTS.get(user_id, set())

    monkeypatch.setattr(main, "authenticated_dashboard_user", fake_user, raising=False)
    monkeypatch.setattr("core.auth.authenticated_dashboard_user", fake_user)
    monkeypatch.setattr(tenancy, "_creator_ids_for_user", fake_creator_ids)
    return TestClient(app=main.app)


def _headers(operator: str) -> dict[str, str]:
    return {
        "X-API-Key": "test-dashboard-secret",
        "Authorization": f"Bearer {operator}",
    }


class _Creators:
    def __init__(self, rows):
        self.rows = rows

    def table(self, _name):
        return self

    def select(self, *_args, **_kwargs):
        return self

    def eq(self, column, value):
        self.rows = [row for row in self.rows if str(row.get(column)) == str(value)]
        return self

    def limit(self, _value):
        return self

    def execute(self):
        return SimpleNamespace(data=self.rows)


def test_cross_tenant_refresh_is_rejected_before_any_api_call(client, monkeypatch):
    called = []

    async def should_not_run(*args, **kwargs):
        called.append(args)
        return {}

    monkeypatch.setattr("services.fansly_lists.sync_fansly_lists", should_not_run)

    response = client.post(
        "/creator/creator-1/sync-fansly-lists",
        headers=_headers("operator-2"),
    )

    # 404 rather than 403: never reveal that another agency's creator exists.
    assert response.status_code == 404
    assert called == []


def test_cross_tenant_read_is_rejected(client, monkeypatch):
    called = []

    async def should_not_run(*args, **kwargs):
        called.append(args)
        return {}

    monkeypatch.setattr("services.fansly_lists.read_lists_sync_state", should_not_run)

    response = client.get(
        "/creator/creator-1/fansly-lists",
        headers=_headers("operator-2"),
    )

    assert response.status_code == 404
    assert called == []


def test_unauthenticated_refresh_is_rejected(client, monkeypatch):
    response = client.post(
        "/creator/creator-1/sync-fansly-lists",
        headers={"X-API-Key": "test-dashboard-secret"},
    )

    assert response.status_code in {401, 404}


def test_assigned_operator_can_refresh_and_the_account_id_comes_from_the_row(
    client, monkeypatch
):
    seen: dict = {}

    async def fake_sync(creator_id, account_id, **_kwargs):
        seen["creator_id"] = creator_id
        seen["account_id"] = account_id
        return {"status": "ok", "created_lists": 1}

    async def fake_state(creator_id):
        return {"last_synced_at": "2026-09-07T00:00:00+00:00", "lists": []}

    monkeypatch.setattr("services.fansly_lists.sync_fansly_lists", fake_sync)
    monkeypatch.setattr("services.fansly_lists.read_lists_sync_state", fake_state)
    monkeypatch.setattr(
        main,
        "get_supabase",
        lambda: _Creators([{"id": "creator-1", "apifansly_account_id": "acct-1"}]),
    )

    response = client.post(
        "/creator/creator-1/sync-fansly-lists",
        headers=_headers("operator-1"),
    )

    assert response.status_code == 200
    assert seen == {"creator_id": "creator-1", "account_id": "acct-1"}
    body = response.json()
    assert body["created_lists"] == 1
    assert body["last_synced_at"] == "2026-09-07T00:00:00+00:00"


def test_refresh_on_a_disconnected_creator_is_a_conflict(client, monkeypatch):
    monkeypatch.setattr(
        main,
        "get_supabase",
        lambda: _Creators([{"id": "creator-1", "apifansly_account_id": None}]),
    )

    response = client.post(
        "/creator/creator-1/sync-fansly-lists",
        headers=_headers("operator-1"),
    )

    assert response.status_code == 409
    assert "not connected" in response.json()["detail"]


def test_access_denied_becomes_a_reconnect_conflict_not_a_500(client, monkeypatch):
    async def denied(*_args, **_kwargs):
        raise ApiFanslyAccountAccessError(
            "API Fansly access denied for account acct-1. Reconnect this creator."
        )

    monkeypatch.setattr("services.fansly_lists.sync_fansly_lists", denied)
    monkeypatch.setattr(
        main,
        "get_supabase",
        lambda: _Creators([{"id": "creator-1", "apifansly_account_id": "acct-1"}]),
    )

    response = client.post(
        "/creator/creator-1/sync-fansly-lists",
        headers=_headers("operator-1"),
    )

    assert response.status_code == 409
    assert "Reconnect this creator" in response.json()["detail"]


def test_refresh_is_a_conflict_while_the_feature_is_disabled(client, monkeypatch):
    monkeypatch.setenv("FANSLY_LISTS_SYNC_ENABLED", "false")

    response = client.post(
        "/creator/creator-1/sync-fansly-lists",
        headers=_headers("operator-1"),
    )

    assert response.status_code == 409


def test_read_returns_mirrors_and_sync_state(client, monkeypatch):
    async def fake_state(creator_id):
        assert creator_id == "creator-1"
        return {
            "last_synced_at": "2026-09-07T00:00:00+00:00",
            "last_error": None,
            "last_failed_at": None,
            "lists": [
                {
                    "id": "list-1",
                    "name": "VIP",
                    "source": "fansly",
                    "external_list_id": "1001",
                    "external_archived_at": None,
                }
            ],
        }

    monkeypatch.setattr("services.fansly_lists.read_lists_sync_state", fake_state)

    response = client.get(
        "/creator/creator-1/fansly-lists",
        headers=_headers("operator-1"),
    )

    assert response.status_code == 200
    body = response.json()
    assert body["creator_id"] == "creator-1"
    assert body["enabled"] is True
    assert body["lists"][0]["external_list_id"] == "1001"


# --- lifecycle integration ----------------------------------------------------


class _CreatorState:
    def __init__(self, row):
        self.row = row

    def table(self, _name):
        return self

    def select(self, *_args, **_kwargs):
        return self

    def eq(self, _column, _value):
        return self

    def limit(self, _value):
        return self

    def execute(self):
        return SimpleNamespace(data=[self.row])


def test_full_sync_always_refreshes_lists(monkeypatch):
    import asyncio

    monkeypatch.setenv("FANSLY_LISTS_SYNC_ENABLED", "true")
    calls = []

    async def fake_sync(creator_id, account_id):
        calls.append((creator_id, account_id))
        return {"status": "ok"}

    monkeypatch.setattr("services.fansly_lists.sync_fansly_lists", fake_sync)

    result = asyncio.run(
        main._sync_fansly_lists_if_due("creator-1", "acct-1", force=True)
    )

    assert result == {"status": "ok"}
    assert calls == [("creator-1", "acct-1")]


def test_incremental_pass_skips_a_recently_synced_mirror(monkeypatch):
    import asyncio
    from datetime import datetime, timezone

    monkeypatch.setenv("FANSLY_LISTS_SYNC_ENABLED", "true")
    monkeypatch.setenv("FANSLY_LISTS_SYNC_INTERVAL_HOURS", "6")
    calls = []

    async def fake_sync(creator_id, account_id):
        calls.append(creator_id)
        return {"status": "ok"}

    monkeypatch.setattr("services.fansly_lists.sync_fansly_lists", fake_sync)
    monkeypatch.setattr(
        main,
        "get_supabase",
        lambda: _CreatorState(
            {"last_fansly_lists_sync_at": datetime.now(timezone.utc).isoformat()}
        ),
    )

    result = asyncio.run(
        main._sync_fansly_lists_if_due("creator-1", "acct-1", force=False)
    )

    # No API Fansly credits spent for a mirror that is already fresh.
    assert result is None
    assert calls == []


def test_incremental_pass_refreshes_a_stale_mirror(monkeypatch):
    import asyncio
    from datetime import datetime, timedelta, timezone

    monkeypatch.setenv("FANSLY_LISTS_SYNC_ENABLED", "true")
    monkeypatch.setenv("FANSLY_LISTS_SYNC_INTERVAL_HOURS", "6")
    calls = []

    async def fake_sync(creator_id, account_id):
        calls.append(creator_id)
        return {"status": "ok"}

    monkeypatch.setattr("services.fansly_lists.sync_fansly_lists", fake_sync)
    monkeypatch.setattr(
        main,
        "get_supabase",
        lambda: _CreatorState(
            {
                "last_fansly_lists_sync_at": (
                    datetime.now(timezone.utc) - timedelta(hours=7)
                ).isoformat()
            }
        ),
    )

    asyncio.run(main._sync_fansly_lists_if_due("creator-1", "acct-1", force=False))

    assert calls == ["creator-1"]


def test_a_list_failure_never_fails_the_surrounding_chat_sync(monkeypatch):
    import asyncio

    monkeypatch.setenv("FANSLY_LISTS_SYNC_ENABLED", "true")

    async def boom(creator_id, account_id):
        raise RuntimeError("upstream 500")

    monkeypatch.setattr("services.fansly_lists.sync_fansly_lists", boom)

    result = asyncio.run(
        main._sync_fansly_lists_if_due("creator-1", "acct-1", force=True)
    )

    # Explicit in the sync result rather than swallowed silently.
    assert result == {"status": "error", "detail": "upstream 500"}


def test_access_denied_during_chat_sync_is_reported_not_retried(monkeypatch):
    import asyncio

    monkeypatch.setenv("FANSLY_LISTS_SYNC_ENABLED", "true")

    async def denied(creator_id, account_id):
        raise ApiFanslyAccountAccessError("access denied; Reconnect this creator")

    monkeypatch.setattr("services.fansly_lists.sync_fansly_lists", denied)

    result = asyncio.run(
        main._sync_fansly_lists_if_due("creator-1", "acct-1", force=True)
    )

    assert result["status"] == "access_denied"


def test_lists_are_not_synced_while_the_feature_is_disabled(monkeypatch):
    import asyncio

    monkeypatch.setenv("FANSLY_LISTS_SYNC_ENABLED", "false")
    calls = []

    async def fake_sync(creator_id, account_id):
        calls.append(creator_id)
        return {"status": "ok"}

    monkeypatch.setattr("services.fansly_lists.sync_fansly_lists", fake_sync)

    assert (
        asyncio.run(main._sync_fansly_lists_if_due("creator-1", "acct-1", force=True))
        is None
    )
    assert calls == []
