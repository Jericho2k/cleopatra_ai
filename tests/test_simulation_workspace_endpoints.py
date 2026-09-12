"""The workspace endpoints: owner only, test fans only, and no real-fan reach.

The simulator runs the real pipeline with delivery replaced by local
persistence, which makes every one of these a privileged capability rather than
a product feature. Each test removes exactly one condition and asserts the same
indistinguishable 404 the rest of the simulator uses.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import main
from core import tenancy


OWNER = "11111111-1111-4111-8111-111111111111"
AGENCY = "22222222-2222-4222-8222-222222222222"

ASSIGNMENTS = {OWNER: {"creator-1"}, AGENCY: {"creator-1"}}

FANS = {
    "fan-test": {
        "id": "fan-test",
        "creator_id": "creator-1",
        "platform_fan_id": "test_abc",
        "display_name": "Test fan",
    },
    "fan-real": {
        "id": "fan-real",
        "creator_id": "creator-1",
        "platform_fan_id": "884422113355",
        "display_name": "Real fan",
    },
}


class _Table:
    def __init__(self, store, name):
        self.store = store
        self.name = name
        self.filters: list[tuple[str, str]] = []
        self.payload: dict | None = None

    def select(self, *_a, **_k):
        return self

    def insert(self, payload):
        self.payload = payload
        return self

    def eq(self, column, value):
        self.filters.append((column, str(value)))
        return self

    def limit(self, *_a, **_k):
        return self

    def single(self):
        return self

    def execute(self):
        if self.payload is not None:
            row = dict(self.payload)
            row["id"] = f"fan-new-{len(self.store.fans)}"
            self.store.fans[row["id"]] = row
            self.store.inserts.append(dict(row))
            return SimpleNamespace(data=[dict(row)])
        rows = [
            row
            for row in self.store.fans.values()
            if all(str(row.get(col)) == val for col, val in self.filters)
        ]
        return SimpleNamespace(data=[dict(row) for row in rows])


class _DB:
    def __init__(self):
        self.fans = {k: dict(v) for k, v in FANS.items()}
        self.inserts: list[dict] = []

    def table(self, name):
        return _Table(self, name)


@pytest.fixture
def store():
    return _DB()


@pytest.fixture
def client(monkeypatch, store):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("DASHBOARD_API_SECRET", "test-dashboard-secret")
    monkeypatch.setenv("AUTO_SIMULATION_ENABLED", "true")
    monkeypatch.setenv("AUTO_SIMULATION_ALLOWED_USER_IDS", OWNER)

    async def fake_user(authorization):
        token = str(authorization or "").split(" ")[-1]
        return token if token in ASSIGNMENTS else None

    async def fake_creator_ids(user_id):
        return ASSIGNMENTS.get(user_id, set())

    monkeypatch.setattr(main, "authenticated_dashboard_user", fake_user, raising=False)
    monkeypatch.setattr("core.auth.authenticated_dashboard_user", fake_user)
    monkeypatch.setattr(tenancy, "_creator_ids_for_user", fake_creator_ids)
    monkeypatch.setattr(main, "get_supabase", lambda: store)
    monkeypatch.setattr(tenancy, "get_supabase", lambda: store)
    monkeypatch.setattr("services.simulation_workspace.get_supabase", lambda: store)
    return TestClient(app=main.app)


def headers(user: str) -> dict[str, str]:
    return {"X-API-Key": "test-dashboard-secret", "Authorization": f"Bearer {user}"}


# --- creating a test fan ----------------------------------------------------


def test_the_owner_can_create_a_test_fan(client, store):
    response = client.post(
        "/simulation/test-fans",
        headers=headers(OWNER),
        json={"creator_id": "creator-1", "display_name": "Fan A"},
    )

    assert response.status_code == 200
    fan = response.json()["fan"]
    assert fan["platform_fan_id"].startswith("test_")
    assert fan["simulation"] is True


def test_an_agency_account_cannot_create_a_test_fan(client, store):
    response = client.post(
        "/simulation/test-fans",
        headers=headers(AGENCY),
        json={"creator_id": "creator-1"},
    )

    assert response.status_code == 404
    assert store.inserts == []


def test_a_client_supplied_platform_id_is_not_accepted(client, store):
    """The field does not exist. Even an owner cannot make a fan that later
    turns out to be a real Fansly account."""
    response = client.post(
        "/simulation/test-fans",
        headers=headers(OWNER),
        json={
            "creator_id": "creator-1",
            "display_name": "Sneaky",
            "platform_fan_id": "884422113355",
        },
    )

    assert response.status_code == 200
    assert store.inserts[-1]["platform_fan_id"].startswith("test_")
    assert store.inserts[-1]["platform_fan_id"] != "884422113355"


def test_a_test_fan_cannot_be_created_under_an_unassigned_creator(client, store):
    response = client.post(
        "/simulation/test-fans",
        headers=headers(OWNER),
        json={"creator_id": "creator-999"},
    )

    assert response.status_code == 404
    assert store.inserts == []


# --- reading state and firing actions --------------------------------------


def test_state_and_run_now_refuse_a_real_fan(client):
    state = client.get(
        "/creator/creator-1/fan/fan-real/simulation-state", headers=headers(OWNER)
    )
    run_now = client.post(
        "/creator/creator-1/fan/fan-real/simulation-actions/a1/run-now",
        headers=headers(OWNER),
        json={},
    )

    assert state.status_code == 404
    assert run_now.status_code == 404


def test_an_agency_account_reaches_none_of_the_workspace(client):
    responses = [
        client.get(
            "/creator/creator-1/fan/fan-test/simulation-state", headers=headers(AGENCY)
        ),
        client.post(
            "/creator/creator-1/fan/fan-test/simulation-actions/a1/run-now",
            headers=headers(AGENCY),
            json={},
        ),
        client.post(
            "/creator/creator-1/simulation-media-previews",
            headers=headers(AGENCY),
            json={"media_ids": ["sim:abc:1"]},
        ),
    ]

    assert [response.status_code for response in responses] == [404, 404, 404]
    assert {response.json()["detail"] for response in responses} == {
        "Resource not found"
    }


def test_the_whole_workspace_disappears_when_the_feature_is_off(client, monkeypatch):
    monkeypatch.setenv("AUTO_SIMULATION_ENABLED", "false")

    responses = [
        client.post(
            "/simulation/test-fans",
            headers=headers(OWNER),
            json={"creator_id": "creator-1"},
        ),
        client.get(
            "/creator/creator-1/fan/fan-test/simulation-state", headers=headers(OWNER)
        ),
        client.post(
            "/creator/creator-1/simulation-media-previews",
            headers=headers(OWNER),
            json={"media_ids": []},
        ),
    ]

    assert [response.status_code for response in responses] == [404, 404, 404]
