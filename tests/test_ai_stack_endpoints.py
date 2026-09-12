"""Who may administer the AI stack, and what a client is allowed to send.

Choosing which brain answers every fan of a creator is an operational
capability, not an agency product feature, so these routes are gated by the same
owner allowlist as the simulator. An ordinary agency account must neither see
the controls nor successfully call the endpoints.

The second property is just as important: there is no way for any client — owner
included — to submit a provider or a model. The only thing that crosses the wire
is a stable profile identifier, validated against the backend registry.
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
        "ai_stack_profile": None,
    },
    "fan-real": {
        "id": "fan-real",
        "creator_id": "creator-1",
        "platform_fan_id": "884422113355",
        "display_name": "Real fan",
        "ai_stack_profile": None,
    },
}

CREATORS = {"creator-1": {"id": "creator-1", "ai_stack_profile": None}}


class _Table:
    def __init__(self, store, name):
        self.store = store
        self.name = name
        self.rows = store.creators if name == "creators" else store.fans
        self.filters: list[tuple[str, str]] = []
        self.payload: dict | None = None

    def select(self, *_a, **_k):
        return self

    def update(self, payload):
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
        matched = [
            row
            for row in self.rows.values()
            if all(str(row.get(col)) == val for col, val in self.filters)
        ]
        if self.payload is not None:
            for row in matched:
                row.update(self.payload)
            self.store.writes.append(dict(self.payload))
        return SimpleNamespace(data=[dict(row) for row in matched])


class _DB:
    def __init__(self):
        self.creators = {k: dict(v) for k, v in CREATORS.items()}
        self.fans = {k: dict(v) for k, v in FANS.items()}
        self.writes: list[dict] = []

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
    monkeypatch.setenv("AI_STACK_PROFILE", "cleo_v2")
    monkeypatch.setenv("AI_STACK_CACHE_SECONDS", "0")

    from services import ai_stack

    ai_stack.clear_ai_stack_cache()

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
    monkeypatch.setattr(ai_stack, "get_supabase", lambda: store)
    return TestClient(app=main.app)


def headers(user: str) -> dict[str, str]:
    return {"X-API-Key": "test-dashboard-secret", "Authorization": f"Bearer {user}"}


# --- who may read the registry ---------------------------------------------


def test_the_owner_can_read_every_profile(client):
    response = client.get("/ai-stack/profiles", headers=headers(OWNER))

    assert response.status_code == 200
    body = response.json()
    assert {row["profile_id"] for row in body["profiles"]} == {
        "cleo_legacy_v1",
        "cleo_v2",
    }
    assert body["environment_profile"] == "cleo_v2"
    assert body["environment_variable"] == "AI_STACK_PROFILE"


def test_an_agency_account_cannot_see_the_registry(client):
    assert client.get("/ai-stack/profiles", headers=headers(AGENCY)).status_code == 404


def test_the_registry_never_returns_an_api_key(client):
    body = client.get("/ai-stack/profiles", headers=headers(OWNER)).text

    assert "API_KEY" not in body
    assert "sk-" not in body


# --- who may change a creator's brain --------------------------------------


def test_an_agency_account_cannot_read_or_mutate_a_creator_override(client, store):
    assert client.get("/creator/creator-1/ai-stack", headers=headers(AGENCY)).status_code == 404
    response = client.put(
        "/creator/creator-1/ai-stack",
        headers=headers(AGENCY),
        json={"ai_stack_profile": "cleo_legacy_v1"},
    )

    assert response.status_code == 404
    # And nothing was written on the way to the refusal.
    assert store.writes == []
    assert store.creators["creator-1"]["ai_stack_profile"] is None


def test_the_owner_can_set_and_clear_a_creator_override(client, store):
    response = client.put(
        "/creator/creator-1/ai-stack",
        headers=headers(OWNER),
        json={"ai_stack_profile": "cleo_legacy_v1"},
    )

    assert response.status_code == 200
    assert response.json()["override"] == "cleo_legacy_v1"
    assert response.json()["effective"]["ai_stack_profile"] == "cleo_legacy_v1"
    assert response.json()["effective"]["ai_stack_source"] == "creator"
    assert store.creators["creator-1"]["ai_stack_profile"] == "cleo_legacy_v1"

    cleared = client.put(
        "/creator/creator-1/ai-stack",
        headers=headers(OWNER),
        json={"ai_stack_profile": None},
    )

    assert cleared.json()["override"] is None
    assert cleared.json()["effective"]["ai_stack_profile"] == "cleo_v2"


@pytest.mark.parametrize(
    "value",
    [
        "cleo_v99",
        "moonshotai/kimi-k2.6",
        "openrouter",
        "anthropic:claude-opus-5",
        "../../etc/passwd",
    ],
)
def test_an_arbitrary_provider_or_model_string_is_rejected(client, store, value):
    response = client.put(
        "/creator/creator-1/ai-stack",
        headers=headers(OWNER),
        json={"ai_stack_profile": value},
    )

    assert response.status_code == 400
    assert store.creators["creator-1"]["ai_stack_profile"] is None


def test_there_is_no_field_for_a_provider_or_a_model(client, store):
    """Even an owner cannot name one: the extra keys are simply not read."""
    response = client.put(
        "/creator/creator-1/ai-stack",
        headers=headers(OWNER),
        json={
            "ai_stack_profile": "cleo_v2",
            "provider": "openai",
            "model": "gpt-4o",
            "writer_model": "whatever",
        },
    )

    assert response.status_code == 200
    assert store.creators["creator-1"]["ai_stack_profile"] == "cleo_v2"
    assert all("model" not in write for write in store.writes)
    assert all("provider" not in write for write in store.writes)


# --- the simulation fan override -------------------------------------------


def test_a_test_fan_can_be_pinned_to_a_profile(client, store):
    response = client.put(
        "/creator/creator-1/fan/fan-test/ai-stack",
        headers=headers(OWNER),
        json={"ai_stack_profile": "cleo_legacy_v1"},
    )

    assert response.status_code == 200
    assert store.fans["fan-test"]["ai_stack_profile"] == "cleo_legacy_v1"


def test_a_real_fan_cannot_be_pinned_at_all(client, store):
    response = client.put(
        "/creator/creator-1/fan/fan-real/ai-stack",
        headers=headers(OWNER),
        json={"ai_stack_profile": "cleo_legacy_v1"},
    )

    assert response.status_code == 404
    assert store.fans["fan-real"]["ai_stack_profile"] is None


def test_an_agency_account_cannot_pin_even_a_test_fan(client, store):
    response = client.put(
        "/creator/creator-1/fan/fan-test/ai-stack",
        headers=headers(AGENCY),
        json={"ai_stack_profile": "cleo_legacy_v1"},
    )

    assert response.status_code == 404
    assert store.fans["fan-test"]["ai_stack_profile"] is None


def test_every_refusal_is_the_same_indistinguishable_404(client):
    """An agency must not be able to tell "not allowed" from "does not exist"."""
    responses = [
        client.get("/ai-stack/profiles", headers=headers(AGENCY)),
        client.get("/creator/creator-1/ai-stack", headers=headers(AGENCY)),
        client.put(
            "/creator/creator-1/ai-stack",
            headers=headers(AGENCY),
            json={"ai_stack_profile": "cleo_v2"},
        ),
        client.put(
            "/creator/creator-1/fan/fan-test/ai-stack",
            headers=headers(AGENCY),
            json={"ai_stack_profile": "cleo_v2"},
        ),
    ]

    assert [response.status_code for response in responses] == [404, 404, 404, 404]
    assert {response.json()["detail"] for response in responses} == {"Resource not found"}
