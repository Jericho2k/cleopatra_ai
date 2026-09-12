"""Who may configure pricing, and what the read endpoint is honest about.

Configuring pricing is an agency's own business, so unlike the AI stack these
routes are ordinary creator-scoped operator endpoints. What they must never
allow is an operator reaching another agency's policy — which is why no scope id
is ever accepted from the client: the agency scope a write lands in is resolved
from the creator's own membership.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import main
from core import tenancy
from db import pricing_policy_queries
from services.pricing_presets import AGGRESSIVE, BALANCED, PRESETS


AGENCY_A = "22222222-2222-4222-8222-222222222222"
AGENCY_B = "33333333-3333-4333-8333-333333333333"

ASSIGNMENTS = {AGENCY_A: {"creator-a"}, AGENCY_B: {"creator-b"}}


class _Store:
    def __init__(self) -> None:
        self.policies: dict[tuple[str, str], dict] = {}
        self.memberships = {"creator-a": "agency-a", "creator-b": "agency-b"}

    def table(self, name):
        return _Table(self, name)


class _Table:
    def __init__(self, store: _Store, name: str) -> None:
        self.store = store
        self.name = name
        self.filters: dict[str, str] = {}
        self.payload: dict | None = None

    def select(self, *_a, **_k):
        return self

    def upsert(self, payload, **_k):
        self.payload = payload
        return self

    def eq(self, column, value):
        self.filters[column] = str(value)
        return self

    def limit(self, *_a, **_k):
        return self

    def single(self):
        return self

    def execute(self):
        if self.payload is not None:
            key = (self.payload["scope_type"], self.payload["scope_id"])
            self.store.policies[key] = dict(self.payload["settings"])
            return SimpleNamespace(data=[dict(self.payload)])
        if self.name == "creator_pricing_scope_memberships":
            scope = self.store.memberships.get(self.filters.get("creator_id", ""))
            return SimpleNamespace(data=[{"agency_scope_id": scope}] if scope else [])
        if self.name == "price_learning_policy_scopes":
            key = (self.filters.get("scope_type", ""), self.filters.get("scope_id", ""))
            settings = self.store.policies.get(key)
            return SimpleNamespace(data=[{"settings": settings}] if settings else [])
        return SimpleNamespace(data=[])


@pytest.fixture
def store():
    return _Store()


@pytest.fixture
def client(monkeypatch, store):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("DASHBOARD_API_SECRET", "test-dashboard-secret")
    monkeypatch.setenv("PRICE_LEARNING_ENABLED", "true")
    monkeypatch.setenv("PRICE_LEARNING_POLICY_CACHE_SECONDS", "0")
    pricing_policy_queries.clear_price_learning_policy_cache()

    async def fake_user(authorization):
        token = str(authorization or "").split(" ")[-1]
        return token if token in ASSIGNMENTS else None

    async def fake_creator_ids(user_id):
        return ASSIGNMENTS.get(user_id, set())

    monkeypatch.setattr(main, "authenticated_dashboard_user", fake_user, raising=False)
    monkeypatch.setattr("core.auth.authenticated_dashboard_user", fake_user)
    monkeypatch.setattr(tenancy, "_creator_ids_for_user", fake_creator_ids)
    monkeypatch.setattr(tenancy, "get_supabase", lambda: store)
    monkeypatch.setattr(pricing_policy_queries, "get_supabase", lambda: store)
    return TestClient(app=main.app)


def headers(user: str) -> dict[str, str]:
    return {"X-API-Key": "test-dashboard-secret", "Authorization": f"Bearer {user}"}


def test_the_read_reports_the_layers_and_the_effective_policy(client):
    response = client.get("/creator/creator-a/pricing-policy", headers=headers(AGENCY_A))

    assert response.status_code == 200
    body = response.json()
    assert body["agency_scope_id"] == "agency-a"
    assert body["effective_preset"] == BALANCED
    assert [row["preset"] for row in body["presets"]] == [
        "conservative",
        "balanced",
        "aggressive",
    ]
    assert "environment_defaults" in body
    assert "effective" in body


def test_the_read_states_the_feature_gate_honestly(client, monkeypatch):
    on = client.get("/creator/creator-a/pricing-policy", headers=headers(AGENCY_A)).json()
    assert on["price_learning_enabled"] is True

    monkeypatch.setenv("PRICE_LEARNING_ENABLED", "false")
    off = client.get("/creator/creator-a/pricing-policy", headers=headers(AGENCY_A)).json()

    assert off["price_learning_enabled"] is False
    assert off["price_learning_env_var"] == "PRICE_LEARNING_ENABLED"


def test_an_agency_writes_its_own_scope_and_the_creator_reads_it_back(client, store):
    response = client.put(
        "/creator/creator-a/pricing-policy",
        headers=headers(AGENCY_A),
        json={"scope": "agency", "preset": AGGRESSIVE},
    )

    assert response.status_code == 200
    assert response.json()["scope_id"] == "agency-a"
    assert store.policies[("AGENCY", "agency-a")]["cold_start_probe_bps"] == (
        PRESETS[AGGRESSIVE]["cold_start_probe_bps"]
    )
    assert response.json()["effective"]["cold_start_probe_bps"] == (
        PRESETS[AGGRESSIVE]["cold_start_probe_bps"]
    )


def test_an_operator_cannot_touch_another_agencys_creator(client, store):
    response = client.put(
        "/creator/creator-b/pricing-policy",
        headers=headers(AGENCY_A),
        json={"scope": "agency", "preset": AGGRESSIVE},
    )

    # The tenancy layer's indistinguishable rejection: an operator cannot tell
    # "not allowed" from "does not exist", so it cannot enumerate other agencies.
    assert response.status_code == 404
    assert response.json()["detail"] == "Resource not found"
    assert store.policies == {}


def test_no_scope_id_is_ever_accepted_from_the_client(client, store):
    """Naming another agency's scope has no effect: the scope is resolved from
    the creator's own membership, and the extra key is simply not read."""
    response = client.put(
        "/creator/creator-a/pricing-policy",
        headers=headers(AGENCY_A),
        json={"scope": "agency", "scope_id": "agency-b", "preset": AGGRESSIVE},
    )

    assert response.status_code == 200
    assert ("AGENCY", "agency-b") not in store.policies
    assert ("AGENCY", "agency-a") in store.policies


def test_a_creator_scope_write_beats_the_agency(client, store):
    client.put(
        "/creator/creator-a/pricing-policy",
        headers=headers(AGENCY_A),
        json={"scope": "agency", "preset": AGGRESSIVE},
    )
    response = client.put(
        "/creator/creator-a/pricing-policy",
        headers=headers(AGENCY_A),
        json={"scope": "creator", "preset": "conservative"},
    )

    assert response.json()["effective"]["cold_start_probe_bps"] == (
        PRESETS["conservative"]["cold_start_probe_bps"]
    )


def test_an_unknown_preset_is_refused(client, store):
    response = client.put(
        "/creator/creator-a/pricing-policy",
        headers=headers(AGENCY_A),
        json={"scope": "creator", "preset": "maximal"},
    )

    assert response.status_code == 400
    assert store.policies == {}


def test_an_invalid_advanced_value_is_refused_before_it_is_stored(client, store):
    response = client.put(
        "/creator/creator-a/pricing-policy",
        headers=headers(AGENCY_A),
        json={"scope": "creator", "settings": {"min_offer_cents": 900000}},
    )

    assert response.status_code == 400
    assert store.policies == {}


def test_advanced_settings_apply_on_top_of_a_preset(client, store):
    """"Aggressive, but cap the ceiling" is expressible in one request."""
    client.put(
        "/creator/creator-a/pricing-policy",
        headers=headers(AGENCY_A),
        json={
            "scope": "creator",
            "preset": AGGRESSIVE,
            "settings": {"max_offer_cents": 9000},
        },
    )

    stored = store.policies[("CREATOR", "creator-a")]
    assert stored["max_offer_cents"] == 9000
    assert stored["cold_start_probe_bps"] == PRESETS[AGGRESSIVE]["cold_start_probe_bps"]


def test_a_creator_with_no_agency_scope_is_told_rather_than_silently_redirected(
    client, store
):
    store.memberships.pop("creator-a")

    response = client.put(
        "/creator/creator-a/pricing-policy",
        headers=headers(AGENCY_A),
        json={"scope": "agency", "preset": AGGRESSIVE},
    )

    assert response.status_code == 400
    assert "agency pricing scope" in response.json()["detail"]
    assert store.policies == {}


def test_the_category_price_table_is_readable_and_marks_free_categories(client):
    response = client.get("/content-price-ranges", headers=headers(AGENCY_A))

    assert response.status_code == 200
    rows = {row["category"]: row for row in response.json()["categories"]}
    assert rows["nude_photo"]["min_dollars"] == 15
    assert rows["nude_photo"]["max_dollars"] == 80
    assert rows["nude_photo"]["priced"] is True
    # A free/teaser/unclear category is NOT an approved commercial range.
    assert rows["teaser_clothed"]["priced"] is False
    assert rows["other"]["priced"] is False
