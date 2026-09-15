"""An agency tenant cannot obtain or mirror another tenant's creator or media.

The sprint that opened the simulator to agency operators moved the load-bearing
boundary. It used to be the owner allowlist: almost nothing was reachable, so
almost nothing had to be checked carefully. Now an agency operator legitimately
holds creators, legitimately simulates them, and legitimately creates test fans
— which means ORDINARY TENANCY is what stands between one agency and another's
creators, fans, vault and mirrored media.

So this file does not test the happy path (test_simulation_authorization.py
does). It calls every simulator surface the way a hostile tenant with a valid
session would, naming another tenant's ids directly, and asserts:

* the same indistinguishable 404 every time, so nothing is learned;
* that no side effect happened on the way to the refusal — nothing mirrored,
  nothing inserted, nothing read out of another tenant's vault.

Nothing here goes through the dashboard. Hiding a control is not a boundary.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import main
from core import tenancy


OWNER = "11111111-1111-4111-8111-111111111111"
AGENCY = "22222222-2222-4222-8222-222222222222"

# AGENCY holds creator-1 only. creator-2 belongs to somebody else and is what
# every attack below tries to reach.
ASSIGNMENTS = {
    OWNER: {"creator-1"},
    AGENCY: {"creator-1"},
}

CREATORS = {
    "creator-1": {"id": "creator-1", "platform_username": "agency-own"},
    "creator-2": {"id": "creator-2", "platform_username": "other-tenant"},
}

FANS = {
    "fan-test": {
        "id": "fan-test",
        "creator_id": "creator-1",
        "platform_fan_id": "test_own",
        "display_name": "Own test fan",
    },
    "fan-other": {
        "id": "fan-other",
        "creator_id": "creator-2",
        "platform_fan_id": "test_other",
        "display_name": "Other tenant's test fan",
    },
}

# A mirrored row sitting on the AGENCY's own creator whose provenance points at
# creator-2. This is the subtle case: the row is on a creator the agency holds,
# so tenancy on the ROW passes — and the pixels still belong to another tenant.
MIRRORED_MEDIA = [
    {
        "id": "mirror-1",
        "creator_id": "creator-1",
        "media_id": "sim:creator2:999",
        "source_creator_id": "creator-2",
        "source_media_id": "999",
        "mimetype": "image/jpeg",
        "simulation_only": True,
    },
]

OTHER_TENANT_VAULT = [
    {
        "id": "other-1",
        "creator_id": "creator-2",
        "media_id": "999",
        "url": "https://cdn3.fansly.com/other-tenant/secret.jpg",
        "thumbnail_url": "https://cdn3.fansly.com/other-tenant/secret-thumb.jpg",
        "mimetype": "image/jpeg",
    },
]


class _Query:
    """One filtered read or write against one table, recording what it touched."""

    def __init__(self, store: "_Store", name: str) -> None:
        self.store = store
        self.name = name
        self.rows = [dict(row) for row in store.tables.get(name, [])]

    def select(self, *_a, **_k):
        self.store.reads.append(self.name)
        return self

    def eq(self, column, value):
        self.rows = [r for r in self.rows if str(r.get(column)) == str(value)]
        return self

    def in_(self, column, values):
        allowed = {str(v) for v in values}
        self.rows = [r for r in self.rows if str(r.get(column)) in allowed]
        return self

    def like(self, column, pattern):
        prefix = pattern.rstrip("%")
        self.rows = [
            r for r in self.rows if str(r.get(column) or "").startswith(prefix)
        ]
        return self

    def order(self, *_a, **_k):
        return self

    def limit(self, *_a, **_k):
        return self

    def single(self):
        return self

    def insert(self, rows):
        self.store.inserts.append((self.name, rows))
        return self

    def upsert(self, rows, **_k):
        self.store.inserts.append((self.name, rows))
        return self

    def update(self, values):
        self.store.updates.append((self.name, values))
        return self

    def delete(self):
        self.store.deletes.append(self.name)
        return self

    def execute(self):
        return SimpleNamespace(data=list(self.rows), count=len(self.rows))


class _Store:
    def __init__(self) -> None:
        self.tables = {
            "creators": list(CREATORS.values()),
            "fans": list(FANS.values()),
            "creator_vault_media": [*MIRRORED_MEDIA, *OTHER_TENANT_VAULT],
            "vault_sets": [],
            "scheduled_actions": [],
        }
        self.reads: list[str] = []
        self.inserts: list[tuple] = []
        self.updates: list[tuple] = []
        self.deletes: list[str] = []

    def table(self, name):
        return _Query(self, name)


@pytest.fixture
def store():
    return _Store()


@pytest.fixture
def client(monkeypatch, store):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("DASHBOARD_API_SECRET", "test-dashboard-secret")
    monkeypatch.setenv("AUTO_SIMULATION_ENABLED", "true")
    monkeypatch.setenv("AUTO_SIMULATION_ALLOWED_USER_IDS", OWNER)
    monkeypatch.delenv("AUTO_SIMULATION_AGENCY_ACCESS", raising=False)

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
    monkeypatch.setattr("services.simulation_catalog.get_supabase", lambda: store)
    return TestClient(app=main.app)


def _headers(user: str) -> dict[str, str]:
    return {
        "X-API-Key": "test-dashboard-secret",
        "Authorization": f"Bearer {user}",
    }


NOT_FOUND = {"detail": "Resource not found"}


# ---------------------------------------------------------------------------
# 1. Another tenant's creator cannot be simulated, listed, or written to
# ---------------------------------------------------------------------------


def test_an_agency_cannot_simulate_another_tenants_creator(client, monkeypatch):
    async def never_runs(**_kwargs):
        raise AssertionError("Full Auto must not run for another tenant's creator")

    monkeypatch.setattr("services.suggestions.run_simulated_inbound", never_runs)
    response = client.post(
        "/creator/creator-2/fan/fan-other/simulate-inbound",
        headers=_headers(AGENCY),
        json={"message": "hi", "fast": True},
    )

    assert response.status_code == 404
    assert response.json() == NOT_FOUND


def test_the_simulator_creator_listing_never_names_another_tenant(client):
    body = client.get("/simulation/creators", headers=_headers(AGENCY)).json()

    ids = [row["id"] for row in body["creators"]]
    assert ids == ["creator-1"]
    assert "creator-2" not in client.get(
        "/simulation/creators", headers=_headers(AGENCY)
    ).text


def test_an_agency_cannot_create_a_test_fan_under_another_tenants_creator(
    client, store
):
    response = client.post(
        "/simulation/test-fans",
        headers=_headers(AGENCY),
        json={"creator_id": "creator-2", "display_name": "wedge"},
    )

    assert response.status_code == 404
    assert store.inserts == [], "no row may be written on the way to a refusal"


def test_an_agency_cannot_pin_another_tenants_fan_to_an_ai_stack(client, store):
    response = client.put(
        "/creator/creator-2/fan/fan-other/ai-stack",
        headers=_headers(AGENCY),
        json={"ai_stack_profile": "cleo_v2"},
    )

    assert response.status_code == 404
    assert store.updates == []


# ---------------------------------------------------------------------------
# 2. Mirror-source discovery is invisible to an agency
# ---------------------------------------------------------------------------


def test_an_agency_cannot_call_mirror_source_discovery(client):
    """The one route that enumerates every creator in the deployment."""
    response = client.get("/simulation/catalog/sources", headers=_headers(AGENCY))

    assert response.status_code == 404
    assert response.json() == NOT_FOUND


def test_mirror_source_discovery_is_not_reached_at_all_for_an_agency(
    client, monkeypatch
):
    """The refusal is authorization, not an empty result: the cross-tenant
    listing function must never be entered."""

    async def never_runs():
        raise AssertionError("cross-tenant source discovery must not run")

    monkeypatch.setattr(
        "services.simulation_catalog.list_mirror_source_creators", never_runs
    )
    assert client.get(
        "/simulation/catalog/sources", headers=_headers(AGENCY)
    ).status_code == 404


def test_the_owner_still_gets_the_cross_tenant_source_listing(client, monkeypatch):
    """The capability is preserved, not removed — it just became owner-only."""

    async def sources():
        from services.simulation_catalog import MirrorSource

        return [
            MirrorSource(
                creator_id="creator-2",
                name="other-tenant",
                approved_sets=3,
                media_items=40,
            )
        ]

    monkeypatch.setattr(
        "services.simulation_catalog.list_mirror_source_creators", sources
    )
    response = client.get("/simulation/catalog/sources", headers=_headers(OWNER))

    assert response.status_code == 200, response.text
    assert response.json()["sources"][0]["creator_id"] == "creator-2"


# ---------------------------------------------------------------------------
# 3. An agency cannot mirror anything, in any direction
# ---------------------------------------------------------------------------


@pytest.fixture
def mirror_spy(monkeypatch):
    calls: list[tuple] = []

    async def fake_mirror(*, source_creator_id, target_creator_id):
        calls.append(("mirror", source_creator_id, target_creator_id))
        from services.simulation_catalog import MirrorResult

        return MirrorResult(
            source_creator_id=source_creator_id,
            target_creator_id=target_creator_id,
            media_mirrored=1,
            sets_mirrored=1,
            media_removed=0,
            sets_removed=0,
        )

    async def fake_delete(*, source_creator_id, target_creator_id):
        calls.append(("delete", source_creator_id, target_creator_id))
        from services.simulation_catalog import MirrorResult

        return MirrorResult(
            source_creator_id=source_creator_id,
            target_creator_id=target_creator_id,
            media_mirrored=0,
            sets_mirrored=0,
            media_removed=1,
            sets_removed=1,
        )

    monkeypatch.setattr(
        "services.simulation_catalog.mirror_creator_catalog", fake_mirror
    )
    monkeypatch.setattr(
        "services.simulation_catalog.delete_creator_catalog_mirror", fake_delete
    )
    return calls


@pytest.mark.parametrize(
    "source,target,description",
    [
        ("creator-2", "creator-1", "another tenant's vault INTO its own creator"),
        ("creator-1", "creator-2", "its own vault INTO another tenant's creator"),
        ("creator-2", "creator-2", "entirely between another tenant's creators"),
        ("creator-1", "creator-1", "between creators it legitimately holds"),
    ],
)
def test_an_agency_cannot_mirror_in_any_direction(
    client, mirror_spy, source, target, description
):
    """Including the last case, which is the important one.

    Mirroring creator-1 onto creator-1 involves no other tenant at all, and it
    is STILL refused — because the mirror is the owner tier as a whole, not a
    cross-tenant check bolted onto an agency-accessible route. If the agency
    could invoke it here, the only thing standing between it and creator-2
    would be the source id it chose to type.
    """
    response = client.post(
        "/simulation/catalog/mirror",
        headers=_headers(AGENCY),
        json={"source_creator_id": source, "target_creator_id": target},
    )

    assert response.status_code == 404, description
    assert response.json() == NOT_FOUND
    assert mirror_spy == [], "no mirror may run for a rejected caller"


@pytest.mark.parametrize("source,target", [("creator-2", "creator-1"), ("creator-1", "creator-1")])
def test_an_agency_cannot_delete_a_mirror(client, mirror_spy, source, target):
    response = client.post(
        "/simulation/catalog/mirror/delete",
        headers=_headers(AGENCY),
        json={"source_creator_id": source, "target_creator_id": target},
    )

    assert response.status_code == 404
    assert mirror_spy == []


def test_the_owner_mirror_workflow_still_works(client, mirror_spy):
    """Preserved end to end: read another tenant's vault, write the owner's."""
    response = client.post(
        "/simulation/catalog/mirror",
        headers=_headers(OWNER),
        json={"source_creator_id": "creator-2", "target_creator_id": "creator-1"},
    )

    assert response.status_code == 200, response.text
    assert response.json()["simulation_only"] is True
    assert mirror_spy == [("mirror", "creator-2", "creator-1")]

    removed = client.post(
        "/simulation/catalog/mirror/delete",
        headers=_headers(OWNER),
        json={"source_creator_id": "creator-2", "target_creator_id": "creator-1"},
    )
    assert removed.status_code == 200
    assert mirror_spy[-1] == ("delete", "creator-2", "creator-1")


def test_the_owner_still_cannot_mirror_INTO_a_creator_they_do_not_hold(
    client, mirror_spy
):
    """Owner authority is read-across, never write-across."""
    response = client.post(
        "/simulation/catalog/mirror",
        headers=_headers(OWNER),
        json={"source_creator_id": "creator-1", "target_creator_id": "creator-2"},
    )

    assert response.status_code == 404
    assert mirror_spy == []


# ---------------------------------------------------------------------------
# 4. Mirrored cross-tenant MEDIA does not resolve for an agency
# ---------------------------------------------------------------------------


def test_an_agency_cannot_resolve_another_tenants_media_through_a_mirror(client):
    """The subtle one.

    ``mirror-1`` lives on creator-1, which the agency holds, so tenancy on the
    row passes. Its provenance points at creator-2. Resolving it would render
    another tenant's vault image inside a creator of one's own, which is
    exactly the thing the mirror exists to keep separate.
    """
    response = client.post(
        "/creator/creator-1/simulation-media-previews",
        headers=_headers(AGENCY),
        json={"media_ids": ["sim:creator2:999"]},
    )

    assert response.status_code == 200
    resolved = response.json()["media"]["sim:creator2:999"]
    assert resolved["url"] is None
    assert resolved["thumbnail_url"] is None
    assert "secret" not in response.text


def test_the_owner_may_resolve_the_mirror_they_created(client):
    response = client.post(
        "/creator/creator-1/simulation-media-previews",
        headers=_headers(OWNER),
        json={"media_ids": ["sim:creator2:999"]},
    )

    assert response.status_code == 200
    resolved = response.json()["media"]["sim:creator2:999"]
    assert resolved["url"] == "https://cdn3.fansly.com/other-tenant/secret.jpg"
    assert resolved["source"] == "simulation_mirror"


def test_an_agency_cannot_resolve_media_on_another_tenants_creator(client):
    response = client.post(
        "/creator/creator-2/simulation-media-previews",
        headers=_headers(AGENCY),
        json={"media_ids": ["sim:creator2:999"]},
    )

    assert response.status_code == 404


# ---------------------------------------------------------------------------
# 5. Every cross-tenant refusal looks identical
# ---------------------------------------------------------------------------


def test_every_cross_tenant_refusal_is_indistinguishable(client, mirror_spy):
    """An agency must not be able to tell 'that exists but is not yours' from
    'that does not exist' from 'that capability is not yours'."""
    bodies = {
        client.get("/simulation/catalog/sources", headers=_headers(AGENCY)).text,
        client.post(
            "/simulation/catalog/mirror",
            headers=_headers(AGENCY),
            json={"source_creator_id": "creator-2", "target_creator_id": "creator-1"},
        ).text,
        client.post(
            "/simulation/catalog/mirror",
            headers=_headers(AGENCY),
            json={
                "source_creator_id": "creator-does-not-exist",
                "target_creator_id": "creator-1",
            },
        ).text,
        client.post(
            "/simulation/test-fans",
            headers=_headers(AGENCY),
            json={"creator_id": "creator-2"},
        ).text,
        client.get(
            "/creator/creator-2/fan/fan-other/simulation-state",
            headers=_headers(AGENCY),
        ).text,
    }

    assert bodies == {'{"detail":"Resource not found"}'}


# ---------------------------------------------------------------------------
# 6. An agency's simulated turn still cannot reach the platform
# ---------------------------------------------------------------------------


def test_an_agency_simulated_turn_cannot_reach_real_fansly():
    """Zero delivery to real Fansly is a property of the transport, and it now
    has to hold for a tier that is not the owner's.

    The scope refuses the call before the network, so this asserts on the
    transport rather than on a list of callers somebody remembered to patch.
    """
    import pytest as _pytest

    from core.apifansly_gate import (
        ApiFanslyDisabledError,
        REASON_SIMULATION,
        simulation_scope,
    )
    from services import apifansly

    # Both tiers, because the mirrored-catalog flag widens what a turn may PLAN
    # against and must never widen what it may REACH.
    for mirrored in (False, True):
        with simulation_scope(include_mirrored_catalog=mirrored):
            with _pytest.raises(ApiFanslyDisabledError) as caught:
                apifansly.headers()
            assert caught.value.reason == REASON_SIMULATION


@pytest.mark.asyncio
async def test_an_agency_simulated_turn_cannot_download_protected_media():
    from core.apifansly_gate import ApiFanslyDisabledError, simulation_scope
    from services import apifansly

    with simulation_scope():
        with pytest.raises(ApiFanslyDisabledError):
            await apifansly.download_media(
                "https://cdn3.fansly.com/account/video.mp4"
            )
