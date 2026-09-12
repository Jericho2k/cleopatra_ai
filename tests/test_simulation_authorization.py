"""Who may reach the owner-only Full Auto simulator, and who may not.

The security boundary is the backend. The dashboard hides the simulator using
GET /simulation-capabilities, but hiding is convenience: every test here calls
the backend directly, the way an ordinary agency account with a valid session
could.

Six conditions must all hold. Each test removes exactly one and asserts the same
indistinguishable 404, so a caller can never learn which condition it failed, nor
that another tenant's creator or fan exists.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import main
from core import simulation, tenancy


OWNER = "11111111-1111-4111-8111-111111111111"
AGENCY = "22222222-2222-4222-8222-222222222222"
OTHER_TENANT = "33333333-3333-4333-8333-333333333333"

ASSIGNMENTS = {
    OWNER: {"creator-1"},
    AGENCY: {"creator-1"},
    OTHER_TENANT: {"creator-2"},
}

FANS = {
    "fan-test": {
        "id": "fan-test",
        "creator_id": "creator-1",
        "platform_fan_id": "test_jostar",
        "display_name": "Jostar",
    },
    "fan-real": {
        "id": "fan-real",
        "creator_id": "creator-1",
        "platform_fan_id": "884422113355",
        "display_name": "Real Fan",
    },
    "fan-other": {
        "id": "fan-other",
        "creator_id": "creator-2",
        "platform_fan_id": "test_other",
        "display_name": "Other Tenant Test Fan",
    },
}


class _Fans:
    """Minimal fans table honouring the .eq() filters the route applies."""

    def __init__(self) -> None:
        self.rows = list(FANS.values())

    def table(self, _name):
        return self

    def select(self, *_a, **_k):
        return self

    def eq(self, column, value):
        self.rows = [r for r in self.rows if str(r.get(column)) == str(value)]
        return self

    def limit(self, *_a, **_k):
        return self

    def single(self):
        return self

    def execute(self):
        return SimpleNamespace(data=list(self.rows))


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("DASHBOARD_API_SECRET", "test-dashboard-secret")
    monkeypatch.setenv("AUTO_SIMULATION_ENABLED", "true")
    monkeypatch.setenv("AUTO_SIMULATION_ALLOWED_USER_IDS", OWNER)

    async def fake_user(authorization):
        token = str(authorization or "").split(" ")[-1]
        return token if token in ASSIGNMENTS else None

    async def fake_creator_ids(user_id):
        return ASSIGNMENTS.get(user_id, set())

    async def never_runs(**_kwargs):
        raise AssertionError("Full Auto must not run for a rejected caller")

    monkeypatch.setattr(main, "authenticated_dashboard_user", fake_user, raising=False)
    monkeypatch.setattr("core.auth.authenticated_dashboard_user", fake_user)
    monkeypatch.setattr(tenancy, "_creator_ids_for_user", fake_creator_ids)
    monkeypatch.setattr(main, "get_supabase", _Fans)
    # core.tenancy binds get_supabase at import time, so it needs its own patch.
    monkeypatch.setattr(tenancy, "get_supabase", _Fans)
    monkeypatch.setattr("services.suggestions.run_simulated_inbound", never_runs)
    return TestClient(app=main.app)


def _headers(user: str) -> dict[str, str]:
    return {
        "X-API-Key": "test-dashboard-secret",
        "Authorization": f"Bearer {user}",
    }


def _simulate(client, user, creator="creator-1", fan="fan-test"):
    return client.post(
        f"/creator/{creator}/fan/{fan}/simulate-inbound",
        headers=_headers(user),
        json={"message": "hiii", "fast": True},
    )


# --- 10: the master switch --------------------------------------------------


def test_nobody_can_simulate_when_the_feature_is_off(client, monkeypatch):
    monkeypatch.setenv("AUTO_SIMULATION_ENABLED", "false")
    assert _simulate(client, OWNER).status_code == 404
    assert (
        client.get("/simulation-capabilities", headers=_headers(OWNER)).json()
        == {"auto_simulation": False}
    )


def test_allowlist_alone_is_not_enough(monkeypatch):
    monkeypatch.setenv("AUTO_SIMULATION_ENABLED", "false")
    monkeypatch.setenv("AUTO_SIMULATION_ALLOWED_USER_IDS", OWNER)
    assert simulation.user_may_simulate(OWNER) is False


def test_enabled_alone_is_not_enough(monkeypatch):
    monkeypatch.setenv("AUTO_SIMULATION_ENABLED", "true")
    monkeypatch.setenv("AUTO_SIMULATION_ALLOWED_USER_IDS", "")
    assert simulation.user_may_simulate(OWNER) is False
    assert simulation.allowed_simulation_user_ids() == frozenset()


def test_there_is_no_development_bypass(monkeypatch):
    """core.auth and core.tenancy relax under APP_ENV=development. This must
    not, or one misread variable hands the simulator to a production tenant."""
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("AUTO_SIMULATION_ENABLED", "true")
    monkeypatch.setenv("AUTO_SIMULATION_ALLOWED_USER_IDS", "")
    assert simulation.user_may_simulate(None) is False
    assert simulation.user_may_simulate(OWNER) is False


def test_blank_allowlist_entries_never_match_a_blank_user(monkeypatch):
    monkeypatch.setenv("AUTO_SIMULATION_ENABLED", "true")
    monkeypatch.setenv("AUTO_SIMULATION_ALLOWED_USER_IDS", " , ,, ")
    assert simulation.allowed_simulation_user_ids() == frozenset()
    assert simulation.user_may_simulate("") is False
    assert simulation.user_may_simulate(" ") is False


def test_allowlist_tolerates_whitespace_and_case(monkeypatch):
    monkeypatch.setenv("AUTO_SIMULATION_ENABLED", "true")
    monkeypatch.setenv(
        "AUTO_SIMULATION_ALLOWED_USER_IDS", f"  {OWNER.upper()} ,\n{AGENCY}  ,"
    )
    assert simulation.user_may_simulate(OWNER) is True
    assert simulation.user_may_simulate(OWNER.upper()) is True
    assert simulation.user_may_simulate(OTHER_TENANT) is False


# --- 11, 12, 13: allowlist and capability -----------------------------------


def test_enabled_but_not_allowlisted_is_rejected(client):
    assert _simulate(client, AGENCY).status_code == 404


def test_allowlisted_user_sees_capability_true(client):
    response = client.get("/simulation-capabilities", headers=_headers(OWNER))
    assert response.status_code == 200
    assert response.json() == {"auto_simulation": True}


def test_ordinary_agency_account_sees_capability_false(client):
    response = client.get("/simulation-capabilities", headers=_headers(AGENCY))
    assert response.status_code == 200
    assert response.json() == {"auto_simulation": False}


def test_capability_response_leaks_nothing(client):
    """No allowlist, no environment variables, no user ids, no reason."""
    for user in (OWNER, AGENCY):
        body = client.get("/simulation-capabilities", headers=_headers(user)).text
        assert set(client.get("/simulation-capabilities", headers=_headers(user)).json()) == {
            "auto_simulation"
        }
        assert OWNER not in body
        assert AGENCY not in body
        assert "AUTO_SIMULATION" not in body
        assert "allow" not in body.lower()


def test_unauthenticated_caller_has_no_capability(client):
    response = client.get(
        "/simulation-capabilities", headers={"X-API-Key": "test-dashboard-secret"}
    )
    # The dashboard secret alone is not an identity; in production the user
    # token is required, so this is a 401 before the route is reached.
    assert response.status_code in (200, 401)
    if response.status_code == 200:
        assert response.json() == {"auto_simulation": False}


# --- 16: tenancy ------------------------------------------------------------


def test_wrong_tenant_is_rejected(client):
    """OTHER_TENANT is allowlisted for nothing and assigned to creator-2."""
    assert _simulate(client, OTHER_TENANT, creator="creator-2", fan="fan-other").status_code == 404


def test_allowlisted_owner_cannot_cross_tenants(client, monkeypatch):
    """Being the product owner does not create a global admin bypass: the
    ordinary creator assignment still decides."""
    response = _simulate(client, OWNER, creator="creator-2", fan="fan-other")
    assert response.status_code == 404
    assert response.json() == {"detail": "Resource not found"}


def test_fan_belonging_to_another_creator_is_rejected(client):
    """creator-1 is the caller's own creator, but fan-other is not its fan."""
    assert _simulate(client, OWNER, creator="creator-1", fan="fan-other").status_code == 404


# --- 17, 18: the test_ fan boundary -----------------------------------------


def test_real_fan_is_rejected_even_for_the_allowlisted_owner(client):
    """Knowing a real fan's UUID must never be enough."""
    response = _simulate(client, OWNER, fan="fan-real")
    assert response.status_code == 404
    assert response.json() == {"detail": "Resource not found"}


def test_test_fan_is_accepted(client, monkeypatch):
    captured: dict = {}

    async def fake_run(**kwargs):
        captured.update(kwargs)
        return {"status": "ok", "simulation": True, "fan_message_id": "m1", "creator_messages": []}

    monkeypatch.setattr("services.suggestions.run_simulated_inbound", fake_run)
    response = _simulate(client, OWNER)
    assert response.status_code == 200, response.text
    assert response.json()["simulation"] is True
    assert captured["fan_id"] == "fan-test"
    assert captured["creator_id"] == "creator-1"
    assert captured["message"] == "hiii"
    assert captured["fast"] is True


def test_every_rejection_is_indistinguishable(client, monkeypatch):
    """A caller must not be able to tell 'not allowlisted' from 'wrong tenant'
    from 'not a test fan' from 'does not exist'."""
    bodies = set()
    bodies.add(_simulate(client, AGENCY).text)
    bodies.add(_simulate(client, OWNER, fan="fan-real").text)
    bodies.add(_simulate(client, OWNER, fan="fan-other").text)
    bodies.add(_simulate(client, OWNER, creator="creator-2", fan="fan-other").text)
    bodies.add(_simulate(client, OWNER, fan="fan-does-not-exist").text)
    assert bodies == {'{"detail":"Resource not found"}'}


# --- the boundary is re-checked on every mutation ---------------------------


@pytest.mark.parametrize(
    "path,payload",
    [
        ("simulate-inbound", {"message": "hi"}),
        ("simulate-purchase", None),
        ("simulate-decline", {}),
    ],
)
def test_test_fan_boundary_applies_to_every_simulation_mutation(client, path, payload):
    response = client.post(
        f"/creator/creator-1/fan/fan-real/{path}",
        headers=_headers(OWNER),
        json=payload if payload is not None else {},
    )
    assert response.status_code == 404


@pytest.mark.parametrize(
    "path,payload",
    [
        ("simulate-inbound", {"message": "hi"}),
        ("simulate-purchase", None),
        ("simulate-decline", {}),
    ],
)
def test_non_allowlisted_user_is_rejected_on_every_mutation(client, path, payload):
    response = client.post(
        f"/creator/creator-1/fan/fan-test/{path}",
        headers=_headers(AGENCY),
        json=payload if payload is not None else {},
    )
    assert response.status_code == 404


# --- creator listing, and the production columns it may read ----------------
#
# db/ci_baseline_schema.sql is a TEST FIXTURE, not the authoritative production
# schema. It carries ``creators.name``; the production table does not. Selecting
# it returned PostgREST 42703 and made GET /simulation/creators 404 in
# production while /simulation-capabilities answered 200. The fakes below model
# production, not the fixture, so the same mistake fails here first.


# Exactly what a production creators row exposes to this route. ``name`` is
# deliberately absent: the display field is platform_username, as /my-creators
# has always used.
PRODUCTION_CREATOR_COLUMNS = {"id", "platform_username"}


class PostgrestUndefinedColumn(Exception):
    """What PostgREST actually raises for a column that does not exist."""

    def __init__(self, table: str, column: str) -> None:
        super().__init__(
            {"message": f"column {table}.{column} does not exist", "code": "42703"}
        )


class _Query:
    """One query against one table, so concurrent reads cannot share state."""

    def __init__(self, listing: "_Listing", name: str) -> None:
        self.listing = listing
        self.name = name
        self.columns: list[str] = []

    def select(self, columns="*", **_k):
        self.columns = [c.strip() for c in str(columns).split(",") if c.strip()]
        if self.name == "creators":
            self.listing.creator_columns.update(self.columns)
            for column in self.columns:
                if column not in PRODUCTION_CREATOR_COLUMNS:
                    raise PostgrestUndefinedColumn("creators", column)
        return self

    def in_(self, *_a, **_k):
        return self

    def like(self, *_a, **_k):
        self.listing.filtered = True
        return self

    def order(self, *_a, **_k):
        return self

    def limit(self, *_a, **_k):
        return self

    def execute(self):
        if self.name == "creators":
            return SimpleNamespace(
                data=[{"id": "creator-1", "platform_username": "Sophia"}]
            )
        return SimpleNamespace(
            data=[
                FANS["fan-test"],
                # A real fan smuggled past the database filter must still be
                # dropped by the Python-side boundary.
                FANS["fan-real"],
            ]
        )


class _Listing:
    """A creators/fans database that only has production's columns."""

    def __init__(self) -> None:
        self.creator_columns: set[str] = set()
        self.filtered = False

    def table(self, name):
        # A fresh query per call: the route reads creators and fans
        # concurrently, so a shared mutable builder would let one read's table
        # name clobber the other's.
        return _Query(self, name)


def test_creator_listing_is_owner_only_and_test_fans_only(client, monkeypatch):
    assert client.get("/simulation/creators", headers=_headers(AGENCY)).status_code == 404

    listing = _Listing()
    monkeypatch.setattr(main, "get_supabase", lambda: listing)
    response = client.get("/simulation/creators", headers=_headers(OWNER))
    assert response.status_code == 200, response.text
    body = response.json()
    # The API contract is unchanged: the dashboard still receives "name",
    # regardless of which column it is read from.
    assert body["creators"][0]["name"] == "Sophia"
    assert [fan["id"] for fan in body["creators"][0]["test_fans"]] == ["fan-test"]


def test_creator_listing_never_queries_the_fixture_only_name_column(client, monkeypatch):
    """Regression guard for the 42703 production outage.

    The fake raises the real PostgREST error for any creators column that does
    not exist in production, so reintroducing ``creators.name`` fails here
    instead of on Railway.
    """
    listing = _Listing()
    monkeypatch.setattr(main, "get_supabase", lambda: listing)
    response = client.get("/simulation/creators", headers=_headers(OWNER))

    assert response.status_code == 200, response.text
    assert "name" not in listing.creator_columns
    assert listing.creator_columns == {"id", "platform_username"}


def test_creator_listing_falls_back_to_the_id_when_username_is_null(client, monkeypatch):
    """platform_username is nullable, so the picker must still have a label."""

    class _Nameless(_Listing):
        def table(self, name):
            query = _Query(self, name)
            if name == "creators":
                query.execute = lambda: SimpleNamespace(
                    data=[{"id": "creator-1", "platform_username": None}]
                )
            return query

    listing = _Nameless()
    monkeypatch.setattr(main, "get_supabase", lambda: listing)
    body = client.get("/simulation/creators", headers=_headers(OWNER)).json()
    assert body["creators"][0]["name"] == "creator-1"


def test_creator_listing_retries_a_transient_read_instead_of_returning_empty(
    client, monkeypatch
):
    """A PostgREST connection recycle must not read as "you have no creators"."""
    from httpx import ConnectError

    attempts = {"creators": 0}

    class _Flaky(_Listing):
        def table(self, name):
            query = _Query(self, name)
            if name == "creators":
                original = query.execute

                def flaky():
                    attempts["creators"] += 1
                    if attempts["creators"] == 1:
                        raise ConnectError("server disconnected")
                    return original()

                query.execute = flaky
            return query

    listing = _Flaky()
    monkeypatch.setattr(main, "get_supabase", lambda: listing)
    response = client.get("/simulation/creators", headers=_headers(OWNER))

    assert response.status_code == 200, response.text
    assert attempts["creators"] >= 2, "the read must be retried, not abandoned"
    assert response.json()["creators"][0]["name"] == "Sophia"


def test_creator_listing_reports_a_failed_read_rather_than_an_empty_list(
    client, monkeypatch
):
    """When retries are exhausted, say the read failed.

    Authorization has already passed at this point, so a distinguishable status
    tells a verified owner nothing they may not know — and a 404 here would be
    a claim about the DATA (no creators) rather than a report of a failed read.
    """
    from httpx import ConnectError

    class _Broken(_Listing):
        def table(self, name):
            query = _Query(self, name)
            if name == "creators":
                query.execute = lambda: (_ for _ in ()).throw(
                    ConnectError("server disconnected")
                )
            return query

    listing = _Broken()
    monkeypatch.setattr(main, "get_supabase", lambda: listing)
    response = client.get("/simulation/creators", headers=_headers(OWNER))

    assert response.status_code == 503
    assert response.json() != {"creators": []}


# --- 30: the simulator is not blocked by the connector switch ---------------


def test_disabled_connector_does_not_block_the_authorized_simulator(
    client, monkeypatch
):
    monkeypatch.setenv("APIFANSLY_ENABLED", "false")
    called: list[dict] = []

    async def fake_run(**kwargs):
        called.append(kwargs)
        return {"status": "ok", "simulation": True, "fan_message_id": "m1", "creator_messages": []}

    monkeypatch.setattr("services.suggestions.run_simulated_inbound", fake_run)
    response = _simulate(client, OWNER)
    assert response.status_code == 200, response.text
    assert len(called) == 1


def test_no_user_id_is_hardcoded_in_source():
    """The allowlist is configuration, never source."""
    import inspect
    import re

    for module in (simulation, main):
        source = inspect.getsource(module)
        assert not re.search(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
            source,
            re.IGNORECASE,
        ), f"{module.__name__} contains a literal UUID"


# --- the owner-only simulation catalog mirror -------------------------------
#
# The mirror copies one creator's vault metadata into another's TEST catalog. It
# is an owner capability over creators the caller already holds: the allowlist
# decides WHO, ordinary tenancy decides WHICH. An agency account must not be
# able to invoke it, and must not be able to learn that it exists.


def _mirror(client, user, *, source="creator-1", target="creator-1", delete=False):
    path = "/simulation/catalog/mirror/delete" if delete else "/simulation/catalog/mirror"
    return client.post(
        path,
        headers=_headers(user),
        json={"source_creator_id": source, "target_creator_id": target},
    )


@pytest.fixture
def mirror_spy(monkeypatch):
    """Record every mirror the routes would perform, and perform none of them."""
    calls: list[tuple] = []

    async def fake_mirror(*, source_creator_id, target_creator_id):
        calls.append(("mirror", source_creator_id, target_creator_id))
        from services.simulation_catalog import MirrorResult

        return MirrorResult(
            source_creator_id=source_creator_id,
            target_creator_id=target_creator_id,
            media_mirrored=2,
            sets_mirrored=2,
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
            media_removed=2,
            sets_removed=2,
        )

    monkeypatch.setattr(
        "services.simulation_catalog.mirror_creator_catalog", fake_mirror
    )
    monkeypatch.setattr(
        "services.simulation_catalog.delete_creator_catalog_mirror", fake_delete
    )
    return calls


def test_an_agency_account_cannot_invoke_the_mirror(client, mirror_spy):
    """Same 404 as every other simulator rejection: an agency must not learn
    that this capability exists at all."""
    response = _mirror(client, AGENCY, source="creator-1", target="creator-1")

    assert response.status_code == 404
    assert response.json() == {"detail": "Resource not found"}
    assert mirror_spy == [], "no mirror may be performed for a rejected caller"


def test_an_agency_account_cannot_delete_a_mirror(client, mirror_spy):
    assert _mirror(client, AGENCY, delete=True).status_code == 404
    assert mirror_spy == []


def test_the_mirror_is_off_when_the_feature_switch_is_off(client, mirror_spy, monkeypatch):
    monkeypatch.setenv("AUTO_SIMULATION_ENABLED", "false")
    assert _mirror(client, OWNER).status_code == 404
    assert mirror_spy == []


def test_the_owner_cannot_mirror_a_creator_they_are_not_assigned(client, mirror_spy):
    """Owner-only does not mean owner-of-everything: ordinary tenancy still
    decides which creators are in reach."""
    assert _mirror(client, OWNER, source="creator-2").status_code == 404
    assert _mirror(client, OWNER, target="creator-2").status_code == 404
    assert mirror_spy == []


def test_an_unauthenticated_caller_cannot_invoke_the_mirror(client, mirror_spy):
    response = client.post(
        "/simulation/catalog/mirror",
        headers={"X-API-Key": "test-dashboard-secret"},
        json={"source_creator_id": "creator-1", "target_creator_id": "creator-1"},
    )
    assert response.status_code in (401, 404)
    assert mirror_spy == []


def test_the_owner_may_mirror_between_creators_they_hold(client, mirror_spy):
    response = _mirror(client, OWNER, source="creator-1", target="creator-1")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["simulation_only"] is True
    assert mirror_spy == [("mirror", "creator-1", "creator-1")]


def test_the_owner_may_delete_a_mirror_they_created(client, mirror_spy):
    response = _mirror(client, OWNER, delete=True)

    assert response.status_code == 200
    assert response.json()["sets_removed"] == 2
    assert mirror_spy == [("delete", "creator-1", "creator-1")]
