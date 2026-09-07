"""SEC-005 — the /test/* helpers must be unavailable in production and safe in dev.

Two independent defects were fixed together:

1. Neither route was environment-gated, so both were reachable in production
   with an ordinary operator session.
2. ``simulate_ppv_purchase`` read ``sales_log`` from a row that never selected
   it, so appending one entry replaced the fan's entire sales history, and
   ``test_inject_message`` hardcoded ``auto_mode=True``, so the helper could
   trigger a real Full Auto send for a creator whose Auto was off.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import main
from main import app


@pytest.fixture
def client():
    return TestClient(app)


# --- production availability ------------------------------------------------


ROUTES = [
    ("/test/simulate-ppv-purchase", {"fan_id": "fan-1"}),
    (
        "/test/inject-message",
        {"fan_id": "fan-1", "creator_id": "creator-1", "content": "hi"},
    ),
]


@pytest.fixture
def authorized_operator(monkeypatch):
    """Get past the API middleware and the tenancy checks.

    These routes were always tenancy-authorized — the SEC-005 claim is that an
    operator who legitimately owns the fan could still call them in production.
    Passing tenancy is therefore essential: a 404 from an unowned resource would
    prove nothing, since _forbidden() is also a 404.
    """
    monkeypatch.setenv("DASHBOARD_API_SECRET", "secret")

    async def _user(_authorization):
        return "operator-1"

    async def _creator_ids(_user_id):
        return {"creator-1"}

    async def _fan_creator(_fan_id):
        return "creator-1"

    monkeypatch.setattr(main, "authenticated_dashboard_user", _user)
    monkeypatch.setattr("core.tenancy._creator_ids_for_user", _creator_ids)
    monkeypatch.setattr("core.tenancy._fan_creator_id", _fan_creator)
    return {"X-API-Key": "secret", "Authorization": "Bearer token"}


@pytest.mark.parametrize("path, params", ROUTES)
@pytest.mark.parametrize("app_env", ["production", "prod", "staging", None])
def test_test_endpoints_are_404_outside_development(
    monkeypatch, client, authorized_operator, path, params, app_env
):
    """Production — and any unrecognised APP_ENV — must not expose these."""
    if app_env is None:
        monkeypatch.delenv("APP_ENV", raising=False)
    else:
        monkeypatch.setenv("APP_ENV", app_env)

    response = client.post(path, params=params, headers=authorized_operator)

    assert response.status_code == 404
    assert response.json() == {"detail": "Not Found"}


@pytest.mark.parametrize("path, params", ROUTES)
@pytest.mark.parametrize("app_env", ["development", "test"])
def test_test_endpoints_are_reachable_in_development(
    monkeypatch, client, authorized_operator, path, params, app_env
):
    """The counterpart: the same authorized call is NOT a 404 in dev/test.

    Without this, the 404 above could be produced by the tenancy check rather
    than the environment guard, and the test would prove nothing.
    """
    monkeypatch.setenv("APP_ENV", app_env)
    monkeypatch.setattr(main, "get_supabase", lambda: _creator_db(False))

    async def _save_message(*_args, **_kwargs):
        return "message-1"

    async def _process(*_args, **_kwargs):
        return None

    monkeypatch.setattr("db.queries.save_message", _save_message)
    monkeypatch.setattr(main, "process_incoming_fan_message", _process)

    response = client.post(path, params=params, headers=authorized_operator)

    assert response.status_code != 404


@pytest.mark.parametrize("app_env", ["development", "test"])
def test_route_guard_allows_development_and_test(monkeypatch, app_env):
    import asyncio

    from main import require_local_test_endpoints

    monkeypatch.setenv("APP_ENV", app_env)
    assert asyncio.run(require_local_test_endpoints()) is None


# --- simulate_ppv_purchase preserves sales history -------------------------


class _FakeTable:
    def __init__(self, store, name):
        self._store = store
        self._name = name
        self._payload = None

    def select(self, columns):
        self._store["selected_columns"] = columns
        return self

    def update(self, payload):
        self._payload = payload
        return self

    def eq(self, *_args):
        return self

    def single(self):
        return self

    def execute(self):
        if self._payload is not None:
            self._store["updates"].append(self._payload)
            return SimpleNamespace(data=[self._payload])
        selected = {
            column.strip(): self._store["fan_row"].get(column.strip())
            for column in self._store["selected_columns"].split(",")
        }
        return SimpleNamespace(data=selected)


class _FakeDB:
    def __init__(self, store):
        self._store = store

    def table(self, name):
        return _FakeTable(self._store, name)


@pytest.fixture
def purchase_store(monkeypatch):
    store = {
        "fan_row": {
            "pending_ppv_check": {"media_id": "media-9", "price": 25},
            "total_spent": 100,
            "active_session": None,
            "ai_summary": {},
            "sales_log": [
                {"date": "01.01.2026", "item": "PPV media 1", "amount": 40,
                 "chatter": "AI"},
                {"date": "02.01.2026", "item": "PPV media 2", "amount": 60,
                 "chatter": "AI"},
            ],
        },
        "updates": [],
        "selected_columns": "",
    }
    monkeypatch.setattr(main, "get_supabase", lambda: _FakeDB(store))

    async def _no_session(_fan_id):
        return None

    monkeypatch.setattr("db.queries.get_fan_session", _no_session)
    return store


def test_simulate_purchase_selects_sales_log(monkeypatch, purchase_store):
    """The column must actually be read, or the append below is a wipe."""
    import asyncio

    monkeypatch.setenv("APP_ENV", "development")
    asyncio.run(main.simulate_ppv_purchase("fan-1", request=None))

    assert "sales_log" in purchase_store["selected_columns"]


def test_simulate_purchase_appends_rather_than_replaces(monkeypatch, purchase_store):
    import asyncio

    monkeypatch.setenv("APP_ENV", "development")
    result = asyncio.run(main.simulate_ppv_purchase("fan-1", request=None))

    assert result["status"] == "ok"
    written = purchase_store["updates"][0]
    assert len(written["sales_log"]) == 3, "historical sales_log entries were lost"
    assert written["sales_log"][0]["item"] == "PPV media 1"
    assert written["sales_log"][1]["item"] == "PPV media 2"
    assert written["sales_log"][2]["item"] == "PPV media media-9"
    assert written["total_spent"] == 125


def test_simulate_purchase_refuses_a_non_list_sales_log(monkeypatch, purchase_store):
    import asyncio

    monkeypatch.setenv("APP_ENV", "development")
    purchase_store["fan_row"]["sales_log"] = {"corrupt": True}

    result = asyncio.run(main.simulate_ppv_purchase("fan-1", request=None))

    assert result["status"] == "error"
    assert purchase_store["updates"] == [], "refusal must not write anything"


# --- test_inject_message respects the creator's real Auto state -------------


@pytest.fixture
def inject_calls(monkeypatch):
    calls = {}

    async def _save_message(*_args, **_kwargs):
        return "message-1"

    async def _process(fan_id, creator_id, content, auto_mode, message_id):
        calls["auto_mode"] = auto_mode

    monkeypatch.setattr("db.queries.save_message", _save_message)
    monkeypatch.setattr(main, "process_incoming_fan_message", _process)
    return calls


def _creator_db(auto_mode):
    class _Table:
        def select(self, _columns):
            return self

        def eq(self, *_args):
            return self

        def single(self):
            return self

        def execute(self):
            return SimpleNamespace(data={"auto_mode": auto_mode})

    class _DB:
        def table(self, _name):
            return _Table()

    return _DB()


@pytest.mark.parametrize(
    "creator_auto, requested, expected",
    [
        # The old code passed True unconditionally; the first row is the bug.
        (False, None, False),
        (True, None, True),
        (False, True, False),
        (True, False, False),
        (True, True, True),
    ],
)
def test_inject_message_resolves_real_auto_state(
    monkeypatch, inject_calls, creator_auto, requested, expected
):
    import asyncio

    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setattr(main, "get_supabase", lambda: _creator_db(creator_auto))

    result = asyncio.run(
        main.test_inject_message(
            fan_id="fan-1",
            creator_id="creator-1",
            content="hello",
            auto_mode=requested,
        )
    )

    assert inject_calls["auto_mode"] is expected
    assert result["auto_mode"] is expected
    assert result["creator_auto_mode"] is creator_auto


def test_inject_message_fails_closed_when_creator_row_is_unreadable(
    monkeypatch, inject_calls
):
    """An unreadable creator row must not authorise an autonomous send."""
    import asyncio

    monkeypatch.setenv("APP_ENV", "development")

    class _ExplodingDB:
        def table(self, _name):
            raise RuntimeError("supabase down")

    monkeypatch.setattr(main, "get_supabase", lambda: _ExplodingDB())

    asyncio.run(
        main.test_inject_message(
            fan_id="fan-1",
            creator_id="creator-1",
            content="hello",
            auto_mode=True,
        )
    )

    assert inject_calls["auto_mode"] is False
