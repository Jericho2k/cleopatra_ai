"""APIFANSLY_ENABLED — the deployment-wide remote-connector switch.

The invariant these tests defend is narrow and absolute: with the switch off,
nothing in this process may send an HTTP request to the API Fansly provider, and
nothing may report success for work that did not happen. Everything else — the
stored data, Assisted generation, health — must be exactly as it was.

The transport is spied at the one place a request can physically leave
(``httpx.AsyncClient``), so a path nobody thought to check still counts.
"""
from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

import main
from core import tenancy
from core.apifansly_gate import (
    ApiFanslyDisabledError,
    REASON_DISABLED,
    REASON_SIMULATION,
    apifansly_enabled,
    simulation_scope,
)
from services import apifansly


ASSIGNMENTS = {"operator-1": {"creator-1"}}


class SpyTransport(httpx.AsyncBaseTransport):
    """Counts every request that reaches the network layer."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(200, json={"success": True, "response": []})


@pytest.fixture
def spy(monkeypatch):
    transport = SpyTransport()
    client = httpx.AsyncClient(transport=transport)
    apifansly.set_shared_client(client)
    yield transport
    apifansly.set_shared_client(None)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("DASHBOARD_API_SECRET", "test-dashboard-secret")

    async def fake_user(authorization):
        token = str(authorization or "").split(" ")[-1]
        return token if token in ASSIGNMENTS else None

    async def fake_creator_ids(user_id):
        return ASSIGNMENTS.get(user_id, set())

    monkeypatch.setattr(main, "authenticated_dashboard_user", fake_user, raising=False)
    monkeypatch.setattr("core.auth.authenticated_dashboard_user", fake_user)
    monkeypatch.setattr(tenancy, "_creator_ids_for_user", fake_creator_ids)
    return TestClient(app=main.app)


def _headers(operator: str = "operator-1") -> dict[str, str]:
    return {
        "X-API-Key": "test-dashboard-secret",
        "Authorization": f"Bearer {operator}",
    }


# --- 1 & 2: unset and true preserve current live behaviour -----------------


def test_unset_preserves_live_behaviour(monkeypatch):
    monkeypatch.delenv("APIFANSLY_ENABLED", raising=False)
    assert apifansly_enabled() is True
    # No exception: headers() is the universal chokepoint, so this proves the
    # switch is inert when unset.
    assert apifansly.headers()["x-api-key"]


@pytest.mark.parametrize("value", ["true", "TRUE", "1", "yes", "on", " true "])
def test_explicit_true_values_preserve_live_behaviour(monkeypatch, value):
    monkeypatch.setenv("APIFANSLY_ENABLED", value)
    assert apifansly_enabled() is True
    assert apifansly.headers()["x-api-key"]


@pytest.mark.parametrize("value", ["false", "FALSE", "0", "no", "off", "disabled"])
def test_recognised_false_values_disable_the_connector(monkeypatch, value):
    monkeypatch.setenv("APIFANSLY_ENABLED", value)
    assert apifansly_enabled() is False


def test_unrecognised_value_stays_enabled(monkeypatch):
    """Backward compatibility beats cleverness: a typo must not silently take
    a paying deployment offline. Only the documented false spellings disable."""
    monkeypatch.setenv("APIFANSLY_ENABLED", "maybe")
    assert apifansly_enabled() is True


# --- transport-level enforcement -------------------------------------------


def test_disabled_refuses_before_any_socket(monkeypatch, spy):
    monkeypatch.setenv("APIFANSLY_ENABLED", "false")
    import asyncio

    with pytest.raises(ApiFanslyDisabledError) as excinfo:
        asyncio.run(apifansly.request("GET", "x", operation="probe"))
    assert excinfo.value.reason == REASON_DISABLED
    assert spy.requests == []


def test_disabled_refuses_raw_call_sites_via_headers(monkeypatch):
    """The six raw httpx call sites in main.py and suggestions.py never go
    through request(); they all build auth headers here."""
    monkeypatch.setenv("APIFANSLY_ENABLED", "false")
    with pytest.raises(ApiFanslyDisabledError):
        apifansly.headers()
    with pytest.raises(ApiFanslyDisabledError):
        apifansly.headers(json_content=True)


def test_disabled_refuses_media_download(monkeypatch, spy):
    monkeypatch.setenv("APIFANSLY_ENABLED", "false")
    import asyncio

    with pytest.raises(ApiFanslyDisabledError):
        asyncio.run(apifansly.download_media("https://cdn3.fansly.com/a.jpg"))
    assert spy.requests == []


def test_simulation_scope_refuses_even_while_enabled(monkeypatch, spy):
    monkeypatch.setenv("APIFANSLY_ENABLED", "true")
    with simulation_scope():
        with pytest.raises(ApiFanslyDisabledError) as excinfo:
            apifansly.headers()
        assert excinfo.value.reason == REASON_SIMULATION
    # Scope is restored on exit.
    assert apifansly.headers()["x-api-key"]
    assert spy.requests == []


def test_simulation_scope_propagates_into_spawned_tasks(monkeypatch, spy):
    """contextvars are copied by create_task, which is what makes the guard hold
    for work the simulated turn spawns rather than awaits."""
    import asyncio

    monkeypatch.setenv("APIFANSLY_ENABLED", "true")

    async def scenario() -> str:
        async def child() -> str:
            try:
                apifansly.headers()
            except ApiFanslyDisabledError as exc:
                return exc.reason
            return "allowed"

        with simulation_scope():
            return await asyncio.create_task(child())

    assert asyncio.run(scenario()) == REASON_SIMULATION
    assert spy.requests == []


# --- 3, 4, 5: background paths make zero calls ------------------------------


def test_chat_reconciliation_makes_zero_calls_when_disabled(monkeypatch, spy):
    """The scheduler skips the tick before it reads a single creator row."""
    import asyncio

    monkeypatch.setenv("APIFANSLY_ENABLED", "false")
    monkeypatch.setenv("CHAT_RECONCILE_TICK_MINUTES", "1")

    sleeps: list[float] = []
    reconciled: list[object] = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) >= 2:
            raise asyncio.CancelledError

    async def should_not_run(*args, **kwargs):
        reconciled.append(args)
        return {}

    monkeypatch.setattr(main.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(main, "_reconcile_chat_creators_once", should_not_run)
    monkeypatch.setattr(main, "sync_chats", should_not_run)
    monkeypatch.setattr(
        main,
        "get_supabase",
        lambda: (_ for _ in ()).throw(AssertionError("must not read creators")),
    )

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(main.chat_reconciliation_scheduler())

    assert reconciled == []
    assert spy.requests == []


def test_vault_autosync_makes_zero_calls_when_disabled(monkeypatch, spy):
    import asyncio

    monkeypatch.setenv("APIFANSLY_ENABLED", "false")
    sleeps: list[float] = []
    started: list[str] = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) >= 2:
            raise asyncio.CancelledError

    async def should_not_run(creator_id, force=False):
        started.append(creator_id)
        return {}

    monkeypatch.setattr(main.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(main, "sync_vault_start", should_not_run)
    monkeypatch.setattr(
        main,
        "get_supabase",
        lambda: (_ for _ in ()).throw(AssertionError("must not read creators")),
    )

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(main.vault_autosync_scheduler())

    assert started == []
    assert spy.requests == []


def test_fansly_lists_sync_skips_cleanly_when_disabled(monkeypatch, spy):
    import asyncio

    monkeypatch.setenv("APIFANSLY_ENABLED", "false")
    monkeypatch.setenv("FANSLY_LISTS_SYNC_ENABLED", "true")

    async def should_not_run(*args, **kwargs):
        raise AssertionError("list sync must not run")

    monkeypatch.setattr(
        "services.fansly_lists.sync_fansly_lists_single_flight", should_not_run
    )
    result = asyncio.run(
        main._sync_fansly_lists_if_due("creator-1", "account-1", force=True)
    )
    assert result == {"status": "skipped", "reason": REASON_DISABLED}
    assert spy.requests == []


def test_remote_delivery_actions_are_postponed_not_failed(monkeypatch, spy):
    """A queued Auto reply for a real fan must not spend an analyzer and a
    writer producing copy with nowhere to go — and must not be failed either,
    because the connector comes back without a redeploy."""
    import asyncio

    from workers import scheduled_actions

    monkeypatch.setenv("APIFANSLY_ENABLED", "false")

    class _Fans:
        def table(self, _name):
            return self

        def select(self, *_a, **_k):
            return self

        def eq(self, *_a, **_k):
            return self

        def limit(self, *_a, **_k):
            return self

        def execute(self):
            return SimpleNamespace(data=[{"platform_fan_id": "fansly-999"}])

    monkeypatch.setattr(scheduled_actions, "get_supabase", _Fans, raising=False)
    monkeypatch.setattr("core.supabase.get_supabase", lambda: _Fans())

    blocked = asyncio.run(
        scheduled_actions._connector_blocks_delivery(
            {"fan_id": "fan-1", "action_type": "AUTO_REPLY"}
        )
    )
    assert blocked is True
    assert "AUTO_REPLY" in scheduled_actions.REMOTE_DELIVERY_ACTIONS
    # Local state work keeps running: neither does remote I/O.
    assert "PPV_RECONCILE" not in scheduled_actions.REMOTE_DELIVERY_ACTIONS
    assert "OFFER_EXPIRY" not in scheduled_actions.REMOTE_DELIVERY_ACTIONS
    assert spy.requests == []


def test_test_fans_are_exempt_from_the_worker_postponement(monkeypatch):
    """The switch is about the remote platform. A test_ fan never used it."""
    import asyncio

    from workers import scheduled_actions

    monkeypatch.setenv("APIFANSLY_ENABLED", "false")

    class _Fans:
        def table(self, _name):
            return self

        def select(self, *_a, **_k):
            return self

        def eq(self, *_a, **_k):
            return self

        def limit(self, *_a, **_k):
            return self

        def execute(self):
            return SimpleNamespace(data=[{"platform_fan_id": "test_jostar"}])

    monkeypatch.setattr("core.supabase.get_supabase", lambda: _Fans())
    assert (
        asyncio.run(
            scheduled_actions._connector_blocks_delivery({"fan_id": "fan-test"})
        )
        is False
    )


def test_ppv_reconciliation_defers_instead_of_freezing(monkeypatch, spy):
    """A deliberate configuration choice must never escalate into freezing a
    fan for human review through the verification-failure counter."""
    import asyncio
    from datetime import datetime, timezone

    from models.commercial import CreatorPolicy
    from services import ppv_reconciliation as reconciliation
    from services.ppv_reconciliation import PPVReconcileDisposition

    monkeypatch.setenv("APIFANSLY_ENABLED", "false")
    now = datetime(2026, 7, 18, 12, 0, tzinfo=timezone.utc)
    pending = {
        "reference": "ref-1",
        "media_id": "m1",
        "media_ids": ["m1"],
        "price": 25.0,
        "sent_at": "2026-07-18T11:00:00+00:00",
        "expires_at": "2026-07-19T11:00:00+00:00",
    }

    async def _value(value):
        return value

    monkeypatch.setattr(
        reconciliation, "get_creator_policy", lambda _c: _value(CreatorPolicy())
    )
    monkeypatch.setattr(
        reconciliation,
        "_load_platform_context",
        lambda _c, _f: _value(
            ({"apifansly_account_id": "acct"}, {"platform_fan_id": "fansly-1", "pending_ppv_check": pending})
        ),
    )
    monkeypatch.setattr(
        reconciliation,
        "_fetch_purchase_amount",
        lambda **_k: (_ for _ in ()).throw(AssertionError("must not call provider")),
    )
    monkeypatch.setattr(
        reconciliation,
        "_persist_pending_check",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("a disabled connector is not a verification failure")
        ),
    )

    result = asyncio.run(
        reconciliation.reconcile_pending_ppv(
            creator_id="creator-1", fan_id="fan-1", now=now
        )
    )
    assert result.disposition == PPVReconcileDisposition.PENDING
    assert result.retry_at is not None
    assert spy.requests == []


# --- 6: live delivery returns a safe connector-disabled result --------------


@pytest.mark.parametrize(
    "method,path",
    [
        ("post", "/reply"),
        ("post", "/sync-chats/creator-1"),
        ("post", "/mark-all-read/creator-1"),
        ("post", "/sync-vault/creator-1"),
        ("post", "/load-history/creator-1/fan-1"),
        ("post", "/creator/creator-1/sync-fansly-lists"),
        ("post", "/fan/fan-1/operator-ppv"),
    ],
)
def test_remote_routes_return_structured_503(monkeypatch, client, spy, method, path):
    monkeypatch.setenv("APIFANSLY_ENABLED", "false")

    # Tenancy would otherwise need real rows; the guard must fire regardless of
    # the order dependencies resolve in, so allow tenancy to pass.
    async def allow(*_args, **_kwargs):
        return None

    async def owning_creator(_fan_id):
        return "creator-1"

    monkeypatch.setattr(main, "require_creator_path_access", allow, raising=False)
    monkeypatch.setattr("core.tenancy.require_creator_access", allow)
    monkeypatch.setattr("core.tenancy._fan_creator_id", owning_creator)

    response = getattr(client, method)(path, headers=_headers(), json={})
    assert response.status_code in (503, 422), response.text
    if response.status_code == 503:
        body = response.json()
        assert body["reason"] == REASON_DISABLED
        assert body["status"] == "connector_disabled"
        # detail stays a human-readable string: the dashboard renders it.
        assert isinstance(body["detail"], str)
    assert spy.requests == []


def test_reply_never_persists_a_message_it_did_not_send(monkeypatch, client, spy):
    """Never report delivery success when nothing was sent."""
    monkeypatch.setenv("APIFANSLY_ENABLED", "false")
    saved: list[tuple] = []

    async def allow(*_args, **_kwargs):
        return None

    async def record(*args, **kwargs):
        saved.append(args)
        return "message-1"

    async def owning_creator(_fan_id):
        return "creator-1"

    monkeypatch.setattr("core.tenancy.require_creator_access", allow)
    monkeypatch.setattr("core.tenancy._fan_creator_id", owning_creator)
    monkeypatch.setattr(main, "save_message", record, raising=False)

    response = client.post(
        "/reply",
        headers=_headers(),
        json={
            "fan_id": "fan-1",
            "creator_id": "creator-1",
            "content": "hi",
            "was_ai_suggested": False,
        },
    )
    assert response.status_code == 503
    assert saved == []
    assert spy.requests == []


# --- 7: passive dashboard refresh no-ops cleanly ---------------------------


def test_sync_fan_messages_no_ops_cleanly(monkeypatch, spy):
    import asyncio

    monkeypatch.setenv("APIFANSLY_ENABLED", "false")
    monkeypatch.setattr(
        main,
        "get_supabase",
        lambda: (_ for _ in ()).throw(AssertionError("must not read bindings")),
    )
    result = asyncio.run(main.sync_recent_fan_messages("creator-1", "fan-1"))
    assert result["status"] == "skipped"
    assert result["reason"] == REASON_DISABLED
    # The dashboard computes changed = imported + media_updated. Both keys must
    # exist with the same meaning as on the success path, or app/page.tsx reads
    # NaN and the caller's retry maths breaks.
    assert result["imported"] == 0
    assert result["media_updated"] == 0
    assert result["inbound"] == 0
    assert result["identity_reconciled"] == 0
    assert spy.requests == []


# --- 8: health is not generically unhealthy --------------------------------


def test_health_reports_disabled_connector_without_degrading(monkeypatch):
    import asyncio

    from services import operational_health

    monkeypatch.setenv("APIFANSLY_ENABLED", "false")
    operational_health.reset_cache()

    async def probe_db():
        return {"reachable": True, "latency_ms": 3}

    async def probe_queue():
        return {"available": True, "pending": 0, "oldest_pending_age_seconds": 0}

    monkeypatch.setattr(operational_health, "probe_database", probe_db)
    monkeypatch.setattr(operational_health, "probe_queue", probe_queue)
    monkeypatch.setattr(
        operational_health,
        "worker_health_snapshot",
        lambda: {"poll_seconds": 5, "cycles_completed": 10, "seconds_since_last_cycle": 1},
    )
    document = asyncio.run(operational_health.collect(use_cache=False))
    operational_health.reset_cache()

    assert document["integrations"]["apifansly"] == "disabled"
    assert document["status"] != "unhealthy"
    assert document["fatal_reasons"] == []
    joined = " ".join(document["degraded_reasons"])
    assert "apifansly" not in joined
    assert "connector" not in joined


def test_integration_health_reports_disabled_not_access_denied(monkeypatch, spy):
    import asyncio

    monkeypatch.setenv("APIFANSLY_ENABLED", "false")

    class _Creators:
        def table(self, _name):
            return self

        def select(self, *_a, **_k):
            return self

        def eq(self, *_a, **_k):
            return self

        def single(self):
            return self

        def execute(self):
            return SimpleNamespace(
                data={
                    "platform": "fansly",
                    "fansly_account_id": "plat-1",
                    "apifansly_account_id": "acct-1",
                }
            )

    monkeypatch.setattr(main, "get_supabase", _Creators)
    document = asyncio.run(main.read_fansly_integration_health("creator-1"))
    assert document["status"] == "connector_disabled"
    assert document["requires_reconnect"] is False
    assert spy.requests == []


# --- 4 (availability): Full Auto must not claim it is usable ---------------


def test_auto_availability_reports_connector_disabled(monkeypatch, spy):
    import asyncio

    monkeypatch.setenv("APIFANSLY_ENABLED", "false")

    class _Sets:
        def table(self, _name):
            return self

        def select(self, *_a, **_k):
            return self

        def eq(self, *_a, **_k):
            return self

        def limit(self, *_a, **_k):
            return self

        def execute(self):
            return SimpleNamespace(data=[], count=7)

    monkeypatch.setattr(main, "get_supabase", _Sets)
    result = asyncio.run(main._creator_auto_availability("creator-1"))
    assert result["auto_available"] is False
    assert result["reason"] == "connector_disabled"
    # Approved sets are still reported: the data is not hidden, only Auto is.
    assert result["approved_sets"] == 7
    assert spy.requests == []


def test_auto_availability_unchanged_when_connector_enabled(monkeypatch):
    import asyncio

    monkeypatch.setenv("APIFANSLY_ENABLED", "true")

    class _Sets:
        def table(self, _name):
            return self

        def select(self, *_a, **_k):
            return self

        def eq(self, *_a, **_k):
            return self

        def limit(self, *_a, **_k):
            return self

        def execute(self):
            return SimpleNamespace(data=[], count=2)

    monkeypatch.setattr(main, "get_supabase", _Sets)
    assert asyncio.run(main._creator_auto_availability("creator-1")) == {
        "auto_available": True,
        "approved_sets": 2,
    }


# --- 9: Assisted still works ------------------------------------------------


def test_assisted_generation_is_untouched_by_the_switch(monkeypatch, spy):
    """Assisted never had a delivery leg — a human sends the copy — so the
    switch must not appear anywhere in its path."""
    import inspect

    from services.suggestions import get_suggestions

    monkeypatch.setenv("APIFANSLY_ENABLED", "false")
    source = inspect.getsource(get_suggestions)
    assert "apifansly" not in source.lower()
    assert "APIFANSLY_ENABLED" not in source
    assert spy.requests == []
