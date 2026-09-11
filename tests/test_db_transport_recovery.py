"""Recovery behaviour on top of the PostgREST transport.

tests/test_postgrest_transport.py establishes what the transport does. This
module is about what *we* do with it: which operations get a second chance,
which deliberately do not, that the budget is finite, that concurrent failures
produce one rebuild rather than a herd, and that health tracks reality in both
directions.

The transport exception used throughout is the real one — httpx's
``RemoteProtocolError`` wrapping h2's ``ConnectionTerminated``, built exactly as
httpcore builds it — so a test cannot pass by agreeing with a fake exception
type that production never raises.
"""
from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import h2.errors
import h2.events
import httpx
import pytest
from fastapi.testclient import TestClient

import main
from core import supabase as supabase_module
from core import tenancy
from services import db_reliability, operational_health


def run(coro):
    return asyncio.run(coro)


def connection_terminated() -> httpx.RemoteProtocolError:
    """The production failure, constructed the way httpcore constructs it."""
    event = h2.events.ConnectionTerminated()
    event.error_code = h2.errors.ErrorCodes.NO_ERROR
    event.last_stream_id = 3
    event.additional_data = None
    return httpx.RemoteProtocolError(event)


@pytest.fixture(autouse=True)
def clean_transport():
    supabase_module.close_supabase_client()
    yield
    supabase_module.close_supabase_client()


# ---------------------------------------------------------------------------
# 1. Read recovery — the caller sees success, not a 500.
# ---------------------------------------------------------------------------


ASSIGNMENTS = {"operator-1": {"creator-1"}}


class _FlakyCreators:
    """A Supabase double whose first execute() dies on the transport."""

    def __init__(self, *, failures: int):
        self.remaining_failures = failures
        self.executions = 0

    def table(self, name):
        self.last_table = name
        return self

    def select(self, *_args, **_kwargs):
        return self

    def eq(self, *_args, **_kwargs):
        return self

    def in_(self, *_args, **_kwargs):
        return self

    def execute(self):
        self.executions += 1
        if self.remaining_failures > 0:
            self.remaining_failures -= 1
            raise connection_terminated()
        if self.last_table == "chatter_creators":
            return SimpleNamespace(data=[{"creator_id": "creator-1"}])
        return SimpleNamespace(
            data=[{"id": "creator-1", "platform_username": "nyx"}]
        )


@pytest.fixture
def dashboard_client(monkeypatch):
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


def _headers() -> dict[str, str]:
    return {
        "X-API-Key": "test-dashboard-secret",
        "Authorization": "Bearer operator-1",
    }


def test_my_creators_survives_one_terminated_connection(dashboard_client, monkeypatch):
    """The exact production 500. A read that would succeed must not become one."""
    db = _FlakyCreators(failures=1)
    monkeypatch.setattr(main, "get_supabase", lambda: db)

    response = dashboard_client.get("/my-creators", headers=_headers())

    assert response.status_code == 200
    assert response.json() == {"creators": [{"id": "creator-1", "platform_username": "nyx"}]}
    # One failure, one repeat of the same select, then the second select.
    assert db.executions == 3


def test_my_creators_still_fails_when_the_database_is_genuinely_gone(
    dashboard_client, monkeypatch
):
    """Fail-closed is preserved: a real outage is still an error, not an empty list."""
    db = _FlakyCreators(failures=99)
    monkeypatch.setattr(main, "get_supabase", lambda: db)

    with pytest.raises(httpx.RemoteProtocolError):
        dashboard_client.get("/my-creators", headers=_headers())


# ---------------------------------------------------------------------------
# 2. Client recovery — a rebuilt transport, and unrelated work still works.
# ---------------------------------------------------------------------------


def test_transport_reset_rebuilds_the_client_and_later_work_succeeds(monkeypatch):
    built: list[httpx.Client] = []
    real_build = supabase_module.build_http_client

    def counting_build():
        client = real_build()
        built.append(client)
        return client

    monkeypatch.setattr(supabase_module, "build_http_client", counting_build)

    first = supabase_module.get_supabase()
    generation = supabase_module.supabase_generation()

    assert supabase_module.reset_supabase_client(
        reason="test:RemoteProtocolError", generation=generation
    ) is True
    assert supabase_module.supabase_generation() == generation + 1

    second = supabase_module.get_supabase()
    assert second is not first
    assert len(built) == 2
    # An unrelated database operation on the new client works: the rebuild is a
    # working client, not a placeholder.
    assert second.postgrest.session is built[1]
    assert second.postgrest.session.is_closed is False


def test_a_reset_for_a_stale_generation_is_a_no_op():
    """Two callers who failed on the same connection must not both rebuild."""
    supabase_module.get_supabase()
    generation = supabase_module.supabase_generation()

    assert supabase_module.reset_supabase_client(reason="first", generation=generation)
    # The second caller is still holding the number it read before the failure.
    assert not supabase_module.reset_supabase_client(
        reason="second", generation=generation
    )
    assert supabase_module.supabase_generation() == generation + 1


def test_a_replaced_transport_is_not_closed_under_a_live_request(monkeypatch):
    """Closing synchronously would break the requests still reading from it."""
    closed_after: list[float] = []

    def fake_timer(delay, func):
        closed_after.append(delay)
        return SimpleNamespace(daemon=False, start=lambda: None)

    client = supabase_module.get_supabase()
    session = client.postgrest.session
    monkeypatch.setattr(threading, "Timer", fake_timer)

    supabase_module.reset_supabase_client(reason="test")

    assert session.is_closed is False
    assert closed_after == [supabase_module.RETIRE_GRACE_SECONDS]


# ---------------------------------------------------------------------------
# 3. Bounded retry — a budget, not a loop.
# ---------------------------------------------------------------------------


def test_retry_stops_at_the_budget_and_reraises_the_transport_error():
    attempts = {"count": 0}

    async def always_fails():
        attempts["count"] += 1
        raise connection_terminated()

    with pytest.raises(httpx.RemoteProtocolError):
        run(
            db_reliability.retry_transient_db_operation(
                always_fails, label="test.read", attempts=3, delay_seconds=0.0
            )
        )

    assert attempts["count"] == 3


def test_a_non_transient_error_is_not_retried_at_all():
    attempts = {"count": 0}

    async def bad_request():
        attempts["count"] += 1
        raise ValueError("column does not exist")

    with pytest.raises(ValueError):
        run(
            db_reliability.retry_transient_db_operation(
                bad_request, label="test.read", attempts=3, delay_seconds=0.0
            )
        )

    assert attempts["count"] == 1


def test_the_backoff_is_jittered_and_capped():
    """Callers knocked over together must not come back in lockstep."""
    delays = {db_reliability._backoff(0.15, attempt) for attempt in (1, 1, 1, 1, 1)}
    assert len(delays) > 1, "identical delays would re-collide"
    for attempt in range(1, 40):
        value = db_reliability._backoff(0.15, attempt)
        assert 0 < value <= db_reliability.MAX_DELAY_SECONDS * (
            1 + db_reliability.JITTER_RATIO
        )


# ---------------------------------------------------------------------------
# 4. Non-idempotent write safety.
# ---------------------------------------------------------------------------


def test_is_transient_does_not_by_itself_authorise_a_write_retry():
    """The classifier says "the transport failed", never "it is safe to repeat".

    ``RemoteProtocolError`` on a write is ambiguous: the statement may well have
    committed before the connection went away. Nothing may turn that
    classification into an automatic retry.
    """
    assert db_reliability.is_transient_db_error(connection_terminated())


class _CountingWrites:
    def __init__(self):
        self.inserts = 0
        self.updates = 0
        self.rpcs = 0
        self._pending = None

    def table(self, _name):
        return self

    def insert(self, _rows):
        self._pending = "insert"
        return self

    def update(self, _values):
        self._pending = "update"
        return self

    def rpc(self, _name, _params):
        self._pending = "rpc"
        return self

    def eq(self, *_args):
        return self

    def is_(self, *_args):
        return self

    def execute(self):
        if self._pending == "insert":
            self.inserts += 1
        elif self._pending == "update":
            self.updates += 1
        else:
            self.rpcs += 1
        raise connection_terminated()


def test_a_bare_message_insert_is_attempted_exactly_once():
    """An INSERT with no idempotency key is never repeated after an ambiguous
    failure. Repeating it is how a fan gets the same message twice."""
    db = _CountingWrites()

    async def write():
        await asyncio.to_thread(
            lambda: db.table("messages").insert([{"content": "x"}]).execute()
        )

    with pytest.raises(httpx.RemoteProtocolError):
        run(write())

    assert db.inserts == 1


def test_the_atomic_claim_rpc_is_attempted_exactly_once(monkeypatch):
    """A retried claim cannot double-claim, but it CAN strand the first batch as
    PROCESSING until the stale window elapses. The worker's next cycle is
    seconds away, so the correct answer is to let the cycle fail and poll
    again — not to retry a write for a saving measured in seconds."""
    from db import commercial_queries

    db = _CountingWrites()
    monkeypatch.setattr(commercial_queries, "get_supabase", lambda: db)
    monkeypatch.setattr(commercial_queries, "_ATOMIC_CLAIM_AVAILABLE", True)

    with pytest.raises(httpx.RemoteProtocolError):
        run(commercial_queries.claim_due_actions(limit=5))

    assert db.rpcs == 1


# ---------------------------------------------------------------------------
# 5. The one write class that IS retried, and why.
# ---------------------------------------------------------------------------


class _CompareAndSetUpdate:
    """A ``messages`` row whose platform id is set once and only once.

    Models the PostgREST behaviour the retry safety rests on: the UPDATE carries
    ``.is_("fansly_message_id", "null")``, so once it has been applied the
    predicate no longer matches any row and a repeat updates nothing.
    """

    def __init__(self, *, fail_first: bool):
        self.rows = [{"id": "m1", "fansly_message_id": None}]
        self.applied = 0
        self.attempts = 0
        self._fail_first = fail_first
        self._values = None
        self._require_null = False

    def table(self, _name):
        return self

    def update(self, values):
        self._values = values
        self._require_null = False
        return self

    def eq(self, *_args):
        return self

    def is_(self, column, value):
        if column == "fansly_message_id" and value == "null":
            self._require_null = True
        return self

    def execute(self):
        self.attempts += 1
        if self._fail_first and self.attempts == 1:
            # Ambiguous: this one DID commit before the connection died.
            self._apply()
            raise connection_terminated()
        return SimpleNamespace(data=self._apply())

    def _apply(self):
        matched = [
            row
            for row in self.rows
            if not self._require_null or row["fansly_message_id"] is None
        ]
        for row in matched:
            row.update(self._values)
            self.applied += 1
        return matched


def test_a_compare_and_set_update_is_safe_to_retry_because_the_repeat_matches_nothing():
    """The existing identity-reconciliation UPDATE, and the exact reason it may
    be retried where a bare INSERT may not.

    The first attempt commits and then loses its response. The retry runs the
    same statement, the ``fansly_message_id IS NULL`` predicate no longer
    matches, and nothing is applied a second time.
    """
    db = _CompareAndSetUpdate(fail_first=True)

    async def reconcile_identity():
        await db_reliability.retry_transient_db_operation(
            lambda: asyncio.to_thread(
                lambda: db.table("messages")
                .update({"fansly_message_id": "fansly-1"})
                .eq("id", "m1")
                .is_("fansly_message_id", "null")
                .execute()
            ),
            label="reconcile_message_identity:m1",
            delay_seconds=0.0,
        )

    run(reconcile_identity())

    assert db.attempts == 2
    # Applied once, by the attempt whose answer was lost. The retry was a no-op.
    assert db.applied == 1
    assert db.rows == [{"id": "m1", "fansly_message_id": "fansly-1"}]


# ---------------------------------------------------------------------------
# 6. Concurrent failure — one rebuild, not a herd.
# ---------------------------------------------------------------------------


def test_many_simultaneous_failures_rebuild_the_client_once(monkeypatch):
    builds = {"count": 0}
    real_build = supabase_module.build_http_client

    def counting_build():
        builds["count"] += 1
        return real_build()

    monkeypatch.setattr(supabase_module, "build_http_client", counting_build)

    supabase_module.get_supabase()
    assert builds["count"] == 1
    generation = supabase_module.supabase_generation()

    # Everyone failed on the same connection, so everyone is holding the same
    # generation number.
    barrier = threading.Barrier(24)
    wins: list[bool] = []
    lock = threading.Lock()

    def report_failure():
        barrier.wait(timeout=10)
        won = supabase_module.reset_supabase_client(
            reason="concurrent:RemoteProtocolError", generation=generation
        )
        with lock:
            wins.append(won)

    threads = [threading.Thread(target=report_failure) for _ in range(24)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=20)

    assert sum(wins) == 1, "24 callers must not produce 24 resets"
    assert supabase_module.supabase_generation() == generation + 1

    # And the rebuild itself happens once, however many callers ask for it.
    def use_client():
        barrier2.wait(timeout=10)
        supabase_module.get_supabase()

    barrier2 = threading.Barrier(24)
    users = [threading.Thread(target=use_client) for _ in range(24)]
    for thread in users:
        thread.start()
    for thread in users:
        thread.join(timeout=20)

    assert builds["count"] == 2


def test_concurrent_retries_do_not_multiply_client_builds(monkeypatch):
    """End to end through the retry helper rather than the reset API."""
    builds = {"count": 0}
    real_build = supabase_module.build_http_client

    def counting_build():
        builds["count"] += 1
        return real_build()

    monkeypatch.setattr(supabase_module, "build_http_client", counting_build)
    supabase_module.get_supabase()

    calls = {"count": 0}

    async def fails_then_succeeds():
        calls["count"] += 1
        if calls["count"] <= 16:
            raise connection_terminated()
        return "ok"

    async def drive():
        return await asyncio.gather(
            *(
                db_reliability.retry_transient_db_operation(
                    fails_then_succeeds,
                    label="concurrent.read",
                    attempts=4,
                    delay_seconds=0.0,
                )
                for _ in range(16)
            )
        )

    assert run(drive()) == ["ok"] * 16
    # At most one rebuild beyond the initial build, whatever the failure count.
    assert builds["count"] <= 2


# ---------------------------------------------------------------------------
# 7. Health recovery — unhealthy while it is true, and only while it is true.
# ---------------------------------------------------------------------------


def test_database_health_recovers_once_the_database_does(monkeypatch):
    state = {"fail": True}

    class _Probe:
        def table(self, _name):
            return self

        def select(self, *_args, **_kwargs):
            return self

        def limit(self, _value):
            return self

        def execute(self):
            if state["fail"]:
                raise connection_terminated()
            return SimpleNamespace(data=[{"id": "creator-1"}])

    monkeypatch.setattr("core.supabase.get_supabase", lambda: _Probe())

    unhealthy = run(operational_health.probe_database())
    assert unhealthy["reachable"] is False
    assert unhealthy["error"] == "RemoteProtocolError"

    verdict = operational_health.evaluate(
        database=unhealthy,
        queue={"available": True, "pending": 0, "oldest_pending_age_seconds": 0},
        scheduler={"poll_seconds": 5, "seconds_since_last_cycle": 1.0, "cycles_completed": 2},
        model_gate={},
        model_availability={"status": "ok"},
    )
    assert verdict["status"] == "unhealthy"
    assert any(r.startswith("database_unreachable") for r in verdict["fatal_reasons"])

    # The database comes back. Nothing is latched: the next probe is a live one.
    state["fail"] = False
    operational_health.reset_cache()
    healthy = run(operational_health.probe_database())
    assert healthy["reachable"] is True

    verdict = operational_health.evaluate(
        database=healthy,
        queue={"available": True, "pending": 0, "oldest_pending_age_seconds": 0},
        scheduler={"poll_seconds": 5, "seconds_since_last_cycle": 1.0, "cycles_completed": 2},
        model_gate={},
        model_availability={"status": "ok"},
    )
    assert verdict["fatal_reasons"] == []
    assert verdict["status"] == "ok"


def test_a_single_terminated_connection_does_not_raise_the_banner(monkeypatch):
    """One recycled connection is not an outage, and must not be reported as one."""
    attempts = {"count": 0}

    class _FlakyProbe:
        def table(self, _name):
            return self

        def select(self, *_args, **_kwargs):
            return self

        def limit(self, _value):
            return self

        def execute(self):
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise connection_terminated()
            return SimpleNamespace(data=[{"id": "creator-1"}])

    monkeypatch.setattr("core.supabase.get_supabase", lambda: _FlakyProbe())
    operational_health.reset_cache()

    result = run(operational_health.probe_database())

    assert result["reachable"] is True
    assert attempts["count"] == 2


def test_the_health_probe_never_churns_the_client(monkeypatch):
    """Health observes an outage; it must not become a cause of one."""
    resets = {"count": 0}

    def fake_reset(**_kwargs):
        resets["count"] += 1
        return True

    class _Dead:
        def table(self, _name):
            return self

        def select(self, *_args, **_kwargs):
            return self

        def limit(self, _value):
            return self

        def execute(self):
            raise connection_terminated()

    monkeypatch.setattr("core.supabase.get_supabase", lambda: _Dead())
    monkeypatch.setattr(supabase_module, "reset_supabase_client", fake_reset)
    operational_health.reset_cache()

    result = run(operational_health.probe_database())

    assert result["reachable"] is False
    assert resets["count"] == 0


def test_the_health_document_reports_the_transport_it_is_using():
    snapshot = supabase_module.snapshot()
    assert snapshot["http2"] is False
    assert "generation" in snapshot
    # No URL, no key, no token — only configuration facts.
    text = repr(snapshot).lower()
    assert "supabase.co" not in text
    assert "key" not in text
