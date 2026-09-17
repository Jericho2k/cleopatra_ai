"""OBS-001: health that can actually detect a failure, without causing one.

The two things that must both hold: a real infrastructure failure is visible and
fails readiness, and a wobbly model provider is visible but never fails the
liveness probe Railway restarts on.
"""
from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime

import pytest

import main
from services import operational_health


def run(coro):
    return asyncio.run(coro)


HEALTHY_DB = {"reachable": True, "latency_ms": 4, "error": None}
HEALTHY_QUEUE = {
    "available": True,
    "pending": 3,
    "processing": 1,
    "failed": 0,
    "pending_inbound_messages": 1,
    "oldest_due_execute_at": "2026-09-08T07:00:00+00:00",
    "oldest_pending_age_seconds": 12.0,
    "oldest_due_action_type": "AUTO_REPLY",
    "error": None,
}
HEALTHY_SCHEDULER = {
    "poll_seconds": 5,
    "seconds_since_last_cycle": 2.0,
    "cycles_completed": 40,
    "last_error": None,
    "last_claimed": 4,
    "last_sent": 3,
}


def wire(monkeypatch, *, db=None, queue=None, scheduler=None, availability=None, gate=None):
    operational_health.reset_cache()
    monkeypatch.setattr(
        operational_health, "probe_database", lambda: _value(db or HEALTHY_DB)
    )
    monkeypatch.setattr(
        operational_health, "probe_queue", lambda: _value(queue or HEALTHY_QUEUE)
    )
    monkeypatch.setattr(
        operational_health,
        "worker_health_snapshot",
        lambda: dict(scheduler or HEALTHY_SCHEDULER),
    )
    monkeypatch.setattr(
        operational_health,
        "current_model_availability",
        lambda: availability or {"status": "healthy", "checked_at": None},
    )
    if gate is not None:
        monkeypatch.setattr(
            operational_health.MODEL_GATE, "snapshot", lambda: gate
        )


async def _value(v):
    return v


@pytest.fixture(autouse=True)
def clear_cache():
    operational_health.reset_cache()
    yield
    operational_health.reset_cache()


def test_healthy_deployment_reports_ok(monkeypatch):
    wire(monkeypatch)
    document = run(operational_health.collect(use_cache=False))

    assert document["status"] == "ok"
    assert document["liveness"] == "ok"
    assert document["degraded_reasons"] == []
    assert document["fatal_reasons"] == []
    assert document["queue"]["pending"] == 3
    assert document["scheduler"]["cycles_completed"] == 40
    assert document["model"]["gate"]["limit"] >= 1


def test_unreachable_database_is_fatal_and_visible(monkeypatch):
    """A probe result with no confirmation record still fails closed.

    The reason string is now ``database_unavailable`` — the confirmed-outage
    name — because a confirmed outage is the only thing that reaches fatal once
    the hysteresis ladder is in play (tests/test_db_health_hysteresis.py).
    """
    wire(monkeypatch, db={"reachable": False, "latency_ms": 1500, "error": "timeout"})
    document = run(operational_health.collect(use_cache=False))

    assert document["status"] == "unhealthy"
    assert any("database_unavailable" in r for r in document["fatal_reasons"])


def test_unavailable_model_provider_is_degraded_never_fatal(monkeypatch):
    wire(monkeypatch, availability={"status": "unavailable", "checked_at": None})
    document = run(operational_health.collect(use_cache=False))

    assert document["status"] == "degraded"
    assert document["fatal_reasons"] == [], (
        "a provider incident must never take the container out of service"
    )
    assert "model_unavailable" in document["degraded_reasons"]


def test_backlog_age_and_depth_produce_degraded(monkeypatch):
    monkeypatch.setenv("HEALTH_QUEUE_MAX_AGE_SECONDS", "300")
    monkeypatch.setenv("HEALTH_QUEUE_MAX_DEPTH", "100")
    wire(
        monkeypatch,
        queue={**HEALTHY_QUEUE, "oldest_pending_age_seconds": 900.0, "pending": 400},
    )
    document = run(operational_health.collect(use_cache=False))

    assert document["status"] == "degraded"
    assert "queue_oldest_pending_age_exceeds_300s" in document["degraded_reasons"]
    assert "queue_depth_exceeds_100" in document["degraded_reasons"]
    assert document["fatal_reasons"] == []


def test_normal_composition_delay_does_not_trip_the_age_threshold(monkeypatch):
    """A fan waiting 20 s for a human-like reply is not a degraded deployment."""
    monkeypatch.delenv("HEALTH_QUEUE_MAX_AGE_SECONDS", raising=False)
    wire(monkeypatch, queue={**HEALTHY_QUEUE, "oldest_pending_age_seconds": 45.0})
    document = run(operational_health.collect(use_cache=False))

    assert document["status"] == "ok"
    assert operational_health.queue_max_age_seconds() >= 300


def test_stale_scheduler_is_degraded(monkeypatch):
    wire(
        monkeypatch,
        scheduler={**HEALTHY_SCHEDULER, "seconds_since_last_cycle": 3600.0},
    )
    document = run(operational_health.collect(use_cache=False))

    assert document["status"] == "degraded"
    assert any(r.startswith("scheduler_stale_for_") for r in document["degraded_reasons"])


def test_scheduler_error_is_surfaced(monkeypatch):
    wire(monkeypatch, scheduler={**HEALTHY_SCHEDULER, "last_error": "boom"})
    document = run(operational_health.collect(use_cache=False))
    assert "scheduler_last_cycle_errored" in document["degraded_reasons"]


def test_fresh_process_is_not_degraded_before_its_first_cycle(monkeypatch):
    wire(
        monkeypatch,
        scheduler={
            "poll_seconds": 5,
            "seconds_since_last_cycle": None,
            "cycles_completed": 0,
            "last_error": None,
        },
    )
    monkeypatch.setattr(operational_health, "process_uptime_seconds", lambda: 3.0)
    assert run(operational_health.collect(use_cache=False))["status"] == "ok"

    monkeypatch.setattr(operational_health, "process_uptime_seconds", lambda: 600.0)
    operational_health.reset_cache()
    document = run(operational_health.collect(use_cache=False))
    assert "scheduler_has_not_completed_a_cycle" in document["degraded_reasons"]


def test_saturated_model_gate_is_degraded(monkeypatch):
    wire(
        monkeypatch,
        gate={
            "limit": 8, "inflight": 8, "waiting": 12, "max_inflight": 8,
            "acquired_total": 200, "avg_wait_ms": 900, "max_wait_ms": 4000,
            "recent_wait_ms": 850,
        },
    )
    document = run(operational_health.collect(use_cache=False))
    assert "model_gate_saturated" in document["degraded_reasons"]
    assert document["model"]["gate"]["recent_wait_ms"] == 850


def test_health_endpoint_stays_200_while_ready_refuses(monkeypatch):
    wire(monkeypatch, db={"reachable": False, "latency_ms": 1, "error": "AuthError"})

    payload = run(main.health())
    assert payload["status"] == "unhealthy"
    assert payload["vault_classifier_version"]

    class Resp:
        status_code = 200

    response = Resp()
    ready = run(main.health_ready(response))
    assert response.status_code == 503
    assert ready["status"] == "unhealthy"


def test_ready_stays_200_for_a_throttled_provider(monkeypatch):
    wire(monkeypatch, availability={"status": "unavailable", "checked_at": None})

    class Resp:
        status_code = 200

    response = Resp()
    run(main.health_ready(response))
    assert response.status_code == 200, "no restart loop over a provider incident"


def test_health_exposes_no_message_content_or_credentials(monkeypatch):
    wire(monkeypatch)
    document = run(operational_health.collect(use_cache=False))
    blob = json.dumps(document).lower()

    for forbidden in (
        "test-webhook-secret",
        "test-together-key",
        "test-anthropic-key",
        "authorization",
        "api_key",
        "content",
        "fan_name",
        "display_name",
        "prompt",
    ):
        assert forbidden not in blob, f"health leaked {forbidden}"
    # Fan and creator identifiers are not in the payload either.
    assert "fan_id" not in blob and "creator_id" not in blob


def test_health_is_cached_so_probing_cannot_become_load(monkeypatch):
    calls = {"db": 0}

    async def counting_db():
        calls["db"] += 1
        return HEALTHY_DB

    operational_health.reset_cache()
    monkeypatch.setattr(operational_health, "probe_database", counting_db)
    monkeypatch.setattr(operational_health, "probe_queue", lambda: _value(HEALTHY_QUEUE))
    monkeypatch.setattr(
        operational_health, "worker_health_snapshot", lambda: dict(HEALTHY_SCHEDULER)
    )
    monkeypatch.setattr(
        operational_health, "current_model_availability",
        lambda: {"status": "healthy", "checked_at": None},
    )

    async def hammer():
        for _ in range(25):
            await operational_health.collect()

    run(hammer())
    assert calls["db"] == 1


def test_health_stays_within_a_latency_budget(monkeypatch):
    wire(monkeypatch)

    async def timed():
        started = time.perf_counter()
        await operational_health.collect(use_cache=False)
        return (time.perf_counter() - started) * 1000

    assert run(timed()) < 250


def test_probes_are_time_bounded_and_never_raise(monkeypatch):
    class HangingDB:
        def table(self, _n):
            return self

        def select(self, *_a):
            return self

        def limit(self, *_a):
            return self

        def execute(self):
            time.sleep(0.5)

    monkeypatch.setattr(operational_health, "_DB_PROBE_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr("core.supabase.get_supabase", lambda: HangingDB())

    result = run(operational_health.probe_database())
    assert result["reachable"] is False
    assert result["error"] == "timeout"


def test_probe_reports_error_type_not_message(monkeypatch):
    class ExplodingDB:
        def table(self, _n):
            raise RuntimeError("postgres://user:hunter2@host/db refused")

    monkeypatch.setattr("core.supabase.get_supabase", lambda: ExplodingDB())
    result = run(operational_health.probe_database())

    assert result["reachable"] is False
    assert result["error"] == "RuntimeError"
    assert "hunter2" not in json.dumps(result)


class _Request:
    def __init__(self, headers: dict | None = None):
        self.headers = headers or {}


# --- the anonymous redaction gate -------------------------------------------
#
# What an anonymous prober may receive is a fixed, small schema: the verdict and
# the reason categories, plus the build identity that answers "which commit is
# this". Everything else in the document is operational detail.
#
# These sets are written out here rather than imported from ``main`` on purpose.
# Importing the allowlist would make the test agree with whatever the allowlist
# says, so widening it would silently widen the gate too. Spelled out, adding a
# public key fails this test until someone confirms the new key is safe to serve
# to an unauthenticated caller.
_PUBLIC_HEALTH_KEYS = {
    "status",
    "liveness",
    "checked_at",
    "degraded_reasons",
    "fatal_reasons",
    "vault_classifier_version",
    "vault_semantics_configured",
    "build_sha",
    "flags_digest",
}

# Sections of the document that carry deployment state. None of these key names
# may appear at ANY depth of an anonymous response.
_OPERATIONAL_SECTIONS = {
    "integrations",
    "thresholds",
    "database",
    "queue",
    "scheduler",
    "model",
    "vault",
    "db_executor",
    "db_transport",
}

# Typed sentinels planted in the probes below. They are compared by value AND
# type, never as digit substrings of the serialized document: the previous
# ``"4321" not in json.dumps(payload)`` form matched the microseconds of an
# ordinary ``checked_at`` timestamp and failed CI run 35219091193 with nothing
# leaked. ``test_an_ordinary_timestamp_is_not_a_leak`` pins that exact value.
_QUEUE_DEPTH = 4321
_SCHEDULER_CYCLES = 9271
_DB_LATENCY = 8143
_PROBE_ERROR = "postgres://operator:hunter2@db.internal/cleopatra"
_SENTINELS = (_QUEUE_DEPTH, _SCHEDULER_CYCLES, _DB_LATENCY, _PROBE_ERROR)


class _FrozenClock(datetime):
    """``checked_at`` exactly as CI run 35219091193 produced it."""

    @classmethod
    def now(cls, tz=None):
        return datetime(2026, 9, 8, 12, 8, 25, 432182, tzinfo=tz)


def _scalars(node, path="$"):
    """Every leaf value in a JSON-able document, with the path that reached it."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _scalars(value, f"{path}.{key}")
    elif isinstance(node, (list, tuple)):
        for index, value in enumerate(node):
            yield from _scalars(value, f"{path}[{index}]")
    else:
        yield path, node


def _keys(node):
    """Every key name in a JSON-able document, at any depth."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield key
            yield from _keys(value)
    elif isinstance(node, (list, tuple)):
        for value in node:
            yield from _keys(value)


def _leaks(payload) -> list[str]:
    """Operational detail found in ``payload``, as human-readable findings.

    Two independent checks, because either alone has a hole: a section could be
    renamed (so key names are not enough), and a value could be reshaped (so
    value matching is not enough).
    """
    findings = []
    for key in _keys(payload):
        if key in _OPERATIONAL_SECTIONS:
            findings.append(f"operational section {key!r} present")
    for path, value in _scalars(payload):
        for sentinel in _SENTINELS:
            # `type(...) is type(...)` first: in Python `True == 1`, so a bare
            # `==` would let a boolean field trip an integer sentinel.
            if type(value) is type(sentinel) and value == sentinel:
                findings.append(f"{path} == {value!r} (planted operational value)")
    return findings


def _wire_with_sentinels(monkeypatch):
    wire(
        monkeypatch,
        db={"reachable": True, "latency_ms": _DB_LATENCY, "error": _PROBE_ERROR},
        queue={**HEALTHY_QUEUE, "pending": _QUEUE_DEPTH},
        scheduler={**HEALTHY_SCHEDULER, "cycles_completed": _SCHEDULER_CYCLES},
    )


def test_anonymous_probe_gets_status_without_operational_detail(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("DASHBOARD_API_SECRET", "dashboard-secret")
    _wire_with_sentinels(monkeypatch)

    payload = run(main.health(_Request()))

    assert payload["status"] == "degraded"
    assert payload["liveness"] == "ok"
    assert payload["degraded_reasons"]
    # A healthcheck needs the verdict, not the deployment's queue depth.
    assert set(payload) == _PUBLIC_HEALTH_KEYS
    assert _leaks(payload) == []


def test_anonymous_readiness_is_redacted_on_the_same_terms(monkeypatch):
    """/health/ready shares the redaction, so it needs the same gate.

    It returns the public document without the build identity ``/health`` adds,
    so its schema is the allowlist alone.
    """
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("DASHBOARD_API_SECRET", "dashboard-secret")
    _wire_with_sentinels(monkeypatch)

    class Resp:
        status_code = 200

    payload = run(main.health_ready(Resp(), _Request()))

    assert set(payload) <= _PUBLIC_HEALTH_KEYS
    assert _leaks(payload) == []


def test_the_redaction_gate_can_actually_fail(monkeypatch):
    """The negative tests above must not be able to pass vacuously.

    Two ways they could: the sentinels never reach the document at all, or
    ``_leaks`` finds nothing anywhere. The authenticated document proves the
    first, and it is itself the proof of the second — every planted value is
    in it, so the same scan that returns clean for an anonymous caller returns
    findings here.
    """
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("DASHBOARD_API_SECRET", "dashboard-secret")
    _wire_with_sentinels(monkeypatch)

    document = run(main.health(_Request({"x-api-key": "dashboard-secret"})))

    assert document["queue"]["pending"] == _QUEUE_DEPTH
    assert document["scheduler"]["cycles_completed"] == _SCHEDULER_CYCLES
    assert document["database"]["latency_ms"] == _DB_LATENCY
    findings = _leaks(document)
    assert any("queue" in finding for finding in findings)
    assert any(repr(_PROBE_ERROR) in finding for finding in findings)


def test_an_ordinary_timestamp_is_not_a_leak(monkeypatch):
    """Regression for CI run 35219091193 — 1 failed, 2336 passed.

    ``checked_at`` was ``2026-09-08T12:08:25.432182+00:00``. Its microseconds
    contain the digits of the planted queue depth, so asserting on the
    serialized document's digits reported a redaction failure on a correctly
    redacted response. The timestamp is pinned here so that assertion shape
    cannot come back.
    """
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("DASHBOARD_API_SECRET", "dashboard-secret")
    monkeypatch.setattr(operational_health, "datetime", _FrozenClock)
    _wire_with_sentinels(monkeypatch)

    payload = run(main.health(_Request()))

    assert payload["checked_at"] == "2026-09-08T12:08:25.432182+00:00"
    # The digits are right there in the serialized document ...
    assert str(_QUEUE_DEPTH) in json.dumps(payload)
    # ... and the response is correctly redacted all the same.
    assert set(payload) == _PUBLIC_HEALTH_KEYS
    assert _leaks(payload) == []


def test_dashboard_caller_gets_the_full_document(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("DASHBOARD_API_SECRET", "dashboard-secret")
    wire(monkeypatch)

    payload = run(main.health(_Request({"x-api-key": "dashboard-secret"})))

    assert payload["queue"]["pending"] == 3
    assert payload["scheduler"]["cycles_completed"] == 40
    assert payload["model"]["gate"]["limit"] >= 1


def test_anonymous_readiness_still_refuses_on_a_fatal_condition(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("DASHBOARD_API_SECRET", "dashboard-secret")
    wire(monkeypatch, db={"reachable": False, "latency_ms": 1, "error": "timeout"})

    class Resp:
        status_code = 200

    response = Resp()
    payload = run(main.health_ready(response, _Request()))

    assert response.status_code == 503
    assert payload["status"] == "unhealthy"
    assert "queue" not in payload
