from fastapi.testclient import TestClient

from main import app


def test_auth_failure_keeps_dashboard_cors_headers(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("DASHBOARD_API_SECRET", "expected-dashboard-key")

    response = TestClient(app).get(
        "/fan/test/operator-ppv-options?creator_id=test",
        headers={
            "Origin": "https://cleopatra-dashboard.vercel.app",
            "X-API-Key": "wrong-dashboard-key",
        },
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "Missing or invalid credentials"}
    assert (
        response.headers["access-control-allow-origin"]
        == "https://cleopatra-dashboard.vercel.app"
    )



def test_health_probes_stay_reachable_without_credentials(monkeypatch):
    """The platform healthcheck has no dashboard key, so both must be public."""
    from services import operational_health

    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("DASHBOARD_API_SECRET", "expected-dashboard-key")
    _stub_health(monkeypatch, operational_health)

    client = TestClient(app)
    operational_health.reset_cache()
    health = client.get("/health")
    operational_health.reset_cache()
    ready = client.get("/health/ready")

    assert health.status_code == 200
    assert ready.status_code == 200
    assert health.json()["status"] == "ok"
    # An anonymous prober gets the verdict, not the deployment's queue state.
    assert "queue" not in health.json()
    assert "scheduler" not in health.json()


def test_health_detail_is_returned_to_the_dashboard(monkeypatch):
    from services import operational_health

    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("DASHBOARD_API_SECRET", "expected-dashboard-key")
    _stub_health(monkeypatch, operational_health)

    operational_health.reset_cache()
    response = TestClient(app).get(
        "/health", headers={"X-API-Key": "expected-dashboard-key"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["queue"]["pending"] == 7
    assert body["scheduler"]["cycles_completed"] == 9
    assert body["thresholds"]["queue_max_depth"] >= 1


def _stub_health(monkeypatch, operational_health):
    async def database():
        return {"reachable": True, "latency_ms": 2, "error": None}

    async def queue():
        return {
            "available": True,
            "pending": 7,
            "processing": 0,
            "failed": 0,
            "pending_inbound_messages": 0,
            "oldest_due_execute_at": None,
            "oldest_due_action_type": None,
            "oldest_pending_age_seconds": 0.0,
            "error": None,
        }

    monkeypatch.setattr(operational_health, "probe_database", database)
    monkeypatch.setattr(operational_health, "probe_queue", queue)
    monkeypatch.setattr(
        operational_health,
        "worker_health_snapshot",
        lambda: {
            "poll_seconds": 5,
            "seconds_since_last_cycle": 1.0,
            "cycles_completed": 9,
            "last_error": None,
        },
    )
    monkeypatch.setattr(
        operational_health,
        "current_model_availability",
        lambda: {"status": "healthy", "checked_at": None},
    )


# ---------------------------------------------------------------------------
# An unhandled exception must be READABLE by the browser
# ---------------------------------------------------------------------------
#
# Starlette answers an exception that escapes a route from its OUTERMOST
# middleware, so that response never passes back through the CORS layer. The
# browser therefore sees a response with no Access-Control-Allow-Origin, blocks
# it, and rejects the fetch with the opaque TypeError "Failed to fetch" — which
# is precisely what the Simulator surfaced while the backend had already logged
# a traceback nobody could correlate with it.


def test_an_unhandled_route_error_is_a_cors_visible_json_500(monkeypatch):
    from fastapi.testclient import TestClient

    import main

    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("DASHBOARD_API_SECRET", "test-dashboard-secret")

    async def fake_user(_authorization):
        return "operator-1"

    monkeypatch.setattr(main, "authenticated_dashboard_user", fake_user, raising=False)
    monkeypatch.setattr("core.auth.authenticated_dashboard_user", fake_user)

    @main.app.get("/__boom_for_test")
    async def _boom():  # pragma: no cover - exercised through the client
        raise KeyError("a genuinely unexpected failure")

    client = TestClient(app=main.app, raise_server_exceptions=False)
    response = client.get(
        "/__boom_for_test",
        headers={"X-API-Key": "test-dashboard-secret", "Origin": "http://localhost:3000"},
    )

    assert response.status_code == 500
    # The half that made it invisible: without this header the browser refuses
    # to hand the body to the page at all.
    assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"

    body = response.json()
    assert body["error_type"] == "KeyError"
    assert body["error_id"]
    assert body["error_id"] in body["detail"]
    # ...and nothing about internals reaches an ordinary agency account.
    assert "a genuinely unexpected failure" not in body["detail"]
    assert "Traceback" not in response.text
