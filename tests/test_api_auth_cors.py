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
