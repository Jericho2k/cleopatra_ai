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

