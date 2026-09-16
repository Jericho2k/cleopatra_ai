"""The Simulator's HTTP contract, after a turn stopped being one request.

Three things a dashboard depends on and which the old synchronous endpoint
could not offer:

* POST returns a turn id and ``processing`` immediately, whatever the writer
  is about to spend recovering;
* GET resolves that turn — and, for a browser that reloaded, the newest turn
  for the conversation — without ever running the pipeline again;
* a second concurrent submission for one fan is refused, and the refusal names
  the turn the operator should be watching instead.

Authorization is unchanged and is asserted in tests/test_simulation_authorization.py.
What matters here is that the NEW read routes sit behind exactly the same gate:
a poll endpoint that skipped it would hand another tenant's turn — and the
creator messages attached to it — to anyone who guessed a uuid.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import main
from core import tenancy
from services import simulation_turns

OWNER = "11111111-1111-4111-8111-111111111111"
AGENCY = "22222222-2222-4222-8222-222222222222"
OUTSIDER = "33333333-3333-4333-8333-333333333333"

ASSIGNMENTS = {OWNER: {"creator-1"}, AGENCY: {"creator-1"}, OUTSIDER: {"creator-2"}}

FANS = {
    "fan-test": {
        "id": "fan-test",
        "creator_id": "creator-1",
        "platform_fan_id": "test_abc",
        "display_name": "Test fan",
    },
    "fan-real": {
        "id": "fan-real",
        "creator_id": "creator-1",
        "platform_fan_id": "884422113355",
        "display_name": "Real fan",
    },
    "fan-other": {
        "id": "fan-other",
        "creator_id": "creator-2",
        "platform_fan_id": "test_other",
        "display_name": "Other tenant's test fan",
    },
}

CREATORS = {
    "creator-1": {"id": "creator-1", "platform_username": "sophia"},
    "creator-2": {"id": "creator-2", "platform_username": "eliz"},
}


class _Rows:
    """Creators and fans, honouring the filters the simulator routes apply."""

    def __init__(self) -> None:
        self._tables = {"fans": list(FANS.values()), "creators": list(CREATORS.values())}
        self.rows: list[dict] = []

    def table(self, name):
        clone = _Rows.__new__(_Rows)
        clone._tables = self._tables
        clone.rows = list(self._tables.get(name, []))
        return clone

    def select(self, *_a, **_k):
        return self

    def eq(self, column, value):
        self.rows = [r for r in self.rows if str(r.get(column)) == str(value)]
        return self

    def in_(self, column, values):
        allowed = {str(v) for v in values}
        self.rows = [r for r in self.rows if str(r.get(column)) in allowed]
        return self

    def order(self, *_a, **_k):
        return self

    def limit(self, *_a, **_k):
        return self

    def single(self):
        return self

    def execute(self):
        return SimpleNamespace(data=list(self.rows))


@pytest.fixture
def transcript(monkeypatch):
    rows: list[dict] = []

    async def fake_rows(_fan_id):
        return [dict(row) for row in rows]

    monkeypatch.setattr("services.suggestions._recent_creator_message_rows", fake_rows)
    return rows


@pytest.fixture
def pipeline(monkeypatch, transcript):
    """The real Full Auto path, replaced by one that writes a known reply."""
    calls: list[dict] = []

    async def fake_run(**kwargs):
        calls.append(kwargs)
        row = {
            "id": f"creator-message-{len(transcript) + 1}",
            "role": "creator",
            "content": "hey you",
            "sent_at": None,
            "media_context": {"ai_stack": {"profile": "cleo_v3", "model": "kimi"}},
        }
        transcript.append(row)
        return {
            "status": "ok",
            "simulation": True,
            "fan_message_id": "fan-message-1",
            "creator_messages": [row],
            "outcome": "replied",
        }

    monkeypatch.setattr("services.suggestions.run_simulated_inbound", fake_run)
    return calls


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

    monkeypatch.setattr(main, "authenticated_dashboard_user", fake_user, raising=False)
    monkeypatch.setattr("core.auth.authenticated_dashboard_user", fake_user)
    monkeypatch.setattr(tenancy, "_creator_ids_for_user", fake_creator_ids)
    monkeypatch.setattr(main, "get_supabase", _Rows)
    monkeypatch.setattr(tenancy, "get_supabase", _Rows)
    return TestClient(app=main.app)


def _headers(user: str) -> dict[str, str]:
    return {"X-API-Key": "test-dashboard-secret", "Authorization": f"Bearer {user}"}


def _send(client, user, key="send-1", creator="creator-1", fan="fan-test"):
    return client.post(
        f"/creator/{creator}/fan/{fan}/simulate-inbound",
        headers=_headers(user),
        json={"message": "hii", "fast": True, "idempotency_key": key},
    )


def _poll(client, user, turn_id, creator="creator-1", fan="fan-test"):
    return client.get(
        f"/creator/{creator}/fan/{fan}/simulation/turn/{turn_id}",
        headers=_headers(user),
    )


# --- the new contract -------------------------------------------------------


def test_the_post_returns_a_turn_id_and_processing(client, pipeline):
    response = _send(client, AGENCY)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "processing"
    assert body["created"] is True
    assert body["turn_id"]
    # The reply is NOT in the POST response. It arrives with the completed
    # turn, which is what stops a slow recovery becoming a browser timeout.
    assert "creator_messages" not in body


def test_polling_returns_the_completed_turn_and_its_reply(client, pipeline):
    turn_id = _send(client, AGENCY).json()["turn_id"]

    polled = _poll(client, AGENCY, turn_id)

    assert polled.status_code == 200, polled.text
    body = polled.json()
    assert body["status"] == "completed"
    assert body["outcome"] == "replied"
    assert [row["id"] for row in body["creator_messages"]] == ["creator-message-1"]
    assert body["creator_messages"][0]["content"] == "hey you"


def test_polling_never_runs_the_pipeline_again(client, pipeline):
    turn_id = _send(client, AGENCY).json()["turn_id"]
    assert len(pipeline) == 1

    for _ in range(10):
        assert _poll(client, AGENCY, turn_id).json()["status"] == "completed"

    assert len(pipeline) == 1, "a poll is a read"


def test_a_repeated_post_with_the_same_key_does_not_start_a_second_turn(
    client, pipeline
):
    first = _send(client, AGENCY, key="send-1").json()
    second = _send(client, AGENCY, key="send-1").json()

    assert second["turn_id"] == first["turn_id"]
    assert first["created"] is True
    assert second["created"] is False
    assert len(pipeline) == 1


def test_a_reloaded_browser_recovers_the_turn_from_the_fan_alone(client, pipeline):
    turn_id = _send(client, AGENCY).json()["turn_id"]

    latest = client.get(
        "/creator/creator-1/fan/fan-test/simulation/turn", headers=_headers(AGENCY)
    )

    assert latest.status_code == 200, latest.text
    assert latest.json()["turn_id"] == turn_id
    assert latest.json()["status"] == "completed"
    assert len(pipeline) == 1


def test_a_conversation_with_no_turns_yet_says_so_rather_than_erroring(client):
    latest = client.get(
        "/creator/creator-1/fan/fan-test/simulation/turn", headers=_headers(AGENCY)
    )

    assert latest.status_code == 200
    assert latest.json()["turn_id"] is None


def test_a_second_concurrent_turn_is_refused_with_the_one_already_running(
    client, monkeypatch, pipeline
):
    """409, and the active turn id, so the browser can go and watch it."""
    held: list = []

    async def hold(coro, _name):
        coro.close()
        held.append(_name)

    monkeypatch.setattr(simulation_turns, "schedule_turn_execution", hold)

    first = _send(client, AGENCY, key="send-1")
    assert first.status_code == 200

    second = _send(client, AGENCY, key="send-2")

    assert second.status_code == 409
    assert second.headers["X-Simulation-Turn-Id"] == first.json()["turn_id"]
    assert "still running" in second.json()["detail"]


# --- the read routes sit behind the same gate -------------------------------


def test_another_tenants_turn_is_not_readable_even_with_its_id(client, pipeline):
    turn_id = _send(client, AGENCY).json()["turn_id"]

    stolen = client.get(
        f"/creator/creator-1/fan/fan-test/simulation/turn/{turn_id}",
        headers=_headers(OUTSIDER),
    )

    assert stolen.status_code == 404
    assert stolen.json() == {"detail": "Resource not found"}


def test_an_unknown_turn_id_is_indistinguishable_from_a_forbidden_one(client, pipeline):
    turn_id = _send(client, AGENCY).json()["turn_id"]

    unknown = _poll(client, AGENCY, "00000000-0000-0000-0000-000000000000")
    forbidden = client.get(
        f"/creator/creator-1/fan/fan-test/simulation/turn/{turn_id}",
        headers=_headers(OUTSIDER),
    )

    assert unknown.status_code == forbidden.status_code == 404
    assert unknown.text == forbidden.text


def test_a_turn_id_from_one_fan_cannot_be_read_through_another(client, pipeline):
    """Scoped by BOTH ids, so a turn cannot be walked across conversations."""
    turn_id = _send(client, AGENCY).json()["turn_id"]

    crossed = client.get(
        f"/creator/creator-1/fan/fan-real/simulation/turn/{turn_id}",
        headers=_headers(AGENCY),
    )

    # fan-real is not a test fan at all, so this is the test-fan boundary as
    # well as the scoping one — both give the same 404.
    assert crossed.status_code == 404


@pytest.mark.parametrize("path", ["simulation/turn", "simulation/turn/some-id"])
def test_the_read_routes_disappear_when_the_simulator_is_off(
    client, monkeypatch, path
):
    monkeypatch.setenv("AUTO_SIMULATION_ENABLED", "false")

    response = client.get(
        f"/creator/creator-1/fan/fan-test/{path}", headers=_headers(AGENCY)
    )

    assert response.status_code == 404


def test_an_agency_poll_is_redacted_and_an_owner_poll_is_not(client, pipeline):
    """Same boundary as the synchronous response carried, on the new surface."""
    agency_turn = _send(client, AGENCY, key="agency-1").json()["turn_id"]
    agency = _poll(client, AGENCY, agency_turn)
    assert agency.json()["creator_messages"][0]["media_context"]["ai_stack"] == {
        "profile": "cleo_v3"
    }
    assert "kimi" not in agency.text.lower()

    owner_turn = _send(client, OWNER, key="owner-1").json()["turn_id"]
    owner = _poll(client, OWNER, owner_turn)
    assert (
        owner.json()["creator_messages"][0]["media_context"]["ai_stack"]["model"]
        == "kimi"
    )
