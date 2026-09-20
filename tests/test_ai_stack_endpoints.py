"""Who may administer the AI stack, and what a client is allowed to send.

These routes share the simulator's gate, which is what "AI stack selection is
part of the simulator" means here — so as the simulator opened to agency
operators, so did these, for the creators those operators already hold.
Comparing profiles turn for turn is the reason to simulate at all.

Two boundaries survive that widening and are what this file asserts:

* a profile choice reaches ONE creator, decided by the ordinary tenancy check.
  An operator cannot read or pin a creator it is not assigned, and a fan-level
  pin still requires a ``test_`` fan of that creator;
* there is no way for any client — owner included — to submit a provider or a
  model. The only thing that crosses the wire is a stable profile identifier,
  validated against the backend registry.

And one the widening added: an agency operator is told a profile's id and its
display name, and nothing about what it routes to. Provider names, model
identifiers, fallback chains, stage routing, prompt versions and generation
configuration are platform-owner diagnostics, redacted on the RESPONSE rather
than merely left unrendered by the dashboard.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import main
from ai.stack_profiles import PROFILE_IDS
from core import tenancy


OWNER = "11111111-1111-4111-8111-111111111111"
AGENCY = "22222222-2222-4222-8222-222222222222"

ASSIGNMENTS = {OWNER: {"creator-1"}, AGENCY: {"creator-1"}}

FANS = {
    "fan-test": {
        "id": "fan-test",
        "creator_id": "creator-1",
        "platform_fan_id": "test_abc",
        "display_name": "Test fan",
        "ai_stack_profile": None,
        "conversation_core": None,
    },
    "fan-real": {
        "id": "fan-real",
        "creator_id": "creator-1",
        "platform_fan_id": "884422113355",
        "display_name": "Real fan",
        "ai_stack_profile": None,
        "conversation_core": None,
    },
}

CREATORS = {
    "creator-1": {
        "id": "creator-1",
        "ai_stack_profile": None,
        "conversation_core": None,
    }
}


class _Table:
    def __init__(self, store, name):
        self.store = store
        self.name = name
        self.rows = store.creators if name == "creators" else store.fans
        self.filters: list[tuple[str, str]] = []
        self.payload: dict | None = None

    def select(self, *_a, **_k):
        return self

    def update(self, payload):
        self.payload = payload
        return self

    def eq(self, column, value):
        self.filters.append((column, str(value)))
        return self

    def limit(self, *_a, **_k):
        return self

    def single(self):
        return self

    def execute(self):
        matched = [
            row
            for row in self.rows.values()
            if all(str(row.get(col)) == val for col, val in self.filters)
        ]
        if self.payload is not None:
            for row in matched:
                row.update(self.payload)
            self.store.writes.append(dict(self.payload))
        return SimpleNamespace(data=[dict(row) for row in matched])


class _DB:
    def __init__(self):
        self.creators = {k: dict(v) for k, v in CREATORS.items()}
        self.fans = {k: dict(v) for k, v in FANS.items()}
        self.writes: list[dict] = []

    def table(self, name):
        return _Table(self, name)


@pytest.fixture
def store():
    return _DB()


@pytest.fixture
def client(monkeypatch, store):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("DASHBOARD_API_SECRET", "test-dashboard-secret")
    monkeypatch.setenv("AUTO_SIMULATION_ENABLED", "true")
    monkeypatch.setenv("AUTO_SIMULATION_ALLOWED_USER_IDS", OWNER)
    monkeypatch.setenv("AI_STACK_PROFILE", "cleo_v2")
    monkeypatch.setenv("AI_STACK_CACHE_SECONDS", "0")
    monkeypatch.setenv("CONVERSATION_CORE_CACHE_SECONDS", "0")

    from services import ai_stack, conversation_core

    ai_stack.clear_ai_stack_cache()
    conversation_core.clear_conversation_core_cache()

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
    monkeypatch.setattr(ai_stack, "get_supabase", lambda: store)
    monkeypatch.setattr(conversation_core, "get_supabase", lambda: store)
    client = TestClient(app=main.app)
    yield client
    ai_stack.clear_ai_stack_cache()
    conversation_core.clear_conversation_core_cache()


def headers(user: str) -> dict[str, str]:
    return {"X-API-Key": "test-dashboard-secret", "Authorization": f"Bearer {user}"}


# --- who may read the registry ---------------------------------------------


def test_the_owner_can_read_every_profile(client):
    response = client.get("/ai-stack/profiles", headers=headers(OWNER))

    assert response.status_code == 200
    body = response.json()
    # The registry is served from ai/stack_profiles.py rather than restated
    # here, so adding a profile makes it selectable without touching a second
    # list. The two frozen ones are asserted by name because a comparison
    # baseline disappearing is exactly the regression worth catching.
    served = {row["profile_id"] for row in body["profiles"]}
    assert served == set(PROFILE_IDS)
    assert {"cleo_legacy_v1", "cleo_v2"} <= served
    assert body["environment_profile"] == "cleo_v2"
    assert body["environment_variable"] == "AI_STACK_PROFILE"


def test_an_agency_account_sees_the_registry_it_must_choose_from(client):
    """The registry is deployment configuration — the same list for every
    tenant, naming no creator, fan or sale — and an operator cannot pick a
    profile without being able to read what the choices are."""
    response = client.get("/ai-stack/profiles", headers=headers(AGENCY))

    assert response.status_code == 200, response.text
    assert response.json()["profiles"]


def test_the_registry_disappears_when_the_simulator_is_off(client, monkeypatch):
    monkeypatch.setenv("AUTO_SIMULATION_ENABLED", "false")

    assert client.get("/ai-stack/profiles", headers=headers(AGENCY)).status_code == 404
    assert client.get("/ai-stack/profiles", headers=headers(OWNER)).status_code == 404


def test_the_registry_never_returns_an_api_key(client):
    body = client.get("/ai-stack/profiles", headers=headers(OWNER)).text

    assert "API_KEY" not in body
    assert "sk-" not in body


# --- selecting the conversational architecture ----------------------------


def test_only_the_platform_owner_can_read_conversation_cores(client):
    owner = client.get("/conversation-cores", headers=headers(OWNER))
    agency = client.get("/conversation-cores", headers=headers(AGENCY))

    assert owner.status_code == 200
    assert {row["id"] for row in owner.json()["cores"]} == {
        "legacy",
        "semantic_v1",
        "semantic_v2",
    }
    assert agency.status_code == 403


def test_owner_can_select_and_rollback_creator_conversation_core(client, store):
    selected = client.put(
        "/creator/creator-1/conversation-core",
        headers=headers(OWNER),
        json={"conversation_core": "semantic_v1"},
    )
    assert selected.status_code == 200, selected.text
    assert selected.json()["effective"]["conversation_core"] == "semantic_v1"
    assert store.creators["creator-1"]["conversation_core"] == "semantic_v1"

    rolled_back = client.put(
        "/creator/creator-1/conversation-core",
        headers=headers(OWNER),
        json={"conversation_core": None},
    )
    assert rolled_back.status_code == 200, rolled_back.text
    assert rolled_back.json()["effective"]["conversation_core"] == "legacy"


def test_agency_cannot_change_conversation_architecture(client, store):
    response = client.put(
        "/creator/creator-1/conversation-core",
        headers=headers(AGENCY),
        json={"conversation_core": "semantic_v1"},
    )

    assert response.status_code == 403
    assert store.creators["creator-1"]["conversation_core"] is None


def test_owner_can_pin_only_a_test_fan_to_the_new_core(client, store):
    selected = client.put(
        "/creator/creator-1/fan/fan-test/conversation-core",
        headers=headers(OWNER),
        json={"conversation_core": "semantic_v1"},
    )
    refused = client.put(
        "/creator/creator-1/fan/fan-real/conversation-core",
        headers=headers(OWNER),
        json={"conversation_core": "semantic_v1"},
    )

    assert selected.status_code == 200, selected.text
    assert selected.json()["effective"]["conversation_core"] == "semantic_v1"
    assert store.fans["fan-test"]["conversation_core"] == "semantic_v1"
    assert refused.status_code in {403, 404}
    assert store.fans["fan-real"]["conversation_core"] is None


# --- who may change a creator's brain --------------------------------------


def test_an_agency_account_may_set_its_own_creators_override(client, store):
    assert client.get("/creator/creator-1/ai-stack", headers=headers(AGENCY)).status_code == 200
    response = client.put(
        "/creator/creator-1/ai-stack",
        headers=headers(AGENCY),
        json={"ai_stack_profile": "cleo_legacy_v1"},
    )

    assert response.status_code == 200, response.text
    assert store.creators["creator-1"]["ai_stack_profile"] == "cleo_legacy_v1"


def test_an_agency_account_cannot_reach_another_tenants_creator(client, store):
    """The widening was about WHO may use these routes, never about WHICH
    creators they reach. Tenancy is untouched and still decides."""
    assert client.get("/creator/creator-2/ai-stack", headers=headers(AGENCY)).status_code == 404
    response = client.put(
        "/creator/creator-2/ai-stack",
        headers=headers(AGENCY),
        json={"ai_stack_profile": "cleo_legacy_v1"},
    )

    assert response.status_code == 404
    # And nothing was written on the way to the refusal.
    assert store.writes == []
    assert store.creators["creator-1"]["ai_stack_profile"] is None


def test_the_owner_can_set_and_clear_a_creator_override(client, store):
    response = client.put(
        "/creator/creator-1/ai-stack",
        headers=headers(OWNER),
        json={"ai_stack_profile": "cleo_legacy_v1"},
    )

    assert response.status_code == 200
    assert response.json()["override"] == "cleo_legacy_v1"
    assert response.json()["effective"]["ai_stack_profile"] == "cleo_legacy_v1"
    assert response.json()["effective"]["ai_stack_source"] == "creator"
    assert store.creators["creator-1"]["ai_stack_profile"] == "cleo_legacy_v1"

    cleared = client.put(
        "/creator/creator-1/ai-stack",
        headers=headers(OWNER),
        json={"ai_stack_profile": None},
    )

    assert cleared.json()["override"] is None
    assert cleared.json()["effective"]["ai_stack_profile"] == "cleo_v2"


@pytest.mark.parametrize(
    "value",
    [
        "cleo_v99",
        "moonshotai/kimi-k2.6",
        "openrouter",
        "anthropic:claude-opus-5",
        "../../etc/passwd",
    ],
)
def test_an_arbitrary_provider_or_model_string_is_rejected(client, store, value):
    response = client.put(
        "/creator/creator-1/ai-stack",
        headers=headers(OWNER),
        json={"ai_stack_profile": value},
    )

    assert response.status_code == 400
    assert store.creators["creator-1"]["ai_stack_profile"] is None


def test_there_is_no_field_for_a_provider_or_a_model(client, store):
    """Even an owner cannot name one: the extra keys are simply not read."""
    response = client.put(
        "/creator/creator-1/ai-stack",
        headers=headers(OWNER),
        json={
            "ai_stack_profile": "cleo_v2",
            "provider": "openai",
            "model": "gpt-4o",
            "writer_model": "whatever",
        },
    )

    assert response.status_code == 200
    assert store.creators["creator-1"]["ai_stack_profile"] == "cleo_v2"
    assert all("model" not in write for write in store.writes)
    assert all("provider" not in write for write in store.writes)


# --- the simulation fan override -------------------------------------------


def test_a_test_fan_can_be_pinned_to_a_profile(client, store):
    response = client.put(
        "/creator/creator-1/fan/fan-test/ai-stack",
        headers=headers(OWNER),
        json={"ai_stack_profile": "cleo_legacy_v1"},
    )

    assert response.status_code == 200
    assert store.fans["fan-test"]["ai_stack_profile"] == "cleo_legacy_v1"


def test_a_real_fan_cannot_be_pinned_at_all(client, store):
    response = client.put(
        "/creator/creator-1/fan/fan-real/ai-stack",
        headers=headers(OWNER),
        json={"ai_stack_profile": "cleo_legacy_v1"},
    )

    assert response.status_code == 404
    assert store.fans["fan-real"]["ai_stack_profile"] is None


def test_an_agency_account_may_pin_its_own_test_fan(client, store):
    response = client.put(
        "/creator/creator-1/fan/fan-test/ai-stack",
        headers=headers(AGENCY),
        json={"ai_stack_profile": "cleo_legacy_v1"},
    )

    assert response.status_code == 200, response.text
    assert store.fans["fan-test"]["ai_stack_profile"] == "cleo_legacy_v1"


def test_an_agency_account_cannot_pin_a_real_fan(client, store):
    """The ``test_`` boundary is unchanged for every tier."""
    response = client.put(
        "/creator/creator-1/fan/fan-real/ai-stack",
        headers=headers(AGENCY),
        json={"ai_stack_profile": "cleo_legacy_v1"},
    )

    assert response.status_code == 404
    assert store.fans["fan-real"]["ai_stack_profile"] is None


def test_every_refusal_is_the_same_indistinguishable_404(client):
    """A caller must not be able to tell "not allowed" from "does not exist"."""
    responses = [
        client.get("/creator/creator-2/ai-stack", headers=headers(AGENCY)),
        client.put(
            "/creator/creator-2/ai-stack",
            headers=headers(AGENCY),
            json={"ai_stack_profile": "cleo_v2"},
        ),
        client.put(
            "/creator/creator-1/fan/fan-real/ai-stack",
            headers=headers(AGENCY),
            json={"ai_stack_profile": "cleo_v2"},
        ),
        client.put(
            "/creator/creator-1/fan/fan-missing/ai-stack",
            headers=headers(AGENCY),
            json={"ai_stack_profile": "cleo_v2"},
        ),
    ]

    assert [response.status_code for response in responses] == [404, 404, 404, 404]
    assert {response.json()["detail"] for response in responses} == {"Resource not found"}


# --- what each tier is TOLD about a profile ---------------------------------
#
# The simulator opened to agency operators, and with it the AI stack selector.
# Selecting a stack needs a stable id and a display name. It does not need the
# platform's supply chain, and an agency must not receive it: no provider, no
# model identifier, no fallback chain, no stage routing, no prompt version and
# no generation configuration.
#
# Enforced on the RESPONSE (services/ai_stack_visibility.py), because a field
# the dashboard declines to render has still been delivered to the browser.

# Every internal identity that appears somewhere in the registry. Asserted as
# substrings of the whole serialised body, so a leak through a key nobody
# thought of — a new stage field, a summary rewritten to name a model — fails
# this test rather than shipping.
ROUTING_IDENTITIES = (
    "openrouter",
    "together",
    "anthropic",
    "kimi",
    "qwen",
    "moonshotai",
    "glm",
    "llama",
    "claude",
    "gpt-oss",
    "writer_v1",
    "writer_v2",
    "writer_v3",
    "analyzer_v1",
)


def test_an_agency_account_gets_profile_ids_and_display_names_only(client):
    body = client.get("/ai-stack/profiles", headers=headers(AGENCY)).json()

    # The same registry, in the same order, with the same ids: an agency picks
    # from exactly what the owner does.
    assert [row["id"] for row in body["profiles"]] == list(PROFILE_IDS)
    # And nothing else at all. Asserted as an exact key set rather than a list
    # of absences, so a field added to the public view later has to be a
    # deliberate decision made here.
    assert all(set(row) == {"id", "name"} for row in body["profiles"])
    assert {"cleo_v3": "Cleo V3"}.items() <= {
        row["id"]: row["name"] for row in body["profiles"]
    }.items()
    assert body["diagnostics"] is False
    # The deployment default is a profile id — product-level in exactly the way
    # the ids above are. The NAME of the variable behind it is not.
    assert body["environment_profile"] == "cleo_v2"
    assert "environment_variable" not in body


def test_an_agency_account_cannot_obtain_routing_through_the_registry(client):
    raw = client.get("/ai-stack/profiles", headers=headers(AGENCY)).text.lower()

    for identity in ROUTING_IDENTITIES:
        assert identity not in raw, f"agency registry leaked {identity!r}"
    for field in ("provider", "model", "fallback", "stages", "prompt_version"):
        assert field not in raw, f"agency registry leaked {field!r}"


def test_the_owner_still_gets_the_full_diagnostics(client):
    body = client.get("/ai-stack/profiles", headers=headers(OWNER)).json()

    assert body["diagnostics"] is True
    assert body["environment_variable"] == "AI_STACK_PROFILE"
    by_id = {row["profile_id"]: row for row in body["profiles"]}
    assert set(by_id) == set(PROFILE_IDS)

    v3 = by_id["cleo_v3"]
    assert v3["summary"]
    writer = next(row for row in v3["stages"] if row["stage"] == "writer_default")
    assert writer["provider"] == "openrouter"
    assert writer["model"] == "moonshotai/kimi-k2.6"
    assert writer["fallback_provider"] == "together"
    assert writer["fallback_model"] == "Qwen/Qwen3.7-Plus"
    assert writer["prompt_version"] == "writer_v3"
    # Every stage, not just the writer: the owner's view is the whole stack.
    assert len(v3["stages"]) == 7


def test_an_agency_account_can_still_select_cleo_v3(client, store):
    """The reduced representation must remain a working selector.

    The id an agency reads out of the registry is the id it sends back, and the
    backend validates it against the same registry. Nothing about redacting the
    routing changes what may be chosen.
    """
    listed = client.get("/ai-stack/profiles", headers=headers(AGENCY)).json()
    chosen = next(row for row in listed["profiles"] if row["name"] == "Cleo V3")

    creator = client.put(
        "/creator/creator-1/ai-stack",
        headers=headers(AGENCY),
        json={"ai_stack_profile": chosen["id"]},
    )
    assert creator.status_code == 200, creator.text
    assert store.creators["creator-1"]["ai_stack_profile"] == "cleo_v3"
    assert creator.json()["effective"]["ai_stack_profile"] == "cleo_v3"

    # And the Simulator's own control — pinning one test fan to a stack.
    fan = client.put(
        "/creator/creator-1/fan/fan-test/ai-stack",
        headers=headers(AGENCY),
        json={"ai_stack_profile": chosen["id"]},
    )
    assert fan.status_code == 200, fan.text
    assert store.fans["fan-test"]["ai_stack_profile"] == "cleo_v3"

    # Confirming the choice back to the agency names the profile and still not
    # what it routes to.
    assert fan.json()["ai_stack_profile"] == "cleo_v3"
    assert "kimi" not in fan.text.lower()


def test_the_agency_registry_is_not_simply_the_owners_with_keys_hidden(client):
    """Same profiles, different depth — asserted against each other."""
    owner = client.get("/ai-stack/profiles", headers=headers(OWNER)).json()
    agency = client.get("/ai-stack/profiles", headers=headers(AGENCY)).json()

    assert [row["id"] for row in agency["profiles"]] == [
        row["profile_id"] for row in owner["profiles"]
    ]
    assert [row["name"] for row in agency["profiles"]] == [
        row["label"] for row in owner["profiles"]
    ]


# --- the same boundary on the turn a simulated message came back from -------


def test_an_agency_turn_reports_the_profile_and_not_the_model(
    client, monkeypatch
):
    """The durable marker on a creator message names the model that wrote it.

    Persisted deliberately (services.suggestions.message_ai_stack_metadata) so
    "which brain produced this?" is answerable months later from the row alone.
    It is diagnostics, so it is redacted on the way out to an agency — and the
    rest of the message's media_context, which is ordinary product state, is
    not touched.

    Asserted on the POLL response rather than the POST, because that is where
    the turn's creator messages now come back: a simulated turn is durable and
    pollable, so the POST returns a turn id and the transcript arrives with the
    completed turn. The boundary is the same one, on the surface that now
    carries it.
    """
    from services import simulation_turns, suggestions

    message_row = {
        "id": "message-1",
        "role": "creator",
        "content": "hey you",
        "sent_at": None,
        "media_context": {
            "ppv": {"media_ids": ["media-1"], "price_cents": 2500},
            "ai_stack": {
                "profile": "cleo_v3",
                "route": "commercial_complex",
                "prompt_version": "writer_v3",
                "provider": "openrouter",
                "model": "moonshotai/kimi-k2.6",
            },
        },
    }

    async def fake_turn(**_kwargs):
        return {
            "status": "ok",
            "simulation": True,
            "fan_message_id": "fan-message-1",
            "outcome": "replied",
            "creator_messages": [message_row],
        }

    async def fake_rows(_fan_id):
        return [message_row]

    monkeypatch.setattr(suggestions, "run_simulated_inbound", fake_turn)
    monkeypatch.setattr(suggestions, "_recent_creator_message_rows", fake_rows)
    monkeypatch.setattr(
        simulation_turns, "_creator_message_ids", lambda _fan_id: _empty_snapshot()
    )

    def _run(user: str, key: str):
        started = client.post(
            "/creator/creator-1/fan/fan-test/simulate-inbound",
            headers=headers(user),
            json={"message": "hi", "fast": True, "idempotency_key": key},
        )
        assert started.status_code == 200, started.text
        assert started.json()["status"] == "processing"
        turn_id = started.json()["turn_id"]
        return client.get(
            f"/creator/creator-1/fan/fan-test/simulation/turn/{turn_id}",
            headers=headers(user),
        )

    agency = _run(AGENCY, "key-agency")
    assert agency.status_code == 200, agency.text
    assert agency.json()["status"] == "completed"
    context = agency.json()["creator_messages"][0]["media_context"]
    assert context["ai_stack"] == {"profile": "cleo_v3"}
    # Untouched: this is a stack-routing boundary, not a general scrubber.
    assert context["ppv"] == {"media_ids": ["media-1"], "price_cents": 2500}
    assert "kimi" not in agency.text.lower()
    assert "openrouter" not in agency.text.lower()

    owner = _run(OWNER, "key-owner")
    assert owner.status_code == 200, owner.text
    owner_marker = owner.json()["creator_messages"][0]["media_context"]["ai_stack"]
    assert owner_marker["provider"] == "openrouter"
    assert owner_marker["model"] == "moonshotai/kimi-k2.6"
    assert owner_marker["route"] == "commercial_complex"
    assert owner_marker["prompt_version"] == "writer_v3"


async def _empty_snapshot():
    """The pre-turn transcript snapshot, for a fan with no history."""
    return set()


# --- and on the health banner, which named the failing model in prose -------


def test_model_health_gives_an_agency_the_verdict_without_the_supply_chain(
    client, monkeypatch
):
    monkeypatch.setattr(
        main,
        "current_model_availability",
        lambda: {
            "status": "degraded",
            "checked_at": "2026-09-15T00:00:00+00:00",
            "detail": "Configured model unavailable: openrouter:moonshotai/kimi-k2.6.",
            "models": [
                {
                    "role": "ordinary_writer",
                    "provider": "openrouter",
                    "model": "moonshotai/kimi-k2.6",
                    "available": False,
                }
            ],
        },
    )

    agency = client.get("/model-runtime-health", headers=headers(AGENCY)).json()
    # It still learns that replies are degraded, and when that was checked.
    assert agency["status"] == "degraded"
    assert agency["checked_at"] == "2026-09-15T00:00:00+00:00"
    assert agency["models"] == []
    assert "kimi" not in agency["detail"].lower()
    assert "openrouter" not in agency["detail"].lower()
    # The analyzer counters name no provider or model and are unredacted.
    assert agency["analyzer"]["window_hours"] == 1

    owner = client.get("/model-runtime-health", headers=headers(OWNER)).json()
    assert owner["models"][0]["model"] == "moonshotai/kimi-k2.6"
    assert "kimi" in owner["detail"].lower()
