"""One simulated fan turn must be processed once, by the simulator, only.

The simulator persists its fan message with an ordinary INSERT into
``messages``. That is precisely what the production Supabase database webhook
fires on, so the row also arrived at POST /generate-suggestions and ran the
whole ordinary inbound pipeline — situation analysis, commercial state, price
learning, the conversation director — alongside the Full Auto turn the
simulator was itself driving. Railway showed it plainly: an
``[AUTO MODE] effective_auto=False reason=no_approved_sets`` block from the
webhook path, then a second, unrelated ``[WRITER ROUTE] mode=auto`` from the
simulator. Every commercial reading the simulator produced was measuring two
overlapping passes.

The production webhook is not disabled and ordinary fan messages are not
skipped. Only a row the simulator explicitly marked is ignored — and it is
still answered 2xx, so Supabase records the delivery as handled instead of
retrying it forever.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import main
from core.simulation import is_simulation_message, simulation_message_marker


@pytest.fixture
def webhook(monkeypatch):
    """The real /generate-suggestions route with the pipeline behind it spied."""
    processed: list[tuple] = []

    async def fake_process(fan_id, creator_id, content, auto_mode, message_id):
        processed.append((fan_id, creator_id, content, auto_mode, message_id))

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
            return SimpleNamespace(data={"auto_mode": True})

    monkeypatch.setenv("WEBHOOK_SECRET", "test-webhook-secret")
    monkeypatch.setattr(main, "process_incoming_fan_message", fake_process)
    monkeypatch.setattr(main, "get_supabase", _Creators)
    main._processed_messages.clear()
    return TestClient(app=main.app), processed


def _post(client, record):
    return client.post(
        "/generate-suggestions",
        headers={"x-webhook-secret": "test-webhook-secret"},
        json={"type": "INSERT", "record": record},
    )


def _row(**overrides):
    row = {
        "id": "msg-1",
        "fan_id": "fan-test",
        "creator_id": "creator-1",
        "role": "fan",
        "content": "hii",
    }
    row.update(overrides)
    return row


# --- the simulator's own event is ignored, and answered 2xx -----------------


def test_simulator_marked_insert_is_ignored_by_the_webhook(webhook):
    client, processed = webhook

    response = _post(client, _row(media_context=simulation_message_marker()))

    assert response.status_code == 200, "Supabase must not be asked to retry"
    assert response.json() == {"status": "skipped - owner simulation"}
    assert processed == [], "the simulator is the sole processor of its own event"


def test_marker_delivered_as_raw_json_text_is_still_ignored(webhook):
    """jsonb may arrive decoded or as text; a transport detail must not decide
    whether a simulated turn is processed twice."""
    import json

    client, processed = webhook

    response = _post(
        client,
        _row(media_context=json.dumps(simulation_message_marker())),
    )

    assert response.status_code == 200
    assert response.json() == {"status": "skipped - owner simulation"}
    assert processed == []


def test_the_simulator_and_the_webhook_agree_on_the_marker():
    """The producer and the consumer must not drift apart."""
    assert is_simulation_message(simulation_message_marker()) is True


# --- ordinary production traffic is completely unchanged --------------------


def test_ordinary_fan_message_is_still_processed(webhook):
    client, processed = webhook

    response = _post(client, _row())

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert processed == [("fan-test", "creator-1", "hii", True, "msg-1")]


@pytest.mark.parametrize(
    "media_context",
    [
        None,
        {},
        {"media_ids": ["m-1"]},
        # Shaped like the marker but not it: neither half alone may skip a turn.
        {"simulation": True},
        {"simulation_source": "owner_auto_simulator"},
        {"simulation": "true", "simulation_source": "owner_auto_simulator"},
        {"simulation": True, "simulation_source": "something_else"},
        "not json at all",
    ],
)
def test_only_the_explicit_marker_skips_a_message(webhook, media_context):
    """Never by shape, never by guesswork: an ordinary production fan message
    that happens to carry metadata must still be processed."""
    client, processed = webhook

    response = _post(client, _row(media_context=media_context))

    assert response.json() == {"status": "ok"}
    assert len(processed) == 1


def test_a_real_fan_on_a_test_prefixed_account_is_not_skipped(webhook, monkeypatch):
    """The ``test_`` prefix says a fan is SIMULATABLE, not that this particular
    message came from the simulator. Typing into a test fan's chat by hand must
    still run the ordinary pipeline."""
    client, processed = webhook

    response = _post(client, _row(fan_id="fan-test", media_context=None))

    assert response.json() == {"status": "ok"}
    assert len(processed) == 1


def test_existing_webhook_skips_are_unchanged(webhook):
    """The pre-existing escape hatches must not have been disturbed."""
    client, processed = webhook

    assert _post(client, _row(role="creator")).json() == {"status": "skipped"}
    assert _post(
        client, _row(id="msg-2", fansly_message_id="plat-9")
    ).json() == {"status": "skipped - handled by fansly webhook"}

    assert _post(client, _row(id="msg-3")).json() == {"status": "ok"}
    assert _post(client, _row(id="msg-3")).json() == {"status": "duplicate"}

    non_insert = client.post(
        "/generate-suggestions",
        headers={"x-webhook-secret": "test-webhook-secret"},
        json={"type": "UPDATE", "record": _row()},
    )
    assert non_insert.json() == {"status": "skipped"}


# ---------------------------------------------------------------------------
# The REAL Supabase database-webhook payload
# ---------------------------------------------------------------------------
#
# Railway kept showing one simulated INSERT followed by a full ordinary pipeline
# pass — [WEBHOOK], [AUTO MODE], [PRICE LEARNING], [SITUATION], [COMMERCIAL] —
# even though the marker check above passes its own unit tests. The lesson is
# that those tests assert against a payload WE invented. Supabase builds the
# real one in ``supabase_functions.http_request``: five top-level keys, the row
# under ``record``, and a jsonb column whose transport encoding is not ours to
# assume.
#
# So ownership no longer rests on the payload's shape at all. It is established
# three ways, cheapest first, and the last one asks the database.


def supabase_webhook_payload(record: dict, *, event: str = "INSERT") -> dict:
    """The payload shape ``supabase_functions.http_request`` actually POSTs."""
    return {
        "type": event,
        "table": "messages",
        "schema": "public",
        "record": record,
        "old_record": None,
    }


def _post_payload(client, payload):
    return client.post(
        "/generate-suggestions",
        headers={"x-webhook-secret": "test-webhook-secret"},
        json=payload,
    )


def _production_row(**overrides) -> dict:
    """Every column a production ``messages`` row carries, as the trigger sends it."""
    row = {
        "id": "3f2a9c1e-0b44-4d8b-9a51-6f0e2c7d8a90",
        "fan_id": "0d0f6f31-1b6f-4a2e-9b1a-2f5c4e8a1234",
        "creator_id": "9a1c2b3d-4e5f-4071-8293-a4b5c6d7e8f9",
        "role": "fan",
        "content": "hey you",
        "sent_at": "2026-09-12T10:00:00+00:00",
        "was_ai_suggested": False,
        "fansly_message_id": None,
        "media_context": None,
    }
    row.update(overrides)
    return row


def test_the_real_supabase_payload_with_the_marker_is_skipped(webhook):
    client, processed = webhook

    response = _post_payload(
        client,
        supabase_webhook_payload(
            _production_row(media_context=simulation_message_marker())
        ),
    )

    assert response.status_code == 200
    assert response.json() == {"status": "skipped - owner simulation"}
    assert processed == []


def test_the_real_supabase_payload_without_a_marker_is_processed(webhook):
    """The production path is completely unchanged."""
    client, processed = webhook

    response = _post_payload(client, supabase_webhook_payload(_production_row()))

    assert response.json() == {"status": "ok"}
    assert len(processed) == 1


def test_a_record_with_no_media_context_key_asks_the_database(webhook, monkeypatch):
    """The failure mode that survived the first fix.

    A payload that simply does not mention ``media_context`` is not evidence
    that the row has none. Reading that absence as "ordinary message" is what
    ran a simulated turn twice. The row itself is authoritative.
    """
    client, processed = webhook

    asked: list[str] = []

    async def fake_row_lookup(message_id):
        asked.append(str(message_id))
        return True

    monkeypatch.setattr(
        "core.simulation.message_row_is_simulation_owned", fake_row_lookup
    )

    row = _production_row()
    row.pop("media_context")
    response = _post_payload(client, supabase_webhook_payload(row))

    assert asked == [row["id"]], "the database must be consulted, not guessed at"
    assert response.json() == {"status": "skipped - owner simulation"}
    assert processed == []


def test_the_database_answering_no_still_processes_the_message(webhook, monkeypatch):
    client, processed = webhook

    async def fake_row_lookup(_message_id):
        return False

    monkeypatch.setattr(
        "core.simulation.message_row_is_simulation_owned", fake_row_lookup
    )

    row = _production_row()
    row.pop("media_context")
    response = _post_payload(client, supabase_webhook_payload(row))

    assert response.json() == {"status": "ok"}
    assert len(processed) == 1


def test_a_failed_database_read_falls_back_to_ordinary_processing(webhook, monkeypatch):
    """Never positively identified means never skipped. A broken read must not
    be able to silence a real fan's message."""
    client, processed = webhook

    async def exploding_lookup(_message_id):
        raise RuntimeError("PostgREST connection terminated")

    monkeypatch.setattr(
        "core.simulation.message_row_is_simulation_owned", exploding_lookup
    )

    row = _production_row()
    row.pop("media_context")
    # The helper swallows its own failures; this patches one that does not, so
    # the claim under test is about the ROUTE, not about the helper.
    response = _post_payload(client, supabase_webhook_payload(row))

    assert response.status_code == 200, (
        "an ownership check must never turn a fan message into a 500 and a "
        "Supabase redelivery loop"
    )
    assert response.json() == {"status": "ok"}
    assert len(processed) == 1


def test_the_in_process_registry_skips_without_reading_anything(webhook, monkeypatch):
    """The single-process case Railway actually runs: no payload parsing, no
    database read, just the id the simulator recorded before the turn began."""
    from core.simulation import (
        mark_simulation_owned_message,
        reset_simulation_owned_message_ids,
    )

    client, processed = webhook
    reset_simulation_owned_message_ids()

    async def must_not_be_called(_message_id):
        pytest.fail("the registry answered; nothing else should be consulted")

    monkeypatch.setattr(
        "core.simulation.message_row_is_simulation_owned", must_not_be_called
    )

    row = _production_row()
    row.pop("media_context")
    mark_simulation_owned_message(row["id"])

    assert _post_payload(client, supabase_webhook_payload(row)).json() == {
        "status": "skipped - owner simulation"
    }
    assert processed == []
    reset_simulation_owned_message_ids()


def test_the_registry_is_bounded_and_keeps_the_newest_ids():
    """A long-lived process must not accumulate ids forever; the database
    fallback covers anything evicted."""
    from core.simulation import (
        _OWNED_ID_LIMIT,
        is_simulation_owned_message_id,
        mark_simulation_owned_message,
        reset_simulation_owned_message_ids,
    )

    reset_simulation_owned_message_ids()
    for index in range(_OWNED_ID_LIMIT + 10):
        mark_simulation_owned_message(f"msg-{index}")

    assert is_simulation_owned_message_id("msg-0") is False
    assert is_simulation_owned_message_id(f"msg-{_OWNED_ID_LIMIT + 9}") is True
    reset_simulation_owned_message_ids()


def test_double_encoded_jsonb_is_still_recognised():
    """One hop serialises the jsonb, the next string-encodes it. Failing to
    unwrap twice is a silent "not a simulation" — the exact class of transport
    assumption that made the first fix look correct in tests and fail in prod."""
    import json

    from core.simulation import is_simulation_message

    once = json.dumps(simulation_message_marker())
    twice = json.dumps(once)
    assert is_simulation_message(once) is True
    assert is_simulation_message(twice) is True


def test_the_simulator_registers_its_message_before_running_the_turn():
    """A webhook delivery that lands mid-analysis must already be covered."""
    source = (
        Path(__file__).resolve().parents[1] / "services" / "suggestions.py"
    ).read_text(encoding="utf-8")

    mark_at = source.index("mark_simulation_owned_message(fan_message_id)")
    turn_at = source.index("with simulation_scope():")
    assert mark_at < turn_at
