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
