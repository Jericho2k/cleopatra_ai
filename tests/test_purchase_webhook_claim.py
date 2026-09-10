"""REL-003, handler half — the ppv.purchased route actually uses the ledger.

tests/test_purchase_idempotency.py proves the database guarantee. These tests
prove the webhook is wired to it, which is the part a correct migration cannot
enforce on its own:

  * a duplicate claim short-circuits BEFORE any downstream effect;
  * every path that returns without recording a purchase releases the claim,
    because a claim left behind for work that never happened would make the
    platform's redelivery a silent no-op and lose the sale;
  * a deployment that has not applied the migration yet keeps working.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

import main
from tests.fake_supabase import FakeSupabase

CREATOR_ID = "creator-1"
ACCOUNT_ID = "apifansly-account-1"
PLATFORM_FAN_ID = "platform-fan-1"
FAN_ID = "fan-1"
ORDER_ID = "order-1"
MEDIA_ID = "media-1"


@pytest.fixture
def env(monkeypatch):
    state: dict = {
        "claims": [],
        "completions": [],
        "releases": [],
        "claim_result": "claimed",
        "recorded": [],
        "record_raises": False,
        "sales_log": [],
        "pending": {"media_ids": [MEDIA_ID], "price_cents": 1000},
    }

    db = FakeSupabase({
        "creators": [{
            "id": CREATOR_ID,
            "apifansly_account_id": ACCOUNT_ID,
            "fansly_account_id": "creator-platform-1",
            "auto_mode": False,
            "auto_mode_new_fans": False,
        }],
        "fans": [{
            "id": FAN_ID,
            "creator_id": CREATOR_ID,
            "platform_fan_id": PLATFORM_FAN_ID,
            "pending_ppv_check": state["pending"],
            "sales_log": state["sales_log"],
        }],
        "ppv_deliveries": [],
    })
    monkeypatch.setattr(main, "get_supabase", lambda: db)
    state["db"] = db

    async def fake_claim(*, creator_id, platform_order_id, **_kwargs):
        state["claims"].append((creator_id, platform_order_id))
        return state["claim_result"]

    async def fake_complete(creator_id, platform_order_id):
        state["completions"].append((creator_id, platform_order_id))

    async def fake_release(creator_id, platform_order_id):
        state["releases"].append((creator_id, platform_order_id))

    monkeypatch.setattr(main, "_claim_platform_purchase", fake_claim)
    monkeypatch.setattr(main, "_complete_platform_purchase", fake_complete)
    monkeypatch.setattr(main, "_release_platform_purchase", fake_release)

    import services.suggestions as suggestions

    async def fake_record(fan_id, media_id, price, **kwargs):
        if state["record_raises"]:
            raise RuntimeError("downstream commercial write failed")
        state["recorded"].append((fan_id, media_id, price))

    monkeypatch.setattr(suggestions, "record_ppv_purchase", fake_record)

    monkeypatch.setattr(
        main, "normalize_media_ids", lambda ids: [str(i) for i in ids if i]
    )
    return state


def _deliver(*, order_id=ORDER_ID, media_id=MEDIA_ID, price_cents=1000):
    """Invoke the ppv.purchased branch directly.

    The route body is reached through the module function rather than an HTTP
    client so the test is about purchase identity, not signature verification —
    which has its own tests.
    """
    request = SimpleNamespace(
        body=_body(order_id, media_id, price_cents),
        headers={},
    )
    return asyncio.run(main.fansly_webhook(request))


def _body(order_id, media_id, price_cents):
    import json

    payload = json.dumps({
        "event": "ppv.purchased",
        "accountId": ACCOUNT_ID,
        "data": {
            "accountId": PLATFORM_FAN_ID,
            "accountMediaId": media_id,
            "orderId": order_id,
            "orderMetadata": {"accountMediaPrice": price_cents},
        },
    }).encode()

    async def _read():
        return payload

    return _read


@pytest.fixture(autouse=True)
def _dev_webhook(monkeypatch):
    """Signature verification is a separate concern with its own tests."""
    monkeypatch.delenv("APIFANSLY_WEBHOOK_SECRET", raising=False)
    monkeypatch.delenv("WEBHOOK_SECRET", raising=False)
    monkeypatch.setattr(main, "_is_dev", lambda: True)


def test_a_first_delivery_claims_records_and_settles(env) -> None:
    result = _deliver()

    assert result["status"] == "ok"
    assert env["claims"] == [(CREATOR_ID, ORDER_ID)]
    assert len(env["recorded"]) == 1
    assert env["completions"] == [(CREATOR_ID, ORDER_ID)]
    assert env["releases"] == []


def test_a_duplicate_delivery_does_no_downstream_work(env) -> None:
    """The whole finding: the second concurrent delivery must not apply spend,
    lifecycle, PPV purchase or follow-up cancellation."""
    env["claim_result"] = "duplicate"

    result = _deliver()

    assert result["status"] == "duplicate"
    assert env["recorded"] == [], "a duplicate applied a second purchase"
    assert env["completions"] == []
    assert env["releases"] == []


def test_an_unmatched_price_releases_the_claim(env) -> None:
    """Otherwise a redelivery of a genuinely unprocessed order would be told it
    is a duplicate and the sale would be lost."""
    result = _deliver(price_cents=999_00)

    assert result["status"] == "unmatched_ppv_purchase"
    assert env["recorded"] == []
    assert env["releases"] == [(CREATOR_ID, ORDER_ID)]


def test_an_unmatched_media_releases_the_claim(env) -> None:
    result = _deliver(media_id="a-different-media")

    assert result["status"] == "unmatched_ppv_purchase"
    assert env["recorded"] == []
    assert env["releases"] == [(CREATOR_ID, ORDER_ID)]


def test_a_failed_downstream_write_releases_the_claim(env) -> None:
    env["record_raises"] = True

    with pytest.raises(RuntimeError):
        _deliver()

    assert env["releases"] == [(CREATOR_ID, ORDER_ID)]
    assert env["completions"] == []


def test_an_order_already_in_sales_log_is_a_duplicate(env) -> None:
    """Orders processed before this migration exist only in sales_log. They must
    keep deduplicating, and the ledger should adopt the identity so the scan
    stops being needed."""
    env["db"].tables["fans"][0]["sales_log"] = [
        {"platform_order_id": ORDER_ID, "media_ids": [MEDIA_ID]}
    ]

    result = _deliver()

    assert result["status"] == "duplicate"
    assert env["recorded"] == []
    assert env["completions"] == [(CREATOR_ID, ORDER_ID)]
    assert env["releases"] == []


def test_a_deployment_without_the_migration_still_processes(env) -> None:
    """Rolling deploy in either order. Without the ledger the handler falls back
    to the pre-existing sales_log scan rather than refusing the sale."""
    env["claim_result"] = "unavailable"

    result = _deliver()

    assert result["status"] == "ok"
    assert len(env["recorded"]) == 1
    # Nothing to settle or release: no claim was ever taken.
    assert env["completions"] == []
    assert env["releases"] == []


def test_an_event_without_an_order_id_is_still_processed(env) -> None:
    """No platform identity to deduplicate on; never invent one, and never drop
    the sale because of it."""
    result = _deliver(order_id="")

    assert result["status"] == "ok"
    assert env["claims"] == [], "claimed on an event with no order id"
    assert len(env["recorded"]) == 1
