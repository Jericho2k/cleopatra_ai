"""Sprint 1 — a confirmed purchase survives every other writer.

Finding I of ``docs/autonomy_architecture_review.md``. ``transition_delivery``
updated ``ppv_deliveries`` by reference with no expected prior status, while
``abandon_delivery_if_active`` in the same module used a status predicate. Four
writers reach that function from different clocks:

* ``services/ppv_delivery.py`` writes ``delivered_pending`` *after* a round trip
  to the platform to verify the payment lock;
* ``record_ppv_purchase`` writes ``purchased`` from a purchase event;
* ``services/ppv_reconciliation.py`` writes ``abandoned`` when the window expires;
* ``services/ppv_recovery.py`` writes ``voided`` when an operator confirms
  nothing was sent.

Unordered, a slow ``delivered_pending`` could undo a purchase that landed while
it was verifying, and an expiry sweep could mark a paid item unsold. That is the
review's sentence: *a delayed pending update must not reverse a confirmed
purchase.*

These are source-level races, written as tests because no live incident was
reproduced. Passing them does not mean every payment race in this codebase is
resolved — the review says so explicitly, and so does this file.
"""

from __future__ import annotations

import asyncio

import pytest

from services import ppv_delivery_ledger as ledger
from tests.fake_supabase import FakeSupabase


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def ledger_db(monkeypatch):
    """One claimed delivery, in a PostgREST-shaped double."""
    db = FakeSupabase(
        {
            "ppv_deliveries": [
                {
                    "reference": "ref-1",
                    "creator_id": "creator-1",
                    "fan_id": "fan-1",
                    "status": "claimed",
                    "media_ids": ["m1"],
                    "price_cents": 2500,
                    "source": "auto",
                }
            ]
        }
    )
    monkeypatch.setattr(ledger, "get_supabase", lambda: db)
    return db


def _status(db) -> str:
    return db.tables["ppv_deliveries"][0]["status"]


# --- the ordinary path is unchanged -----------------------------------------


def test_the_normal_lifecycle_still_runs(ledger_db):
    assert run(ledger.transition_delivery("ref-1", "delivered_pending")) is True
    assert _status(ledger_db) == "delivered_pending"
    assert run(ledger.transition_delivery("ref-1", "purchased")) is True
    assert _status(ledger_db) == "purchased"


def test_an_unsupported_status_is_still_rejected_outright(ledger_db):
    with pytest.raises(ValueError, match="unsupported"):
        run(ledger.transition_delivery("ref-1", "refunded"))


# --- nothing may move a row out of purchased --------------------------------


def test_a_late_pending_acknowledgement_cannot_reverse_a_purchase(ledger_db):
    """The interleaving the review names by name.

    The platform round trip in ppv_delivery.py finishes AFTER the purchase
    event has already been recorded. Before the predicate, this wrote
    ``delivered_pending`` straight over ``purchased``.
    """
    run(ledger.transition_delivery("ref-1", "purchased", amount_paid_cents=2500))

    assert run(
        ledger.transition_delivery(
            "ref-1", "delivered_pending", platform_message_id="platform-1"
        )
    ) is False
    assert _status(ledger_db) == "purchased"
    assert ledger_db.tables["ppv_deliveries"][0]["amount_paid_cents"] == 2500


def test_an_expiry_sweep_cannot_mark_a_paid_item_unsold(ledger_db):
    run(ledger.transition_delivery("ref-1", "purchased", amount_paid_cents=2500))

    assert run(ledger.transition_delivery("ref-1", "abandoned")) is False
    assert _status(ledger_db) == "purchased"


def test_an_operator_void_cannot_erase_a_purchase(ledger_db):
    run(ledger.transition_delivery("ref-1", "purchased"))

    assert run(ledger.transition_delivery("ref-1", "voided")) is False
    assert _status(ledger_db) == "purchased"


def test_a_send_failure_report_cannot_erase_a_purchase(ledger_db):
    run(ledger.transition_delivery("ref-1", "purchased"))

    assert run(ledger.transition_delivery("ref-1", "failed", error="boom")) is False
    assert _status(ledger_db) == "purchased"


# --- a purchase is authoritative even when it arrives late ------------------


def test_a_purchase_after_an_expiry_is_still_recorded(ledger_db):
    """A fan can pay an offer this backend has already given up on.

    Money arriving is a fact about the world. Refusing to record it would hide
    a real payment from the operator, which is worse than a surprising status
    transition.
    """
    run(ledger.transition_delivery("ref-1", "abandoned"))

    assert run(
        ledger.transition_delivery("ref-1", "purchased", amount_paid_cents=2500)
    ) is True
    assert _status(ledger_db) == "purchased"


def test_a_duplicate_purchase_event_is_a_no_op_that_reports_success(ledger_db):
    """A redelivered webhook must not read as a refusal."""
    run(ledger.transition_delivery("ref-1", "purchased", amount_paid_cents=2500))

    assert run(ledger.transition_delivery("ref-1", "purchased")) is True
    assert _status(ledger_db) == "purchased"


# --- the other terminal statuses are one-way too ----------------------------


def test_a_terminal_row_is_not_reopened_by_a_later_writer(ledger_db):
    run(ledger.transition_delivery("ref-1", "voided"))

    assert run(ledger.transition_delivery("ref-1", "delivered_pending")) is False
    assert run(ledger.transition_delivery("ref-1", "abandoned")) is False
    assert _status(ledger_db) == "voided"


def test_an_unknown_reference_is_reported_rather_than_silently_ignored(ledger_db):
    assert run(ledger.transition_delivery("no-such-ref", "purchased")) is False
    assert len(ledger_db.tables["ppv_deliveries"]) == 1


def test_a_retry_whose_first_attempt_already_landed_reports_success(
    ledger_db, monkeypatch
):
    """The guard must not turn a lost response into a phantom refusal.

    ``retry_transient_db_operation`` re-runs the update after a transport
    disconnect. If the first attempt actually applied, the retry matches no row
    — because the predicate no longer holds — and that is a success, not a
    conflict.
    """
    calls = {"n": 0}
    real_table = ledger_db.table

    def flaky_table(name):
        query = real_table(name)
        real_execute = query.execute

        def execute():
            result = real_execute()
            calls["n"] += 1
            if calls["n"] == 1:
                raise ConnectionError("server disconnected without sending a response")
            return result

        query.execute = execute
        return query

    monkeypatch.setattr(
        ledger, "get_supabase", lambda: type("_DB", (), {"table": staticmethod(flaky_table)})()
    )

    assert run(ledger.transition_delivery("ref-1", "purchased")) is True
    assert _status(ledger_db) == "purchased"


# --- the cheap pre-check commercial state uses ------------------------------


def test_delivery_is_paid_reports_the_ledger_not_a_guess(ledger_db):
    assert run(ledger.delivery_is_paid("ref-1")) is False
    run(ledger.transition_delivery("ref-1", "purchased"))
    assert run(ledger.delivery_is_paid("ref-1")) is True
    assert run(ledger.delivery_is_paid("no-such-ref")) is False


def test_abandon_if_active_still_reports_who_won_the_race(ledger_db):
    """It answers a different question from transition_delivery, on purpose."""
    assert run(ledger.abandon_delivery_if_active("ref-1")) is True
    assert run(ledger.abandon_delivery_if_active("ref-1")) is False
    assert run(ledger.transition_delivery("ref-1", "abandoned")) is True


# --- and the expiry path in reconciliation respects it ----------------------


def test_the_expiry_sweep_leaves_a_purchased_delivery_alone(monkeypatch):
    """Commercial state must not record "not sold" for something that sold."""
    from services import ppv_reconciliation

    db = FakeSupabase(
        {
            "ppv_deliveries": [
                {
                    "reference": "ref-1",
                    "creator_id": "creator-1",
                    "fan_id": "fan-1",
                    "status": "purchased",
                }
            ],
            "fans": [
                {
                    "id": "fan-1",
                    "pending_ppv_check": {"reference": "ref-1", "media_id": "m1"},
                    "not_sold_log": [],
                }
            ],
        }
    )
    monkeypatch.setattr(ledger, "get_supabase", lambda: db)
    monkeypatch.setattr(ppv_reconciliation, "get_supabase", lambda: db)

    from datetime import datetime, timezone

    finalized = run(
        ppv_reconciliation._finalize_abandonment(
            creator_id="creator-1",
            fan_id="fan-1",
            pending={"reference": "ref-1", "media_id": "m1", "price": 25.0},
            fan_row={"not_sold_log": []},
            now=datetime.now(timezone.utc),
        )
    )

    assert finalized is False
    assert db.tables["fans"][0]["not_sold_log"] == []
    assert db.tables["fans"][0]["pending_ppv_check"] is not None
    assert db.tables["ppv_deliveries"][0]["status"] == "purchased"


# --- concurrent purchase events must not lose a customer's money ------------


def _purchase_db(total_spent: int = 100) -> FakeSupabase:
    return FakeSupabase(
        {
            "fans": [
                {
                    "id": "fan-1",
                    "creator_id": "creator-1",
                    "total_spent": total_spent,
                    "spend_tier": "active",
                    "sales_log": [],
                    "not_sold_log": [],
                    "pending_ppv_check": None,
                    "needs_human_review": False,
                }
            ]
        }
    )


def test_a_purchase_writes_against_the_total_it_was_computed_from(monkeypatch):
    from db import queries

    db = _purchase_db(100)
    monkeypatch.setattr(queries, "get_supabase", lambda: db)

    written = run(
        queries.apply_purchase_to_fan(
            "fan-1",
            merge=lambda row: {"total_spent": 125, "spend_tier": "active"},
            expected_total_spent=100,
        )
    )

    assert written is not None
    assert db.tables["fans"][0]["total_spent"] == 125


def test_a_purchase_event_that_lost_the_race_recomputes_instead_of_overwriting(
    monkeypatch,
):
    """The lost-update this closes.

    Two events read $100. The first writes $125 while the second is still
    awaiting the ledger. Before the predicate, the second wrote $125 too and
    one $25 purchase vanished from the customer's spend, silently.
    """
    from db import queries

    db = _purchase_db(100)
    monkeypatch.setattr(queries, "get_supabase", lambda: db)

    # The other event lands first.
    db.tables["fans"][0]["total_spent"] = 125
    db.tables["fans"][0]["sales_log"] = [{"payment_reference": "ref-other", "amount": 25}]

    seen: list[dict | None] = []

    def merge(row):
        seen.append(row)
        base = 100 if row is None else int(row["total_spent"])
        return {"total_spent": base + 40, "spend_tier": "active"}

    written = run(
        queries.apply_purchase_to_fan(
            "fan-1", merge=merge, expected_total_spent=100
        )
    )

    assert written is not None
    # 125 + 40, not 100 + 40: the second event merged into what was there.
    assert db.tables["fans"][0]["total_spent"] == 165
    assert seen[0] is None, "the first attempt uses the caller's snapshot"
    assert seen[1] is not None, "the retry is handed the row as it actually is"
    assert seen[1]["total_spent"] == 125


def test_merge_may_decline_when_the_other_event_recorded_the_same_purchase(
    monkeypatch,
):
    from db import queries

    db = _purchase_db(100)
    monkeypatch.setattr(queries, "get_supabase", lambda: db)
    db.tables["fans"][0]["total_spent"] = 125

    def merge(row):
        if row is None:
            return {"total_spent": 125, "spend_tier": "active"}
        return None

    assert run(
        queries.apply_purchase_to_fan(
            "fan-1", merge=merge, expected_total_spent=100
        )
    ) is None
    assert db.tables["fans"][0]["total_spent"] == 125


def test_money_that_cannot_be_written_down_raises_rather_than_disappearing(
    monkeypatch,
):
    """Silence is how the loss became invisible in the first place."""
    from db import queries

    db = _purchase_db(100)
    monkeypatch.setattr(queries, "get_supabase", lambda: db)

    def merge(row):
        # Every attempt is beaten by another event before it can write.
        db.tables["fans"][0]["total_spent"] += 1
        return {"total_spent": 999, "spend_tier": "whale"}

    with pytest.raises(queries.PurchaseAggregateConflict):
        run(
            queries.apply_purchase_to_fan(
                "fan-1", merge=merge, expected_total_spent=100, attempts=3
            )
        )
    assert db.tables["fans"][0]["total_spent"] != 999
