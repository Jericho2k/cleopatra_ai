"""A free repair happens at most once, and clears only the hold it was about.

THE THREE FAILURES, REPRODUCED AT BACKEND 4a1683a
-------------------------------------------------
services/content_access.py called the platform adapter and then wrote the
receipt, with nothing durable in between:

  1. platform accepts -> save_message raises -> the operator clicks Resend
     again -> TWO provider sends, and the hold is still set, so they are
     invited to click a third time;
  2. two operators click at the same moment -> TWO provider sends, both
     reporting success;
  3. a crisis hold raised DURING the platform call is cleared by the repair on
     its way out, because clear_fan_review was an unconditional update by fan
     id. A conversation frozen for crisis language silently resumed.

Each is a test below, written as the failure rather than as the fix, so it
stays readable as "this is the thing that must not happen".

The uniqueness half runs against real PostgreSQL. The claim IS a unique index,
and whether a unique index rejects a concurrent insert is a question about
PostgreSQL, not about our beliefs — the same reasoning
tests/test_browser_least_privilege.py gives for its RLS tests.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from pathlib import Path

import pytest

from services import content_access
from services import content_access_repairs as repairs
from tests.test_content_access_recovery import _stub_platform, _world


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def world(monkeypatch):
    db = _world()
    monkeypatch.setattr(content_access, "get_supabase", lambda: db)
    monkeypatch.setattr("services.ppv_delivery_ledger.get_supabase", lambda: db)
    monkeypatch.setattr("db.queries.get_supabase", lambda: db)
    monkeypatch.setattr("services.content_access_repairs.get_supabase", lambda: db)
    monkeypatch.setattr("services.message_diagnostics.get_supabase", lambda: db)
    return db


def _repairs(db) -> list[dict]:
    return db.tables.get("content_access_repairs", [])


# ===========================================================================
# 1. A retry reconciles instead of sending again
# ===========================================================================


def test_a_db_failure_after_a_successful_send_does_not_cause_a_second_send(
    world, monkeypatch
):
    """The platform has the media. The local write is what failed."""
    sends: list = []
    _stub_platform(monkeypatch, messages=[], account_media=[], sends=sends)

    async def exploding_save(*_a, **_k):
        raise RuntimeError("supabase down")

    monkeypatch.setattr(content_access, "save_message", exploding_save)

    with pytest.raises(RuntimeError):
        run(content_access.resolve_content_access(
            "fan-1", resolution=content_access.RESOLUTION_RESEND))

    # The operator sees a failure and clicks again, which is the correct thing
    # for them to do and must not be the thing that duplicates.
    with pytest.raises(content_access.ContentAccessError) as refused:
        run(content_access.resolve_content_access(
            "fan-1", resolution=content_access.RESOLUTION_RESEND))

    assert len(sends) == 1
    assert "second copy" in str(refused.value)
    # The claim settled as confirmed before the local write was attempted, so
    # the send is proven even though the message row is missing.
    assert _repairs(world)[0]["status"] == repairs.STATUS_CONFIRMED


def test_two_operators_clicking_together_send_once(world, monkeypatch):
    sends: list = []
    _stub_platform(monkeypatch, messages=[], account_media=[], sends=sends)

    async def both():
        return await asyncio.gather(
            content_access.resolve_content_access(
                "fan-1", resolution=content_access.RESOLUTION_RESEND),
            content_access.resolve_content_access(
                "fan-1", resolution=content_access.RESOLUTION_RESEND),
            return_exceptions=True,
        )

    results = asyncio.run(both())

    assert len(sends) == 1
    succeeded = [r for r in results if isinstance(r, dict)]
    refused = [r for r in results if isinstance(r, content_access.ContentAccessError)]
    assert len(succeeded) == 1
    assert len(refused) == 1
    # And the loser is told something true and actionable, not "failed".
    assert "already" in str(refused[0])


def test_a_repair_for_a_different_purchase_is_not_blocked(world, monkeypatch):
    """The claim is per purchase, not per customer.

    Two paid items can both be broken. Refusing the second because the first
    was repaired would be the same bug in the other direction.
    """
    world.tables["ppv_deliveries"].append({
        "reference": "ref-2",
        "creator_id": "creator-1",
        "fan_id": "fan-1",
        "status": "purchased",
        "media_ids": ["m9"],
        "price_cents": 900,
        "platform_message_id": "platform-2",
        "purchased_at": "2026-09-16T11:00:00+00:00",
        "claimed_at": "2026-09-16T10:30:00+00:00",
    })
    sends: list = []
    _stub_platform(monkeypatch, messages=[], account_media=[], sends=sends)

    run(content_access.resend_paid_content("fan-1", reference="ref-1"))
    run(content_access.resend_paid_content("fan-1", reference="ref-2"))

    assert len(sends) == 2
    assert {row["reference"] for row in _repairs(world)} == {"ref-1", "ref-2"}


def test_a_new_hold_may_repair_the_same_purchase_again(world, monkeypatch):
    """The media went missing twice. That is a real second complaint."""
    sends: list = []
    _stub_platform(monkeypatch, messages=[], account_media=[], sends=sends)

    run(content_access.resend_paid_content(
        "fan-1", reference="ref-1", review_case_id="case-1"))
    run(content_access.resend_paid_content(
        "fan-1", reference="ref-1", review_case_id="case-2"))

    assert len(sends) == 2
    assert {row["review_case_id"] for row in _repairs(world)} == {"case-1", "case-2"}


# ===========================================================================
# 2. Four outcomes, and `unknown` is one of them
# ===========================================================================


def test_a_send_with_no_receipt_is_unknown_and_not_retried(world, monkeypatch):
    """Accepted, no message id. It may have arrived; it may not have."""
    from services import apifansly

    sends: list = []

    async def receiptless_send(_account, _chat, **kwargs):
        sends.append(kwargs)
        return {"data": {"data": {"response": {}}}}

    async def fake_list(*_a, **_k):
        return [], [], None

    monkeypatch.setattr(apifansly, "send_message", receiptless_send)
    monkeypatch.setattr(apifansly, "list_chat_messages", fake_list)

    with pytest.raises(content_access.ContentAccessError):
        run(content_access.resend_paid_content("fan-1"))

    assert _repairs(world)[0]["status"] == repairs.STATUS_UNKNOWN

    # A second attempt does not send. This is the case where a retry is the
    # thing that can cost the customer a duplicate.
    with pytest.raises(content_access.ContentAccessError) as refused:
        run(content_access.resend_paid_content("fan-1"))
    assert len(sends) == 1
    assert "not known whether the customer received it" in str(refused.value)


def test_a_platform_call_that_raises_is_unknown_and_not_failed(world, monkeypatch):
    """A timeout on the response side looks exactly like one on the request side."""
    from services import apifansly

    async def exploding_send(*_a, **_k):
        raise TimeoutError("read timeout")

    async def fake_list(*_a, **_k):
        return [], [], None

    monkeypatch.setattr(apifansly, "send_message", exploding_send)
    monkeypatch.setattr(apifansly, "list_chat_messages", fake_list)

    with pytest.raises(content_access.ContentAccessError):
        run(content_access.resend_paid_content("fan-1"))

    row = _repairs(world)[0]
    assert row["status"] == repairs.STATUS_UNKNOWN
    assert "TimeoutError" in row["detail"]


def test_a_failed_claim_may_be_attempted_again(world, monkeypatch):
    """Nothing left the process, so nothing is owed and nothing duplicates."""
    sends: list = []
    _stub_platform(monkeypatch, messages=[], account_media=[], sends=sends)

    claimed = run(repairs.claim(
        creator_id="creator-1", fan_id="fan-1", reference="ref-1",
        review_case_id="case-1", media_ids=["m1"],
    ))
    run(repairs.mark_failed(claimed, detail="route missing"))

    result = run(content_access.resend_paid_content(
        "fan-1", reference="ref-1", review_case_id="case-1"))

    assert result["status"] == "resent"
    assert len(sends) == 1
    assert _repairs(world)[0]["status"] == repairs.STATUS_CONFIRMED
    assert len(_repairs(world)) == 1, "re-opened in place, not a second claim row"


def test_two_operators_retrying_a_failed_repair_still_send_once(world, monkeypatch):
    """Re-opening a failed claim is itself a compare-and-set."""
    sends: list = []
    _stub_platform(monkeypatch, messages=[], account_media=[], sends=sends)

    claimed = run(repairs.claim(
        creator_id="creator-1", fan_id="fan-1", reference="ref-1",
        review_case_id="case-1", media_ids=["m1", "m2"],
    ))
    run(repairs.mark_failed(claimed, detail="route missing"))

    async def both():
        return await asyncio.gather(
            content_access.resend_paid_content(
                "fan-1", reference="ref-1", review_case_id="case-1"),
            content_access.resend_paid_content(
                "fan-1", reference="ref-1", review_case_id="case-1"),
            return_exceptions=True,
        )

    results = asyncio.run(both())
    assert len(sends) == 1
    assert len([r for r in results if isinstance(r, dict)]) == 1


def test_an_abandoned_claim_becomes_unknown_rather_than_retryable(world, monkeypatch):
    """A worker that died mid-send may well have sent.

    Treating a stale claim as free to retry would turn every crash into a
    duplicate, so it becomes a question for a person instead.
    """
    sends: list = []
    _stub_platform(monkeypatch, messages=[], account_media=[], sends=sends)
    monkeypatch.setattr(repairs, "STALE_CLAIM_SECONDS", -1.0)

    run(repairs.claim(
        creator_id="creator-1", fan_id="fan-1", reference="ref-1",
        review_case_id="case-1", media_ids=["m1"],
    ))

    with pytest.raises(content_access.ContentAccessError):
        run(content_access.resend_paid_content(
            "fan-1", reference="ref-1", review_case_id="case-1"))

    assert sends == []
    assert _repairs(world)[0]["status"] == repairs.STATUS_UNKNOWN


def test_a_repair_that_was_never_possible_leaves_no_claim(world, monkeypatch):
    """A refusal must not look like an attempt in the operator's history."""
    world.tables["ppv_deliveries"][0]["status"] = "pending"
    _stub_platform(monkeypatch, messages=[], account_media=[])

    with pytest.raises(content_access.ContentAccessError):
        run(content_access.resend_paid_content("fan-1"))

    assert _repairs(world) == []


# ===========================================================================
# 3. The hold that is cleared is the hold that was resolved
# ===========================================================================


def test_a_repair_does_not_clear_a_newer_hold(world, monkeypatch):
    """The crisis case. This is the one that could resume a frozen conversation."""
    from services import apifansly

    async def fake_list(*_a, **_k):
        return [], [], None

    async def send_then_freeze(*_a, **_kwargs):
        # The analyzer freezes this conversation again, for something else
        # entirely, while the platform call is in flight. Through the real
        # producer, so the new hold gets its own identity as it would live.
        from db.queries import freeze_fan_for_review

        await freeze_fan_for_review("fan-1", "crisis_language")
        return {"data": {"data": {"response": {"id": "platform-resend-1"}}}}

    monkeypatch.setattr(apifansly, "list_chat_messages", fake_list)
    monkeypatch.setattr(apifansly, "send_message", send_then_freeze)

    result = run(content_access.resolve_content_access(
        "fan-1", resolution=content_access.RESOLUTION_RESEND))

    fan = world.tables["fans"][0]
    assert fan["needs_human_review"] is True
    assert fan["review_reason"] == "crisis_language"
    # The repair itself succeeded. Both facts are reported, because they lead
    # the operator to different next actions.
    assert result["status"] == "resent"
    assert result["review_cleared"] is False
    assert result["hold_superseded"] is True


def test_an_ordinary_repair_still_clears_its_own_hold(world, monkeypatch):
    """The compare-and-set must not have broken the normal path."""
    _stub_platform(monkeypatch, messages=[], account_media=[])

    result = run(content_access.resolve_content_access(
        "fan-1", resolution=content_access.RESOLUTION_RESEND))

    assert result["review_cleared"] is True
    assert result["hold_superseded"] is False
    assert world.tables["fans"][0]["needs_human_review"] is False


def test_every_freeze_mints_a_new_case(world):
    """Two holds are never the same hold, even for the same reason."""
    from db.queries import freeze_fan_for_review

    first = run(freeze_fan_for_review("fan-1", content_access.REVIEW_REASON))
    second = run(freeze_fan_for_review("fan-1", content_access.REVIEW_REASON))

    assert first and second and first != second
    assert world.tables["fans"][0]["review_case_id"] == second


def test_clearing_without_a_case_is_still_unconditional(world):
    """An operator clearing a hold by hand acts on whatever is there now."""
    from db.queries import clear_fan_review

    assert run(clear_fan_review("fan-1")) is True
    assert world.tables["fans"][0]["needs_human_review"] is False


# ===========================================================================
# 4. The claim against real PostgreSQL
# ===========================================================================

psycopg = pytest.importorskip("psycopg", reason="psycopg is required for schema tests")

DATABASE_URL = os.getenv("TEST_DATABASE_URL", "").strip()

schema_only = pytest.mark.skipif(
    not DATABASE_URL,
    reason="TEST_DATABASE_URL is not set; uniqueness tests need a real PostgreSQL",
)

DB = Path(__file__).resolve().parents[1] / "db"
OPERATOR = "44444444-4444-4444-4444-444444444444"


def _order() -> list[str]:
    return [
        line.strip()
        for line in (DB / "migration_order.txt").read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


@pytest.fixture(scope="module")
def deployed():
    name = f"cleo_repair_{uuid.uuid4().hex[:12]}"
    connection = psycopg.connect(DATABASE_URL, autocommit=True)

    def scoped(sql: str) -> str:
        return sql.replace("public.", f'"{name}".').replace("'public'", f"'{name}'")

    try:
        with connection.cursor() as cursor:
            cursor.execute(f'create schema "{name}"')
            cursor.execute((DB / "ci_supabase_stubs.sql").read_text())
            cursor.execute(f'set search_path to "{name}", public')
            cursor.execute(scoped((DB / "ci_baseline_schema.sql").read_text()))
            cursor.execute(f'grant usage on schema "{name}" to anon, authenticated')
            cursor.execute(f'grant usage on schema "{name}" to service_role')
            cursor.execute(
                f'grant all on all tables in schema "{name}" to anon, authenticated'
            )
            cursor.execute(
                f'alter default privileges in schema "{name}" '
                "grant all on tables to anon, authenticated"
            )
            cursor.execute(
                f'insert into "{name}".creators (name) values (%s) returning id',
                ("Agency",),
            )
            creator_id = cursor.fetchone()[0]
            cursor.execute(
                f'insert into "{name}".chatter_creators (chatter_id, creator_id) '
                "values (%s, %s)",
                (OPERATOR, creator_id),
            )
            cursor.execute(
                f'insert into "{name}".fans (creator_id, display_name, platform_fan_id, '
                "needs_human_review, review_reason) values (%s, %s, %s, true, %s) "
                "returning id",
                (creator_id, "Fan", "p-1", content_access.REVIEW_REASON),
            )
            fan_id = cursor.fetchone()[0]

            for filename in _order():
                cursor.execute(scoped((DB / filename).read_text()))

        yield connection, name, {"creator_id": creator_id, "fan_id": fan_id}
    finally:
        try:
            with connection.cursor() as cursor:
                cursor.execute(f'drop schema if exists "{name}" cascade')
        finally:
            connection.close()


def _claim(cursor, schema, data, *, reference="ref-1", case="case-1"):
    cursor.execute(
        f'insert into "{schema}".content_access_repairs '
        "(creator_id, fan_id, reference, review_case_id, media_ids) "
        "values (%s, %s, %s, %s, %s) returning id",
        (data["creator_id"], data["fan_id"], reference, case, json.dumps(["m1"])),
    )
    return cursor.fetchone()[0]


@schema_only
def test_postgresql_rejects_a_second_claim_for_the_same_case(deployed):
    """The claim is a unique index. This is the assertion that it exists."""
    connection, name, data = deployed

    with connection.cursor() as cursor:
        cursor.execute("begin")
        try:
            cursor.execute(f'set local search_path to "{name}", public')
            _claim(cursor, name, data)
            with pytest.raises(psycopg.errors.UniqueViolation):
                _claim(cursor, name, data)
        finally:
            cursor.execute("rollback")


@schema_only
def test_postgresql_allows_a_claim_for_a_different_review_case(deployed):
    connection, name, data = deployed

    with connection.cursor() as cursor:
        cursor.execute("begin")
        try:
            cursor.execute(f'set local search_path to "{name}", public')
            first = _claim(cursor, name, data, case="case-1")
            second = _claim(cursor, name, data, case="case-2")
            assert first != second
        finally:
            cursor.execute("rollback")


@schema_only
def test_two_concurrent_transactions_cannot_both_claim(deployed):
    """Two operators, two connections, one index.

    The second INSERT blocks on the first's uncommitted row and then fails
    outright when it commits. That is the behaviour the service relies on, and
    it is a property of PostgreSQL rather than of our code, so it is asserted
    here against the real thing.
    """
    connection, name, data = deployed
    other = psycopg.connect(DATABASE_URL, autocommit=False)
    try:
        with connection.cursor() as first, other.cursor() as second:
            first.execute("begin")
            first.execute(f'set local search_path to "{name}", public')
            _claim(first, name, data, case="concurrent")

            second.execute(f'set search_path to "{name}", public')
            # Do not block the test run on lock wait: the point is that the
            # second transaction cannot proceed while the first holds the key.
            second.execute("set lock_timeout = '500ms'")
            with pytest.raises(psycopg.Error) as blocked:
                _claim(second, name, data, case="concurrent")
            assert isinstance(
                blocked.value,
                (psycopg.errors.LockNotAvailable, psycopg.errors.UniqueViolation),
            )
            other.rollback()
            first.execute("rollback")
    finally:
        other.close()


@schema_only
def test_an_operator_can_read_their_repairs_but_not_forge_one(deployed):
    """The panel has to show the real state; nothing may invent one.

    Deliberately readable, unlike message_diagnostics: an operator who cannot
    see that a repair is in flight is an operator who clicks Resend again.
    """
    connection, name, data = deployed

    with connection.cursor() as cursor:
        cursor.execute("begin")
        try:
            cursor.execute(f'set local search_path to "{name}", public')
            _claim(cursor, name, data, case="visible")
            cursor.execute("set local role authenticated")
            cursor.execute(f"set local request.jwt.claim.sub = '{OPERATOR}'")
            cursor.execute(
                f'select status from "{name}".content_access_repairs '
                "where review_case_id = 'visible'"
            )
            assert cursor.fetchall() == [("claimed",)]

            with pytest.raises(psycopg.Error):
                cursor.execute(
                    f'insert into "{name}".content_access_repairs '
                    "(creator_id, fan_id, reference, review_case_id) "
                    "values (%s, %s, 'forged', 'forged')",
                    (data["creator_id"], data["fan_id"]),
                )
        finally:
            cursor.execute("rollback")


@schema_only
def test_an_existing_hold_was_given_an_identity_by_the_migration(deployed):
    """A hold raised before the column existed still has to be distinguishable."""
    connection, name, data = deployed

    with connection.cursor() as cursor:
        cursor.execute(
            f'select review_case_id from "{name}".fans where id = %s',
            (data["fan_id"],),
        )
        case_id = cursor.fetchone()[0]

    assert case_id, "every hold that predates the column gets its own case id"


# ===========================================================================
# 5. The operator says WHICH purchase, and sees the real state of the operation
# ===========================================================================


def _second_purchase(db) -> None:
    db.tables["ppv_deliveries"].append({
        "reference": "ref-2",
        "creator_id": "creator-1",
        "fan_id": "fan-1",
        "status": "purchased",
        "media_ids": ["m9"],
        "price_cents": 900,
        "platform_message_id": "platform-2",
        "purchased_at": "2026-09-16T11:00:00+00:00",
        "claimed_at": "2026-09-16T10:30:00+00:00",
    })


def test_two_purchases_and_no_selection_is_refused(world, monkeypatch):
    """The failure this replaces: an older item's complaint resent a newer item.

    The customer then has two copies of something that worked and still cannot
    open the thing they wrote in about — and the hold is cleared as though it
    had been repaired.
    """
    _second_purchase(world)
    sends: list = []
    _stub_platform(monkeypatch, messages=[], account_media=[], sends=sends)

    with pytest.raises(content_access.ContentAccessError) as refused:
        run(content_access.resend_paid_content("fan-1"))

    assert sends == []
    assert "chosen rather than assumed" in str(refused.value)
    # And the operator is told what they may choose between.
    assert "ref-1" in str(refused.value) and "ref-2" in str(refused.value)


def test_a_single_purchase_needs_no_selection(world, monkeypatch):
    """Nothing is ambiguous, so nothing is demanded."""
    sends: list = []
    _stub_platform(monkeypatch, messages=[], account_media=[], sends=sends)

    assert run(content_access.resend_paid_content("fan-1"))["reference"] == "ref-1"
    assert len(sends) == 1


def test_the_chosen_purchase_is_the_one_that_is_resent(world, monkeypatch):
    _second_purchase(world)
    sends: list = []
    _stub_platform(monkeypatch, messages=[], account_media=[], sends=sends)

    result = run(content_access.resend_paid_content("fan-1", reference="ref-1"))

    assert result["reference"] == "ref-1"
    assert sends[0]["media_ids"] == ["m1", "m2"]


def test_a_reference_that_names_nothing_does_not_fall_back(world, monkeypatch):
    """Falling back to the newest is the bug, whatever caused the bad reference."""
    sends: list = []
    _stub_platform(monkeypatch, messages=[], account_media=[], sends=sends)

    with pytest.raises(content_access.ContentAccessError) as refused:
        run(content_access.resend_paid_content("fan-1", reference="ref-nonexistent"))

    assert sends == []
    assert "another conversation" in str(refused.value)


def test_the_panel_is_told_a_selection_is_required(world, monkeypatch):
    _second_purchase(world)
    _stub_platform(monkeypatch, messages=[], account_media=[])

    evidence = run(content_access.inspect_content_access("fan-1")).as_dict()

    assert evidence["selection_required"] is True
    assert evidence["repairable_count"] == 2
    assert {item["reference"] for item in evidence["paid_items"]} == {"ref-1", "ref-2"}


def test_the_panel_sees_a_repair_that_is_already_confirmed(world, monkeypatch):
    """A reload has to show the operation's actual state.

    An operator who reloads and sees nothing clicks Resend again — which is
    refused now, but being refused is a worse experience than being told.
    """
    _stub_platform(monkeypatch, messages=[], account_media=[])
    run(content_access.resend_paid_content("fan-1", review_case_id="case-1"))
    world.tables["fans"][0].update({
        "needs_human_review": True,
        "review_reason": content_access.REVIEW_REASON,
        "review_case_id": "case-1",
    })

    evidence = run(content_access.inspect_content_access("fan-1")).as_dict()

    item = next(i for i in evidence["paid_items"] if i["reference"] == "ref-1")
    assert item["repair"]["status"] == repairs.STATUS_CONFIRMED
    assert item["repair"]["settled"] is True
    assert evidence["repairs"][0]["platform_message_id"] == "platform-resend-1"


def test_the_panel_sees_an_unknown_outcome_as_needing_a_decision(world, monkeypatch):
    """Uncertainty is preserved rather than rounded to success or failure."""
    from services import apifansly

    async def receiptless_send(*_a, **_k):
        return {"data": {"data": {"response": {}}}}

    async def fake_list(*_a, **_k):
        return [], [], None

    monkeypatch.setattr(apifansly, "send_message", receiptless_send)
    monkeypatch.setattr(apifansly, "list_chat_messages", fake_list)

    with pytest.raises(content_access.ContentAccessError):
        run(content_access.resend_paid_content("fan-1", review_case_id="case-1"))

    evidence = run(content_access.inspect_content_access("fan-1")).as_dict()

    item = next(i for i in evidence["paid_items"] if i["reference"] == "ref-1")
    assert item["repair"]["status"] == repairs.STATUS_UNKNOWN
    assert item["repair"]["needs_operator_decision"] is True
    assert item["repair"]["settled"] is False


def test_a_repair_from_an_older_hold_is_not_shown_against_the_new_one(
    world, monkeypatch
):
    """A genuinely new complaint about the same purchase is repairable again."""
    _stub_platform(monkeypatch, messages=[], account_media=[])
    run(content_access.resend_paid_content("fan-1", review_case_id="case-0"))
    world.tables["fans"][0].update({
        "needs_human_review": True,
        "review_reason": content_access.REVIEW_REASON,
        "review_case_id": "case-1",
    })

    evidence = run(content_access.inspect_content_access("fan-1")).as_dict()

    item = next(i for i in evidence["paid_items"] if i["reference"] == "ref-1")
    assert item["repair"] is None, "the old attempt does not block the new hold"
    # It is still in the history, because it still happened.
    assert len(evidence["repairs"]) == 1


def test_deciding_on_a_hold_that_has_since_changed_is_refused(world, monkeypatch):
    """The panel sends back the case it rendered."""
    _stub_platform(monkeypatch, messages=[], account_media=[])

    with pytest.raises(content_access.ContentAccessError) as refused:
        run(content_access.resolve_content_access(
            "fan-1",
            resolution=content_access.RESOLUTION_RESEND,
            review_case_id="a-hold-that-is-no-longer-current",
        ))

    assert "changed while you were looking at it" in str(refused.value)
    assert world.tables["fans"][0]["needs_human_review"] is True


def test_the_resolution_records_who_asked_for_it(world, monkeypatch):
    """An auditable resolution needs an actor."""
    _stub_platform(monkeypatch, messages=[], account_media=[])

    run(content_access.resolve_content_access(
        "fan-1",
        resolution=content_access.RESOLUTION_RESEND,
        actor="operator-7",
    ))

    assert _repairs(world)[0]["claimed_by"] == "operator-7"


def test_the_panel_reports_repairs_even_when_the_platform_is_unreachable(
    world, monkeypatch
):
    """Read before the platform call, so uncertainty there costs nothing here."""
    from services import apifansly

    async def exploding_list(*_a, **_k):
        raise RuntimeError("provider down")

    async def fake_send(*_a, **_k):
        return {"data": {"data": {"response": {"id": "platform-resend-1"}}}}

    monkeypatch.setattr(apifansly, "send_message", fake_send)
    run(content_access.resend_paid_content("fan-1", review_case_id="case-1"))
    monkeypatch.setattr(apifansly, "list_chat_messages", exploding_list)

    evidence = run(content_access.inspect_content_access("fan-1")).as_dict()

    assert evidence["platform_error"]
    assert evidence["repairs"][0]["status"] == repairs.STATUS_CONFIRMED
