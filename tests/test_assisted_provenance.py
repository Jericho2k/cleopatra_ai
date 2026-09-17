"""An Assisted reply stays attributable across a restart, or admits it is not.

Full Auto generates and delivers inside one function; Assisted does not. A
person reads the candidates and sends one some time later, so the record has to
survive two HTTP requests.

SUGGESTION_PROVENANCE bridged that with an in-process OrderedDict, and its own
comment said the cost:

    The process-wide store. One per backend replica, which is why a miss is an
    ordinary outcome rather than an error.

A deploy, a crash, an autoscale event, or the second request landing on another
replica lost the record — and the message was then saved with NO provenance at
all. The reply became unattributable and nothing said so, which is the failure
the whole provenance effort exists to remove.

The second half matters more than the first. Durability can still fail, and a
reply that is SILENTLY unattributable is indistinguishable from one nobody ever
tried to attribute.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from services import assisted_provenance as store
from services.reply_provenance import (
    PIPELINE_ASSISTED,
    SUGGESTION_PROVENANCE,
    ReplyProvenance,
)
from tests.fake_supabase import FakeSupabase


def run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _clear_process_store():
    SUGGESTION_PROVENANCE.clear()
    yield
    SUGGESTION_PROVENANCE.clear()


@pytest.fixture
def db(monkeypatch):
    fake = FakeSupabase({"assisted_provenance": []})
    monkeypatch.setattr(store, "get_supabase", lambda: fake)
    return fake


def _recorder() -> ReplyProvenance:
    provenance = ReplyProvenance(
        creator_id="creator-1", fan_id="fan-1", mode=PIPELINE_ASSISTED
    )
    provenance.record_trigger(kind="fan_message", text="are you there?")
    provenance.writer = {"actual": {"provider": "openrouter", "model": "kimi"}}
    return provenance


# ===========================================================================
# 1. It survives losing the process
# ===========================================================================


def test_a_record_is_redeemable_after_the_process_forgets_it(db):
    """The whole point. A deploy between generating and sending."""
    token = run(store.remember(_recorder()))

    SUGGESTION_PROVENANCE.clear()  # the restart

    restored, unavailable = run(
        store.redeem(token, creator_id="creator-1", fan_id="fan-1")
    )

    assert unavailable == ""
    assert restored is not None
    assert restored.writer["actual"]["model"] == "kimi"
    assert restored.trigger["kind"] == "fan_message"


def test_the_turn_identity_survives_too():
    """A restored record has to be the same turn, or it attributes a reply to
    a turn it did not come from."""
    original = _recorder()

    restored = ReplyProvenance.from_state(original.as_state())

    assert restored.turn_id == original.turn_id
    assert restored.started_at == original.started_at
    assert restored.mode == PIPELINE_ASSISTED


def test_the_in_process_store_is_still_the_fast_path(db):
    """The common case is the same replica seconds later, and it should not
    pay a round trip."""
    token = run(store.remember(_recorder()))
    before = len(db.queries)

    restored, _ = run(store.redeem(token, creator_id="creator-1", fan_id="fan-1"))

    assert restored is not None
    # A delete is a write, not a query; no SELECT was needed.
    assert len(db.queries) == before


def test_an_in_process_hit_still_clears_the_durable_row(db):
    """Otherwise a replay on another replica could redeem the same turn."""
    token = run(store.remember(_recorder()))
    run(store.redeem(token, creator_id="creator-1", fan_id="fan-1"))

    assert db.tables["assisted_provenance"] == []


def test_a_record_is_redeemed_exactly_once(db):
    """One generated turn becomes at most one sent message."""
    token = run(store.remember(_recorder()))
    run(store.redeem(token, creator_id="creator-1", fan_id="fan-1"))
    SUGGESTION_PROVENANCE.clear()

    second, unavailable = run(
        store.redeem(token, creator_id="creator-1", fan_id="fan-1")
    )

    assert second is None
    assert unavailable == store.UNAVAILABLE_MISSING


# ===========================================================================
# 2. The wrong record is worse than no record
# ===========================================================================


@pytest.mark.parametrize(
    "creator_id,fan_id",
    [("creator-2", "fan-1"), ("creator-1", "fan-2")],
)
def test_a_record_from_another_conversation_is_a_miss(db, creator_id, fan_id):
    token = run(store.remember(_recorder()))
    SUGGESTION_PROVENANCE.clear()

    restored, unavailable = run(
        store.redeem(token, creator_id=creator_id, fan_id=fan_id)
    )

    assert restored is None
    assert unavailable == store.UNAVAILABLE_MISSING


def test_an_expired_record_is_a_miss_and_says_which(db):
    token = run(store.remember(_recorder()))
    SUGGESTION_PROVENANCE.clear()
    stale = datetime.now(timezone.utc) - store.TTL - timedelta(minutes=1)
    db.tables["assisted_provenance"][0]["created_at"] = stale.isoformat()

    restored, unavailable = run(
        store.redeem(token, creator_id="creator-1", fan_id="fan-1")
    )

    assert restored is None
    assert unavailable == store.UNAVAILABLE_EXPIRED
    assert db.tables["assisted_provenance"] == [], "and it is swept"


def test_no_token_at_all_says_so():
    restored, unavailable = run(store.redeem(""))

    assert restored is None
    assert unavailable == store.UNAVAILABLE_NO_TOKEN


def test_a_record_written_by_an_older_build_is_a_miss_not_a_crash(db):
    db.tables["assisted_provenance"].append(
        {
            "token": "old",
            "creator_id": "creator-1",
            "fan_id": "fan-1",
            "record": {"something": "unrecognised"},
        }
    )

    restored, unavailable = run(
        store.redeem("old", creator_id="creator-1", fan_id="fan-1")
    )

    assert restored is None
    assert unavailable == store.UNAVAILABLE_MISSING


@pytest.mark.parametrize("state", [None, "a string", 42, {}, {"creator_id": "c"}])
def test_rebuilding_never_raises_on_a_shape_it_did_not_expect(state):
    assert ReplyProvenance.from_state(state) is None


# ===========================================================================
# 3. A miss is admitted, never silent
# ===========================================================================


def test_an_unavailable_record_still_produces_a_record():
    """Saving nothing made three different things look identical: a reply
    typed by hand, a reply whose record was lost, and a reply from a build
    that never recorded provenance."""
    metadata = store.unavailable_metadata(
        store.UNAVAILABLE_EXPIRED, creator_id="creator-1", fan_id="fan-1"
    )
    record = metadata["reply_provenance"]

    assert record["attribution_available"] is False
    assert record["attribution_unavailable_because"] == store.UNAVAILABLE_EXPIRED
    assert record["creator_id"] == "creator-1"


def test_an_unavailable_record_invents_nothing():
    """It says what is missing. It must not fill any of it in."""
    record = store.unavailable_metadata("gone", creator_id="c", fan_id="f")[
        "reply_provenance"
    ]

    for invented in ("writer", "turn_id", "decision", "trigger", "context", "build"):
        assert invented not in record


def test_it_uses_the_same_key_a_real_record_would():
    """So a reader looking for provenance finds this rather than finding
    nothing, which is the whole difference."""
    from services.reply_provenance import PROVENANCE_KEY

    assert PROVENANCE_KEY in store.unavailable_metadata("gone", creator_id="c", fan_id="f")


# ===========================================================================
# 4. Nothing here may cost a message
# ===========================================================================


def test_a_database_that_refuses_the_write_still_returns_a_usable_token(monkeypatch):
    class Exploding:
        def table(self, _name):
            raise RuntimeError("supabase down")

    monkeypatch.setattr(store, "get_supabase", lambda: Exploding())

    token = run(store.remember(_recorder()))

    assert token
    # Still valid on this replica, which is strictly better than failing the
    # suggestion — and the miss it may later cause is now admitted.
    restored, _ = run(store.redeem(token, creator_id="creator-1", fan_id="fan-1"))
    assert restored is not None


def test_a_database_that_refuses_the_read_is_a_miss_not_an_error(monkeypatch):
    class Exploding:
        def table(self, _name):
            raise RuntimeError("supabase down")

    monkeypatch.setattr(store, "get_supabase", lambda: Exploding())

    restored, unavailable = run(store.redeem("anything", creator_id="c", fan_id="f"))

    assert restored is None
    assert unavailable == store.UNAVAILABLE_MISSING


def test_clearing_a_record_never_raises(monkeypatch):
    class Exploding:
        def table(self, _name):
            raise RuntimeError("supabase down")

    monkeypatch.setattr(store, "get_supabase", lambda: Exploding())

    run(store.forget("anything"))  # no raise
