"""API-003 and FE-006 — what sync_chats costs a creator with a lot of chats.

API-003: the incremental early-break compares each page of Fansly chats against
the set of platform ids we already know. That set came from an unranged select,
so PostgREST capped it at 1,000 rows and the comparison was almost never true
for a bigger creator. Those creators paginated the *entire* Fansly chat list on
every reconcile pass, every 10-30 minutes, forever.

FE-006: the loop then issued fans.update(...) for every chat whether or not
anything had changed. Every UPDATE is delivered to every subscribed dashboard as
a realtime event, so a 2,000-chat creator produced 2,000 writes and a
2,000-event burst per pass.

Both are measured here against a PostgREST double that reproduces the 1,000-row
cap, because a test double that returns everything would let API-003 pass.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

import main
from tests.fake_supabase import FakeSupabase

CREATOR_ID = "creator-1"
ACCOUNT_ID = "apifansly-account-1"
PAGE_SIZE = 50


def _fan_rows(count: int) -> list[dict]:
    return [
        {
            "id": f"fan-{index:06d}",
            "creator_id": CREATOR_ID,
            "platform_fan_id": f"p-{index:06d}",
            "display_name": f"Fan {index}",
            "fansly_group_id": f"g-{index:06d}",
            "avatar_url": None,
        }
        for index in range(count)
    ]


def _chat(index: int) -> dict:
    return {
        "partnerAccountId": f"p-{index:06d}",
        "groupId": f"g-{index:06d}",
        "partnerUsername": f"Fan {index}",
        "lastMessageId": f"m-{index:06d}",
    }


def _account(index: int) -> dict:
    return {"id": f"p-{index:06d}", "displayName": f"Fan {index}"}


@pytest.fixture
def sync_env(monkeypatch):
    """Wire sync_chats to a fake database and a fake Fansly chat list."""

    state: dict = {"chat_pages_served": 0, "chats": [], "created": []}

    db = FakeSupabase({
        "creators": [{
            "id": CREATOR_ID,
            "apifansly_account_id": ACCOUNT_ID,
            "fansly_account_id": "",
            "auto_mode": False,
        }],
        "fans": [],
    })
    db.rpc = lambda *_args, **_kwargs: SimpleNamespace(
        execute=lambda: SimpleNamespace(data=True)
    )
    state["db"] = db

    monkeypatch.setattr(main, "get_supabase", lambda: db)

    async def fake_list_chats(account_id, *, cursor=None, client=None, **_kwargs):
        state["chat_pages_served"] += 1
        start = int(cursor or 0)
        page = state["chats"][start:start + PAGE_SIZE]
        next_cursor = (
            str(start + PAGE_SIZE)
            if start + PAGE_SIZE < len(state["chats"])
            else None
        )
        indexes = [int(chat["partnerAccountId"].split("-")[1]) for chat in page]
        return page, [_account(index) for index in indexes], next_cursor

    monkeypatch.setattr(main, "apifansly_list_chats", fake_list_chats)

    async def fake_create_fan(creator_id, platform_fan_id, display_name):
        state["created"].append(platform_fan_id)
        return SimpleNamespace(id=f"new-{platform_fan_id}")

    monkeypatch.setattr(main, "create_fan", fake_create_fan)

    async def noop(*_args, **_kwargs):
        return None

    # A forced pass also refreshes the platform audience; that is a separate
    # concern with its own tests and would otherwise reach the network here.
    import services.fansly_audience as fansly_audience

    async def fake_audience(*_args, **_kwargs):
        return {"status": "ok"}

    monkeypatch.setattr(fansly_audience, "sync_fansly_audience", fake_audience)

    monkeypatch.setattr(main, "_stamp_vault_op", noop)
    monkeypatch.setattr(main, "_sync_fansly_lists_if_due", noop)
    monkeypatch.setattr(main, "_sync_recent_fan_messages", noop)

    return state


def _fan_update_writes(db: FakeSupabase) -> list:
    return [
        payload
        for op, table, payload in db.writes
        if table == "fans" and op == "update"
    ]


def _run_incremental(state):
    return asyncio.run(main.sync_chats(CREATOR_ID, incremental=True, force=True))


# --- API-003 ----------------------------------------------------------------


@pytest.mark.parametrize("fan_count", [500, 1500, 10_000])
def test_incremental_sync_stops_after_one_page_of_known_chats(sync_env, fan_count):
    """The early-break must survive past the PostgREST cap.

    Every chat on the first page is already known, so the pass should read one
    page and stop — at 500 fans and at 10,000 alike. Before the fix the known-id
    set was truncated at 1,000, so at 1,500 and 10,000 the subset test failed
    and the whole chat list was paginated.
    """

    state = sync_env
    state["db"].tables["fans"] = _fan_rows(fan_count)
    state["chats"] = [_chat(index) for index in range(fan_count)]

    result = _run_incremental(state)

    assert result["status"] == "ok"
    assert state["chat_pages_served"] == 1
    assert result["synced"] == PAGE_SIZE
    assert result["new_chats"] == 0


def test_the_known_id_set_is_read_with_a_deterministic_total_order(sync_env):
    """Paging without a unique order can drop or repeat rows at a boundary."""

    state = sync_env
    state["db"].tables["fans"] = _fan_rows(2500)
    state["chats"] = [_chat(index) for index in range(2500)]

    _run_incremental(state)

    fan_reads = [
        query
        for query in state["db"].queries_for("fans")
        if query.range is not None
    ]
    assert fan_reads, "the known-id read must be paginated"
    assert all(query.orders == [("id", False)] for query in fan_reads)
    # 2,500 rows is three pages of 1,000, and the short last page ends it.
    assert len(fan_reads) == 3


def test_a_genuinely_new_chat_past_the_cap_is_still_imported(sync_env):
    """The early-break must not skip a new chat sitting on the first page."""

    state = sync_env
    state["db"].tables["fans"] = _fan_rows(1500)
    # Fansly returns most-recently-active first, so a brand new chat leads.
    state["chats"] = [_chat(99_999)] + [_chat(index) for index in range(1500)]

    result = _run_incremental(state)

    assert state["created"] == ["p-099999"]
    assert result["new_chats"] == 1
    # The page carried an unknown id, so it could not end the scan there.
    assert state["chat_pages_served"] == 2


# --- FE-006 -----------------------------------------------------------------


def test_unchanged_chats_write_nothing(sync_env):
    state = sync_env
    state["db"].tables["fans"] = _fan_rows(1500)
    state["chats"] = [_chat(index) for index in range(1500)]

    result = _run_incremental(state)

    assert _fan_update_writes(state["db"]) == []
    assert result["updated"] == 0
    assert result["synced"] == PAGE_SIZE


def test_a_changed_display_name_writes_exactly_once(sync_env):
    state = sync_env
    rows = _fan_rows(1500)
    rows[3]["display_name"] = "Stale Name"
    state["db"].tables["fans"] = rows
    state["chats"] = [_chat(index) for index in range(1500)]

    result = _run_incremental(state)

    writes = _fan_update_writes(state["db"])
    assert writes == [{"display_name": "Fan 3"}]
    assert result["updated"] == 1


def test_a_changed_group_binding_writes_exactly_once(sync_env):
    """Chat binding correctness must not regress into the no-op skip."""

    state = sync_env
    rows = _fan_rows(1500)
    rows[7]["fansly_group_id"] = None
    state["db"].tables["fans"] = rows
    state["chats"] = [_chat(index) for index in range(1500)]

    result = _run_incremental(state)

    writes = _fan_update_writes(state["db"])
    assert writes == [{"fansly_group_id": "g-000007"}]
    assert result["updated"] == 1


def test_a_new_fan_still_gets_its_binding_written(sync_env):
    state = sync_env
    state["db"].tables["fans"] = _fan_rows(100)
    state["chats"] = [_chat(99_999)] + [_chat(index) for index in range(100)]

    result = _run_incremental(state)

    writes = _fan_update_writes(state["db"])
    assert writes == [{"fansly_group_id": "g-099999"}]
    assert result["new_chats"] == 1


def test_one_read_replaces_the_per_chat_fan_lookup(sync_env):
    """get_fan() was a select("*") per chat, pulling every JSONB column.

    A full pass over 200 chats used to be 200 of those plus 200 UPDATEs. It is
    now one paginated read and no writes at all when nothing changed.
    """

    state = sync_env
    state["db"].tables["fans"] = _fan_rows(200)
    state["chats"] = [_chat(index) for index in range(200)]

    asyncio.run(main.sync_chats(CREATOR_ID, incremental=False, force=True))

    fan_reads = state["db"].queries_for("fans")
    assert len(fan_reads) == 1
    assert "*" not in fan_reads[0].columns
    assert _fan_update_writes(state["db"]) == []
