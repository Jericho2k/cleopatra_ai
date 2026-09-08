"""SCALE-003 — Fansly list reconciliation must survive >1,000 fans.

_load_state read `fans` and `fan_list_members` with no .range() and no .order().
PostgREST caps such a select at 1,000 rows and, without an ORDER BY, the
particular 1,000 returned is not stable between calls. _reconcile then treated a
remotely-present fan it could not map as absent and DELETED its mirrored
membership — which the next sync re-added, flapping VIP/Whale targeting every
six hours for any creator above the threshold.

These tests run against tests/fake_supabase.py, which enforces the same 1,000-row
cap and deliberately rotates unordered results. A fake that returned every row
would let the original bug pass.
"""

from __future__ import annotations

import asyncio

import pytest

from services import fansly_lists
from services.fansly_lists import FANSLY_SOURCE, LOCAL_SOURCE, sync_fansly_lists
from tests.fake_supabase import FakeSupabase

CREATOR = "creator-1"
ACCOUNT = "acct-1"
FAN_COUNT = 1500


def _dataset(fan_count: int = FAN_COUNT):
    return {
        "creators": [{"id": CREATOR, "apifansly_account_id": ACCOUNT}],
        "fans": [
            {
                "id": f"fan-{index:05d}",
                "creator_id": CREATOR,
                "platform_fan_id": f"p-{index:05d}",
            }
            for index in range(fan_count)
        ],
        "fan_lists": [],
        "fan_list_members": [],
    }


def _run(db, monkeypatch, *, lists, members, incomplete=(), creator_id=CREATOR):
    async def fake_fetch_lists(_account_id, *, client):
        return list(lists)

    async def fake_fetch_members(_account_id, external_list_id, *, client):
        return (
            list(members.get(external_list_id, [])),
            external_list_id not in incomplete,
        )

    monkeypatch.setattr(fansly_lists, "get_supabase", lambda: db)
    monkeypatch.setattr(fansly_lists, "fetch_remote_lists", fake_fetch_lists)
    monkeypatch.setattr(fansly_lists, "fetch_remote_members", fake_fetch_members)
    monkeypatch.setenv("FANSLY_LISTS_SYNC_ENABLED", "true")
    return asyncio.run(sync_fansly_lists(creator_id, ACCOUNT))


def _remote(external_id, name, count=0):
    return {"external_list_id": external_id, "name": name, "item_count": count}


def _members_of(db, list_id):
    return sorted(
        row["fan_id"]
        for row in db.tables["fan_list_members"]
        if str(row["list_id"]) == str(list_id)
    )


def _mirror_id(db, external_id="1001"):
    return next(
        str(row["id"])
        for row in db.tables["fan_lists"]
        if str(row.get("external_list_id")) == external_id
    )


# --- the core regression ----------------------------------------------------


def test_every_fan_maps_with_1500_local_fans(monkeypatch):
    """Members past row 1,000 must map, not be counted as unmapped."""
    db = FakeSupabase(_dataset())
    # Remote members drawn from across the whole range, including well past the
    # 1,000-row cap that used to truncate the local fan map.
    remote = [f"p-{index:05d}" for index in (5, 999, 1000, 1001, 1250, 1499)]

    result = _run(db, monkeypatch, lists=[_remote("1001", "VIP")], members={"1001": remote})

    assert result["unmapped_members"] == 0
    assert result["added_members"] == 6
    assert result["status"] == "ok"
    assert _members_of(db, _mirror_id(db)) == [
        "fan-00005", "fan-00999", "fan-01000", "fan-01001", "fan-01250", "fan-01499",
    ]


def test_repeat_sync_is_idempotent_and_deletes_nothing(monkeypatch):
    """The flapping regression: run twice, assert no membership churn."""
    db = FakeSupabase(_dataset())
    remote = [f"p-{index:05d}" for index in range(1200)]
    lists = [_remote("1001", "VIP")]

    first = _run(db, monkeypatch, lists=lists, members={"1001": remote})
    before = _members_of(db, _mirror_id(db))
    second = _run(db, monkeypatch, lists=lists, members={"1001": remote})
    after = _members_of(db, _mirror_id(db))

    assert first["added_members"] == 1200
    assert second["added_members"] == 0
    assert second["removed_members"] == 0
    assert second["skipped_removals"] == 0
    assert before == after
    assert len(after) == 1200


def test_membership_beyond_row_1000_is_not_deleted(monkeypatch):
    """A mirrored membership on a high-index fan must survive a resync."""
    db = FakeSupabase(_dataset())
    remote = [f"p-{index:05d}" for index in (1200, 1201, 1499)]
    lists = [_remote("1001", "VIP")]

    _run(db, monkeypatch, lists=lists, members={"1001": remote})
    assert _members_of(db, _mirror_id(db)) == ["fan-01200", "fan-01201", "fan-01499"]

    result = _run(db, monkeypatch, lists=lists, members={"1001": remote})

    assert result["removed_members"] == 0
    assert _members_of(db, _mirror_id(db)) == ["fan-01200", "fan-01201", "fan-01499"]


def test_remote_rename_still_works_past_the_cap(monkeypatch):
    db = FakeSupabase(_dataset())
    members = {"1001": [f"p-{index:05d}" for index in range(1100)]}

    _run(db, monkeypatch, lists=[_remote("1001", "VIP")], members=members)
    result = _run(db, monkeypatch, lists=[_remote("1001", "Whales")], members=members)

    assert result["renamed_lists"] == 1
    mirrors = [row for row in db.tables["fan_lists"] if row.get("external_list_id") == "1001"]
    assert len(mirrors) == 1
    assert mirrors[0]["name"] == "Whales"


def test_local_cleopatra_lists_are_untouched(monkeypatch):
    """Operator-created lists and memberships must never be affected."""
    data = _dataset(fan_count=1200)
    data["fan_lists"].append(
        {"id": "local-1", "creator_id": CREATOR, "name": "My List", "source": LOCAL_SOURCE}
    )
    data["fan_list_members"].append(
        {"list_id": "local-1", "fan_id": "fan-01100", "source": LOCAL_SOURCE}
    )
    db = FakeSupabase(data)

    _run(db, monkeypatch, lists=[_remote("1001", "VIP")], members={"1001": ["p-00001"]})

    assert _members_of(db, "local-1") == ["fan-01100"]
    local = next(row for row in db.tables["fan_lists"] if row["id"] == "local-1")
    assert local["source"] == LOCAL_SOURCE
    assert local["name"] == "My List"


def test_operator_membership_on_a_mirror_survives(monkeypatch):
    """A hand-made membership on a mirrored list is not fansly-sourced."""
    db = FakeSupabase(_dataset(fan_count=1200))
    lists = [_remote("1001", "VIP")]
    _run(db, monkeypatch, lists=lists, members={"1001": ["p-00001"]})
    mirror = _mirror_id(db)
    db.tables["fan_list_members"].append(
        {"list_id": mirror, "fan_id": "fan-01150", "source": LOCAL_SOURCE}
    )

    _run(db, monkeypatch, lists=lists, members={"1001": []})

    assert "fan-01150" in _members_of(db, mirror)


# --- deletion safety --------------------------------------------------------


def test_genuinely_removed_fan_is_removed_when_the_snapshot_is_complete(monkeypatch):
    """The positive case: removal still happens on trustworthy evidence."""
    db = FakeSupabase(_dataset(fan_count=1200))
    lists = [_remote("1001", "VIP")]
    _run(db, monkeypatch, lists=lists, members={"1001": ["p-00001", "p-01100"]})
    assert _members_of(db, _mirror_id(db)) == ["fan-00001", "fan-01100"]

    result = _run(db, monkeypatch, lists=lists, members={"1001": ["p-00001"]})

    assert result["removed_members"] == 1
    assert result["skipped_removals"] == 0
    assert result["status"] == "ok"
    assert _members_of(db, _mirror_id(db)) == ["fan-00001"]


def test_partial_fan_load_performs_no_destructive_removals(monkeypatch):
    """A failing fans read must withhold removals, not delete everything."""
    db = FakeSupabase(_dataset(fan_count=1200))
    lists = [_remote("1001", "VIP")]
    _run(db, monkeypatch, lists=lists, members={"1001": ["p-00001", "p-01100"]})
    mirror = _mirror_id(db)
    assert _members_of(db, mirror) == ["fan-00001", "fan-01100"]

    original_table = db.table

    def exploding_table(name):
        if name == "fans":
            raise RuntimeError("simulated fans read failure")
        return original_table(name)

    db.table = exploding_table  # type: ignore[assignment]
    result = _run(db, monkeypatch, lists=lists, members={"1001": []})
    db.table = original_table  # type: ignore[assignment]

    assert result["removed_members"] == 0
    assert result["skipped_removals"] == 2
    assert result["degraded"] == 1
    assert result["status"] == "degraded"
    assert _members_of(db, mirror) == ["fan-00001", "fan-01100"]


def test_truncated_remote_member_listing_withholds_removals(monkeypatch):
    """A remote listing that hit its page cap is not evidence of absence."""
    db = FakeSupabase(_dataset(fan_count=1200))
    lists = [_remote("1001", "VIP")]
    _run(db, monkeypatch, lists=lists, members={"1001": ["p-00001", "p-01100"]})
    mirror = _mirror_id(db)

    result = _run(
        db,
        monkeypatch,
        lists=lists,
        members={"1001": ["p-00001"]},
        incomplete={"1001"},
    )

    assert result["removed_members"] == 0
    assert result["skipped_removals"] == 1
    assert result["degraded"] == 1
    assert _members_of(db, mirror) == ["fan-00001", "fan-01100"]


def test_membership_for_a_fan_missing_locally_is_kept(monkeypatch):
    """The exact SCALE-003 signature: a membership we cannot justify removing."""
    db = FakeSupabase(_dataset(fan_count=1200))
    lists = [_remote("1001", "VIP")]
    _run(db, monkeypatch, lists=lists, members={"1001": ["p-00001"]})
    mirror = _mirror_id(db)
    # A mirrored membership whose fan row is not in the fans table at all.
    db.tables["fan_list_members"].append(
        {"list_id": mirror, "fan_id": "fan-vanished", "source": FANSLY_SOURCE}
    )

    result = _run(db, monkeypatch, lists=lists, members={"1001": ["p-00001"]})

    assert result["removed_members"] == 0
    assert result["skipped_removals"] == 1
    assert "fan-vanished" in _members_of(db, mirror)


def test_degraded_sync_records_state_rather_than_reporting_success(monkeypatch):
    db = FakeSupabase(_dataset(fan_count=1200))
    lists = [_remote("1001", "VIP")]
    _run(db, monkeypatch, lists=lists, members={"1001": ["p-00001", "p-01100"]})

    _run(db, monkeypatch, lists=lists, members={"1001": ["p-00001"]}, incomplete={"1001"})

    creator = db.tables["creators"][0]
    assert creator["fansly_lists_sync_error"].startswith("degraded:")
    assert creator["fansly_lists_sync_failed_at"]


def test_a_clean_sync_clears_a_previous_degraded_state(monkeypatch):
    db = FakeSupabase(_dataset(fan_count=1200))
    lists = [_remote("1001", "VIP")]
    _run(db, monkeypatch, lists=lists, members={"1001": ["p-00001", "p-01100"]})
    _run(db, monkeypatch, lists=lists, members={"1001": ["p-00001"]}, incomplete={"1001"})
    assert db.tables["creators"][0]["fansly_lists_sync_error"]

    result = _run(db, monkeypatch, lists=lists, members={"1001": ["p-00001"]})

    assert result["status"] == "ok"
    assert db.tables["creators"][0]["fansly_lists_sync_error"] is None
    assert db.tables["creators"][0]["fansly_lists_sync_failed_at"] is None


def test_the_remote_fansly_list_is_never_modified(monkeypatch):
    """Direction of travel is one way; assert no write leaves Cleopatra."""
    db = FakeSupabase(_dataset(fan_count=1200))

    _run(db, monkeypatch, lists=[_remote("1001", "VIP")], members={"1001": ["p-00001"]})

    written_tables = {table for _op, table, _payload in db.writes}
    assert written_tables <= {"fan_lists", "fan_list_members", "creators"}


# --- pagination mechanics ---------------------------------------------------


def test_reads_are_ordered_and_ranged(monkeypatch):
    """Without a deterministic order, paging can drop or duplicate rows."""
    db = FakeSupabase(_dataset(fan_count=1200))

    _run(db, monkeypatch, lists=[_remote("1001", "VIP")], members={"1001": []})

    for table in ("fans", "fan_lists", "fan_list_members"):
        for query in db.queries_for(table):
            assert query.orders, f"{table} read carried no .order()"
            assert query.range is not None, f"{table} read carried no .range()"


@pytest.mark.parametrize("fan_count", [999, 1000, 1001, 2500])
def test_fan_map_is_complete_at_and_around_the_cap(monkeypatch, fan_count):
    db = FakeSupabase(_dataset(fan_count=fan_count))
    remote = [f"p-{index:05d}" for index in range(fan_count)]

    result = _run(db, monkeypatch, lists=[_remote("1001", "VIP")], members={"1001": remote})

    assert result["unmapped_members"] == 0
    assert result["added_members"] == fan_count


def test_preexisting_memberships_past_the_cap_are_not_wiped(monkeypatch):
    """The exact production scenario SCALE-003 describes.

    Three mirrored memberships already exist for fans whose rows sit past the
    1,000-row cap, and all three are still on the remote list. Against the
    original _reconcile this run reported removed_members=3 and emptied the
    list; the next run re-added nothing because the same fans were unmappable,
    so the targeting stayed destroyed.
    """
    data = _dataset(fan_count=1500)
    data["fan_lists"].append(
        {
            "id": "mirror-1",
            "creator_id": CREATOR,
            "name": "VIP",
            "source": FANSLY_SOURCE,
            "external_list_id": "1001",
            "external_archived_at": None,
        }
    )
    data["fan_list_members"].extend(
        {"list_id": "mirror-1", "fan_id": f"fan-{index:05d}", "source": FANSLY_SOURCE}
        for index in (1200, 1201, 1499)
    )
    db = FakeSupabase(data)
    lists = [_remote("1001", "VIP", count=3)]
    members = {"1001": [f"p-{index:05d}" for index in (1200, 1201, 1499)]}
    expected = ["fan-01200", "fan-01201", "fan-01499"]

    for _ in range(3):
        result = _run(db, monkeypatch, lists=lists, members=members)
        assert result["removed_members"] == 0
        assert result["unmapped_members"] == 0
        assert result["skipped_removals"] == 0
        assert _members_of(db, "mirror-1") == expected

    # And the positive case still works: drop one remotely, it goes.
    result = _run(
        db,
        monkeypatch,
        lists=lists,
        members={"1001": [f"p-{index:05d}" for index in (1200, 1201)]},
    )
    assert result["removed_members"] == 1
    assert _members_of(db, "mirror-1") == ["fan-01200", "fan-01201"]
