"""Fansly list mirroring: parsing, reconciliation, and tenancy.

The database is exercised through a fake Supabase table layer that enforces the
constraints the migration creates — in particular the unique
(creator_id, external_list_id) mapping the whole reconciliation depends on.
A Postgres-backed check of the migration itself lives in
tests/test_fansly_lists_schema.py.
"""

from __future__ import annotations

import asyncio
import itertools
from typing import Any

import pytest

from services import fansly_lists
from services.apifansly import (
    ApiFanslyAccountAccessError,
    parse_account_lists,
    parse_list_member_ids,
)
from services.fansly_lists import FANSLY_SOURCE, LOCAL_SOURCE, sync_fansly_lists


# --- 1: response parsing ------------------------------------------------------


def test_lists_are_parsed_from_the_documented_envelope():
    parsed = parse_account_lists(
        {
            "lists": [
                {"id": "1001", "label": "VIP", "itemCount": 12},
                {"id": "1002", "label": "Whales", "itemCount": 3},
            ]
        }
    )

    assert parsed == [
        {"external_list_id": "1001", "name": "VIP", "item_count": 12},
        {"external_list_id": "1002", "name": "Whales", "item_count": 3},
    ]


def test_lists_are_parsed_from_alternate_collection_and_name_keys():
    assert parse_account_lists({"items": [{"listId": "7", "name": "Buyers"}]}) == [
        {"external_list_id": "7", "name": "Buyers", "item_count": 0}
    ]
    assert parse_account_lists([{"id": "8", "title": "Re-engage"}]) == [
        {"external_list_id": "8", "name": "Re-engage", "item_count": 0}
    ]


def test_a_list_without_a_remote_id_is_dropped_not_mirrored():
    # The remote id is the only stable mapping key. Mirroring an id-less list
    # would create a duplicate on every sync.
    assert parse_account_lists({"lists": [{"label": "VIP"}, {"id": "", "label": "X"}]}) == []


def test_an_unnamed_list_still_mirrors_under_a_derived_name():
    assert parse_account_lists({"lists": [{"id": "42"}]}) == [
        {"external_list_id": "42", "name": "Fansly list 42", "item_count": 0}
    ]


def test_members_are_parsed_as_account_ids_and_deduplicated():
    assert parse_list_member_ids(
        {"items": [{"accountId": "f1"}, {"accountId": "f2"}, {"accountId": "f1"}]}
    ) == ["f1", "f2"]
    assert parse_list_member_ids({"accounts": ["f9", "f9", "f8"]}) == ["f9", "f8"]
    assert parse_list_member_ids({"items": [{"id": "f3", "username": "alex"}]}) == ["f3"]


def test_member_parsing_ignores_usernames_and_display_names():
    parsed = parse_list_member_ids(
        {"items": [{"accountId": "f1", "username": "alex", "displayName": "Alex"}]}
    )

    assert parsed == ["f1"]


def test_malformed_member_payloads_yield_nothing_rather_than_guesses():
    assert parse_list_member_ids(None) == []
    assert parse_list_member_ids({"items": [{"username": "alex"}]}) == []


# --- fake Supabase ------------------------------------------------------------


class _Result:
    def __init__(self, data):
        self.data = data


class _Query:
    def __init__(self, db, table):
        self.db = db
        self.table_name = table
        self.filters: list[tuple[str, Any]] = []
        self.in_filters: list[tuple[str, list[str]]] = []
        self.op = None
        self.payload = None

    # -- builders
    def select(self, *_args, **_kwargs):
        self.op = "select"
        return self

    def insert(self, values):
        self.op = "insert"
        self.payload = values
        return self

    def update(self, values):
        self.op = "update"
        self.payload = values
        return self

    def upsert(self, values, on_conflict=None):
        self.op = "upsert"
        self.payload = values
        self.on_conflict = on_conflict
        return self

    def delete(self):
        self.op = "delete"
        return self

    def eq(self, column, value):
        self.filters.append((column, value))
        return self

    def in_(self, column, values):
        self.in_filters.append((column, [str(v) for v in values]))
        return self

    def limit(self, _value):
        return self

    # -- execution
    def _matches(self, row):
        for column, value in self.filters:
            if str(row.get(column)) != str(value):
                return False
        for column, values in self.in_filters:
            if str(row.get(column)) not in values:
                return False
        return True

    def execute(self):
        rows = self.db.tables.setdefault(self.table_name, [])
        if self.op == "select":
            return _Result([dict(row) for row in rows if self._matches(row)])
        if self.op == "insert":
            return _Result([self.db.insert(self.table_name, dict(self.payload))])
        if self.op == "upsert":
            return _Result([self.db.upsert(self.table_name, dict(self.payload))])
        if self.op == "update":
            updated = []
            for row in rows:
                if self._matches(row):
                    row.update(self.payload)
                    updated.append(dict(row))
            self.db.writes.append((self.table_name, "update", dict(self.payload)))
            return _Result(updated)
        if self.op == "delete":
            kept = [row for row in rows if not self._matches(row)]
            removed = len(rows) - len(kept)
            self.db.tables[self.table_name] = kept
            self.db.writes.append(
                (self.table_name, "delete", dict(self.filters))
            )
            return _Result([{"deleted": removed}])
        raise AssertionError(f"unsupported operation {self.op}")


class FakeSupabase:
    """Enough Supabase to enforce the invariants the migration guarantees."""

    def __init__(self):
        self.tables: dict[str, list[dict]] = {}
        self.writes: list[tuple[str, str, dict]] = []
        self._ids = itertools.count(1)

    def table(self, name):
        return _Query(self, name)

    def seed(self, name, rows):
        self.tables.setdefault(name, []).extend(dict(row) for row in rows)

    def insert(self, name, values):
        rows = self.tables.setdefault(name, [])
        if name == "fan_lists":
            values.setdefault("source", LOCAL_SOURCE)
            external_id = values.get("external_list_id")
            if external_id is not None:
                # The partial unique index on (creator_id, external_list_id).
                clash = [
                    row
                    for row in rows
                    if row.get("creator_id") == values.get("creator_id")
                    and row.get("external_list_id") == external_id
                ]
                assert not clash, (
                    "duplicate mirror for the same remote list; the unique index "
                    "in db/fansly_lists_v1.sql would have rejected this"
                )
            # The check constraint tying source to external_list_id.
            assert (values["source"] == FANSLY_SOURCE) == (external_id is not None)
            values.setdefault("id", f"list-{next(self._ids)}")
        if name == "fan_list_members":
            values.setdefault("source", LOCAL_SOURCE)
        rows.append(values)
        return dict(values)

    def upsert(self, name, values):
        rows = self.tables.setdefault(name, [])
        if name == "fan_list_members":
            for row in rows:
                if row.get("list_id") == values.get("list_id") and row.get(
                    "fan_id"
                ) == values.get("fan_id"):
                    row.update(values)
                    return dict(row)
        return self.insert(name, values)

    # -- helpers for assertions
    def lists(self, **where):
        return [
            row
            for row in self.tables.get("fan_lists", [])
            if all(row.get(k) == v for k, v in where.items())
        ]

    def members(self, list_id):
        return sorted(
            (row["fan_id"], row.get("source"))
            for row in self.tables.get("fan_list_members", [])
            if row["list_id"] == list_id
        )


@pytest.fixture
def db(monkeypatch):
    fake = FakeSupabase()
    fake.seed("creators", [{"id": "creator-1", "apifansly_account_id": "acct-1"}])
    fake.seed(
        "fans",
        [
            {"id": "fan-a", "creator_id": "creator-1", "platform_fan_id": "p-a"},
            {"id": "fan-b", "creator_id": "creator-1", "platform_fan_id": "p-b"},
            {"id": "fan-c", "creator_id": "creator-1", "platform_fan_id": "p-c"},
            # Another agency's fan, reachable only if creator scoping is broken.
            {"id": "fan-z", "creator_id": "creator-2", "platform_fan_id": "p-z"},
        ],
    )
    monkeypatch.setattr(fansly_lists, "get_supabase", lambda: fake)
    monkeypatch.setenv("FANSLY_LISTS_SYNC_ENABLED", "true")
    return fake


def run_sync(monkeypatch, *, lists, members, account_id="acct-1", creator_id="creator-1"):
    async def fake_fetch_lists(_account_id, *, client):
        assert _account_id == account_id
        return list(lists)

    async def fake_fetch_members(_account_id, external_list_id, *, client):
        return list(members.get(external_list_id, []))

    monkeypatch.setattr(fansly_lists, "fetch_remote_lists", fake_fetch_lists)
    monkeypatch.setattr(fansly_lists, "fetch_remote_members", fake_fetch_members)
    return asyncio.run(sync_fansly_lists(creator_id, account_id))


def _remote(external_id, name, count=0):
    return {"external_list_id": external_id, "name": name, "item_count": count}


# --- 3 & 4: first sync creates, second sync is idempotent --------------------


def test_first_sync_creates_mirrored_lists(db, monkeypatch):
    result = run_sync(
        monkeypatch,
        lists=[_remote("1001", "VIP", 2), _remote("1002", "Whales")],
        members={"1001": ["p-a", "p-b"]},
    )

    assert result["created_lists"] == 2
    mirrors = db.lists(source=FANSLY_SOURCE)
    assert {row["external_list_id"] for row in mirrors} == {"1001", "1002"}
    assert {row["name"] for row in mirrors} == {"VIP", "Whales"}

    vip = db.lists(external_list_id="1001")[0]
    assert db.members(vip["id"]) == [("fan-a", FANSLY_SOURCE), ("fan-b", FANSLY_SOURCE)]
    assert vip["external_synced_at"]
    assert vip["external_archived_at"] is None


def test_second_sync_is_idempotent(db, monkeypatch):
    payload = dict(
        lists=[_remote("1001", "VIP", 2)],
        members={"1001": ["p-a", "p-b"]},
    )
    run_sync(monkeypatch, **payload)
    before = [dict(row) for row in db.lists(source=FANSLY_SOURCE)]

    result = run_sync(monkeypatch, **payload)

    assert result["created_lists"] == 0
    assert result["added_members"] == 0
    assert result["removed_members"] == 0
    assert len(db.lists(source=FANSLY_SOURCE)) == len(before) == 1
    vip = db.lists(external_list_id="1001")[0]
    assert db.members(vip["id"]) == [("fan-a", FANSLY_SOURCE), ("fan-b", FANSLY_SOURCE)]


# --- 5 & 6: rename in place, and same-name lists stay distinct ---------------


def test_remote_rename_updates_the_existing_mirror(db, monkeypatch):
    run_sync(monkeypatch, lists=[_remote("1001", "VIP")], members={"1001": ["p-a"]})
    original_id = db.lists(external_list_id="1001")[0]["id"]

    result = run_sync(
        monkeypatch,
        lists=[_remote("1001", "VIP Buyers")],
        members={"1001": ["p-a"]},
    )

    assert result["renamed_lists"] == 1
    assert result["created_lists"] == 0
    mirrors = db.lists(source=FANSLY_SOURCE)
    assert len(mirrors) == 1
    assert mirrors[0]["id"] == original_id
    assert mirrors[0]["name"] == "VIP Buyers"
    # Memberships survive a rename because the row is the same row.
    assert db.members(original_id) == [("fan-a", FANSLY_SOURCE)]


def test_two_remote_lists_with_the_same_name_stay_distinct(db, monkeypatch):
    run_sync(
        monkeypatch,
        lists=[_remote("1001", "VIP"), _remote("2002", "VIP")],
        members={"1001": ["p-a"], "2002": ["p-b"]},
    )

    mirrors = db.lists(source=FANSLY_SOURCE, name="VIP")
    assert len(mirrors) == 2
    assert {row["external_list_id"] for row in mirrors} == {"1001", "2002"}
    by_external = {row["external_list_id"]: row["id"] for row in mirrors}
    assert db.members(by_external["1001"]) == [("fan-a", FANSLY_SOURCE)]
    assert db.members(by_external["2002"]) == [("fan-b", FANSLY_SOURCE)]


# --- 7 & 11: local lists and local memberships are never touched -------------


def test_local_list_with_the_same_name_is_untouched(db, monkeypatch):
    db.seed(
        "fan_lists",
        [
            {
                "id": "local-1",
                "creator_id": "creator-1",
                "name": "VIP",
                "source": LOCAL_SOURCE,
                "external_list_id": None,
                "exclude_from_auto": False,
            }
        ],
    )
    db.seed(
        "fan_list_members",
        [{"list_id": "local-1", "fan_id": "fan-c", "source": LOCAL_SOURCE}],
    )

    run_sync(monkeypatch, lists=[_remote("1001", "VIP")], members={"1001": ["p-a"]})

    local = db.lists(id="local-1")[0]
    assert local["name"] == "VIP"
    assert local["source"] == LOCAL_SOURCE
    assert local["external_list_id"] is None
    assert db.members("local-1") == [("fan-c", LOCAL_SOURCE)]
    assert len(db.lists(creator_id="creator-1")) == 2


def test_local_membership_on_a_mirrored_list_survives_remote_removal(db, monkeypatch):
    """An operator can hand-add a fan to a mirror; the sync must not undo it."""
    run_sync(monkeypatch, lists=[_remote("1001", "VIP")], members={"1001": ["p-a"]})
    mirror_id = db.lists(external_list_id="1001")[0]["id"]
    db.seed(
        "fan_list_members",
        [{"list_id": mirror_id, "fan_id": "fan-c", "source": LOCAL_SOURCE}],
    )

    run_sync(monkeypatch, lists=[_remote("1001", "VIP")], members={"1001": []})

    # The remotely-sourced membership is gone; the operator's own is not.
    assert db.members(mirror_id) == [("fan-c", LOCAL_SOURCE)]


# --- 8 & 9: mapping by platform_fan_id, unknown fans deferred ----------------


def test_members_are_mapped_by_platform_fan_id(db, monkeypatch):
    run_sync(monkeypatch, lists=[_remote("1001", "VIP")], members={"1001": ["p-b"]})

    mirror_id = db.lists(external_list_id="1001")[0]["id"]
    assert db.members(mirror_id) == [("fan-b", FANSLY_SOURCE)]


def test_unknown_remote_fan_is_deferred_not_fabricated(db, monkeypatch):
    result = run_sync(
        monkeypatch,
        lists=[_remote("1001", "VIP")],
        members={"1001": ["p-a", "p-not-imported-yet"]},
    )

    assert result["unmapped_members"] == 1
    mirror_id = db.lists(external_list_id="1001")[0]["id"]
    assert db.members(mirror_id) == [("fan-a", FANSLY_SOURCE)]
    # No placeholder fan invented to satisfy the membership.
    assert len(db.tables["fans"]) == 4


def test_a_deferred_fan_is_picked_up_once_it_exists_locally(db, monkeypatch):
    run_sync(
        monkeypatch,
        lists=[_remote("1001", "VIP")],
        members={"1001": ["p-a", "p-later"]},
    )
    db.seed(
        "fans",
        [{"id": "fan-later", "creator_id": "creator-1", "platform_fan_id": "p-later"}],
    )

    result = run_sync(
        monkeypatch,
        lists=[_remote("1001", "VIP")],
        members={"1001": ["p-a", "p-later"]},
    )

    assert result["unmapped_members"] == 0
    assert result["added_members"] == 1
    mirror_id = db.lists(external_list_id="1001")[0]["id"]
    assert db.members(mirror_id) == [
        ("fan-a", FANSLY_SOURCE),
        ("fan-later", FANSLY_SOURCE),
    ]


# --- 10: remote membership removal ------------------------------------------


def test_remote_removal_removes_only_the_mirrored_membership(db, monkeypatch):
    run_sync(
        monkeypatch,
        lists=[_remote("1001", "VIP")],
        members={"1001": ["p-a", "p-b"]},
    )
    mirror_id = db.lists(external_list_id="1001")[0]["id"]

    result = run_sync(
        monkeypatch, lists=[_remote("1001", "VIP")], members={"1001": ["p-a"]}
    )

    assert result["removed_members"] == 1
    assert db.members(mirror_id) == [("fan-a", FANSLY_SOURCE)]


# --- 12: stale/deleted remote list ------------------------------------------


def test_a_remotely_deleted_list_is_archived_not_deleted(db, monkeypatch):
    run_sync(
        monkeypatch,
        lists=[_remote("1001", "VIP"), _remote("1002", "Whales")],
        members={"1001": ["p-a"], "1002": ["p-b"]},
    )
    whales_id = db.lists(external_list_id="1002")[0]["id"]

    result = run_sync(monkeypatch, lists=[_remote("1001", "VIP")], members={"1001": ["p-a"]})

    assert result["archived_lists"] == 1
    whales = db.lists(id=whales_id)[0]
    # Still present, so an Auto Audience rule pointing at it still resolves.
    assert whales["external_archived_at"]
    assert db.members(whales_id) == [("fan-b", FANSLY_SOURCE)]


def test_a_restored_remote_list_is_unarchived_in_place(db, monkeypatch):
    run_sync(monkeypatch, lists=[_remote("1001", "VIP")], members={"1001": ["p-a"]})
    mirror_id = db.lists(external_list_id="1001")[0]["id"]
    run_sync(monkeypatch, lists=[], members={})
    assert db.lists(id=mirror_id)[0]["external_archived_at"]

    result = run_sync(monkeypatch, lists=[_remote("1001", "VIP")], members={"1001": ["p-a"]})

    assert result["restored_lists"] == 1
    assert result["created_lists"] == 0
    assert len(db.lists(source=FANSLY_SOURCE)) == 1
    assert db.lists(id=mirror_id)[0]["external_archived_at"] is None


def test_archiving_is_not_repeated_on_every_later_sync(db, monkeypatch):
    run_sync(monkeypatch, lists=[_remote("1001", "VIP")], members={"1001": []})
    run_sync(monkeypatch, lists=[], members={})

    result = run_sync(monkeypatch, lists=[], members={})

    assert result["archived_lists"] == 0


# --- 13: access errors reuse the reconnect/backoff semantics -----------------


def test_access_denied_records_state_and_reraises(db, monkeypatch):
    async def denied(*_args, **_kwargs):
        raise ApiFanslyAccountAccessError("API Fansly access denied for account acct-1")

    monkeypatch.setattr(fansly_lists, "fetch_remote_lists", denied)

    with pytest.raises(ApiFanslyAccountAccessError):
        asyncio.run(sync_fansly_lists("creator-1", "acct-1"))

    creator = db.tables["creators"][0]
    assert "access denied" in creator["fansly_lists_sync_error"]
    assert creator["fansly_lists_sync_failed_at"]
    # A failed sync must not stamp a successful sync time.
    assert not creator.get("last_fansly_lists_sync_at")


def test_transport_failure_records_state_and_reraises(db, monkeypatch):
    async def boom(*_args, **_kwargs):
        raise RuntimeError("upstream 500")

    monkeypatch.setattr(fansly_lists, "fetch_remote_lists", boom)

    with pytest.raises(RuntimeError):
        asyncio.run(sync_fansly_lists("creator-1", "acct-1"))

    assert db.tables["creators"][0]["fansly_lists_sync_error"] == "upstream 500"


def test_successful_sync_clears_a_previous_failure(db, monkeypatch):
    db.tables["creators"][0]["fansly_lists_sync_error"] = "old failure"
    db.tables["creators"][0]["fansly_lists_sync_failed_at"] = "2026-01-01T00:00:00+00:00"

    run_sync(monkeypatch, lists=[_remote("1001", "VIP")], members={"1001": []})

    creator = db.tables["creators"][0]
    assert creator["fansly_lists_sync_error"] is None
    assert creator["fansly_lists_sync_failed_at"] is None
    assert creator["last_fansly_lists_sync_at"]


def test_sync_is_a_no_op_while_disabled(db, monkeypatch):
    monkeypatch.setenv("FANSLY_LISTS_SYNC_ENABLED", "false")

    assert asyncio.run(sync_fansly_lists("creator-1", "acct-1")) == {"status": "disabled"}
    assert db.lists(source=FANSLY_SOURCE) == []


# --- 14: cross-tenant isolation ----------------------------------------------


def test_sync_never_maps_another_agencys_fan(db, monkeypatch):
    """p-z belongs to creator-2 and must not be pulled into creator-1's mirror."""
    result = run_sync(
        monkeypatch,
        lists=[_remote("1001", "VIP")],
        members={"1001": ["p-a", "p-z"]},
    )

    assert result["unmapped_members"] == 1
    mirror_id = db.lists(external_list_id="1001")[0]["id"]
    assert db.members(mirror_id) == [("fan-a", FANSLY_SOURCE)]


def test_reconciliation_never_reads_or_archives_another_creators_mirror(db, monkeypatch):
    db.seed(
        "fan_lists",
        [
            {
                "id": "other-1",
                "creator_id": "creator-2",
                "name": "Their VIP",
                "source": FANSLY_SOURCE,
                "external_list_id": "1001",
                "external_archived_at": None,
            }
        ],
    )

    run_sync(monkeypatch, lists=[], members={})

    other = db.lists(id="other-1")[0]
    assert other["external_archived_at"] is None
    assert other["name"] == "Their VIP"


def test_the_same_remote_id_mirrors_separately_per_creator(db, monkeypatch):
    db.seed("creators", [{"id": "creator-2", "apifansly_account_id": "acct-2"}])
    db.seed(
        "fans",
        [{"id": "fan-z2", "creator_id": "creator-2", "platform_fan_id": "p-z"}],
    )

    run_sync(monkeypatch, lists=[_remote("1001", "VIP")], members={"1001": ["p-a"]})
    run_sync(
        monkeypatch,
        lists=[_remote("1001", "VIP")],
        members={"1001": ["p-z"]},
        account_id="acct-2",
        creator_id="creator-2",
    )

    mirrors = db.lists(external_list_id="1001")
    assert len(mirrors) == 2
    assert {row["creator_id"] for row in mirrors} == {"creator-1", "creator-2"}
