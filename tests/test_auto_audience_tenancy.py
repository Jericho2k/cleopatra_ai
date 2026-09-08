"""SEC-002 — preview_auto_audience must not scan other agencies' memberships.

The endpoint read the entire fan_list_members table with service-role
credentials and no creator filter, then filtered in Python. Two defects:

- privacy: every agency's membership rows entered a request scoped to one
  creator, with a Python-side filter as the only thing preventing a leak;
- correctness: PostgREST truncates at 1,000 rows *globally*, so past ~1,000
  total memberships the requesting creator's own rows were usually absent,
  `legacy_exclusions` came back empty, and the exclusion policy silently
  stopped applying in the preview operators use to decide whether to enable
  Auto.

These tests run against tests/fake_supabase.py, which enforces the same 1,000-row
cap; a mock that returned everything could not fail on the second defect.
"""

from __future__ import annotations

import asyncio

import pytest

import main
from tests.fake_supabase import FakeSupabase

OURS = "creator-ours"
THEIRS = "creator-theirs"


def _dataset(*, our_fans: int, their_memberships: int, our_excluded_fans: int):
    """Two agencies whose data looks alike, sized to cross the 1,000-row cap."""
    creators = [
        {"id": OURS, "auto_mode": True, "auto_audience_policy": {"scope": "all"}},
        {"id": THEIRS, "auto_mode": True, "auto_audience_policy": {"scope": "all"}},
    ]
    fans = [
        {
            "id": f"our-fan-{index:05d}",
            "creator_id": OURS,
            "auto_mode": None,
            "total_spent": 0,
            "spend_tier": "cold",
            "needs_human_review": False,
        }
        for index in range(our_fans)
    ]
    fans += [
        {
            "id": f"their-fan-{index:05d}",
            "creator_id": THEIRS,
            "auto_mode": None,
            "total_spent": 0,
            "spend_tier": "cold",
            "needs_human_review": False,
        }
        for index in range(50)
    ]

    fan_lists = [
        {"id": "our-vip", "creator_id": OURS, "exclude_from_auto": False},
        {"id": "our-blocked", "creator_id": OURS, "exclude_from_auto": True},
        {"id": "their-blocked", "creator_id": THEIRS, "exclude_from_auto": True},
    ]

    # The other agency's memberships are listed FIRST and are numerous enough to
    # fill the 1,000-row global cap on their own.
    members = [
        {"list_id": "their-blocked", "fan_id": f"their-fan-{index:05d}"}
        for index in range(their_memberships)
    ]
    members += [
        {"list_id": "our-blocked", "fan_id": f"our-fan-{index:05d}"}
        for index in range(our_excluded_fans)
    ]

    return {
        "creators": creators,
        "fans": fans,
        "fan_lists": fan_lists,
        "fan_list_members": members,
        "messages": [],
    }


def _preview(db, creator_id=OURS):
    main.get_supabase = lambda: db  # type: ignore[assignment]
    return asyncio.run(main.preview_auto_audience(creator_id))


@pytest.fixture(autouse=True)
def _restore_supabase():
    original = main.get_supabase
    yield
    main.get_supabase = original


# --- tenancy ----------------------------------------------------------------


def test_membership_query_is_constrained_to_the_creators_own_lists():
    """The constraint must reach Postgres, not a Python filter afterwards."""
    db = FakeSupabase(_dataset(our_fans=5, their_memberships=10, our_excluded_fans=2))

    _preview(db)

    membership_queries = db.queries_for("fan_list_members")
    assert membership_queries, "expected a fan_list_members read"
    for query in membership_queries:
        list_filters = query.filter_values("in", "list_id")
        assert list_filters, "membership read carried no list_id constraint"
        for allowed in list_filters:
            assert set(allowed) <= {"our-vip", "our-blocked"}, (
                "membership query would read another agency's lists"
            )


def test_no_other_agency_membership_row_is_ever_read():
    """Replay each executed query and assert nothing foreign came back."""
    db = FakeSupabase(_dataset(our_fans=5, their_memberships=10, our_excluded_fans=2))

    _preview(db)

    foreign_lists = {"their-blocked"}
    for query in db.queries_for("fan_list_members"):
        for _kind, column, value in query.filters:
            if column == "list_id":
                assert not (set(map(str, value)) & foreign_lists)

    for query in db.queries_for("fan_lists"):
        assert query.filter_values("eq", "creator_id") == [OURS]
    for query in db.queries_for("fans"):
        assert query.filter_values("eq", "creator_id") == [OURS]
    for query in db.queries_for("messages"):
        assert query.filter_values("eq", "creator_id") == [OURS]


def test_requesting_creator_cannot_preview_another_agency():
    """Sanity: asking for the other creator returns only their own totals."""
    db = FakeSupabase(_dataset(our_fans=5, their_memberships=10, our_excluded_fans=2))

    result = _preview(db, creator_id=THEIRS)

    assert result["total"] == 50


# --- the 1,000-row cap ------------------------------------------------------


def test_exclusions_still_apply_past_1000_global_memberships():
    """The regression: another agency's rows used to fill the global cap."""
    db = FakeSupabase(
        _dataset(our_fans=20, their_memberships=1500, our_excluded_fans=8)
    )

    result = _preview(db)

    assert result["total"] == 20
    assert result["reasons"].get("excluded_list") == 8, (
        "the creator's own exclusion list stopped applying past the row cap"
    )
    assert result["eligible"] == 12


def test_more_than_1000_fans_for_one_creator_are_all_counted():
    db = FakeSupabase(
        _dataset(our_fans=1500, their_memberships=1200, our_excluded_fans=25)
    )

    result = _preview(db)

    assert result["total"] == 1500
    assert result["reasons"].get("excluded_list") == 25
    assert result["eligible"] == 1475
    assert result["eligible"] + result["ineligible"] == result["total"]


def test_new_only_scope_counts_creator_messages_past_the_cap():
    """is_new_fan is derived from the messages read, which was also unbounded."""
    data = _dataset(our_fans=1200, their_memberships=0, our_excluded_fans=0)
    data["creators"][0]["auto_audience_policy"] = {"scope": "new_only"}
    # 1,100 creator messages, all to the first 300 fans, so paging must not stop
    # at 1,000 rows or the tail of those fans looks "new".
    data["messages"] = [
        {
            "id": f"msg-{index:05d}",
            "creator_id": OURS,
            "role": "creator",
            "fan_id": f"our-fan-{index % 300:05d}",
        }
        for index in range(1100)
    ]
    db = FakeSupabase(data)

    result = _preview(db)

    assert result["total"] == 1200
    assert result["reasons"].get("not_new") == 300
    assert result["reasons"].get("new_fan") == 900


# --- semantics unchanged ----------------------------------------------------


def test_policy_exclusions_and_legacy_exclusions_both_apply():
    data = _dataset(our_fans=10, their_memberships=5, our_excluded_fans=3)
    data["creators"][0]["auto_audience_policy"] = {
        "scope": "all",
        "exclude_list_ids": ["our-vip"],
    }
    data["fan_list_members"].append(
        {"list_id": "our-vip", "fan_id": "our-fan-00009"}
    )
    db = FakeSupabase(data)

    result = _preview(db)

    # 3 on the legacy exclude_from_auto list plus 1 on the policy exclude list.
    assert result["reasons"].get("excluded_list") == 4
    assert result["eligible"] == 6


def test_creator_auto_off_still_reports_the_if_on_projection():
    data = _dataset(our_fans=10, their_memberships=0, our_excluded_fans=2)
    data["creators"][0]["auto_mode"] = False
    db = FakeSupabase(data)

    result = _preview(db)

    assert result["eligible"] == 0
    assert result["reasons"].get("creator_auto_off") == 10
    assert result["eligible_if_creator_on"] == 8
    assert result["reasons_if_creator_on"].get("excluded_list") == 2
