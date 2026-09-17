"""Seeding a purchase, and refusing to seed one anywhere it must not go.

WHY THE FIXTURE EXISTS
----------------------
"He cannot open something he paid for" is a test of what happens when the
ledger DOES show a purchase and the customer cannot reach it. The fixture's
first turn said "hey i bought that set yesterday" and its own `tests` line
claimed that established a purchase. It established nothing: a customer message
is not a ledger row, and services/content_access.py answers an unpaid
complaint with

    this customer has no confirmed purchase to resend. A complaint is not proof
    of payment [...]

which is correct, and is the opposite branch from the one the trajectory
claims. So the row passed while exercising none of the code it named.

WHY THE GUARD MATTERS MORE
--------------------------
This module writes rows saying a customer paid. Against a real conversation
that is fabricated payment evidence: it would license a free resend of paid
media, move commercial state, and corrupt the one record this system treats as
authoritative about money. The refusal tests come first for that reason.
"""

from __future__ import annotations

import asyncio

import pytest

from services import trajectory_fixtures as fixtures
from tests.fake_supabase import FakeSupabase


def run(coro):
    return asyncio.run(coro)


def _world(*, platform_fan_id: str = "test_evalfan") -> FakeSupabase:
    return FakeSupabase(
        {
            "fans": [
                {
                    "id": "fan-1",
                    "creator_id": "creator-1",
                    "platform_fan_id": platform_fan_id,
                }
            ],
            "creators": [{"id": "creator-1"}],
            "ppv_deliveries": [],
        }
    )


@pytest.fixture
def world(monkeypatch):
    db = _world()
    monkeypatch.setattr(fixtures, "get_supabase", lambda: db)
    return db


# ===========================================================================
# 1. It refuses to write payment evidence anywhere it must not
# ===========================================================================


def test_a_real_customer_is_refused(monkeypatch):
    """The failure that would matter. Not a test fan, so no purchase is written."""
    db = _world(platform_fan_id="9912345")
    monkeypatch.setattr(fixtures, "get_supabase", lambda: db)

    with pytest.raises(fixtures.FixtureRefused) as refused:
        run(fixtures.seed_purchase(
            creator_id="creator-1", fan_id="fan-1",
            reference="r", media_ids=["m1"], price_cents=2500,
        ))

    assert "not a simulator test fan" in str(refused.value)
    assert db.tables["ppv_deliveries"] == []


def test_a_fan_belonging_to_another_creator_is_refused(world):
    """A typo in a CLI argument is how this reaches the wrong conversation."""
    with pytest.raises(fixtures.FixtureRefused) as refused:
        run(fixtures.seed_purchase(
            creator_id="creator-2", fan_id="fan-1",
            reference="r", media_ids=["m1"], price_cents=2500,
        ))

    assert "does not belong to" in str(refused.value)
    assert world.tables["ppv_deliveries"] == []


def test_a_fan_that_does_not_exist_is_refused(world):
    with pytest.raises(fixtures.FixtureRefused):
        run(fixtures.seed_purchase(
            creator_id="creator-1", fan_id="fan-missing",
            reference="r", media_ids=["m1"], price_cents=2500,
        ))


def test_the_guard_reads_the_database_rather_than_trusting_the_caller(monkeypatch):
    """The caller is a CLI argument. It is not evidence about anything."""
    db = _world(platform_fan_id="realcustomer")
    monkeypatch.setattr(fixtures, "get_supabase", lambda: db)

    with pytest.raises(fixtures.FixtureRefused):
        # Passing an id that LOOKS like a test fan changes nothing; the fan row
        # is what decides.
        run(fixtures.seed_purchase(
            creator_id="creator-1", fan_id="fan-1",
            reference="test_looks_fine", media_ids=["m1"], price_cents=100,
        ))


def test_clearing_is_refused_on_a_real_customer(monkeypatch):
    """Deleting deliveries from a real conversation is its own disaster."""
    db = _world(platform_fan_id="9912345")
    db.tables["ppv_deliveries"].append(
        {"reference": "real-1", "creator_id": "creator-1", "fan_id": "fan-1"}
    )
    monkeypatch.setattr(fixtures, "get_supabase", lambda: db)

    with pytest.raises(fixtures.FixtureRefused):
        run(fixtures.clear_seeded("creator-1", "fan-1"))

    assert len(db.tables["ppv_deliveries"]) == 1


# ===========================================================================
# 2. On a test fan, it writes what the ledger would
# ===========================================================================


def test_a_seeded_purchase_is_what_the_ledger_calls_paid(world):
    row = run(fixtures.seed_purchase(
        creator_id="creator-1", fan_id="fan-1",
        reference="content-access-set", media_ids=["m1", "m2"], price_cents=2500,
    ))

    assert row["status"] == "purchased"
    assert row["media_ids"] == ["m1", "m2"]
    assert row["price_cents"] == 2500
    assert world.tables["ppv_deliveries"][0]["status"] == "purchased"


def test_a_seeded_reference_says_it_was_seeded(world):
    """Forever after, to anyone reading the table."""
    row = run(fixtures.seed_purchase(
        creator_id="creator-1", fan_id="fan-1",
        reference="content-access-set", media_ids=["m1"], price_cents=100,
    ))

    assert row["reference"].startswith(fixtures.SEED_PREFIX)
    assert fixtures.is_seeded_reference(row["reference"])
    assert not fixtures.is_seeded_reference("ppv-real-1")


def test_seeding_twice_does_not_create_two_purchases(world):
    """Re-running a trajectory must not double the customer's spend on paper."""
    for _ in range(2):
        run(fixtures.seed_purchase(
            creator_id="creator-1", fan_id="fan-1",
            reference="same", media_ids=["m1"], price_cents=100,
        ))

    assert len(world.tables["ppv_deliveries"]) == 1


def test_a_purchase_can_be_placed_in_the_past(world, monkeypatch):
    """So a conversation can return to it after a simulated absence."""
    row = run(fixtures.seed_purchase(
        creator_id="creator-1", fan_id="fan-1",
        reference="old", media_ids=["m1"], price_cents=100,
        purchased_days_ago=7,
    ))

    from datetime import datetime, timezone

    purchased = datetime.fromisoformat(row["purchased_at"])
    assert (datetime.now(timezone.utc) - purchased).days >= 6


def test_the_seed_block_applies_every_purchase_it_declares(world):
    written = run(fixtures.apply_seed(
        creator_id="creator-1", fan_id="fan-1",
        seed={
            "purchases": [
                {"reference": "a", "media_ids": ["m1"], "price_cents": 100},
                {"reference": "b", "media_ids": ["m2"], "price_cents": 200},
            ]
        },
    ))

    assert len(written) == 2
    assert len(world.tables["ppv_deliveries"]) == 2


def test_an_empty_seed_writes_nothing(world):
    assert run(fixtures.apply_seed(creator_id="creator-1", fan_id="fan-1", seed={})) == []
    assert world.tables["ppv_deliveries"] == []


# ===========================================================================
# 3. Scenarios do not inherit each other's state
# ===========================================================================


def test_clearing_removes_seeded_purchases(world):
    run(fixtures.seed_purchase(
        creator_id="creator-1", fan_id="fan-1",
        reference="a", media_ids=["m1"], price_cents=100,
    ))

    assert run(fixtures.clear_seeded("creator-1", "fan-1")) == 1
    assert world.tables["ppv_deliveries"] == []


def test_clearing_leaves_a_delivery_nobody_seeded(world):
    """A test fan may also carry real rows an operator made by hand."""
    world.tables["ppv_deliveries"].append({
        "reference": "ppv-real-1",
        "creator_id": "creator-1",
        "fan_id": "fan-1",
        "status": "purchased",
    })
    run(fixtures.seed_purchase(
        creator_id="creator-1", fan_id="fan-1",
        reference="a", media_ids=["m1"], price_cents=100,
    ))

    run(fixtures.clear_seeded("creator-1", "fan-1"))

    assert [row["reference"] for row in world.tables["ppv_deliveries"]] == ["ppv-real-1"]


# ===========================================================================
# 4. The trajectory that needed this
# ===========================================================================


def test_the_content_access_trajectory_now_seeds_a_purchase():
    """The row's own claim, checked against the file."""
    import json
    from pathlib import Path

    from services.trajectory_eval import coverage_gaps, load_trajectories, TrajectoryReport

    root = Path(__file__).resolve().parents[1]
    payload = json.loads((root / "eval" / "trajectories.json").read_text())
    trajectories = load_trajectories(payload["trajectories"])
    row = next(
        t for t in trajectories
        if t.name == "he cannot open something he paid for, then asks for more"
    )

    assert row.requires_authoritative_purchase
    assert row.seed["purchases"][0]["price_cents"] == 2500

    # Without the seed applied, the claim is uncovered ...
    bare = TrajectoryReport(trajectory=row.name)
    assert [gap.claim for gap in coverage_gaps(row, bare)] == [
        "a purchase the ledger records"
    ]

    # ... and with it, it is covered.
    seeded = TrajectoryReport(
        trajectory=row.name,
        seeded_purchases=[{"reference": "eval:content-access-set", "price_cents": 2500}],
    )
    assert coverage_gaps(row, seeded) == []
