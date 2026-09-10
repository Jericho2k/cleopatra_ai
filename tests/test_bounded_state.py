"""REL-004 — in-process state stays bounded without losing its whole window.

The audit found three structures that grew or forgot wrongly:

  * ``_processed_messages`` was cleared WHOLESALE at 1,000 entries, so item
    1,001 erased the previous 1,000 identities at once and left the dedupe
    window empty for the messages that followed;
  * ``_chat_last_message_ids`` was never pruned at all (removed entirely by
    API-001 — the checkpoint is a database column now);
  * ``_active_chat_binding_retry_after`` only ever had entries removed on
    SUCCESS, so a fan whose binding never resolved stayed in it forever.

The first is the one that could actually hurt a fan, so it gets the sharpest
test: after 10,000 messages the structure is bounded AND still remembers the
most recent identities, which the old wholesale clear did not.
"""

from __future__ import annotations

import pytest

from core.bounded_state import BoundedIdSet, prune_expired


# --- BoundedIdSet -----------------------------------------------------------


def test_it_never_exceeds_its_bound() -> None:
    seen = BoundedIdSet(maxsize=100)

    for index in range(10_000):
        seen.add(f"m-{index}")

    assert len(seen) == 100


def test_the_dedupe_window_is_never_emptied() -> None:
    """The regression that mattered.

    With the old `if len(seen) > 1000: seen.clear()`, the id added immediately
    before the boundary was forgotten the moment the boundary was crossed. A
    redelivery arriving right then was processed twice.
    """
    seen = BoundedIdSet(maxsize=1000)

    for index in range(1000):
        seen.add(f"m-{index}")
    assert "m-999" in seen

    seen.add("m-1000")  # the item that used to trigger the wholesale clear

    assert "m-1000" in seen
    assert "m-999" in seen, "crossing the bound emptied the dedupe window"
    assert len(seen) == 1000


def test_it_forgets_the_oldest_first() -> None:
    seen = BoundedIdSet(maxsize=3)

    for item in ("a", "b", "c", "d"):
        seen.add(item)

    assert "a" not in seen
    assert list(seen) == ["b", "c", "d"]


def test_re_adding_refreshes_position() -> None:
    """A repeatedly seen id should not be evicted while it is still arriving."""
    seen = BoundedIdSet(maxsize=3)
    for item in ("a", "b", "c"):
        seen.add(item)

    seen.add("a")
    seen.add("d")

    assert "a" in seen
    assert "b" not in seen


def test_ten_thousand_messages_stay_bounded_and_remember_the_newest() -> None:
    seen = BoundedIdSet(maxsize=5000)

    for index in range(10_000):
        seen.add(f"m-{index}")

    assert len(seen) == 5000
    assert "m-9999" in seen
    assert "m-5000" in seen
    assert "m-4999" not in seen


def test_a_bound_below_one_is_rejected() -> None:
    with pytest.raises(ValueError):
        BoundedIdSet(maxsize=0)


def test_discard_and_clear() -> None:
    seen = BoundedIdSet(maxsize=10)
    seen.add("a")
    seen.discard("a")
    seen.discard("never-present")
    assert "a" not in seen

    seen.add("b")
    seen.clear()
    assert len(seen) == 0


# --- prune_expired ----------------------------------------------------------


def test_expired_entries_are_removed() -> None:
    deadlines = {"fan-1": 10.0, "fan-2": 100.0}

    removed = prune_expired(deadlines, now=50.0)

    assert removed == 1
    assert deadlines == {"fan-2": 100.0}


def test_entries_for_things_that_no_longer_exist_are_removed() -> None:
    deadlines = {"creator-1": 100.0, "creator-gone": 100.0}

    removed = prune_expired(deadlines, now=50.0, keep={"creator-1"})

    assert removed == 1
    assert deadlines == {"creator-1": 100.0}


def test_protected_entries_survive_expiry() -> None:
    """An entry with work in flight is also coalescing concurrent callers, and
    that job outlives the backoff deadline."""
    deadlines = {"fan-1": 10.0, "fan-2": 10.0}

    removed = prune_expired(deadlines, now=50.0, protect={"fan-1"})

    assert removed == 1
    assert deadlines == {"fan-1": 10.0}


def test_pruning_is_idempotent_and_reports_zero_when_clean() -> None:
    deadlines = {"fan-1": 100.0}

    assert prune_expired(deadlines, now=50.0) == 0
    assert prune_expired(deadlines, now=50.0) == 0
    assert deadlines == {"fan-1": 100.0}


def test_ten_thousand_stale_fans_do_not_accumulate() -> None:
    """The _active_chat_binding_retry_after failure mode: fans whose binding
    never resolved used to stay for the life of the process."""
    deadlines = {f"fan-{index}": float(index) for index in range(10_000)}

    prune_expired(deadlines, now=9_990.0)

    # Only the entries whose backoff has genuinely not elapsed remain, so the
    # map is bounded by fans in backoff AT ONCE, not by fans ever seen.
    assert set(deadlines) == {f"fan-{index}" for index in range(9_991, 10_000)}
