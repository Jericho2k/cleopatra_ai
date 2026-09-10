"""Bounded in-process state that forgets the oldest thing, not everything.

Audit reference: REL-004.

The pattern this replaces looked like a bound but was not one::

    seen.add(message_id)
    if len(seen) > 1000:
        seen.clear()

Item 1,001 erased the identities of the previous 1,000 messages at once. For the
following moments the deduplication window was EMPTY, so a webhook redelivery or
a poller/webhook overlap arriving in that window was processed a second time.
The failure is invisible in normal operation and shows up as a duplicate reply
to a fan, which is the single most damaging thing this product can do in front
of an agency.

A bounded FIFO forgets one identity to make room for one identity, so the window
is always the full size. Evicting the oldest is right for this data: identities
arrive in roughly time order and a redelivery follows its original closely, so
the entries most worth keeping are the newest.

These are conveniences, not correctness boundaries. Message identity is enforced
by a unique index (REL-002) and purchase identity by another (REL-003); this
layer only avoids the wasted work of discovering that in the database. Losing
the whole structure on restart is therefore fine, which is why it stays in
process memory.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator
from typing import Any


class BoundedIdSet:
    """A set of recent identities that never holds more than ``maxsize``.

    Adding an identity that is already present refreshes its position, so a
    repeatedly seen id is not evicted while it is still being seen.
    """

    __slots__ = ("_items", "_maxsize")

    def __init__(self, maxsize: int = 1000) -> None:
        if maxsize < 1:
            raise ValueError("maxsize must be at least 1")
        self._maxsize = maxsize
        self._items: OrderedDict[Any, None] = OrderedDict()

    def add(self, item: Any) -> None:
        if item in self._items:
            self._items.move_to_end(item)
            return
        self._items[item] = None
        # `while`, not `if`: a maxsize lowered at runtime must still converge.
        while len(self._items) > self._maxsize:
            self._items.popitem(last=False)

    def discard(self, item: Any) -> None:
        self._items.pop(item, None)

    def clear(self) -> None:
        self._items.clear()

    @property
    def maxsize(self) -> int:
        return self._maxsize

    def __contains__(self, item: Any) -> bool:
        return item in self._items

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[Any]:
        return iter(self._items)

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"BoundedIdSet(size={len(self._items)}, maxsize={self._maxsize})"


def prune_expired(
    deadlines: dict[str, float],
    now: float,
    *,
    keep: set[str] | None = None,
    protect: set[str] | None = None,
) -> int:
    """Drop entries whose deadline has passed, and entries no longer relevant.

    ``deadlines`` maps a key to a time before which it should be left alone.
    An entry is removed when

      * its deadline has passed — the backoff it encoded is over, so the entry
        and its absence now mean the same thing; or
      * ``keep`` is supplied and the key is not in it — the fan or creator it
        referred to is gone.

    ``protect`` overrides both: those keys are never removed. Callers use it for
    entries with work in flight, where the entry is doing a second job
    (coalescing concurrent callers onto one attempt) that outlives its deadline.

    Returns how many entries were removed, so callers can log a number rather
    than guess.

    This exists so retry maps can be cleaned during work the process is already
    doing, rather than by a background task whose only job is a few dicts.
    """
    doomed = [
        key
        for key, deadline in deadlines.items()
        if (protect is None or key not in protect)
        and (deadline <= now or (keep is not None and key not in keep))
    ]
    for key in doomed:
        deadlines.pop(key, None)
    return len(doomed)
