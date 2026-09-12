"""The live/simulation boundary for vault content.

One boolean column decides whether a vault row is real inventory or owner-only
test inventory, and this module is the only place that reads it. Two rules, and
nothing else:

*Live planning excludes simulation-only rows.* Every read that can end in a PPV
being offered or delivered applies :func:`exclude_simulation_only`.

*The owner simulator may include them.* It runs inside
``core.apifansly_gate.simulation_scope()``, which is already the process-wide
statement "this turn cannot reach the platform". Reusing it means the inclusion
rule and the no-remote-calls rule are the same fact, and cannot drift apart.

The rewritten media id
----------------------
Mirrored media carries a ``sim:`` id rather than the source creator's platform
media id. That is the second, independent barrier: the flag keeps mirrored
content out of planning, and the id shape keeps it out of the platform even if
some future code path forgets the flag. A ``sim:`` id is not a Fansly media id
and cannot be one, so a delivery built from it fails structurally rather than
sending another account's media from this one.

Schema tolerance
----------------
``simulation_only`` arrives with db/simulation_catalog_v1.sql. Until that is
applied, PostgREST answers an unknown column with 42703 and the whole read
fails — the failure mode that took ``GET /simulation/creators`` down in #31. So
the filter is applied through :func:`run_live_catalog_query`, which retries once
without it and says so in the log. A deployment that ships ahead of its
migration keeps working; it simply has no mirrored rows to exclude yet, which is
true by construction.
"""

from __future__ import annotations

from typing import Any, Callable, TypeVar

from core.apifansly_gate import simulation_active

SIMULATION_ONLY_COLUMN = "simulation_only"

# The prefix every mirrored media id carries.
SIMULATION_MEDIA_PREFIX = "sim:"

# PostgREST / Postgres signals for "that column does not exist here".
_UNDEFINED_COLUMN_MARKERS = (
    "42703",
    "does not exist",
    "undefined column",
    "could not find the",
)

_warned_missing_column = False

T = TypeVar("T")


def simulation_catalog_visible() -> bool:
    """Whether this execution context may see owner-only test content."""
    return simulation_active()


def is_simulation_media_id(media_id: Any) -> bool:
    """Whether one media id belongs to the owner-only simulation catalog."""
    return str(media_id or "").startswith(SIMULATION_MEDIA_PREFIX)


def simulation_media_id(source_creator_id: str, media_id: str) -> str:
    """The mirrored id for one source media item. Stable, so mirrors are idempotent."""
    short = str(source_creator_id or "").replace("-", "")[:8]
    return f"{SIMULATION_MEDIA_PREFIX}{short}:{media_id}"


def contains_simulation_media(media_ids: Any) -> bool:
    """Whether any id in a delivery batch is owner-only test content."""
    if media_ids is None:
        return False
    if isinstance(media_ids, (str, bytes)):
        return is_simulation_media_id(media_ids)
    try:
        return any(is_simulation_media_id(value) for value in media_ids)
    except TypeError:
        return False


def is_undefined_column_error(error: Exception) -> bool:
    text = str(error).lower()
    return any(marker in text for marker in _UNDEFINED_COLUMN_MARKERS) and (
        SIMULATION_ONLY_COLUMN in text or "42703" in text
    )


def exclude_simulation_only(query: Any) -> Any:
    """Narrow a PostgREST query to real, deliverable inventory."""
    return query.eq(SIMULATION_ONLY_COLUMN, False)


def run_live_catalog_query(
    build: Callable[[bool], T],
    *,
    label: str,
    include_simulation: bool | None = None,
) -> T:
    """Run one vault read with the live/simulation filter correctly applied.

    ``build(apply_filter)`` builds and executes the query; it is called a second
    time with ``False`` only when the column does not exist yet.
    """
    global _warned_missing_column

    include = (
        simulation_catalog_visible()
        if include_simulation is None
        else bool(include_simulation)
    )
    if include:
        # Inside the owner simulator: test content is exactly what we want.
        return build(False)

    try:
        return build(True)
    except Exception as exc:
        if not is_undefined_column_error(exc):
            raise
        if not _warned_missing_column:
            _warned_missing_column = True
            print(
                f"[SIMULATION CATALOG] {SIMULATION_ONLY_COLUMN} column missing "
                f"({label}) — apply db/simulation_catalog_v1.sql. No mirrored "
                "content can exist without it, so live planning is unaffected."
            )
        return build(False)


def reset_missing_column_warning() -> None:
    """Test-support only."""
    global _warned_missing_column
    _warned_missing_column = False
