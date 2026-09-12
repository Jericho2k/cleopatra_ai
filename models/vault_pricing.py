"""Pure pricing-boundary helpers for approved vault sets and packages.

The hierarchy this file enforces:

    CONTENT VALUE / APPROVED PRICE BOUNDS
                |
                v
    FAN-SPECIFIC PRICE POSITION / PROBE      (models/price_learning.py)
                |
                v
    ACTUAL APPROVED OFFER

``price_bounds`` answers the first question only: what is this set allowed to
cost? Nothing here knows or cares about a particular fan.
"""

from __future__ import annotations

from typing import Any, Iterable

from models.content_pricing import (
    DEFAULT_PRICE_STEP_CENTS,
    FALLBACK_PRICE_STEP_CENTS,
    human_price_cents,
    row_category_range_cents,
)


def cents_from_row(row: dict[str, Any]) -> int:
    for key in ("base_price_cents", "price_cents"):
        value = _money(row.get(key))
        if value is not None:
            return value
    try:
        return max(0, int(round(float(row.get("suggested_price") or 0) * 100)))
    except (TypeError, ValueError):
        return 0


def price_bounds(row: dict[str, Any]) -> tuple[int, int, int, bool]:
    """Return ``(base, minimum, maximum, dynamic)`` in cents for one set.

    Resolution order, most authoritative first:

    1. An explicit approved band on the row (``min`` < ``max``).
    2. ``dynamic_pricing_enabled = false`` — a deliberate fixed price.
    3. The classifier category range carried in the row's own metadata. This is
       the bridge for legacy rows that only ever received ``suggested_price``,
       and for rows whose ``min``/``max`` were backfilled to equal the base by
       ``adaptive_planning_v1.sql``. Without it a $15-$80 nude set reads as
       having no commercial bounds at all, and an arbitrary package target
       silently becomes its price.
    4. Otherwise the approved price is exactly the approved price. Failing
       closed to a fixed price is always safer than treating content as
       unbounded.
    """
    base = cents_from_row(row)
    dynamic_flag = row.get("dynamic_pricing_enabled", True)
    dynamic = True if dynamic_flag in (None, "") else bool(dynamic_flag)

    minimum = _money(row.get("min_price_cents"))
    maximum = _money(row.get("max_price_cents"))

    if not dynamic:
        anchor = base or minimum or maximum or 0
        return anchor, anchor, anchor, False

    if minimum is not None and maximum is not None and maximum > minimum:
        low = min(minimum, base) if base else minimum
        high = max(maximum, base)
        return (base or low), max(0, low), max(0, high), True

    bridged = row_category_range_cents(row)
    if bridged:
        low, high = bridged
        if base:
            low = min(low, base)
            high = max(high, base)
        anchor = base or ((low + high) // 2)
        return anchor, max(0, low), max(0, high), high > low

    anchor = base or minimum or maximum or 0
    return anchor, anchor, anchor, False


def sequence_bounds(rows: Iterable[dict[str, Any]]) -> tuple[int, int, int]:
    """Return ``(base, minimum, maximum)`` in cents for a whole package."""
    base = minimum = maximum = 0
    for row in rows:
        item_base, item_min, item_max, _ = price_bounds(row)
        base += item_base
        minimum += item_min
        maximum += item_max
    return base, minimum, maximum


def resolve_sequence_price(
    rows: Iterable[dict[str, Any]],
    target_cents: int,
    *,
    step_cents: int = DEFAULT_PRICE_STEP_CENTS,
) -> int:
    """Clamp a requested package price into the approved band, cleanly.

    A target is a request, never permission: it can position the price inside
    the band and nothing more.
    """
    items = list(rows)
    if not items:
        return max(0, int(target_cents))
    base, minimum, maximum = sequence_bounds(items)
    if maximum <= 0:
        # Nothing on this package carries an approved paid value. Selling it is
        # not a pricing decision we are allowed to make.
        return 0
    requested = int(target_cents or base or minimum)
    return human_price_cents(requested, minimum, maximum, step_cents=step_cents)


def allocate_step_prices(
    total_cents: int,
    rows: list[dict[str, Any]],
    *,
    step_cents: int = DEFAULT_PRICE_STEP_CENTS,
) -> list[int] | None:
    """Split one approved session total into per-step PPV prices.

    Every returned price is inside its own step's approved bounds, sits on a
    human price grid, and the list sums to exactly ``total_cents``. When those
    constraints cannot all hold this returns ``None`` rather than inventing a
    distribution: an impossible allocation must fail before an offer is ever
    presented, not become a $10.63 PPV afterwards.
    """
    if not rows:
        return None
    total = int(total_cents)
    if total <= 0:
        return None

    bounds = []
    for row in rows:
        _, minimum, maximum, _ = price_bounds(row)
        if maximum <= 0:
            return None
        bounds.append((minimum, max(minimum, maximum)))

    for grid in _allocation_grids(step_cents):
        allocation = _allocate_on_grid(total, bounds, grid)
        if allocation is not None:
            return allocation
    return None


def _allocation_grids(step_cents: int) -> list[int]:
    step = max(1, int(step_cents))
    grids = [step]
    if FALLBACK_PRICE_STEP_CENTS < step:
        grids.append(FALLBACK_PRICE_STEP_CENTS)
    return grids


def _allocate_on_grid(
    total: int,
    bounds: list[tuple[int, int]],
    grid: int,
) -> list[int] | None:
    if grid <= 0 or total % grid:
        return None
    low: list[int] = []
    high: list[int] = []
    for minimum, maximum in bounds:
        floor = -((-minimum) // grid) * grid
        ceiling = (maximum // grid) * grid
        if floor > maximum or ceiling < minimum or floor > ceiling:
            return None
        low.append(floor)
        high.append(ceiling)

    allocation = list(low)
    remaining = total - sum(low)
    if remaining < 0 or total > sum(high):
        return None

    # Spread the remainder back-to-front so later, more explicit steps carry
    # more of the session value while every step stays on the grid.
    while remaining >= grid:
        moved = False
        for index in range(len(allocation) - 1, -1, -1):
            if remaining < grid:
                break
            if allocation[index] + grid <= high[index]:
                allocation[index] += grid
                remaining -= grid
                moved = True
        if not moved:
            return None
    if remaining:
        return None
    return allocation


def _money(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return None
