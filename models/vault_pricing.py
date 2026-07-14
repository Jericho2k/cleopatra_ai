"""Pure pricing-boundary helpers for approved vault sets and packages."""

from __future__ import annotations

from typing import Any, Iterable


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
    base = cents_from_row(row)

    has_explicit_dynamic_flag = "dynamic_pricing_enabled" in row
    has_explicit_minimum = row.get("min_price_cents") not in (None, "")
    has_explicit_maximum = row.get("max_price_cents") not in (None, "")
    has_explicit_base = row.get("base_price_cents") not in (None, "")

    has_explicit_boundaries = (
        has_explicit_dynamic_flag
        or has_explicit_minimum
        or has_explicit_maximum
        or has_explicit_base
    )

    # Legacy vault rows only have suggested_price. Treat that as a selection
    # signal, not as an authoritative commercial boundary.
    if not has_explicit_boundaries:
        return base, 0, 0, True

    dynamic = bool(row.get("dynamic_pricing_enabled", True))

    minimum = _money(row.get("min_price_cents"))
    maximum = _money(row.get("max_price_cents"))

    minimum = base if minimum is None else minimum
    maximum = base if maximum is None else maximum

    minimum = min(minimum, base) if base else minimum
    maximum = max(maximum, base)

    if not dynamic:
        minimum = maximum = base

    return base, max(0, minimum), max(0, maximum), dynamic


def resolve_sequence_price(
    rows: Iterable[dict[str, Any]],
    target_cents: int,
    *,
    step_cents: int = 100,
) -> int:
    items = list(rows)
    if not items:
        return max(0, int(target_cents))
    minimum = 0
    maximum = 0
    base = 0
    for row in items:
        item_base, item_min, item_max, _ = price_bounds(row)
        base += item_base
        minimum += item_min
        maximum += item_max
    if maximum <= 0:
        return max(0, int(target_cents))
    requested = int(target_cents or base or minimum)
    resolved = max(minimum, min(maximum, requested))
    step = max(1, int(step_cents))
    rounded = int(round(resolved / step) * step)
    return max(minimum, min(maximum, rounded))


def approved_target_from_learning(
    price_learning: dict[str, Any] | None,
    *,
    fallback_cents: int,
    use_ceiling: bool = False,
) -> int:
    context = price_learning or {}
    key = "recommended_ceiling_cents" if use_ceiling else "recommended_target_cents"
    learned = _money(context.get(key))
    if learned is None:
        return int(fallback_cents)
    return learned


def _money(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return None
