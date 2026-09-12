"""Canonical content-value pricing: category ranges and human-facing prices.

This module is the single source of truth for what a piece of vault content is
worth. Everything downstream — packages, sessions, per-step PPV prices — has to
stay inside the bounds computed here.

Two rules make the rest of the commercial layer easy to reason about:

1. Content value comes first. A per-fan price recommendation says *where inside*
   an approved range to probe; it can never move a set outside that range, and a
   global cold-start target can never turn an intrinsically $15-$80 set into a
   $25 maximum.
2. Customer-facing prices look like prices. $10, $15, $20, $25 — never $10.63,
   unless an agency deliberately configures a cent-level step.
"""

from __future__ import annotations

from typing import Any, Iterable

# The agency's approved commercial range per classifier category, in whole
# dollars. ``main.py`` re-exports this as ``VAULT_CATEGORIES`` and the
# classifier writes the chosen range onto ``creator_vault_media`` as
# ``price_min`` / ``price_max``.
VAULT_CATEGORIES: dict[str, dict[str, Any]] = {
    "teaser_clothed":   {"min": 0,   "max": 0,   "label": "Clothed teaser (free)"},
    "teaser_bundle":    {"min": 0,   "max": 0,   "label": "Teaser bundle no nudity (free)"},
    "legs_feet":        {"min": 15,  "max": 70,  "label": "Legs / feet / armpits"},
    "lingerie_photo":   {"min": 10,  "max": 80,  "label": "Lingerie photo"},
    "lingerie_video":   {"min": 15,  "max": 90,  "label": "Lingerie video"},
    "nude_photo":       {"min": 15,  "max": 80,  "label": "Nude photo"},
    "nude_video":       {"min": 20,  "max": 110, "label": "Nude video"},
    "striptease_video": {"min": 15,  "max": 100, "label": "Striptease video"},
    "closeup_photo":    {"min": 25,  "max": 130, "label": "Closeup photo"},
    "closeup_video":    {"min": 25,  "max": 130, "label": "Closeup video"},
    "dictate_video":    {"min": 15,  "max": 50,  "label": "Dictate / dirty talk video"},
    "solo_toy_video":   {"min": 30,  "max": 150, "label": "Solo / toy / orgasm video"},
    "solo_toy_photo":   {"min": 20,  "max": 80,  "label": "Solo / toy photo"},
    "explicit_photo":   {"min": 25,  "max": 130, "label": "Explicit solo photo"},
    "explicit_video":   {"min": 35,  "max": 170, "label": "Explicit solo video"},
    "bg_content":       {"min": 50,  "max": 300, "label": "BG (boy-girl) content"},
    "task":             {"min": 10,  "max": 50,  "label": "Task / custom request"},
    "other":            {"min": 0,   "max": 0,   "label": "Other / unclear"},
}

# Categories with a real paid range. A free/teaser/unclear category is not an
# approved commercial range and must never be used as one.
PRICED_CATEGORIES: dict[str, tuple[int, int]] = {
    name: (int(value["min"]) * 100, int(value["max"]) * 100)
    for name, value in VAULT_CATEGORIES.items()
    if int(value["max"]) > 0
}

# The customer-facing price grid, in cents, most preferred first. An agency that
# genuinely wants cent-level pricing configures a smaller step explicitly.
DEFAULT_PRICE_STEP_CENTS = 500
FALLBACK_PRICE_STEP_CENTS = 100


def normalize_category(value: Any) -> str:
    text = str(value or "").strip().lower()
    return "_".join(part for part in text.replace("-", " ").replace("_", " ").split())


def category_range_cents(category: Any) -> tuple[int, int] | None:
    """Approved (min, max) in cents for one classifier category."""
    return PRICED_CATEGORIES.get(normalize_category(category))


def row_category_range_cents(row: dict[str, Any]) -> tuple[int, int] | None:
    """Best approved price range derivable from a vault row's own metadata.

    Set generation writes the classifier's ``content_category`` into the set's
    tags, so an approved range survives into ``vault_sets`` even when the legacy
    ``suggested_price`` column is all the pricing the row carries. The most
    valuable matching category wins: a mixed set is worth at least what its
    strongest content is worth.
    """
    candidates: list[str] = [str(row.get("content_category") or "")]
    candidates.extend(str(tag) for tag in (row.get("tags") or []))
    ranges = [
        found
        for candidate in candidates
        if (found := category_range_cents(candidate)) is not None
    ]
    if not ranges:
        return None
    return max(ranges, key=lambda pair: (pair[1], pair[0]))


def category_range_for_items(items: Iterable[dict[str, Any]]) -> tuple[int, int] | None:
    """Approved range for a group of classified media items.

    Uses each item's own classifier range when present (``price_min`` /
    ``price_max`` in dollars) and falls back to its category. The group's range
    is the range of its most valuable member, which is how the agency prices a
    set: the strongest photo sets the ceiling.
    """
    best: tuple[int, int] | None = None
    for item in items:
        found = _item_range_cents(item)
        if found is None:
            continue
        if best is None or (found[1], found[0]) > (best[1], best[0]):
            best = found
    return best


def _item_range_cents(item: dict[str, Any]) -> tuple[int, int] | None:
    minimum = _dollars_to_cents(item.get("price_min"))
    maximum = _dollars_to_cents(item.get("price_max"))
    if minimum is not None and maximum is not None and maximum > 0:
        return min(minimum, maximum), max(minimum, maximum)
    return category_range_cents(item.get("content_category"))


def human_price_cents(
    value: int,
    minimum: int,
    maximum: int,
    *,
    step_cents: int = DEFAULT_PRICE_STEP_CENTS,
) -> int:
    """Snap a price to the cleanest grid that still fits inside [min, max].

    Tries the configured step first ($5 by default), then whole dollars, and
    only then keeps the raw value. A band too narrow to contain any clean price
    is a legitimate outcome — an approved fixed price of $17 stays $17.
    """
    low, high = sorted((max(0, int(minimum)), max(0, int(maximum))))
    target = max(low, min(high, int(value)))
    for grid in _grids(step_cents):
        snapped = _snap(target, low, high, grid)
        if snapped is not None:
            return snapped
    return target


def _grids(step_cents: int) -> list[int]:
    step = max(1, int(step_cents))
    grids = [step]
    if FALLBACK_PRICE_STEP_CENTS not in grids and FALLBACK_PRICE_STEP_CENTS < step:
        grids.append(FALLBACK_PRICE_STEP_CENTS)
    return grids


def _snap(target: int, low: int, high: int, grid: int) -> int | None:
    down = (target // grid) * grid
    up = -((-target) // grid) * grid
    options = [value for value in (down, up) if low <= value <= high]
    if not options:
        return None
    return min(options, key=lambda value: (abs(value - target), value))


def _dollars_to_cents(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return max(0, int(round(float(value) * 100)))
    except (TypeError, ValueError):
        return None


__all__ = [
    "DEFAULT_PRICE_STEP_CENTS",
    "FALLBACK_PRICE_STEP_CENTS",
    "PRICED_CATEGORIES",
    "VAULT_CATEGORIES",
    "category_range_cents",
    "category_range_for_items",
    "human_price_cents",
    "normalize_category",
    "row_category_range_cents",
]
