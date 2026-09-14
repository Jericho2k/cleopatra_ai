"""Approved price bounds, and what a target is allowed to do inside them.

These used to run through ``resolve_sequence_price``, which clamped a requested
PACKAGE price into a package's summed band. Packages are gone — the fan is shown
one unlock at a time — and nothing in production called it, so it was removed
with the rest of the two-package machinery. The assertions are unchanged: they
now compose the two functions it composed, ``sequence_bounds`` and
``human_price_cents``, which is what the surviving pricing path actually uses.
"""

from models.content_pricing import human_price_cents
from models.vault_pricing import price_bounds, sequence_bounds


def clamp(rows, target_cents, *, step_cents):
    """What resolve_sequence_price did, spelled out at the call site."""
    _base, minimum, maximum = sequence_bounds(rows)
    if maximum <= 0:
        return 0
    return human_price_cents(target_cents, minimum, maximum, step_cents=step_cents)


def test_dynamic_price_is_clamped_to_approved_bounds():
    row = {
        "base_price_cents": 3500,
        "min_price_cents": 2500,
        "max_price_cents": 4500,
        "dynamic_pricing_enabled": True,
    }
    assert price_bounds(row) == (3500, 2500, 4500, True)
    assert clamp([row], 1800, step_cents=500) == 2500
    assert clamp([row], 6000, step_cents=500) == 4500


def test_fixed_price_ignores_learned_target():
    row = {
        "base_price_cents": 3500,
        "min_price_cents": 2500,
        "max_price_cents": 4500,
        "dynamic_pricing_enabled": False,
    }
    assert clamp([row], 2500, step_cents=500) == 3500

def test_legacy_row_without_a_derivable_range_is_priced_at_its_approved_value():
    """A package target is a request, not permission to reprice content.

    These rows carry no approved band and no classifier category, so there is
    nothing to probe inside. Failing closed to the approved price is what stops
    a global $25 target from repricing a $35 pair of sets.
    """
    rows = [
        {
            "id": "a",
            "suggested_price": 15,
        },
        {
            "id": "b",
            "suggested_price": 20,
        },
    ]

    assert clamp(rows, 2800, step_cents=100) == 3500
    assert clamp(rows, 9000, step_cents=100) == 3500


def test_category_tag_bridges_a_legacy_row_to_its_approved_range():
    """Set generation writes the classifier category into the set's tags.

    Legacy rows therefore still carry an approved commercial range even though
    only ``suggested_price`` was ever written to their pricing columns.
    """
    row = {"id": "a", "suggested_price": 30, "tags": ["nude_photo", "bedroom"]}

    base, minimum, maximum, dynamic = price_bounds(row)
    assert (base, minimum, maximum, dynamic) == (3000, 1500, 8000, True)
    assert clamp([row], 2500, step_cents=500) == 2500
    assert clamp([row], 500, step_cents=500) == 1500
    assert clamp([row], 20_000, step_cents=500) == 8000


def test_backfilled_equal_bounds_do_not_pin_content_to_one_price():
    """adaptive_planning_v1.sql backfilled min = max = base for every old row.

    Read literally that says a $15-$80 nude set may only ever cost its
    suggested price. ``dynamic_pricing_enabled`` is the flag that means fixed;
    an equal-bounds backfill is not.
    """
    row = {
        "id": "a",
        "suggested_price": 25,
        "base_price_cents": 2500,
        "min_price_cents": 2500,
        "max_price_cents": 2500,
        "dynamic_pricing_enabled": True,
        "tags": ["nude_photo"],
    }
    assert price_bounds(row) == (2500, 1500, 8000, True)

    fixed = dict(row, dynamic_pricing_enabled=False)
    assert price_bounds(fixed) == (2500, 2500, 2500, False)

def test_package_bounds_are_sum_of_set_bounds():
    rows = [
        {"base_price_cents": 1500, "min_price_cents": 1000, "max_price_cents": 2000},
        {"base_price_cents": 2500, "min_price_cents": 2000, "max_price_cents": 3500},
    ]
    assert clamp(rows, 4500, step_cents=500) == 4500
    assert clamp(rows, 1000, step_cents=500) == 3000
