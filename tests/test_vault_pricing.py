from models.vault_pricing import price_bounds, resolve_sequence_price


def test_dynamic_price_is_clamped_to_approved_bounds():
    row = {
        "base_price_cents": 3500,
        "min_price_cents": 2500,
        "max_price_cents": 4500,
        "dynamic_pricing_enabled": True,
    }
    assert price_bounds(row) == (3500, 2500, 4500, True)
    assert resolve_sequence_price([row], 1800, step_cents=500) == 2500
    assert resolve_sequence_price([row], 6000, step_cents=500) == 4500


def test_fixed_price_ignores_learned_target():
    row = {
        "base_price_cents": 3500,
        "min_price_cents": 2500,
        "max_price_cents": 4500,
        "dynamic_pricing_enabled": False,
    }
    assert resolve_sequence_price([row], 2500, step_cents=500) == 3500

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

    assert resolve_sequence_price(rows, 2800, step_cents=100) == 3500
    assert resolve_sequence_price(rows, 9000, step_cents=100) == 3500


def test_category_tag_bridges_a_legacy_row_to_its_approved_range():
    """Set generation writes the classifier category into the set's tags.

    Legacy rows therefore still carry an approved commercial range even though
    only ``suggested_price`` was ever written to their pricing columns.
    """
    row = {"id": "a", "suggested_price": 30, "tags": ["nude_photo", "bedroom"]}

    base, minimum, maximum, dynamic = price_bounds(row)
    assert (base, minimum, maximum, dynamic) == (3000, 1500, 8000, True)
    assert resolve_sequence_price([row], 2500, step_cents=500) == 2500
    assert resolve_sequence_price([row], 500, step_cents=500) == 1500
    assert resolve_sequence_price([row], 20_000, step_cents=500) == 8000


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
    assert resolve_sequence_price(rows, 4500, step_cents=500) == 4500
    assert resolve_sequence_price(rows, 1000, step_cents=500) == 3000
