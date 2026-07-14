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

def test_legacy_suggested_price_does_not_override_package_target():
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

    assert resolve_sequence_price(
        rows,
        2800,
        step_cents=100,
    ) == 2800

def test_package_bounds_are_sum_of_set_bounds():
    rows = [
        {"base_price_cents": 1500, "min_price_cents": 1000, "max_price_cents": 2000},
        {"base_price_cents": 2500, "min_price_cents": 2000, "max_price_cents": 3500},
    ]
    assert resolve_sequence_price(rows, 4500, step_cents=500) == 4500
    assert resolve_sequence_price(rows, 1000, step_cents=500) == 3000
