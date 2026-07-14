from models.commercial import PackageOption
from models.price_learning import select_recommended_packages


def package(cents: int) -> PackageOption:
    return PackageOption(
        package_id=f"p-{cents}",
        label=f"${cents / 100:g}",
        price_cents=cents,
        set_id=f"set-{cents}",
    )


def test_selects_approved_packages_around_target():
    options = [package(1500), package(2500), package(4000), package(6000)]
    selected = select_recommended_packages(
        options,
        {
            "mode": "RANGE",
            "recommended_floor_cents": 2000,
            "recommended_target_cents": 3500,
            "recommended_ceiling_cents": 5000,
        },
        max_options=2,
    )
    assert [item.price_cents for item in selected] == [2500, 4000]


def test_exact_mode_never_invents_price():
    options = [package(2500), package(4000)]
    selected = select_recommended_packages(
        options,
        {"mode": "EXACT", "recommended_target_cents": 2800},
        max_options=2,
    )
    assert len(selected) == 1
    assert selected[0].price_cents == 2500
