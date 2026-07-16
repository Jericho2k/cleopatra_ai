from pathlib import Path

from models.commercial import CreatorPolicy
from services.media_packages import build_offer_packages


def _set(
    set_id: str,
    *,
    title: str,
    location: str,
    base: int,
    minimum: int,
    maximum: int,
    tags: list[str],
) -> dict:
    return {
        "id": set_id,
        "title": title,
        "location": location,
        "outfit": "",
        "tags": tags,
        "media_ids": [f"media:{set_id}"],
        "explicit_min": 0.2,
        "explicit_max": 0.8,
        "base_price_cents": base,
        "min_price_cents": minimum,
        "max_price_cents": maximum,
        "dynamic_pricing_enabled": True,
    }


def _policy() -> CreatorPolicy:
    return CreatorPolicy(
        offer_two_packages=False,
        quick_package_target_cents=2500,
        session_min_steps=1,
        session_max_steps=3,
    )


def test_current_experience_outranks_soft_initial_target():
    rows = [
        _set(
            "bedroom-cheap",
            title="Bedroom tease",
            location="bedroom",
            base=2500,
            minimum=2000,
            maximum=3000,
            tags=["bedroom", "lingerie"],
        ),
        _set(
            "shower-premium",
            title="After the shower",
            location="bathroom shower",
            base=5000,
            minimum=4500,
            maximum=7000,
            tags=["shower", "wet"],
        ),
    ]

    packages = build_offer_packages(
        rows,
        _policy(),
        desired_experience="I want to see what happened in the shower",
    )

    assert len(packages) == 1
    assert packages[0].set_ids == ["shower-premium"]
    assert packages[0].price_cents == 4500
    assert "shower" in (packages[0].experience or "").lower()


def test_explicit_current_ceiling_blocks_unaffordable_requested_set():
    rows = [
        _set(
            "bedroom-affordable",
            title="Bedroom tease",
            location="bedroom",
            base=2500,
            minimum=2000,
            maximum=3000,
            tags=["bedroom", "lingerie"],
        ),
        _set(
            "shower-premium",
            title="After the shower",
            location="bathroom shower",
            base=5000,
            minimum=4500,
            maximum=7000,
            tags=["shower", "wet"],
        ),
    ]

    packages = build_offer_packages(
        rows,
        _policy(),
        desired_experience="show me the shower set",
        hard_ceiling_cents=3000,
    )

    assert len(packages) == 1
    assert packages[0].set_ids == ["bedroom-affordable"]
    assert packages[0].price_cents <= 3000
    assert "shower" not in (packages[0].experience or "").lower()


def test_writer_receives_only_approved_experience_contract():
    source = (Path(__file__).parents[1] / "ai" / "prompt_builder.py").read_text()
    assert "approved experience:" in source
    assert "only concrete content you may" in source
    assert "Do not name a requested theme unless it appears" in source


def test_orchestrator_resolves_anchor_before_package_build():
    source = (
        Path(__file__).parents[1] / "services" / "commercial_orchestrator.py"
    ).read_text()
    desired_at = source.index("current_desired =")
    package_at = source.index("package_options = await get_offerable_packages")
    assert desired_at < package_at
    assert "desired_experience=desired_experience or None" in source
    assert "hard_ceiling_cents=hard_ceiling_cents" in source
