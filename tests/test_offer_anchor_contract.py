from pathlib import Path

from models.commercial import CreatorPolicy
from services.media_packages import build_next_offer


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
    return CreatorPolicy(next_offer_target_cents=2500)


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

    offer = build_next_offer(
        rows,
        _policy(),
        desired_experience="I want to see what happened in the shower",
    )

    assert offer is not None
    assert offer.set_id == "shower-premium"
    # The soft content target is $25; the shower set's approved range is
    # $45-$70. The request wins on content, and the content wins on price.
    assert 4500 <= offer.price_cents <= 7000
    assert offer.price_cents % 500 == 0
    assert "shower" in (offer.experience or "").lower()


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

    offer = build_next_offer(
        rows,
        _policy(),
        desired_experience="show me the shower set",
        hard_ceiling_cents=3000,
    )

    assert offer is not None
    assert offer.set_id == "bedroom-affordable"
    assert offer.price_cents <= 3000
    assert "shower" not in (offer.experience or "").lower()


def test_writer_receives_only_approved_experience_contract():
    source = (Path(__file__).parents[1] / "ai" / "prompt_builder.py").read_text()
    assert "approved content:" in source
    assert "only concrete content you may" in source
    assert "Do not name a requested theme unless it appears" in source


def test_orchestrator_resolves_anchor_before_the_offer_is_built():
    source = (
        Path(__file__).parents[1] / "services" / "commercial_orchestrator.py"
    ).read_text()
    desired_at = source.index("current_desired =")
    # The call is the inventory-reporting variant: it returns the offer AND
    # the asset types of the rows it was built from, in one read.
    offer_at = source.index("await get_next_offer_with_inventory")
    assert desired_at < offer_at
    assert "desired_experience=desired_experience or None" in source
    assert "hard_ceiling_cents=hard_ceiling_cents" in source
