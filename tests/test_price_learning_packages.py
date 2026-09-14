"""Price learning moves the PRICE of the next offer, not which menu entry wins.

``select_recommended_packages`` used to pick which of two packages to put in
front of the fan; with one offer at a time there is nothing to pick between, and
the learned evidence does the only job it should ever have done — deciding where
inside the chosen content's own approved range this fan is probed.
"""

from models.commercial import CreatorPolicy
from models.price_learning import PriceLearningPolicy
from services.media_packages import build_next_offer


def rows():
    return [
        {
            "id": "wide-range",
            "title": "bedroom 1",
            "location": "bedroom",
            "outfit": "black lingerie",
            "explicit_min": 2,
            "explicit_max": 3,
            "base_price_cents": 4000,
            "min_price_cents": 1500,
            "max_price_cents": 8000,
            "media_ids": ["m1"],
            "tags": ["nude_photo"],
        }
    ]


def _price(price_learning: dict | None, **kwargs) -> int:
    offer = build_next_offer(
        rows(),
        CreatorPolicy(),
        price_learning=price_learning,
        pricing_policy=PriceLearningPolicy(cold_start_probe_bps=2000),
        **kwargs,
    )
    assert offer is not None
    return offer.price_cents


def test_demonstrated_willingness_raises_the_probe_inside_the_range():
    cold = _price(None)
    warm = _price(
        {
            "mode": "RANGE",
            "confidence": "HIGH",
            "recommended_floor_cents": 4000,
            "recommended_target_cents": 6000,
            "recommended_ceiling_cents": 7000,
            "evidence_summary": {"demonstrated_willingness_cents": 6000},
        }
    )
    assert warm > cold
    assert 1500 <= warm <= 8000


def test_an_explicit_current_ceiling_is_never_exceeded():
    assert _price(None, hard_ceiling_cents=2000) <= 2000


def test_a_ceiling_below_the_approved_floor_yields_no_offer():
    assert (
        build_next_offer(rows(), CreatorPolicy(), hard_ceiling_cents=500) is None
    ), "discounting below approved value is never an option"
