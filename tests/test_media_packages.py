import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.commercial import CreatorPolicy
from services.media_packages import allocate_budget, build_offer_packages, choose_sequence


def rows():
    return [
        {"id": "a", "title": "bedroom 1", "location": "bedroom", "outfit": "black lingerie", "explicit_min": 1, "explicit_max": 2, "suggested_price": 15, "media_ids": ["a1"], "tags": ["lingerie"]},
        {"id": "b", "title": "bedroom 2", "location": "bedroom", "outfit": "black lingerie", "explicit_min": 2, "explicit_max": 3, "suggested_price": 20, "media_ids": ["b1"], "tags": ["lingerie"]},
        {"id": "c", "title": "bedroom 3", "location": "bedroom", "outfit": "black lingerie", "explicit_min": 4, "explicit_max": 5, "suggested_price": 30, "media_ids": ["c1"], "tags": ["toy"]},
        {"id": "x", "title": "bathroom", "location": "bathroom", "outfit": "red", "explicit_min": 2, "explicit_max": 2, "suggested_price": 18, "media_ids": ["x1"], "tags": []},
    ]


def test_sequence_stays_in_one_shoot_and_escalates():
    sequence = choose_sequence(rows(), target_cents=6000, min_steps=2, max_steps=3)
    assert len(sequence) >= 2
    assert {item["location"] for item in sequence} == {"bedroom"}
    assert [item["explicit_min"] for item in sequence] == sorted(item["explicit_min"] for item in sequence)


def test_allocations_equal_confirmed_budget():
    sequence = rows()[:3]
    allocations = allocate_budget(6000, sequence)
    assert sum(allocations) == 6000
    assert allocations[-1] > allocations[0]


def test_offers_are_multi_step_and_exactly_priced():
    policy = CreatorPolicy(
        quick_package_target_cents=2800,
        full_package_target_cents=6000,
        session_min_steps=2,
        session_max_steps=3,
    )
    offers = build_offer_packages(rows(), policy)
    assert offers[0].price_cents == 2800
    assert len(offers[0].set_ids) >= 2
    assert offers[-1].price_cents == 6000
