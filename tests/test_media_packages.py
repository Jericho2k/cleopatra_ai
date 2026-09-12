import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.commercial import CreatorPolicy
from models.price_learning import PriceLearningPolicy
from services.media_packages import (
    allocate_step_pricing,
    build_offer_packages,
    choose_sequence,
    choose_video_finale,
    order_steps_for_progression,
    split_media_types,
)


def rows():
    """Approved photo sets from one bedroom shoot, with classifier categories.

    ``nude_photo`` is $15-$80 and ``solo_toy_photo`` is $20-$80, so these rows
    carry a real approved range even though only ``suggested_price`` was ever
    written to them.
    """
    return [
        {"id": "a", "title": "bedroom 1", "location": "bedroom", "outfit": "black lingerie", "explicit_min": 1, "explicit_max": 2, "suggested_price": 15, "media_ids": ["a1"], "tags": ["nude_photo", "lingerie"]},
        {"id": "b", "title": "bedroom 2", "location": "bedroom", "outfit": "black lingerie", "explicit_min": 2, "explicit_max": 3, "suggested_price": 20, "media_ids": ["b1"], "tags": ["nude_photo", "lingerie"]},
        {"id": "c", "title": "bedroom 3", "location": "bedroom", "outfit": "black lingerie", "explicit_min": 4, "explicit_max": 5, "suggested_price": 30, "media_ids": ["c1"], "tags": ["solo_toy_photo", "toy"]},
        {"id": "x", "title": "bathroom", "location": "bathroom", "outfit": "red", "explicit_min": 2, "explicit_max": 2, "suggested_price": 18, "media_ids": ["x1"], "tags": ["nude_photo"]},
    ]


def test_sequence_stays_in_one_shoot_and_escalates():
    sequence = choose_sequence(rows(), target_cents=6000, min_steps=2, max_steps=3)
    assert len(sequence) >= 2
    assert {item["location"] for item in sequence} == {"bedroom"}
    assert [item["explicit_min"] for item in sequence] == sorted(item["explicit_min"] for item in sequence)


def test_step_prices_are_human_and_sum_to_the_sold_total():
    sequence = rows()[:3]
    allocations = allocate_step_pricing(6000, sequence)
    assert allocations is not None
    assert sum(allocations) == 6000
    assert all(value % 500 == 0 for value in allocations)
    assert allocations[-1] >= allocations[0]


def test_impossible_allocation_fails_instead_of_inventing_one():
    fixed = [
        {"id": "one", "base_price_cents": 1700, "min_price_cents": 1700, "max_price_cents": 1700, "dynamic_pricing_enabled": False, "media_ids": ["m1"]},
        {"id": "two", "base_price_cents": 1700, "min_price_cents": 1700, "max_price_cents": 1700, "dynamic_pricing_enabled": False, "media_ids": ["m2"]},
    ]
    assert allocate_step_pricing(9999, fixed) is None


def test_offers_stay_inside_approved_content_bounds():
    policy = CreatorPolicy(
        quick_package_target_cents=2800,
        full_package_target_cents=6000,
        session_min_steps=2,
        session_max_steps=3,
    )
    offers = build_offer_packages(rows(), policy)
    assert offers, "an approved photo shoot must be offerable"
    for offer in offers:
        assert offer.content_floor_cents <= offer.price_cents <= offer.content_ceiling_cents
        assert offer.price_cents % 500 == 0
        assert allocate_step_pricing(offer.price_cents, rows()[: offer.step_count]) is not None
    assert offers[-1].price_cents >= offers[0].price_cents


def test_video_request_returns_single_video_ppv_options():
    video_rows = [
        {
            "id": f"video-{index}",
            "title": f"Private shower video {index}",
            "description": "A private shower clip.",
            "location": "shower",
            "outfit": "",
            "explicit_min": 4,
            "explicit_max": 4,
            "base_price_cents": price,
            "min_price_cents": price,
            "max_price_cents": price,
            "media_ids": [f"media-video-{index}"],
            "tags": ["shower", "video", "individual_video"],
        }
        for index, price in enumerate((3500, 7000))
    ]
    offers = build_offer_packages(
        [*rows(), *video_rows],
        CreatorPolicy(),
        desired_experience="send me a shower video",
    )
    assert [offer.price_cents for offer in offers] == [3500, 7000]
    assert all(len(offer.set_ids) == 1 for offer in offers)
    assert {offer.set_ids[0] for offer in offers} == {"video-0", "video-1"}


def test_video_only_vault_is_offerable_without_two_steps():
    video = {
        "id": "only-video",
        "title": "Private video",
        "description": "A private clip.",
        "explicit_min": 3,
        "explicit_max": 3,
        "base_price_cents": 4500,
        "min_price_cents": 4000,
        "max_price_cents": 6000,
        "media_ids": ["media-video"],
        "tags": ["video", "individual_video"],
    }
    offers = build_offer_packages([video], CreatorPolicy())
    assert len(offers) == 1
    assert offers[0].set_ids == ["only-video"]
    assert 4000 <= offers[0].price_cents <= 6000


def test_media_types_split_and_order_photo_before_video():
    video = dict(rows()[0], id="v", tags=["individual_video"], explicit_min=2, explicit_max=3)
    photos, videos = split_media_types([*rows(), video])
    assert [row["id"] for row in videos] == ["v"]
    assert "v" not in {row["id"] for row in photos}

    photo_at_same_level = rows()[1]
    ordered = order_steps_for_progression([video, photo_at_same_level])
    assert [row["id"] for row in ordered] == ["b", "v"]


def test_video_finale_prefers_the_coherent_clip():
    sequence = [row for row in rows() if row["location"] == "bedroom"][:2]
    videos = [
        {"id": "v-kitchen", "title": "kitchen", "location": "kitchen", "explicit_min": 5, "explicit_max": 5, "base_price_cents": 6000, "media_ids": ["vk"], "tags": ["individual_video"]},
        {"id": "v-bedroom", "title": "bedroom 4", "location": "bedroom", "outfit": "black lingerie", "explicit_min": 5, "explicit_max": 5, "base_price_cents": 5000, "media_ids": ["vb"], "tags": ["individual_video"]},
    ]
    assert choose_video_finale(sequence, videos)["id"] == "v-bedroom"


def test_premium_package_probes_higher_than_the_opener():
    policy = PriceLearningPolicy(cold_start_probe_bps=2000)
    offers = build_offer_packages(rows(), CreatorPolicy(), pricing_policy=policy)
    assert len(offers) == 2
    assert offers[1].price_cents > offers[0].price_cents
