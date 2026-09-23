"""The next unlock: one approved set, one approved price, no menu.

``build_offer_packages`` used to return two offers so the fan could be shown a
cheap and an expensive branch. It now returns ONE — ``build_next_offer`` — and
the progression it belongs to stays internal (``plan_progression``), so the fan
never learns how far it goes or what it would cost in total.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.commercial import CreatorPolicy
from models.price_learning import PriceLearningPolicy
from services.media_packages import (
    MAX_PURCHASE_PROBE_BONUS_BPS,
    allocate_step_pricing,
    build_next_offer,
    choose_video_finale,
    plan_progression,
    purchase_probe_bonus_bps,
    sets_with_sellable_media_evidence,
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


# --- one offer, never two ---------------------------------------------------


def test_the_next_offer_is_one_set_at_one_approved_price():
    offer = build_next_offer(rows(), CreatorPolicy())
    assert offer is not None
    assert offer.set_ids == [offer.set_id]
    assert offer.content_floor_cents <= offer.price_cents <= offer.content_ceiling_cents
    assert offer.price_cents % 500 == 0
    assert allocate_step_pricing(offer.price_cents, [
        row for row in rows() if row["id"] == offer.set_id
    ]) is not None


def test_an_offer_carries_no_step_count_and_no_session_total():
    offer = build_next_offer(rows(), CreatorPolicy())
    assert offer is not None
    for removed in ("step_count", "is_multi_step", "session_total_cents"):
        assert not hasattr(offer, removed), removed


def test_the_offer_opens_on_the_softest_thing_in_the_scene():
    """Escalation is a ladder, not a jump to the strongest asset."""
    offer = build_next_offer(rows(), CreatorPolicy())
    assert offer is not None
    assert offer.set_id == "a"


def test_the_next_offer_escalates_past_what_he_already_unlocked():
    already = next(row for row in rows() if row["id"] == "a")
    remaining = [row for row in rows() if row["id"] != "a"]
    offer = build_next_offer(remaining, CreatorPolicy(), last_unlocked=already)
    assert offer is not None
    # Same shoot, one rung up — not the bathroom set and not the hardest one.
    assert offer.set_id == "b"


def test_a_concrete_request_still_outranks_the_content_budget():
    offer = build_next_offer(
        rows(), CreatorPolicy(), desired_experience="the bathroom one"
    )
    assert offer is not None
    assert offer.set_id == "x"


# --- the internal ladder, which the fan never sees --------------------------


def test_the_progression_is_planned_but_is_not_the_offer():
    ladder = plan_progression(rows())
    assert [row["id"] for row in ladder][:2] == ["a", "b"]
    assert len(ladder) > 1, "there is a plan"

    offer = build_next_offer(rows(), CreatorPolicy())
    # ...and exactly one rung of it is sellable right now.
    assert offer is not None
    assert offer.set_id == ladder[0]["id"]


def test_the_ladder_stays_inside_one_coherent_scene():
    ladder = plan_progression(rows())
    assert {row["location"] for row in ladder} == {"bedroom"}


def test_the_ladder_ends_on_a_coherent_clip_when_one_exists():
    video = {
        "id": "v-bedroom",
        "title": "bedroom 4",
        "location": "bedroom",
        "outfit": "black lingerie",
        "explicit_min": 5,
        "explicit_max": 5,
        "base_price_cents": 5000,
        "media_ids": ["vb"],
        "tags": ["individual_video"],
    }
    ladder = plan_progression([*rows(), video])
    assert ladder[-1]["id"] == "v-bedroom"


# --- pricing ----------------------------------------------------------------


def test_step_prices_are_human_and_sum_to_the_sold_total():
    allocations = allocate_step_pricing(6000, rows()[:3])
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


def test_a_proven_buyer_is_probed_higher_inside_the_same_range():
    policy = PriceLearningPolicy(cold_start_probe_bps=2000)
    cold = build_next_offer(rows(), CreatorPolicy(), pricing_policy=policy)
    warm = build_next_offer(
        rows(), CreatorPolicy(), pricing_policy=policy, confirmed_purchase_count=3
    )
    assert cold is not None and warm is not None
    assert warm.price_cents >= cold.price_cents
    # ...and never outside what the content itself approves.
    assert warm.price_cents <= warm.content_ceiling_cents


def test_the_purchase_probe_bonus_is_bounded():
    assert purchase_probe_bonus_bps(0) == 0
    assert purchase_probe_bonus_bps(1) > 0
    assert purchase_probe_bonus_bps(50) == MAX_PURCHASE_PROBE_BONUS_BPS


def test_an_explicit_ceiling_can_make_everything_unofferable():
    assert build_next_offer(rows(), CreatorPolicy(), hard_ceiling_cents=100) is None


# --- video ------------------------------------------------------------------


def test_a_video_request_returns_one_video_offer():
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
    offer = build_next_offer(
        [*rows(), *video_rows],
        CreatorPolicy(),
        desired_experience="send me a shower video",
    )
    assert offer is not None
    assert offer.asset_type == "video"
    assert offer.set_id in {"video-0", "video-1"}
    assert offer.price_cents in {3500, 7000}


def test_video_only_vault_is_offerable():
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
    offer = build_next_offer([video], CreatorPolicy())
    assert offer is not None
    assert offer.set_id == "only-video"
    assert offer.asset_type == "video"
    assert 4000 <= offer.price_cents <= 6000


def test_media_types_split_photo_from_video():
    video = dict(rows()[0], id="v", tags=["individual_video"], explicit_min=2, explicit_max=3)
    photos, videos = split_media_types([*rows(), video])
    assert [row["id"] for row in videos] == ["v"]
    assert "v" not in {row["id"] for row in photos}


def test_video_finale_prefers_the_coherent_clip():
    sequence = [row for row in rows() if row["location"] == "bedroom"][:2]
    videos = [
        {"id": "v-kitchen", "title": "kitchen", "location": "kitchen", "explicit_min": 5, "explicit_max": 5, "base_price_cents": 6000, "media_ids": ["vk"], "tags": ["individual_video"]},
        {"id": "v-bedroom", "title": "bedroom 4", "location": "bedroom", "outfit": "black lingerie", "explicit_min": 5, "explicit_max": 5, "base_price_cents": 5000, "media_ids": ["vb"], "tags": ["individual_video"]},
    ]
    assert choose_video_finale(sequence, videos)["id"] == "v-bedroom"


def test_legacy_set_with_only_teaser_children_is_not_sellable():
    legacy = {
        "id": "legacy",
        "title": "old set",
        "media_ids": ["m1", "m2"],
        "tags": [],
        "suggested_price": 30,
    }
    children = [
        {"media_id": "m1", "content_category": "teaser_clothed"},
        {"media_id": "m2", "content_category": "teaser_bundle"},
    ]
    assert sets_with_sellable_media_evidence([legacy], children) == []


def test_mixed_set_with_paid_child_remains_eligible():
    mixed = {
        "id": "mixed",
        "title": "mixed set",
        "media_ids": ["m1", "m2"],
        "tags": [],
        "suggested_price": 30,
    }
    children = [
        {"media_id": "m1", "content_category": "teaser_clothed"},
        {"media_id": "m2", "content_category": "nude_photo"},
    ]
    assert sets_with_sellable_media_evidence([mixed], children) == [mixed]
