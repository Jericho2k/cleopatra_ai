"""The writer may only promise media that actually exists.

Production behaviour this pins down: Full Auto began steering a conversation
toward video for a creator whose approved vault contained photo sets and nothing
else. Nothing in the pipeline had told the writer what existed. ``media_packages``
and ``session_planner`` were already right — a video finale is appended only when
real video rows exist — but the writer never saw their conclusion, only the fan
asking for a clip and an approved-experience description whose source tags can
legitimately contain the word "video".

So the invariant is enforced in two independent places, and both are tested here:

* the inventory is STATED to the writer, built from the same rows the planner
  used, so it can never be inferred;
* the inventory is ENFORCED after generation, so a model that promises a clip
  anyway has that promise repaired or the candidate rejected.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.commercial import ActionType, CommercialDecision, CreatorPolicy, PackageOption
from services.inventory_authority import (
    ASSET_PHOTO_SET,
    ASSET_VIDEO,
    MediaInventory,
    asset_types_from_packages,
    asset_types_from_rows,
    asset_types_from_session,
    build_media_inventory,
    choose_inventory_safe_reply,
    next_step_asset_type,
    promises_unavailable_media,
    render_inventory_block,
    sanitize_media_promises,
)
from services.media_packages import build_offer_packages, describe_sequence


PHOTO_ROWS = [
    {
        "id": "p1",
        "title": "Bedroom · black lingerie",
        "description": "Soft bedroom photos in black lingerie.",
        "location": "bedroom",
        "outfit": "black lingerie",
        "explicit_min": 1,
        "explicit_max": 2,
        "suggested_price": 15,
        "media_ids": ["m1", "m2", "m3"],
        "tags": ["nude_photo", "lingerie"],
    },
    {
        "id": "p2",
        "title": "Bedroom · black lingerie",
        "description": "The same bedroom set, further along.",
        "location": "bedroom",
        "outfit": "black lingerie",
        "explicit_min": 3,
        "explicit_max": 4,
        "suggested_price": 25,
        "media_ids": ["m4", "m5"],
        "tags": ["nude_photo", "lingerie"],
    },
]

VIDEO_ROW = {
    "id": "v1",
    "title": "Bedroom · black lingerie · nude video",
    "description": "A private bedroom clip.",
    "location": "bedroom",
    "outfit": "black lingerie",
    "explicit_min": 5,
    "explicit_max": 5,
    "suggested_price": 45,
    "media_ids": ["v-m1"],
    "tags": ["nude_video", "video", "individual_video"],
}


def photo_only_inventory(**overrides) -> MediaInventory:
    base = dict(
        authorized_asset_types=(ASSET_PHOTO_SET,),
        available_package_asset_types=(ASSET_PHOTO_SET,),
        vault_asset_types=(ASSET_PHOTO_SET,),
    )
    base.update(overrides)
    return MediaInventory(**base)


# ---------------------------------------------------------------------------
# 1. Stating the inventory
# ---------------------------------------------------------------------------


def test_asset_types_come_from_the_rows_the_planner_used():
    assert asset_types_from_rows(PHOTO_ROWS) == (ASSET_PHOTO_SET,)
    assert asset_types_from_rows([*PHOTO_ROWS, VIDEO_ROW]) == (
        ASSET_PHOTO_SET,
        ASSET_VIDEO,
    )
    assert asset_types_from_rows([VIDEO_ROW]) == (ASSET_VIDEO,)


def test_a_photo_only_vault_can_never_produce_a_video_package():
    policy = CreatorPolicy(
        quick_package_target_cents=2500,
        full_package_target_cents=5000,
        session_min_steps=1,
        session_max_steps=3,
    )
    # The fan asked for video in as many words. The offer builder must still
    # only offer what exists.
    offers = build_offer_packages(
        PHOTO_ROWS, policy, desired_experience="any videos?"
    )
    assert offers, "a photo vault is still sellable when he asks for video"
    assert asset_types_from_packages(offers) == (ASSET_PHOTO_SET,)
    for offer in offers:
        assert ASSET_VIDEO not in offer.asset_types


def test_an_approved_video_is_still_offerable():
    policy = CreatorPolicy(
        quick_package_target_cents=4000,
        full_package_target_cents=6000,
        session_min_steps=1,
        session_max_steps=3,
    )
    offers = build_offer_packages(
        [*PHOTO_ROWS, VIDEO_ROW], policy, desired_experience="got any videos?"
    )
    assert offers
    assert ASSET_VIDEO in asset_types_from_packages(offers)


def test_a_generic_session_opens_on_photos_when_good_photo_content_exists():
    """Commercial progression, not a media rule: the opener is lower friction."""
    policy = CreatorPolicy(
        quick_package_target_cents=2500,
        full_package_target_cents=6000,
        session_min_steps=1,
        session_max_steps=3,
        offer_two_packages=True,
    )
    offers = build_offer_packages([*PHOTO_ROWS, VIDEO_ROW], policy)
    assert offers
    cheapest = offers[0]
    assert cheapest.asset_types[0] == ASSET_PHOTO_SET, (
        "a generic offer must not open on the clip when photos are available"
    )


def test_a_photo_package_never_describes_itself_as_video():
    """A photo set shot on a video day carries the word in its own metadata."""
    contaminated = [
        dict(
            PHOTO_ROWS[0],
            description="Bedroom photos from the same shoot as the video.",
            tags=["nude_photo", "video", "bedroom"],
        )
    ]
    described = describe_sequence(contaminated) or ""
    assert "video" not in described.lower()
    # The scene itself survives; only the format word is removed.
    assert "bedroom" in described.lower()

    # A package that really does contain a clip keeps the word.
    assert "video" in (describe_sequence([VIDEO_ROW]) or "").lower()


def test_inventory_is_built_from_the_decision_not_from_the_conversation():
    decision = CommercialDecision(
        action=ActionType.PRESENT_SESSION_OPTIONS,
        package_options=[
            PackageOption(
                package_id="package:quick:p1",
                label="quick private session",
                price_cents=2500,
                set_ids=["p1"],
                asset_types=[ASSET_PHOTO_SET],
            )
        ],
    )
    inventory = build_media_inventory(
        decision=decision,
        package_options=decision.package_options,
        approved_rows=PHOTO_ROWS,
        fan_message="do you have any videos?",
    )
    assert inventory.authorized_asset_types == (ASSET_PHOTO_SET,)
    assert inventory.video_requested is True
    assert inventory.may_promise_video is False
    assert inventory.video_requested_but_unavailable is True


def test_an_active_plan_is_narrower_than_the_offer_menu():
    session = {
        "status": "active",
        "current_index": 0,
        "plan": [
            {"asset_type": ASSET_PHOTO_SET, "sent": True},
            {"asset_type": ASSET_PHOTO_SET, "sent": False},
        ],
    }
    assert asset_types_from_session(session) == (ASSET_PHOTO_SET,)
    assert next_step_asset_type(session) == ASSET_PHOTO_SET

    inventory = build_media_inventory(
        active_session=session,
        package_options=[
            PackageOption(
                package_id="p",
                label="l",
                price_cents=4000,
                asset_types=[ASSET_PHOTO_SET, ASSET_VIDEO],
            )
        ],
    )
    assert inventory.authorized_asset_types == (ASSET_PHOTO_SET,)


# ---------------------------------------------------------------------------
# 2. The writer-facing statement
# ---------------------------------------------------------------------------


def test_the_prompt_block_states_the_absence_of_video_explicitly():
    block = render_inventory_block(photo_only_inventory(video_requested=True))
    lowered = block.lower()
    assert "no video" in lowered
    assert "photo sets" in lowered
    # The pivot instruction, and no leaking of internal machinery.
    assert "do not stall" in lowered
    for forbidden in ("vault", "approved rows", "package_id", "cents"):
        assert forbidden not in lowered


def test_the_prompt_block_permits_video_when_it_is_authorised():
    block = render_inventory_block(
        MediaInventory(
            authorized_asset_types=(ASSET_PHOTO_SET, ASSET_VIDEO),
            available_package_asset_types=(ASSET_PHOTO_SET, ASSET_VIDEO),
            vault_asset_types=(ASSET_PHOTO_SET, ASSET_VIDEO),
        )
    )
    assert "video is authorised" in block.lower()
    assert "no video" not in block.lower()


def test_an_unknown_inventory_states_nothing_rather_than_guessing():
    assert render_inventory_block(MediaInventory(known=False)) == ""
    assert render_inventory_block(None) == ""


def test_the_prompt_renders_the_block_ahead_of_commercial_guidance():
    from ai.prompt_builder import _render_media_inventory

    rendered = _render_media_inventory(photo_only_inventory().to_context())
    assert "CONTENT INVENTORY" in rendered
    assert "NO video" in rendered


# ---------------------------------------------------------------------------
# 3. Enforcing it after generation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "promise",
    [
        "i have a video for you 😏",
        "i've got some clips saved just for you",
        "wait till you see the video",
        "just wait until you see my clip",
        "i'll send you a clip later",
        "lemme send u a vid",
        "sending you the video now",
        "the video gets so much better",
        "want the video?",
        "there's a video waiting for you",
        "i filmed a little clip earlier",
        "i recorded a video for you",
        "you're gonna love the video",
        "you'll die when you see my clip",
        "check out my video",
        "my latest clip is unreal",
    ],
)
def test_every_promise_of_absent_video_is_repaired(promise):
    inventory = photo_only_inventory()
    assert promises_unavailable_media(promise, inventory) is True

    repaired, was_repaired = sanitize_media_promises(
        promise, inventory, decision_action="PRESENT_SESSION_OPTIONS"
    )
    assert was_repaired is True
    assert repaired.strip(), "a repair must leave something sendable"
    assert promises_unavailable_media(repaired, inventory) is False
    for word in ("video", "clip", "vid ", "footage"):
        assert word not in repaired.lower()


@pytest.mark.parametrize(
    "ordinary",
    [
        "do you watch a lot of videos on youtube?",
        "that video you sent me of your dog was so funny",
        "i fell asleep watching movies last night",
        "my phone died halfway through the movie",
    ],
)
def test_ordinary_conversation_about_video_is_untouched(ordinary):
    """Mentioning video is fine; offering creator inventory is not."""
    repaired, was_repaired = sanitize_media_promises(
        ordinary, photo_only_inventory(), decision_action="PRESENT_SESSION_OPTIONS"
    )
    assert was_repaired is False
    assert repaired == ordinary


def test_a_non_commercial_turn_is_never_rewritten():
    repaired, was_repaired = sanitize_media_promises(
        "i have a video of my cat doing something stupid",
        photo_only_inventory(),
        decision_action="CONTINUE_NORMAL_CHAT",
    )
    assert was_repaired is False
    assert repaired.endswith("stupid")


def test_with_nothing_authorised_the_promise_is_removed_not_softened():
    empty = MediaInventory()
    repaired, was_repaired = sanitize_media_promises(
        "hey you | i have a video for you",
        empty,
        decision_action="PRESENT_SESSION_OPTIONS",
    )
    assert was_repaired is True
    assert "video" not in repaired.lower()
    assert repaired == "hey you"


def test_authorised_video_is_left_exactly_as_written():
    inventory = MediaInventory(
        authorized_asset_types=(ASSET_VIDEO,),
        available_package_asset_types=(ASSET_VIDEO,),
        vault_asset_types=(ASSET_PHOTO_SET, ASSET_VIDEO),
    )
    original = "wait till you see the video 😈"
    repaired, was_repaired = sanitize_media_promises(
        original, inventory, decision_action="CREATE_PAID_SESSION"
    )
    assert was_repaired is False
    assert repaired == original


def test_repair_is_idempotent():
    inventory = photo_only_inventory()
    once, _ = sanitize_media_promises(
        "i have a video for you", inventory, decision_action="CREATE_PAID_SESSION"
    )
    twice, repaired_again = sanitize_media_promises(
        once, inventory, decision_action="CREATE_PAID_SESSION"
    )
    assert repaired_again is False
    assert twice == once


def test_the_first_clean_candidate_wins_over_a_repaired_one():
    reply, repaired = choose_inventory_safe_reply(
        [
            "i have a video for you",
            "i've been thinking about you all day",
            "another one",
        ],
        photo_only_inventory(),
        decision_action="PRESENT_SESSION_OPTIONS",
    )
    assert repaired is False
    assert reply == "i've been thinking about you all day"


def test_a_repaired_candidate_is_used_when_none_are_clean():
    reply, repaired = choose_inventory_safe_reply(
        ["i have a video for you 😏"],
        photo_only_inventory(),
        decision_action="PRESENT_SESSION_OPTIONS",
    )
    assert repaired is True
    assert reply is not None
    assert "photo set" in reply


def test_nothing_is_sent_when_no_candidate_survives_repair():
    """The one acceptable silence: every option was a promise we cannot keep."""
    reply, repaired = choose_inventory_safe_reply(
        ["i have a video for you", "wait till you see the clip"],
        MediaInventory(),  # nothing authorised at all
        decision_action="PRESENT_SESSION_OPTIONS",
    )
    assert repaired is True
    assert reply is None
