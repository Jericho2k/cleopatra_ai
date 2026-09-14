"""Commercial-realism regressions from a real simulator conversation.

The observed failure: Cleopatra offered "3 pics for $25. want the link?", then
delivered a $10.63 PPV followed by a $14.37 one, in a conversation where nine
creator turns in ten were exactly two bubbles.

Each test here pins one of the contracts that makes that impossible.
"""

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from models.commercial import CreatorPolicy
from models.content_pricing import human_price_cents
from models.price_learning import PriceLearningPolicy, probe_price_cents
from models.vault_pricing import allocate_step_prices, price_bounds, sequence_bounds
from services.media_packages import build_next_offer
from services.message_shape import apply_message_shape, choose_message_shape
from services.ppv_language import (
    contains_delivery_link_language,
    sanitize_delivery_language,
)
from services.session_lifecycle import (
    mark_step_purchased,
    mark_step_sent,
    next_step,
    session_progress,
)

# nude_photo is the agency's $15-$80 category; set generation writes the
# classifier category into the set's tags, which is how a legacy row still
# carries an approved range.
NUDE_RANGE = (1500, 8000)


def photo_set(set_id, *, price, level, media=3, location="kitchen", category="nude_photo"):
    return {
        "id": set_id,
        "title": f"Kitchen · {set_id}",
        "description": "kitchen set",
        "location": location,
        "outfit": "apron",
        "explicit_min": level,
        "explicit_max": level,
        "suggested_price": price,
        "media_ids": [f"{set_id}-m{index}" for index in range(media)],
        "tags": [category, location],
    }


def video_set(set_id, *, minimum, maximum, level=5, location="kitchen"):
    return {
        "id": set_id,
        "title": f"{location} clip",
        "description": "a private clip",
        "location": location,
        "outfit": "apron",
        "explicit_min": level,
        "explicit_max": level,
        "base_price_cents": (minimum + maximum) // 2,
        "min_price_cents": minimum,
        "max_price_cents": maximum,
        "media_ids": [f"{set_id}-clip"],
        "tags": ["nude_video", location, "video", "individual_video"],
    }


# --- A. approved content range ---------------------------------------------


def test_a_cold_fan_is_priced_inside_the_approved_content_range():
    row = photo_set("kitchen-1", price=25, level=3)
    _, floor, ceiling = sequence_bounds([row])
    assert (floor, ceiling) == NUDE_RANGE

    probe = probe_price_cents(floor, ceiling)
    assert probe is not None
    assert floor <= probe.price_cents <= ceiling
    # The old global $25 cold-start target could not force a $15-$80 set below
    # its floor, and nothing may push it past its ceiling either.
    assert probe.price_cents >= 1500

    offer = build_next_offer(
        [row, photo_set("kitchen-2", price=30, level=4)],
        CreatorPolicy(),
    )
    assert offer is not None
    assert offer.content_floor_cents <= offer.price_cents <= offer.content_ceiling_cents


# --- B. demonstrated spender ------------------------------------------------


def test_b_demonstrated_spender_probes_upward_and_never_resets_to_the_default():
    evidence = {
        "mode": "RANGE",
        "confirmed_purchase_count": 2,
        "recommended_target_cents": 4000,
        "evidence_summary": {
            "demonstrated_willingness_cents": 4000,
            "confirmed_purchase_count": 2,
        },
    }
    cold = probe_price_cents(*NUDE_RANGE)
    proven = probe_price_cents(*NUDE_RANGE, price_learning=evidence)

    assert proven is not None and cold is not None
    assert proven.price_cents > 4000, "a purchase at $40 is a floor, not a ceiling"
    assert proven.price_cents > cold.price_cents
    assert proven.price_cents != 2500, "must not regress to a static cold-start target"
    assert "progressive_upward_probe" in proven.reason_codes
    assert proven.demonstrated_willingness_cents == 4000


# --- C. soft rejection ------------------------------------------------------


def test_c_a_declined_offer_is_soft_resistance_not_a_permanent_ceiling():
    declined = probe_price_cents(
        *NUDE_RANGE,
        price_learning={
            "mode": "RANGE",
            "evidence_summary": {"latest_soft_resistance_cents": 6000},
        },
    )
    assert declined is not None
    assert declined.price_cents < 6000
    assert declined.explicit_ceiling_cents is None, "a decline is not a budget cap"
    assert "soft_resistance_steps_probe_down" in declined.reason_codes

    # The same fan later buys at $65. The old decline must not hold him down.
    recovered = probe_price_cents(
        *NUDE_RANGE,
        price_learning={
            "mode": "RANGE",
            "confirmed_purchase_count": 1,
            "evidence_summary": {
                "latest_soft_resistance_cents": 6000,
                "demonstrated_willingness_cents": 6500,
            },
        },
    )
    assert recovered is not None
    assert recovered.price_cents >= 6500


# --- D. hard current ceiling ------------------------------------------------


def test_d_an_explicit_current_ceiling_caps_every_offer():
    probe = probe_price_cents(
        *NUDE_RANGE,
        price_learning={
            "mode": "RANGE",
            "confirmed_purchase_count": 3,
            "evidence_summary": {
                "demonstrated_willingness_cents": 6000,
                "current_explicit_cap_cents": 2500,
                "confirmed_purchase_count": 3,
            },
        },
    )
    assert probe is not None
    assert probe.price_cents <= 2500
    assert probe.explicit_ceiling_cents == 2500

    offer = build_next_offer(
        [photo_set("kitchen-1", price=25, level=3)],
        CreatorPolicy(),
        hard_ceiling_cents=2500,
    )
    assert offer is not None
    assert offer.price_cents <= 2500


def test_d_content_above_the_stated_ceiling_is_not_discounted_into_range():
    expensive = video_set("bg", minimum=5000, maximum=15000)
    assert (
        build_next_offer(
            [expensive],
            CreatorPolicy(),
            desired_experience="send me a video",
            hard_ceiling_cents=2500,
        )
        is None
    )


# --- E. selected offer authority --------------------------------------------


def test_e_a_single_set_offer_is_delivered_at_exactly_its_offered_price():
    row = photo_set("kitchen-1", price=25, level=3)
    offered = build_next_offer([row], CreatorPolicy())
    assert offered is not None
    assert offered.set_ids == [row["id"]]

    allocation = allocate_step_prices(offered.price_cents, [row])
    assert allocation == [offered.price_cents]


def test_e_a_selected_offer_price_is_authoritative_for_price_learning():
    probe = probe_price_cents(
        *NUDE_RANGE,
        price_learning={"mode": "EXACT", "recommended_target_cents": 4000},
    )
    assert probe is not None
    assert probe.price_cents == 4000
    assert "selected_offer_is_authoritative" in probe.reason_codes


# --- F. multi-step allocation ------------------------------------------------


def test_f_multi_step_prices_are_human_bounded_and_sum_to_the_exact_total():
    steps = [
        photo_set("kitchen-1", price=20, level=2),
        photo_set("kitchen-2", price=30, level=4),
    ]
    allocation = allocate_step_prices(6000, steps)
    assert allocation is not None
    assert sum(allocation) == 6000
    assert all(value % 500 == 0 for value in allocation), "no $10.63 style prices"
    for value, row in zip(allocation, steps, strict=True):
        _, minimum, maximum, _ = price_bounds(row)
        assert minimum <= value <= maximum


def test_f_the_observed_25_dollar_split_can_no_longer_happen():
    """The real failure, reproduced: $25 across two sets became $10.63 + $14.37.

    Two $15-$80 sets are worth at least $30 together, so $25 is not a price
    those two steps can carry at all. The old code divided the number anyway.
    Now the total is refused up front, and the price that IS presented for the
    same content allocates cleanly.
    """
    steps = [
        photo_set("kitchen-1", price=15, level=2),
        photo_set("kitchen-2", price=15, level=3),
    ]
    assert allocate_step_prices(2500, steps) is None

    # One unlock is one set, so it is only ever priced against its own bounds
    # and can never be a total split across two steps at all.
    offer = build_next_offer(steps, CreatorPolicy())
    assert offer is not None
    allocation = allocate_step_prices(
        offer.price_cents, [row for row in steps if row["id"] == offer.set_id]
    )
    assert allocation == [offer.price_cents]
    assert offer.price_cents % 500 == 0


def test_f_an_impossible_allocation_fails_before_the_offer_is_presented():
    fixed = [
        {
            "id": "one",
            "base_price_cents": 3000,
            "min_price_cents": 3000,
            "max_price_cents": 3000,
            "dynamic_pricing_enabled": False,
            "media_ids": ["a"],
        },
        {
            "id": "two",
            "base_price_cents": 3000,
            "min_price_cents": 3000,
            "max_price_cents": 3000,
            "dynamic_pricing_enabled": False,
            "media_ids": ["b"],
        },
    ]
    assert allocate_step_prices(2500, fixed) is None

    # And no offer is ever built at a price its own content cannot carry.
    offer = build_next_offer(fixed, CreatorPolicy())
    if offer is not None:
        row = next(item for item in fixed if item["id"] == offer.set_id)
        assert allocate_step_prices(offer.price_cents, [row])


def test_f_allocation_is_deterministic():
    steps = [photo_set("a", price=20, level=2), photo_set("b", price=30, level=4)]
    first = allocate_step_prices(6000, steps)
    for _ in range(5):
        assert allocate_step_prices(6000, steps) == first


@pytest.mark.parametrize("value", [1063, 1437, 2499, 3333])
def test_f_prices_snap_to_a_human_grid(value):
    snapped = human_price_cents(value, 1000, 8000)
    assert snapped % 500 == 0


# --- G. PPV language --------------------------------------------------------


def test_g_delivery_link_language_is_repaired_on_commercial_turns():
    original = "i have a kitchen set that's pure trouble, 3 pics for $25 | want the link?"
    repaired, changed = sanitize_delivery_language(
        original, decision_action="OFFER_NEXT_UNLOCK"
    )
    assert changed
    assert "link" not in repaired.lower()
    assert "want it" in repaired.lower()
    assert "3 pics for $25" in repaired

    for phrasing in (
        "here's the link babe",
        "i'll send you the link in a sec",
        "click the link when you're ready",
        "sending you the link now",
    ):
        assert contains_delivery_link_language(phrasing)
        cleaned, _ = sanitize_delivery_language(
            phrasing, decision_action="SEND_NEXT_PPV_STEP"
        )
        assert "link" not in cleaned.lower()


def test_g_unrelated_uses_of_link_are_untouched():
    for phrasing in (
        "my ex used to link up with that crowd lol",
        "send me the link to your spotify",
        "the missing link between us is sleep",
    ):
        assert not contains_delivery_link_language(phrasing)
        assert sanitize_delivery_language(
            phrasing, decision_action="OFFER_NEXT_UNLOCK"
        ) == (phrasing, False)
        assert sanitize_delivery_language(
            phrasing, decision_action="CONTINUE_NORMAL_CHAT"
        ) == (phrasing, False)


def test_g_a_ppv_bubble_survives_sanitizing_and_shaping():
    reply = "want the link? | [PPV:media-1:25]"
    cleaned, changed = sanitize_delivery_language(
        reply, decision_action="SEND_NEXT_PPV_STEP"
    )
    assert changed and "link" not in cleaned.lower()
    shaped = apply_message_shape(cleaned, 1)
    assert "[PPV:media-1:25]" in shaped
    assert shaped.split("|")[-1].strip().startswith("[PPV:")


# --- H. message shapes ------------------------------------------------------


def _run_shape_sequence(writer_reply, *, fan_id="fan-simulated", turns=36, keyed=False):
    """Replay a conversation where the writer always produces the same shape."""
    history: list[int] = []
    shapes: list[int] = []
    for turn in range(turns):
        shape = choose_message_shape(
            fan_id=fan_id,
            recent_counts=history,
            writer_word_count=20,
            **(
                {"turn_key": f"fan message {turn}"}
                if keyed
                else {"turn_index": turn}
            ),
        )
        sent = apply_message_shape(writer_reply, shape.target_bubbles)
        count = len([part for part in sent.split("|") if part.strip()])
        shapes.append(count)
        history = (history + [count])[-6:]
    return shapes


def test_h_a_two_bubble_writer_no_longer_produces_a_two_bubble_conversation():
    shapes = _run_shape_sequence("hey you | what are you up to")
    counts = Counter(shapes)
    singles = counts[1] / len(shapes)
    doubles = counts[2] / len(shapes)

    assert 0.55 <= singles <= 0.75, counts
    assert 0.20 <= doubles <= 0.40, counts
    assert "2222" not in "".join(str(value) for value in shapes)


def test_h_three_bubble_bursts_are_possible_but_occasional():
    shapes = _run_shape_sequence("one | two | three")
    counts = Counter(shapes)
    assert counts[3] >= 1, "a genuine three-bubble burst must be reachable"
    assert counts[3] / len(shapes) <= 0.15, counts
    assert counts[1] > counts[2] > counts[3]


def test_h_shaping_only_merges_and_never_splits():
    assert apply_message_shape("one flowing thought here", 3) == "one flowing thought here"
    assert apply_message_shape("a | b | c", 2) == "a, b | c"
    assert apply_message_shape("a. | b | c", 1) == "a. b, c"


def test_h_shape_selection_is_deterministic():
    assert _run_shape_sequence("a | b") == _run_shape_sequence("a | b")
    assert _run_shape_sequence("a | b", keyed=True) == _run_shape_sequence(
        "a | b", keyed=True
    )


def test_h_production_keying_survives_a_truncated_history():
    """Production has no monotonic turn counter, only the last N messages.

    Keying off the message being answered keeps the same distribution in a long
    conversation, where a counter derived from the visible history would stop
    advancing and freeze one shape forever.
    """
    shapes = _run_shape_sequence("one | two | three", turns=240, keyed=True)
    counts = Counter(shapes)
    assert 0.55 <= counts[1] / len(shapes) <= 0.78, counts
    assert 0.15 <= counts[2] / len(shapes) <= 0.35, counts
    assert 0 < counts[3] / len(shapes) <= 0.15, counts


def test_h_commercial_max_messages_still_caps_the_shape():
    shape = choose_message_shape(
        fan_id="fan-simulated", recent_counts=[1, 1], turn_index=10, max_messages=1
    )
    assert shape.target_bubbles == 1


# --- I / J. photo -> video progression ---------------------------------------


def _mixed_vault():
    return [
        photo_set("kitchen-1", price=20, level=2),
        photo_set("kitchen-2", price=25, level=3),
        photo_set("kitchen-3", price=30, level=4),
        video_set("kitchen-clip", minimum=4000, maximum=9000),
    ]


def test_i_a_generic_offer_opens_on_photos_and_the_clip_stays_for_later():
    from services.media_packages import plan_progression

    offer = build_next_offer(_mixed_vault(), CreatorPolicy())
    assert offer is not None
    assert offer.asset_type == "photo_set", "do not burn the clip as the opener"

    # The clip is where the INTERNAL progression ends. It is not part of the
    # offer, is not priced with it, and is never mentioned to the fan.
    ladder = plan_progression(_mixed_vault())
    assert ladder[0]["id"] == offer.set_id
    assert ladder[-1]["id"] == "kitchen-clip"


def test_j_an_explicit_video_request_outranks_photo_first():
    offer = build_next_offer(
        _mixed_vault(),
        CreatorPolicy(),
        desired_experience="can i get a video of you in the kitchen",
    )
    assert offer is not None
    assert offer.asset_type == "video"
    assert offer.set_id == "kitchen-clip"


def test_i_a_video_only_vault_still_sells():
    offer = build_next_offer(
        [video_set("only", minimum=4000, maximum=9000)], CreatorPolicy()
    )
    assert offer is not None
    assert offer.set_id == "only"


# --- K. session continuation --------------------------------------------------


def _two_step_session():
    return {
        "status": "active",
        "current_index": 0,
        "awaiting_purchase_index": None,
        "total_budget_cents": 6500,
        "plan": [
            {
                "step_number": 1,
                "media_ids": ["m1", "m2", "m3"],
                "media_id": "m1",
                "price": 25.0,
                "price_cents": 2500,
                "set_id": "kitchen-1",
                "scene_key": "Kitchen",
                "location": "kitchen",
                "outfit": "apron",
                "explicit_min": 3,
                "explicit_max": 3,
                "asset_type": "photo_set",
                "sent": False,
                "purchased": False,
            },
            {
                "step_number": 2,
                "media_ids": ["v1"],
                "media_id": "v1",
                "price": 40.0,
                "price_cents": 4000,
                "set_id": "kitchen-clip",
                "scene_key": "Kitchen",
                "location": "kitchen",
                "outfit": "apron",
                "explicit_min": 5,
                "explicit_max": 5,
                "asset_type": "video",
                "sent": False,
                "purchased": False,
            },
        ],
    }


def test_k_after_a_purchase_the_session_exposes_the_next_planned_step():
    session = mark_step_sent(_two_step_session())
    assert session_progress(session)["awaiting_purchase"] is True

    session, completed = mark_step_purchased(session, media_id="m1")
    assert completed is False

    progress = session_progress(session)
    assert progress["purchased_steps"] == 1
    assert progress["total_steps"] == 2
    assert progress["has_next_step"] is True
    assert progress["just_purchased"]["asset_type"] == "photo_set"
    assert progress["just_purchased"]["media_count"] == 3
    assert progress["next_step"]["asset_type"] == "video"
    assert progress["continuity"] == {
        "same_location": True,
        "same_outfit": True,
        "same_scene": True,
        "escalates": True,
        "changes_asset_type": True,
    }


def test_k_purchase_gating_survives_the_choreography_context():
    session = mark_step_sent(_two_step_session())
    # Awaiting confirmation: the next step exists in the plan but must not be
    # reachable as "the step to send now".
    assert next_step(session)["step_number"] == 2
    unchanged = mark_step_sent(session)
    assert unchanged["awaiting_purchase_index"] == 0
    assert unchanged["plan"][1]["sent"] is False


def test_k_a_finished_session_does_not_promise_more():
    session = mark_step_sent(_two_step_session())
    session, _ = mark_step_purchased(session, media_id="m1")
    session = mark_step_sent(session)
    session, completed = mark_step_purchased(session, media_id="v1")
    assert completed is True
    progress = session_progress(session)
    assert progress["has_next_step"] is False
    assert "next_step" not in progress


# --- L. sent media ------------------------------------------------------------


def test_l_already_sold_content_cannot_reappear_as_a_fresh_paid_step():
    from services.media_packages import usable_sets

    vault = _mixed_vault()
    remaining = usable_sets(vault, sent_set_ids={"kitchen-1", "kitchen-clip"})
    assert {row["id"] for row in remaining} == {"kitchen-2", "kitchen-3"}

    offer = build_next_offer(remaining, CreatorPolicy())
    assert offer is not None
    assert offer.set_id not in {"kitchen-1", "kitchen-clip"}


# --- M. the observed conversation, end to end ---------------------------------


def test_m_the_observed_simulator_conversation_is_now_well_formed():
    """The exact shape of the real failure, as one contract check."""
    vault = _mixed_vault()
    offered = build_next_offer(vault, CreatorPolicy())
    assert offered is not None

    # The presented price is approved, human-looking, and inside content bounds.
    assert offered.price_cents % 500 == 0
    assert offered.content_floor_cents <= offered.price_cents <= offered.content_ceiling_cents

    # It is deliverable at exactly that price, as one step, with no fractions.
    steps = [row for row in vault if row["id"] == offered.set_id]
    allocation = allocate_step_prices(offered.price_cents, steps)
    assert allocation == [offered.price_cents]

    # The copy that presents it cannot offer a link.
    copy = f"i have a kitchen set that's pure trouble, ${offered.price_cents / 100:g} | want the link?"
    cleaned, changed = sanitize_delivery_language(
        copy, decision_action="OFFER_NEXT_UNLOCK"
    )
    assert changed and "link" not in cleaned.lower()

    # And the conversation around it is not a wall of two-bubble replies.
    counts = Counter(_run_shape_sequence("hey you | what are you up to"))
    assert counts[1] > counts[2]


# --- Writer prompt contracts --------------------------------------------------


def _prompt(**overrides):
    from ai.prompt_builder import build_prompt
    from models.schemas import ConversationContext, Fan, Persona, StageType

    base = dict(
        fan_profile=Fan(id="fan-1", display_name="Alex"),
        creator_persona=Persona(),
        creator_name="Sophia",
        creator_legend={"name": "Sophia"},
        conversation_stage=StageType.FLIRTING,
        conversation_history=[],
        similar_exchanges=[],
        fan_message="that bikini photo got me",
        situation={
            "fan_mood": "eager",
            "conversation_energy": "high",
            "strategic_move": "build tension",
        },
        ppv_offers=[],
        sent_ppv=[],
    )
    base.update(overrides)
    messages = build_prompt(ConversationContext(**base))
    from ai.prompt_blocks import flatten_message_content

    return flatten_message_content(messages[0]["content"]) + "\n" + messages[1]["content"]


def test_platform_context_forbids_link_language_for_our_own_media():
    from ai.prompt_builder import PLATFORM_CONTEXT

    lowered = PLATFORM_CONTEXT.lower()
    assert "attached directly to a chat message" in lowered
    assert "there is no link involved" in lowered
    assert "unrelated link the fan himself brings up" in lowered


def test_a_selling_turn_is_told_how_delivery_actually_works():
    text = _prompt(
        commercial_decision={
            "action": "OFFER_NEXT_UNLOCK",
            "goal": "offer him the one next thing",
            "must_not_send_media": True,
            "may_be_explicit": True,
            "next_offer": {
                "offer_id": "offer:kitchen-2",
                "label": "private photo set",
                "price_cents": 3000,
                "set_id": "kitchen-2",
                "legal_description": "kitchen, apron",
                "media_count": 6,
                "asset_type": "photo_set",
            },
        }
    )
    assert "DELIVERY LANGUAGE" in text
    assert "want the link" in text.lower()
    assert "sending it now" in text.lower()
    # One thing, one price, and nothing about a total or a sequence.
    assert "THE ONE NEXT THING YOU MAY OFFER: private photo set at $30" in text
    assert "no second option" in text
    assert "NEVER tell him how much he might spend in total" in text
    assert "total for" not in text


def test_a_selling_turn_can_never_be_handed_two_prices():
    text = _prompt(
        commercial_decision={
            "action": "OFFER_NEXT_UNLOCK",
            "goal": "offer him the one next thing",
            "must_not_send_media": True,
            "next_offer": {
                "offer_id": "offer:kitchen-2",
                "label": "private photo set",
                "price_cents": 3000,
                "set_id": "kitchen-2",
            },
        }
    )
    # Scoped to the commercial block: the writer VOICE block for V1 legitimately
    # talks about its three assisted options, which is a different thing.
    commercial = text[text.index("FINAL COMMERCIAL POLICY"):]
    assert commercial.count("$30") >= 1
    for menu_language in ("quick private", "full private", "1) ", "2) ", "ORIGINAL ORDER"):
        assert menu_language not in commercial, menu_language


def test_message_shape_target_reaches_the_writer():
    single = _prompt(message_shape={"target_bubbles": 1, "reason": "shape_cycle"})
    assert "MESSAGE SHAPE FOR THIS TURN" in single
    assert "as ONE message" in single

    triple = _prompt(message_shape={"target_bubbles": 3, "reason": "shape_cycle"})
    assert "as 3 message bubbles" in triple
    assert "Never split" in triple

    assert "MESSAGE SHAPE FOR THIS TURN" not in _prompt()


def test_post_purchase_turn_gets_the_next_planned_step_to_bridge_to():
    session = mark_step_sent(_two_step_session())
    session, _ = mark_step_purchased(session, media_id="m1")
    text = _prompt(active_session=session)

    assert "PAID SESSION CHOREOGRAPHY" in text
    assert "step 1 of 2 purchased" in text
    assert "he just unlocked" in text
    assert "next planned step" in text
    assert "it goes further than what he just got." in text
    assert "want 'more?'" not in text.lower()


def test_a_completed_session_does_not_tell_the_writer_to_close():
    """The commercial plan being finished is not the conversation being over.

    Under one-unlock sessions EVERY purchase leaves "nothing further planned",
    so the old "close the experience warmly" line closed the conversation after
    every single sale. What may still be SOLD is stated here; what happens next
    conversationally belongs to the SCENE block.
    """
    session = mark_step_sent(_two_step_session())
    session, _ = mark_step_purchased(session, media_id="m1")
    session = mark_step_sent(session)
    session, _ = mark_step_purchased(session, media_id="v1")
    text = _prompt(active_session=session)
    assert "close the experience warmly" not in text
    assert "nothing further is authorised" in text
    assert "next planned step" not in text


def test_an_agency_can_still_configure_cent_level_pricing_deliberately():
    """Clean prices are the default, not a hard rule.

    Cent-level prices are refused because nobody chose them, not because the
    system cannot express them. An agency that sets a 1-cent grid gets one.
    """
    cents = PriceLearningPolicy(customer_price_step_cents=1, cold_start_probe_bps=3_333)
    probe = probe_price_cents(*NUDE_RANGE, policy=cents)
    assert probe is not None
    assert probe.price_cents == 1500 + (6500 * 3_333) // 10_000

    default = probe_price_cents(*NUDE_RANGE)
    assert default is not None and default.price_cents % 500 == 0

    steps = [photo_set("a", price=20, level=2), photo_set("b", price=30, level=4)]
    allocation = allocate_step_prices(6013, steps, step_cents=1)
    assert allocation is not None and sum(allocation) == 6013


def test_the_operator_send_path_enforces_the_same_content_bounds():
    """An operator and the commercial layer must price the same set the same way.

    A row backfilled to min = max = base would otherwise let the AI price it
    across its approved category range while rejecting the operator for the
    exact same price.
    """
    import inspect

    import main

    source = inspect.getsource(main.send_operator_ppv)
    assert "price_bounds(approved_set)" in source
    assert "dynamic_pricing_enabled" in source

    row = {
        "status": "approved",
        "suggested_price": 25,
        "base_price_cents": 2500,
        "min_price_cents": 2500,
        "max_price_cents": 2500,
        "dynamic_pricing_enabled": True,
        "tags": ["nude_photo"],
    }
    _, minimum, maximum, _ = price_bounds(row)
    assert (minimum, maximum) == NUDE_RANGE
