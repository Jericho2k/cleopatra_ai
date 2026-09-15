"""The conversation from the failing Simulator run, pinned end to end.

WHAT WENT WRONG
---------------
A fan arrived already saying what he wanted::

    Fan: came from TikTok, you're sexy
    Fan: your bikini post is hot
    Fan: your body is my ideal type
    Fan: I'm hard / I'd like to see underneath

and got a checkout::

    want me to send it?          -> yes
    quick $60 or full $140?      -> $140
    so which one?                -> $140
    sending three parts
    want part 1?                 -> yes
    here is part 1

Five separate things were wrong, and each one is a separate assertion below:

1. a menu with two prices and a disclosed session total;
2. a mandatory qualification ladder before anything could be offered, driven by
   message counts rather than by what the fan had actually said;
3. a confirmation loop — asked to say yes three times for one purchase;
4. "sending it" as free text the writer had to serialise a tag alongside, which
   it frequently did not, so nothing was attached;
5. the next paid step following a purchase immediately, with no moment in
   between.

This module drives the real deterministic engine — no stubs of the policy, the
offer builder, the session planner's contract, or the delivery planner — and
asserts the corrected shape of every one of them.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai.prompt_builder import build_prompt
from models.commercial import (
    ActionType,
    CreatorPolicy,
    FanCommercialState,
    FanStatus,
    Offer,
    SextingMode,
)
from models.schemas import ConversationContext, Fan, Message, Persona, StageType
from services.commercial_events import (
    augment_pending_offer_events,
    extract_events,
    normalize_commercial_facts,
)
from services.commercial_policy import CommercialContext, decide_next_action
from services.media_packages import build_next_offer, plan_progression
from services.ppv_turn import plan_ppv_step_delivery, strip_ppv_tags
from services.session_lifecycle import mark_step_purchased, mark_step_sent


# --- the creator's actual approved vault ------------------------------------


def vault() -> list[dict]:
    """One coherent beach/bikini shoot that escalates, plus the clip."""
    return [
        {
            "id": "bikini-1",
            "title": "beach day",
            "location": "beach",
            "outfit": "white bikini",
            "explicit_min": 1,
            "explicit_max": 2,
            "media_ids": ["m1", "m2", "m3"],
            "base_price_cents": 2000,
            "min_price_cents": 1500,
            "max_price_cents": 4000,
            "dynamic_pricing_enabled": True,
            "tags": ["bikini", "beach"],
        },
        {
            "id": "bikini-2",
            "title": "beach day",
            "location": "beach",
            "outfit": "white bikini",
            "explicit_min": 3,
            "explicit_max": 4,
            "media_ids": ["m4", "m5"],
            "base_price_cents": 3000,
            "min_price_cents": 2000,
            "max_price_cents": 6000,
            "dynamic_pricing_enabled": True,
            "tags": ["bikini", "topless"],
        },
        {
            "id": "bikini-3",
            "title": "beach day",
            "location": "beach",
            "outfit": "white bikini",
            "explicit_min": 5,
            "explicit_max": 6,
            "media_ids": ["m6", "m7"],
            "base_price_cents": 4500,
            "min_price_cents": 3000,
            "max_price_cents": 9000,
            "dynamic_pricing_enabled": True,
            "tags": ["nude_photo"],
        },
    ]


def _situation(message: str, creator_messages: list[str] | None = None) -> dict:
    """The deterministic backstop's reading of one fan message.

    The analyzer model is not involved: ``normalize_commercial_facts`` is the
    regex layer that runs on every turn regardless, and using only it keeps this
    test about the engine rather than about a model's mood.
    """
    return normalize_commercial_facts({}, message, creator_messages or [])


def _events(message: str, creator_messages: list[str] | None = None):
    return extract_events(_situation(message, creator_messages))


def _ctx(**overrides) -> CommercialContext:
    base = dict(
        approved_sets_available=True,
        within_daily_caps=True,
    )
    base.update(overrides)
    return CommercialContext(**base)


# --- 1. the arrival: high intent, no ceremony -------------------------------


def test_an_opening_that_states_what_he_wants_reaches_an_offer_immediately():
    """No twenty-message qualification. He said it in his first four messages."""
    offer = build_next_offer(vault(), CreatorPolicy())
    assert offer is not None

    # HYBRID_TEASER is the default, and the teaser allowance is untouched: a fan
    # who has explicitly asked does not have to spend it first.
    decision = decide_next_action(
        CreatorPolicy(sexting_mode=SextingMode.HYBRID_TEASER, teaser_max_messages=4),
        FanCommercialState(teaser_messages_used=0),
        _events("im so hard rn, i wanna see whats underneath"),
        _ctx(next_offer=offer),
    )

    assert decision.action == ActionType.OFFER_NEXT_UNLOCK
    assert decision.next_offer is offer


def test_the_offer_is_one_thing_at_one_price_with_no_menu_and_no_total():
    offer = build_next_offer(vault(), CreatorPolicy())
    assert offer is not None
    decision = decide_next_action(
        CreatorPolicy(),
        FanCommercialState(),
        _events("show me more, i want to see underneath"),
        _ctx(next_offer=offer),
    )

    # One offer object, not a list. There is no shape here that can hold two.
    assert decision.next_offer is not None
    assert not isinstance(decision.next_offer, list)
    assert decision.mention_price == offer.price_cents // 100

    # And the internal ladder that exists behind it is not on the decision.
    ladder = plan_progression(vault())
    assert len(ladder) > 1, "there IS a planned progression"
    for field in decision.model_dump(mode="json"):
        assert "ladder" not in field
        assert "total" not in field
        assert "step_count" not in field


def test_the_prompt_never_names_a_session_total_or_a_second_option():
    offer = build_next_offer(vault(), CreatorPolicy())
    assert offer is not None
    decision = decide_next_action(
        CreatorPolicy(),
        FanCommercialState(),
        _events("i want to see underneath"),
        _ctx(next_offer=offer),
    )

    text = _prompt(commercial_decision=decision.model_dump(mode="json"))
    commercial = text[text.index("FINAL COMMERCIAL POLICY"):]

    assert "THE ONE NEXT THING YOU MAY OFFER" in commercial
    assert "no second option" in commercial
    assert "NEVER tell him how much he might spend in total" in commercial
    for banned in ("quick ", "full private", "total for", "1) ", "2) ", "which one"):
        assert banned not in commercial, banned
    # Exactly one price is quoted.
    assert commercial.count(f"${offer.price_cents // 100}") >= 1
    assert "$140" not in commercial and "$60 " not in commercial


# --- 2. one yes is enough ---------------------------------------------------


def test_one_acceptance_goes_straight_to_delivery():
    offer = build_next_offer(vault(), CreatorPolicy())
    assert offer is not None
    state = FanCommercialState(status=FanStatus.OFFER_PENDING, pending_offer=offer)

    events = _events("yes send it", [f"it's ${offer.price_cents // 100}"])
    augment_pending_offer_events(events, "yes send it", offer)

    decision = decide_next_action(CreatorPolicy(), state, events, _ctx(next_offer=offer))

    assert decision.action == ActionType.SEND_NEXT_PPV_STEP
    assert decision.must_not_send_media is False
    assert decision.accepted_offer_set_id == offer.set_id
    # Nothing in the decision asks him anything.
    assert decision.must_not_ask_question is True
    assert decision.conversation_continuation == "none"


def test_the_send_turn_is_told_not_to_ask_again():
    offer = build_next_offer(vault(), CreatorPolicy())
    assert offer is not None
    session = _session_for(offer)
    delivery = plan_ppv_step_delivery(
        decision={"action": "SEND_NEXT_PPV_STEP"}, active_session=session
    )
    assert delivery is not None

    text = _prompt(
        active_session=session,
        ppv_delivery=delivery.writer_context(),
        commercial_decision={
            "action": "SEND_NEXT_PPV_STEP",
            "goal": "he said yes, so this is the send",
            "must_not_send_media": False,
            "must_not_ask_question": True,
        },
    )

    assert "THIS MESSAGE CARRIES THE UNLOCK" in text
    assert "Do not ask whether he wants it: he already said yes." in text
    assert "Do not write a tag" in text
    # And the writer is never handed anything to serialise.
    assert "[PPV:" not in text
    assert delivery.media_id not in text


# --- 3. delivery is deterministic ------------------------------------------


def _session_for(offer: Offer) -> dict:
    row = next(item for item in vault() if item["id"] == offer.set_id)
    return {
        "status": "active",
        "current_index": 0,
        "awaiting_purchase_index": None,
        "plan": [
            {
                "step_number": 1,
                "step_count": 1,
                "media_ids": list(row["media_ids"]),
                "media_id": row["media_ids"][0],
                "price": round(offer.price_cents / 100, 2),
                "price_cents": offer.price_cents,
                "set_id": offer.set_id,
                "asset_type": offer.asset_type,
                "description": "beach day bundle",
                "sent": False,
                "purchased": False,
            }
        ],
    }


def test_the_backend_decides_media_and_price_before_the_writer_runs():
    offer = build_next_offer(vault(), CreatorPolicy())
    assert offer is not None
    delivery = plan_ppv_step_delivery(
        decision={"action": "SEND_NEXT_PPV_STEP"},
        active_session=_session_for(offer),
    )

    assert delivery is not None
    assert delivery.media_ids == ["m1", "m2", "m3"]
    assert delivery.price_cents == offer.price_cents
    assert delivery.set_id == offer.set_id
    assert delivery.asset_type == "photo_set"


def test_the_writer_cannot_cause_a_delivery():
    """A tag in free text is stripped, not obeyed."""
    assert plan_ppv_step_delivery(
        decision={"action": "CONTINUE_NORMAL_CHAT"},
        active_session=_session_for(build_next_offer(vault(), CreatorPolicy())),
    ) is None

    cleaned, stripped = strip_ppv_tags("here it is 😏 [PPV:m1:20]")
    assert stripped is True
    assert cleaned == "here it is 😏"
    assert "PPV" not in cleaned


def test_the_writer_cannot_reprice_or_readdress_a_delivery():
    offer = build_next_offer(vault(), CreatorPolicy())
    delivery = plan_ppv_step_delivery(
        decision={"action": "SEND_NEXT_PPV_STEP"},
        active_session=_session_for(offer),
    )
    assert delivery is not None
    # Whatever the model wrote, these came from the plan.
    cleaned, _ = strip_ppv_tags("sending it now [PPV:some-other-media:5]")
    assert "some-other-media" not in cleaned
    assert delivery.media_id == "m1"
    assert delivery.price_cents == offer.price_cents


def test_a_locked_unsold_step_is_never_sent_twice():
    offer = build_next_offer(vault(), CreatorPolicy())
    session = mark_step_sent(_session_for(offer))
    assert plan_ppv_step_delivery(
        decision={"action": "SEND_NEXT_PPV_STEP"}, active_session=session
    ) is None, "purchase gating: one locked PPV at a time"


def test_the_media_context_persisted_is_the_deterministic_one():
    """What the Simulator renders as a PPV card comes from the plan, not the copy."""
    offer = build_next_offer(vault(), CreatorPolicy())
    delivery = plan_ppv_step_delivery(
        decision={"action": "SEND_NEXT_PPV_STEP"},
        active_session=_session_for(offer),
    )
    assert delivery is not None
    context = delivery.media_context(payment_reference="ref-1")["ppv"]

    assert context["media_ids"] == ["m1", "m2", "m3"]
    assert context["price_cents"] == offer.price_cents
    assert context["access_type"] == "ppv"
    assert context["set_id"] == offer.set_id
    assert context["payment_reference"] == "ref-1"


# --- 4. purchase gating, then the moment, then the next step ----------------


def test_progression_is_locked_behind_a_confirmed_purchase():
    offer = build_next_offer(vault(), CreatorPolicy())
    session = mark_step_sent(_session_for(offer))

    decision = decide_next_action(
        CreatorPolicy(),
        FanCommercialState(status=FanStatus.OFFER_SELECTED),
        [],
        _ctx(
            next_offer=offer,
            session_exists=True,
            session_has_pending_purchase=True,
            session_has_remaining_steps=True,
        ),
    )
    assert decision.action == ActionType.CONTINUE_NORMAL_CHAT
    assert decision.must_not_send_media is True
    assert "not imply" in decision.goal
    assert plan_ppv_step_delivery(
        decision=decision, active_session=session
    ) is None


def test_after_a_purchase_the_conversation_comes_first():
    offer = build_next_offer(vault(), CreatorPolicy())
    session = mark_step_sent(_session_for(offer))
    session, completed = mark_step_purchased(session, media_id="m1")
    assert completed is True, "one unlock is one step; it is done when it is paid"

    # And because it is done, the SESSION can no longer be what holds the
    # conversation back — which is exactly why the Experience Director exists.
    # The veto now comes from the scene, not from a counter on a dead session.
    decision = decide_next_action(
        CreatorPolicy(),
        FanCommercialState(status=FanStatus.PAID_SESSION_ACTIVE),
        _events("that was so hot"),
        _ctx(
            next_offer=offer,
            session_exists=True,
            experience_allows_new_offer=False,
        ),
    )
    assert decision.action == ActionType.CONTINUE_NORMAL_CHAT
    assert decision.must_not_send_media is True
    assert decision.mention_price is None
    assert decision.next_offer is None, "no paid step in the same breath"


def test_the_next_offer_is_incremental_grounded_and_more_than_the_last():
    """He bought the opener. The next thing is the next rung, not a bundle."""
    first = build_next_offer(vault(), CreatorPolicy())
    assert first is not None and first.set_id == "bikini-1"

    already = next(row for row in vault() if row["id"] == first.set_id)
    remaining = [row for row in vault() if row["id"] != first.set_id]
    second = build_next_offer(
        remaining,
        CreatorPolicy(),
        last_unlocked=already,
        confirmed_purchase_count=1,
    )

    assert second is not None
    assert second.set_id == "bikini-2", "one rung up, same scene"
    assert second.set_id != first.set_id, "never re-sell what he already has"
    # Real inventory, real approved bounds.
    assert second.content_floor_cents <= second.price_cents <= second.content_ceiling_cents
    # ...and it is worth more than the opener, because the content is.
    assert second.price_cents >= first.price_cents


def test_he_is_never_told_where_the_ladder_ends():
    ladder = plan_progression(vault())
    offer = build_next_offer(vault(), CreatorPolicy())
    assert offer is not None
    assert len(ladder) >= 3

    rendered = offer.model_dump(mode="json")
    for later in ladder[1:]:
        assert later["id"] not in str(rendered), (
            "the next rungs exist internally and appear nowhere in the offer"
        )
    total_of_ladder = sum(row["base_price_cents"] for row in ladder)
    assert str(total_of_ladder) not in str(rendered)


# --- 5. affordability and pricing invariants are untouched ------------------


def test_a_stated_ceiling_still_caps_the_next_offer():
    offer = build_next_offer(vault(), CreatorPolicy(), hard_ceiling_cents=2000)
    assert offer is not None
    assert offer.price_cents <= 2000


def test_cannot_afford_it_still_stops_selling_entirely():
    offer = build_next_offer(vault(), CreatorPolicy())
    decision = decide_next_action(
        CreatorPolicy(),
        FanCommercialState(status=FanStatus.OFFER_PENDING, pending_offer=offer),
        _events("i cant afford it right now, i get paid friday"),
        _ctx(next_offer=offer),
    )
    assert decision.action == ActionType.PAUSE_UNTIL_PAYDAY
    assert decision.must_not_send_media is True
    assert decision.next_offer is None


def test_no_approved_inventory_means_no_offer_rather_than_an_invented_one():
    decision = decide_next_action(
        CreatorPolicy(sexting_mode=SextingMode.PAID_ONLY),
        FanCommercialState(),
        _events("show me everything"),
        _ctx(next_offer=None, approved_sets_available=False),
    )
    assert decision.action != ActionType.OFFER_NEXT_UNLOCK
    assert decision.next_offer is None
    assert decision.mention_price is None


# --- prompt helper ----------------------------------------------------------


def _prompt(**overrides) -> str:
    context = dict(
        fan_message="i want to see underneath",
        conversation_history=[
            Message(role="fan", content="came from tiktok, you're sexy"),
            Message(role="fan", content="your bikini post is hot"),
            Message(role="fan", content="your body is my ideal type"),
            Message(role="fan", content="i'm hard, i want to see underneath"),
        ],
        fan_profile=Fan(id="fan-1", display_name="Jostar"),
        creator_persona=Persona(),
        similar_exchanges=[],
        conversation_stage=StageType.PRE_UPSELL,
        situation={"strategic_move": "build_tension", "purchase_signal": "none"},
    )
    context.update(overrides)
    prompt = build_prompt(
        ConversationContext(**context),
        prompt_version="writer_v3",
        reply_mode="auto",
    )
    system = prompt[0]["content"]
    if isinstance(system, list):
        system = "\n".join(str(block.get("text", "")) for block in system)
    return f"{system}\n{prompt[1]['content']}"
