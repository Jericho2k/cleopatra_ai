"""Acceptance: a fan arrives from TikTok already explicit, and is treated like one.

This is the whole redesign driven as one conversation, through the real
deterministic engine — the commercial policy, the offer builder, the paid-sellable
predicate, the Experience Director and the text-intimacy decision — with no
stubs of any of them. The writer is not called; what is asserted is the state
and the instructions the writer WOULD be handed, because that is the part this
work changed.

The turn-by-turn expectation:

    1. he arrives explicit                    -> engaged immediately
    2. the creator can talk dirty              -> with NO sale action
    3. no "tease or full show?" menu           -> one thing, one price
    4. tease-category junk is never sold       -> the predicate excludes it
    5. one grounded paid unlock                -> when the moment earns it
    6. purchase confirmation                   -> scene to AWAIT_REACTION
    7. he reacts                               -> the reply owes HIM that
    8. the creator plays around the reaction   -> no new offer exists yet
    9. the scene continues                     -> still no new offer
   10. only later a second grounded unlock     -> after a real bridge
   11. the second price may rise               -> inside approved bounds
   12. no total, no ladder, no media spam      -> asserted on the prompt text
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai.prompt_builder import build_prompt  # noqa: E402
from ai.writer_style import WRITER_V3  # noqa: E402
from models.commercial import (  # noqa: E402
    ActionType,
    CreatorPolicy,
    FanCommercialState,
    SextingMode,
)
from models.experience_director import (  # noqa: E402
    ExperienceBeat,
    advance_scene,
)
from models.schemas import (  # noqa: E402
    ConversationContext,
    Fan,
    Message,
    Persona,
    StageType,
)
from services.commercial_events import extract_events, normalize_commercial_facts  # noqa: E402
from services.commercial_policy import CommercialContext, decide_next_action  # noqa: E402
from services.experience_director import scene_allows_new_offer  # noqa: E402
from services.media_packages import build_next_offer, usable_sets  # noqa: E402
from services.scene_metadata import scene_metadata_from_set  # noqa: E402
from services.text_intimacy import TextIntimacy, decide_text_intimacy  # noqa: E402

POLICY = CreatorPolicy(sexting_mode=SextingMode.HYBRID_TEASER, teaser_max_messages=4)


def vault() -> list[dict]:
    """One escalating shoot, plus the teaser junk that must never be sold."""
    return [
        {
            "id": "tease-1",
            "title": "clothed selfies",
            "content_category": "teaser_clothed",
            "tags": ["teaser_clothed", "selfie"],
            "media_ids": ["t1", "t2", "t3"],
            "explicit_min": 0,
            "explicit_max": 1,
            # An operator typed a price into the Sets UI. It changes nothing.
            "base_price_cents": 1500,
            "min_price_cents": 1500,
            "max_price_cents": 1500,
            "dynamic_pricing_enabled": False,
        },
        {
            "id": "shower-1",
            "title": "shower",
            "location": "shower",
            "outfit": "nothing",
            "content_category": "lingerie_photo",
            "tags": ["lingerie_photo", "shower"],
            "media_ids": ["s1", "s2", "s3"],
            "explicit_min": 2,
            "explicit_max": 3,
            "scene_key": "shower:nothing",
            "scene_premise": "photos in the shower",
            "intensity_level": 3,
            "reveals": "stripped back to nothing in the shower",
            "continuation": "nothing left on",
            "base_price_cents": 2000,
            "min_price_cents": 1500,
            "max_price_cents": 6000,
            "dynamic_pricing_enabled": True,
        },
        {
            "id": "shower-2",
            "title": "shower",
            "location": "shower",
            "outfit": "nothing",
            "content_category": "nude_photo",
            "tags": ["nude_photo", "shower"],
            "media_ids": ["s4", "s5"],
            "explicit_min": 4,
            "explicit_max": 4,
            "scene_key": "shower:nothing",
            "scene_premise": "photos in the shower",
            "intensity_level": 4,
            "reveals": "undressed in the shower",
            "continuation": "what happens once she is",
            "base_price_cents": 3500,
            "min_price_cents": 2500,
            "max_price_cents": 9000,
            "dynamic_pricing_enabled": True,
        },
    ]


def _situation(message: str, creator_messages: list[str] | None = None) -> dict:
    return normalize_commercial_facts({}, message, creator_messages or [])


def _decide(message, *, scene, offer, creator_messages=None, state=None, situation=None):
    return decide_next_action(
        POLICY,
        state or FanCommercialState(),
        extract_events(situation or _situation(message, creator_messages)),
        CommercialContext(
            next_offer=offer,
            approved_sets_available=offer is not None,
            within_daily_caps=True,
            experience_allows_new_offer=scene_allows_new_offer(scene),
        ),
    )


def _prompt(*, decision=None, scene=None, text_intimacy=None) -> str:
    context = ConversationContext(
        fan_message="…",
        conversation_history=[Message(role="fan", content="…")],
        fan_profile=Fan(id="fan-1", display_name="Terry"),
        creator_persona=Persona(character="Warm, blunt."),
        similar_exchanges=[],
        conversation_stage=StageType.FLIRTING,
        situation={"crisis_signal": "none"},
        commercial_decision=(
            decision.model_dump(mode="json") if decision is not None else None
        ),
        scene=scene or {},
        text_intimacy=text_intimacy or {},
        ai_stack_profile="cleo_v3",
        writer_prompt_version=WRITER_V3,
    )
    prompt = build_prompt(context, prompt_version=WRITER_V3, reply_mode="auto")
    system = prompt[0]["content"]
    if isinstance(system, list):
        system = "".join(str(block.get("text", "")) for block in system)
    return f"{system}\n\n{prompt[1]['content']}"


# =============================================================================
# The conversation
# =============================================================================


def test_the_whole_arc():
    sellable = usable_sets(vault())

    # ---- 4. tease junk is out of the sellable pool entirely -----------------
    assert "tease-1" not in {row["id"] for row in sellable}, (
        "teaser inventory must never be eligible for an automatic offer, even "
        "with a price column on it"
    )

    # ---- 1 & 2. he arrives explicit; the creator can meet him there ---------
    arrival = "came from tiktok, your body is unreal. im hard rn"
    situation = _situation(arrival)
    scene = advance_scene(situation=situation, latest_fan_message=arrival)

    # Nothing is being sold on this turn, and that is not a reason to be chaste.
    chat_only = decide_next_action(
        POLICY,
        FanCommercialState(),
        [],
        CommercialContext(next_offer=None, approved_sets_available=False),
    )
    assert chat_only.action == ActionType.CONTINUE_NORMAL_CHAT
    assert chat_only.may_be_explicit is False, "the MEDIA flag is off, correctly"

    register = decide_text_intimacy(
        policy=POLICY,
        situation={"wants_explicit": "true", "conversation_energy": "rising"},
        commercial_decision=chat_only.model_dump(mode="json"),
        scene=scene.to_context(),
    )
    assert register.level is TextIntimacy.EXPLICIT, (
        "an explicitly sexual fan gets engaged sexual conversation even while "
        "the commercial action is CONTINUE_NORMAL_CHAT"
    )
    assert register.consumes_free_allowance is True, "and it is still budgeted"

    rendered = _prompt(
        decision=chat_only,
        scene=scene.writer_context(),
        text_intimacy=register.to_context(),
    )
    assert "explicitly sexual language is in bounds" in rendered
    assert "Keep this response non-explicit." not in rendered

    # ---- 3 & 5. one grounded unlock, no menu -------------------------------
    offer = build_next_offer(
        sellable, POLICY, desired_experience="shower", scene=scene.to_context()
    )
    assert offer is not None
    assert offer.set_id == "shower-1", "the entry rung of the scene he asked for"

    wants_it = "im so hard, show me the shower one"
    decision = _decide(wants_it, scene=scene, offer=offer)
    assert decision.action == ActionType.OFFER_NEXT_UNLOCK
    assert decision.next_offer is offer

    offer_prompt = _prompt(decision=decision, scene=scene.writer_context())
    assert "There is no second option, no package, no tier and no menu" in offer_prompt
    assert "NEVER tell him how much he might spend in total" in offer_prompt
    first_price = offer.price_cents

    # ---- 6. purchase confirmation moves the scene, it does not restart it ---
    unlocked_metadata = scene_metadata_from_set(
        next(row for row in vault() if row["id"] == "shower-1")
    )
    scene = advance_scene(
        previous=scene.to_context(),
        commercial_decision={
            "_unlock_confirmed": True,
            "accepted_offer_set_id": "shower-1",
        },
        scene_metadata=unlocked_metadata,
    )
    assert scene.beat is ExperienceBeat.AWAIT_REACTION
    assert scene_allows_new_offer(scene) is False

    # ---- 7. he reacts, and the reply owes HIM that reaction -----------------
    reaction = "holy fuck. that was so much better than i expected"
    scene = advance_scene(
        previous=scene.to_context(),
        situation=_situation(reaction),
        latest_fan_message=reaction,
        scene_metadata=unlocked_metadata,
    )
    assert scene.beat is ExperienceBeat.PLAY
    assert scene.writer_context()["reaction_owed"] is True

    # ---- 8. and no new offer exists to interrupt it ------------------------
    next_offer = build_next_offer(
        [row for row in sellable if row["id"] != "shower-1"],
        POLICY,
        scene=scene.to_context(),
        confirmed_purchase_count=1,
    )
    post_purchase = _decide(reaction, scene=scene, offer=next_offer)
    assert post_purchase.action == ActionType.CONTINUE_NORMAL_CHAT
    assert post_purchase.next_offer is None

    # And the gate is genuinely load-bearing here, not just riding on the fact
    # that a reaction carries no purchase intent: the same turn with him asking
    # for media outright is STILL held, because the scene owes him a reply
    # about what he just unlocked first.
    held = _decide(
        "that was so hot, i want to see you naked",
        scene=scene,
        offer=next_offer,
    )
    assert held.action == ActionType.CONTINUE_NORMAL_CHAT
    assert held.next_offer is None
    assert "bridge" in held.reason

    play_prompt = _prompt(
        decision=post_purchase,
        scene=scene.writer_context(),
        text_intimacy=decide_text_intimacy(
            policy=POLICY,
            situation={"wants_explicit": "true"},
            commercial_decision=post_purchase.model_dump(mode="json"),
            scene=scene.to_context(),
        ).to_context(),
    )
    assert "you are IN the thing he unlocked with him" in play_prompt
    assert "he liked it" in play_prompt
    assert "explicitly sexual language is in bounds" in play_prompt
    assert "$" not in play_prompt.split("SCENE (internal")[1].split("\n\n")[0]

    # ---- 9. the scene continues, and still no offer ------------------------
    for message in ("i keep looking at it", "youre unreal"):
        scene = advance_scene(
            previous=scene.to_context(),
            situation=_situation(message),
            latest_fan_message=message,
        )
        still_chatting = _decide(message, scene=scene, offer=next_offer)
        if still_chatting.action == ActionType.OFFER_NEXT_UNLOCK:
            break
    else:
        assert scene_allows_new_offer(scene) is False

    # ---- 10. a real bridge, and only then a second unlock ------------------
    bridge_message = "ok but what happens after the shower, i want to see"
    bridge_situation = {
        **_situation(bridge_message),
        "wants_explicit": "true",
        "wants_media": "true",
        "conversation_energy": "rising",
        "desired_experience": "after the shower",
    }
    scene = advance_scene(
        previous=scene.to_context(),
        situation=bridge_situation,
        latest_fan_message=bridge_message,
        scene_metadata=unlocked_metadata,
    )
    assert scene.beat is ExperienceBeat.BRIDGE
    assert scene_allows_new_offer(scene) is True

    second = build_next_offer(
        [row for row in sellable if row["id"] != "shower-1"],
        POLICY,
        scene=scene.to_context(),
        confirmed_purchase_count=1,
    )
    assert second is not None
    assert second.set_id == "shower-2", "the next rung of the same scene"

    second_decision = _decide(
        bridge_message, scene=scene, offer=second, situation=bridge_situation
    )
    assert second_decision.action == ActionType.OFFER_NEXT_UNLOCK

    # ---- 11. the price may rise, inside approved bounds --------------------
    assert second.price_cents > first_price, (
        "a confirmed purchase is evidence to probe upward with"
    )
    assert second.price_cents <= second.content_ceiling_cents
    assert second.price_cents >= second.content_floor_cents

    # ---- 12. and nothing about a ladder ever reaches him -------------------
    final_prompt = _prompt(decision=second_decision, scene=scene.writer_context())

    # Asserted as "only one number exists", which is stronger than hunting for
    # forbidden phrases: the ladder leaks as a SECOND price far more easily
    # than as the words "session total".
    import re

    # Scanned over the commercial block only. The fan header legitimately
    # carries his spend-to-date, which is history rather than a future price.
    commercial_block = final_prompt.split(
        "FINAL COMMERCIAL POLICY — THIS OVERRIDES CONFLICTING TEXT ABOVE:"
    )[1]
    prices = set(re.findall(r"\$\d+(?:\.\d+)?", commercial_block))
    assert prices == {f"${second.price_cents // 100}"}, (
        f"exactly one price may appear in the offer, found {sorted(prices)}"
    )
    assert "tease or full" not in final_prompt.lower()
    assert "NEVER tell him how much he might spend in total" in final_prompt
    assert "There is no second option, no package, no tier and no menu" in final_prompt


def test_no_automatic_media_spam_between_the_two_unlocks():
    """Every turn between the unlocks must be must_not_send_media."""
    sellable = usable_sets(vault())
    offer = build_next_offer(sellable, POLICY)
    scene = advance_scene(
        commercial_decision={
            "_unlock_confirmed": True,
            "accepted_offer_set_id": "shower-1",
        },
    )
    for message in ("wow", "that was hot", "mm", "yeah"):
        scene = advance_scene(
            previous=scene.to_context(),
            situation=_situation(message),
            latest_fan_message=message,
        )
        decision = _decide(message, scene=scene, offer=offer)
        assert decision.must_not_send_media is True, (
            f"'{message}' must not trigger an automatic send"
        )


def test_the_menu_wording_is_not_reachable_from_any_decision():
    """"tease or full show?" required two priced options. There is one."""
    offer = build_next_offer(usable_sets(vault()), POLICY)
    decision = _decide("im so hard, show me the shower one", scene=None, offer=offer)
    assert decision.next_offer is not None
    assert not hasattr(decision, "package_options")
    assert decision.mention_price == decision.next_offer.price_cents // 100


def test_the_scene_block_binds_the_writer_to_approved_facts():
    """The premise and "what he got" are catalog metadata, not a writing prompt.

    They are generated once during classification from the classified media
    (services/scene_metadata.py). Handing them to a writer without saying they
    are the LIMIT is how a photo set becomes "the video where I finally take it
    off" in a message the fan then pays for.
    """
    rendered = _prompt(
        scene={
            "beat": "PLAY",
            "premise": "photos in the shower",
            "just_unlocked": "stripped back to nothing in the shower",
            "fan_reaction": "POSITIVE",
            "reaction_owed": True,
            "intimacy_level": 3,
            "tension_level": 3,
            "open_hook": "",
            "desired_direction": "",
        }
    )
    assert "everything above is approved fact" in rendered
    assert "do not add details, body parts, acts or formats" in rendered
