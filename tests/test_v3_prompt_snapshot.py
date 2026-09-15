"""The whole V3 Full Auto prompt, as the model actually receives it.

Every other V3 test looks at one layer. This one assembles the real thing — the
same ``build_prompt`` call the Auto path makes, against a turn that has a
commercial decision, an inventory statement, a director, a session strategy, a
legend and a transcript — and asserts the properties the V3 pass was for. The
point is that those properties survive ASSEMBLY: a rule removed from the writer
voice does not help if the director, the stage instruction or the expression
calibration puts it back three blocks later.

To read the prompt by hand:

    SHOW_V3_PROMPT=1 pytest -s tests/test_v3_prompt_snapshot.py -k printed
"""
from __future__ import annotations

import os

import pytest

from ai.prompt_builder import build_prompt
from ai.writer_style import MODE_ASSISTED, MODE_AUTO, WRITER_V2, WRITER_V3
from models.schemas import (
    ConversationContext,
    Fan,
    Message,
    Persona,
    StageType,
)


def realistic_context() -> ConversationContext:
    """One mid-conversation commercial turn, with every layer populated."""
    return ConversationContext(
        fan_message="ok but what's your favorite color though",
        conversation_history=[
            Message(role="fan", content="hey you"),
            Message(role="creator", content="hey, how was the drive"),
            Message(role="fan", content="long lol. finally home"),
            Message(role="creator", content="worth it though?"),
            Message(role="fan", content="ok but what's your favorite color though"),
        ],
        fan_profile=Fan(
            id="fan-1",
            display_name="Marcus",
            total_spent=40,
            spend_tier="casual",
        ),
        creator_persona=Persona(
            character="Warm, dry sense of humour, blunt when it matters.",
            communication_style="",
            emoji_style="one emoji at most",
            upsell_style="never pushy",
            example_greetings=["hey stranger"],
            example_flirts=["you're going to get me in trouble"],
        ),
        similar_exchanges=[],
        conversation_stage=StageType.FLIRTING,
        creator_name="Sophia",
        situation={
            "fan_mood": "playful",
            "conversation_energy": "high",
            "strategic_move": "build connection",
            "purchase_signal": "none",
            "crisis_signal": "none",
        },
        creator_legend={"name": "Sophia", "other": ["has a cat named Milo"]},
        media_inventory={
            "known": True,
            "authorized_asset_types": ("photo_set",),
            "vault_asset_types": ("photo_set",),
            "video_requested": True,
        },
        commercial_decision={
            "action": "OFFER_NEXT_UNLOCK",
            "goal": "offer him the one next thing",
            "may_be_explicit": False,
            "next_offer": {
                "offer_id": "offer:afternoon",
                "label": "afternoon set",
                "price_cents": 2500,
                "set_id": "afternoon",
                "legal_description": "12 photos, hotel window light",
                "media_count": 12,
                "asset_type": "photo_set",
            },
            "mention_price": 25,
        },
        conversation_director={
            "phase": "FLIRT",
            "action": "PLAYFUL_FLIRT",
            "turns_in_phase": 2,
            "transition_reason": "he is engaged",
            "question_due": True,
            "offer_eligible": True,
            "recent_actions": ["RESPOND_AND_OPEN"],
        },
        session_strategy={
            "goal": "CONVERT",
            "phase": "SOFT_OFFER",
            "next_action": "SEED_PREMIUM_CONTENT",
            "writer_goal": "keep it warm and let the option land",
            "max_messages": 2,
        },
        buyer_lifecycle={"stage": "FIRST_TIME_BUYER", "purchase_count": 1},
        # What the Auto path passes under V1/V2. V3 must ignore it.
        message_shape={"target_bubbles": 2, "reason": "shape_cycle"},
        ai_stack_profile="cleo_v3",
        writer_prompt_version=WRITER_V3,
    )


def _ctx(**overrides) -> ConversationContext:
    """The realistic turn with one or two layers replaced."""
    context = realistic_context()
    return context.model_copy(update=overrides)


def _render_ctx(context: ConversationContext, version: str = WRITER_V3) -> str:
    prompt = build_prompt(context, prompt_version=version, reply_mode=MODE_AUTO)
    system = prompt[0]["content"]
    if isinstance(system, list):
        system = "".join(str(block.get("text", "")) for block in system)
    return f"{system}\n\n{prompt[1]['content']}"


def rendered(version: str, mode: str) -> str:
    prompt = build_prompt(
        realistic_context(), prompt_version=version, reply_mode=mode
    )
    system = prompt[0]["content"]
    if isinstance(system, list):
        system = "".join(str(block.get("text", "")) for block in system)
    return f"{system}\n\n{prompt[1]['content']}"


@pytest.fixture(scope="module")
def v3_auto() -> str:
    return rendered(WRITER_V3, MODE_AUTO)


@pytest.fixture(scope="module")
def v2_auto() -> str:
    return rendered(WRITER_V2, MODE_AUTO)


# --- the six properties this pass exists to guarantee -----------------------


def test_she_is_simply_the_creator_not_his_favourite(v3_auto, v2_auto):
    assert v3_auto.startswith(
        "This conversation takes place inside a paid adult creator subscription platform."
    )
    assert (
        "You are the creator replying to a fan in private messages on a paid "
        "creator platform. Your name is Sophia." in v3_auto
    )
    assert "favorite creator" not in v3_auto
    # And the baseline still says it, so the comparison is real.
    assert "You are Marcus's favorite creator." in v2_auto


def test_nothing_in_the_assembled_prompt_tells_her_to_mirror_him(v3_auto, v2_auto):
    lowered = v3_auto.lower()
    assert "mirrors energy" not in lowered
    assert "match his energy" not in lowered
    assert "match his heat" not in lowered
    assert "match this energy exactly" not in lowered
    assert "match this rhythm and vocabulary" not in lowered
    # V2 assembles at least one of them, which is what V3 removed.
    assert "mirrors energy" in v2_auto.lower() or "match his energy" in v2_auto.lower()


def test_it_does_not_ask_for_three_reply_options(v3_auto, v2_auto):
    assert "Write ONE reply." in v3_auto
    assert '{"messages": ["first message", "second message"]}' in v3_auto
    assert "These are not alternatives" in v3_auto
    for banned in (
        "Write 3 reply options",
        "JSON array of 3 strings",
        "all 3 options",
        "auto mode may send option 1",
        "OPTION ORDER MATTERS",
        "At least one option should be a single message",
    ):
        assert banned not in v3_auto, banned

    assert "Write 3 reply options" in v2_auto
    assert "auto mode may send option 1" in v2_auto


def test_it_imposes_no_deterministic_bubble_count(v3_auto, v2_auto):
    for banned in (
        "MESSAGE SHAPE FOR THIS TURN",
        "Write this reply as ONE message",
        "message bubbles separated by",
        "One bubble is the default",
        "Two bubbles when you genuinely have two thoughts",
        "Three should be rare",
    ):
        assert banned not in v3_auto, banned

    assert "Use one or several message bubbles when it feels natural" in v3_auto
    assert "Do not split a single thought unnaturally" in v3_auto

    # The deterministic policy is still in force for the frozen baseline, from
    # the very same context object.
    assert "MESSAGE SHAPE FOR THIS TURN" in v2_auto


def test_it_allows_safe_personal_improvisation_that_will_persist(v3_auto):
    assert "improvise a plausible one that fits your persona" in v3_auto
    assert "becomes part of your canon" in v3_auto
    assert "Established facts about you are canon. Never contradict them." in v3_auto
    # Identity is not improvisable, and the established canon is present.
    assert "are identity facts" in v3_auto
    assert "FACTS YOU'VE ALREADY ESTABLISHED ABOUT YOURSELF" in v3_auto
    assert "has a cat named Milo" in v3_auto


def test_it_still_carries_every_commercial_and_inventory_constraint(v3_auto):
    # The platform contract.
    assert "The supplied commercial decision and active session are authoritative" in v3_auto
    assert "never independently change whether to sell" in v3_auto
    # What exists.
    assert "content you may offer, promise or describe as yours this turn" in v3_auto
    # The decision itself, verbatim and authoritative.
    assert "FINAL COMMERCIAL POLICY — THIS OVERRIDES CONFLICTING TEXT ABOVE" in v3_auto
    assert "DECIDED ACTION: OFFER_NEXT_UNLOCK" in v3_auto
    assert (
        "THE ONE NEXT THING YOU MAY OFFER: afternoon set at $25 (12 pieces) "
        "— approved content" in v3_auto
    )
    assert "NEVER tell him how much he might spend in total" in v3_auto
    assert "approved content: 12 photos, hotel window light" in v3_auto
    assert "do not invent one" in v3_auto
    assert "The exact price is $25." in v3_auto
    # And the writer block agrees rather than arguing with it.
    assert "decided outside this conversation and supplied to you separately" in v3_auto


def test_the_commercial_decision_no_longer_dictates_the_text_register():
    """``may_be_explicit: False`` is about the MEDIA, and used to gag the words.

    The fixture's decision carries ``may_be_explicit: False`` — the default on
    every ordinary decision — and that used to render "Keep this response
    non-explicit." Whether the creator may speak sexually is now decided by
    services/text_intimacy.py and arrives in its own block, so a turn with
    nothing to sell no longer silently becomes a chaperone.
    """
    rendered_prompt = _render_ctx(
        _ctx(
            commercial_decision={
                "action": "CONTINUE_NORMAL_CHAT",
                "goal": "keep the conversation going",
                "may_be_explicit": False,
            }
        )
    )
    assert "Keep this response non-explicit." not in rendered_prompt
    assert "TEXT INTIMACY" not in rendered_prompt, (
        "no register supplied, no register block"
    )


def test_the_register_block_is_what_grants_or_withholds_explicit_text():
    def _render(text_intimacy):
        return _render_ctx(_ctx(text_intimacy=text_intimacy))

    explicit = _render({"level": "EXPLICIT"})
    assert "explicitly sexual language is in bounds" in explicit
    assert "What you may OFFER, PRICE or SEND is decided separately" in explicit

    flirty = _render({"level": "FLIRTY"})
    assert "flirty and suggestive is in bounds; graphic is not" in flirty

    none = _render({"level": "NONE"})
    assert "nothing sexual and nothing flirtatious" in none


def test_the_scene_block_never_carries_a_price_or_a_step_count():
    rendered_prompt = _render_ctx(
        _ctx(
            scene={
                "beat": "AWAIT_REACTION",
                "premise": "photos in the shower",
                "just_unlocked": "undressed in the shower",
                "fan_reaction": "NONE",
                "reaction_owed": True,
                "intimacy_level": 4,
                "tension_level": 3,
                "open_hook": "",
                "desired_direction": "",
            }
        )
    )
    scene_block = rendered_prompt.split("SCENE (internal, authoritative choreography):")[1]
    scene_block = scene_block.split("\n\n")[0]

    assert "he has just unlocked something" in scene_block
    assert "Nothing new is being offered" in scene_block
    assert "$" not in scene_block
    assert "step" not in scene_block.lower()
    assert "price" not in scene_block.lower()


# --- no duplicated stylistic pressure ---------------------------------------


def test_the_v3_prompt_is_meaningfully_simpler_than_v2(v3_auto, v2_auto):
    assert len(v3_auto) < len(v2_auto), (
        f"V3 assembled to {len(v3_auto)} chars against V2's {len(v2_auto)}"
    )
    # Not a rounding difference: the whole point was to remove layers.
    assert len(v2_auto) - len(v3_auto) > 2000


def test_only_one_layer_dictates_how_a_sentence_should_sound(v3_auto, v2_auto):
    """Expression calibration is style, and V3 keeps style in the voice block."""
    assert "EXPRESSION CALIBRATION" not in v3_auto
    assert "EXPRESSION CALIBRATION" in v2_auto

    # The director still says what has to happen this turn.
    assert "CONVERSATION DIRECTOR (internal):" in v3_auto
    assert "required move: PLAYFUL_FLIRT" in v3_auto
    assert "find out more about what he actually wants" in v3_auto
    # As an objective, never as a mandated sentence.
    assert "MANDATORY" not in v3_auto
    # And not how to phrase it.
    assert "wrap the question inside a personal or playful response" not in v3_auto
    assert "wrap the question inside a personal or playful response" in v2_auto


def test_the_session_strategy_still_states_its_constraints(v3_auto):
    assert "ADAPTIVE SESSION STRATEGY (internal):" in v3_auto
    assert "maximum message bubbles: 2" in v3_auto
    assert "Never invent an offer, price, discount, content set, or promise." in v3_auto


def test_a_safety_turn_still_gets_the_grounded_tone_instruction():
    """The one part of expression calibration that is not style survives."""
    context = realistic_context()
    context.conversation_director = {"phase": "SAFETY", "action": "HAND_OFF"}
    prompt = build_prompt(context, prompt_version=WRITER_V3, reply_mode=MODE_AUTO)
    text = str(prompt[1]["content"])
    assert "EXPRESSION CALIBRATION (internal):" in text
    assert "do not force flirtation" in text


# --- assisted is a different question ---------------------------------------


def test_assisted_under_v3_still_offers_the_operator_three(v3_auto):
    assisted = rendered(WRITER_V3, MODE_ASSISTED)
    assert "Write 3 reply options" in assisted
    assert "Return ONLY a JSON array of 3 strings" in assisted
    assert "Write ONE reply." not in assisted
    # And it is the same voice, not a different profile.
    assert "Do not adapt to the mechanics of how he types" in assisted
    assert "favorite creator" not in assisted
    # The director agrees with the cardinality it was assembled for.
    # The director states one objective, whatever the cardinality.
    assert "find out more about what he actually wants" in assisted


# --- readable by a human ----------------------------------------------------


def test_the_prompt_can_be_printed_for_review(v3_auto):
    """SHOW_V3_PROMPT=1 pytest -s ... prints the exact text the model sees."""
    if os.getenv("SHOW_V3_PROMPT"):
        print("\n" + "=" * 72)
        print(v3_auto)
        print("=" * 72)
    assert v3_auto.strip()
