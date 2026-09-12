"""What writer_v2 actually tells the writer, and what it refuses to.

Not an LLM evaluation — writing quality is judged by hand in the Simulator.
These are deterministic assertions about the PROMPT: that the V2 principles are
present, that the frozen V1 voice has not moved, and above all that neither
version reintroduces manipulative relationship scripting or takes a pricing or
inventory decision back into the writer's hands.
"""
from __future__ import annotations

import pytest

from ai.writer_style import (
    WRITER_PROMPT_VERSIONS,
    WRITER_V1,
    WRITER_V2,
    content_rules,
    emoji_rules,
    normalize_writer_prompt_version,
    response_instructions,
    voice_rules,
)


def v2_text() -> str:
    return "\n".join(
        (
            voice_rules(WRITER_V2),
            emoji_rules(WRITER_V2),
            content_rules(WRITER_V2),
            response_instructions(WRITER_V2),
        )
    )


def v1_text() -> str:
    return "\n".join(
        (
            voice_rules(WRITER_V1),
            emoji_rules(WRITER_V1),
            content_rules(WRITER_V1),
            response_instructions(WRITER_V1),
        )
    )


# --- the V2 principles ------------------------------------------------------


def test_react_before_advancing():
    text = voice_rules(WRITER_V2)
    assert "REACT BEFORE YOU ADVANCE" in text
    assert "His latest message is the hook" in text
    assert "A reply that only reacts is a complete reply" in text


def test_specific_over_generic_without_a_replacement_catchphrase():
    text = voice_rules(WRITER_V2)
    # The actual filler, named so the writer cannot reach for it.
    for phrase in (
        "I love that energy",
        "careful now",
        "don't get too carried away",
        "glad it's working",
        "you're trouble",
        "I like where this is going",
    ):
        assert phrase in text, phrase
    # And the trap that "just ban those six" would walk into.
    assert "Repeating your own signature line is the same failure" in text


def test_a_normal_message_is_allowed_to_be_normal():
    text = voice_rules(WRITER_V2)
    assert "MOST MESSAGES CAN JUST BE NORMAL" in text
    assert "Nothing has to be memorable" in text


def test_one_bubble_is_the_default_and_three_is_rare():
    text = voice_rules(WRITER_V2)
    assert "One short natural bubble is the default" in text
    assert "Three should be rare" in text
    assert "Never chop a single thought into pieces" in text


def test_questions_are_not_required_every_turn():
    text = voice_rules(WRITER_V2)
    assert "Do not put a question in every turn" in text
    assert "A statement that ends is allowed" in text


def test_no_invented_current_life_facts():
    text = voice_rules(WRITER_V2)
    assert "DO NOT INVENT YOUR CURRENT LIFE" in text
    assert "No \"someone called me hot today\"" in text
    assert "no describing what you are doing at this moment" in text


def test_the_creator_does_not_have_to_know_everything():
    text = voice_rules(WRITER_V2)
    assert "YOU DO NOT KNOW EVERYTHING" in text
    assert "asking him to explain" in text


def test_no_deliberate_typos():
    text = voice_rules(WRITER_V2)
    assert "Casual is not misspelled" in text
    assert "no deliberate typos" in text


def test_commercial_writing_uses_the_approved_metadata():
    text = content_rules(WRITER_V2)
    assert "use the approved details you were actually given" in text
    assert "the scene, what you're wearing, where it is, how it progresses" in text
    # The generic-hype failure this replaces.
    assert "my hottest pack" in text


def test_a_purchase_earns_a_reaction_beat_before_the_next_paid_step():
    text = content_rules(WRITER_V2)
    assert "Right after he buys something, react to it like a person" in text
    assert "Do not move to the next paid step unless the instructions" in text


def test_price_talk_is_allowed_but_guilt_is_not():
    text = content_rules(WRITER_V2)
    assert "Talking about money plainly is fine" in text
    assert "never imply he doesn't want you" in text


def test_a_negative_reaction_is_acknowledged_rather_than_re_pitched():
    text = content_rules(WRITER_V2)
    assert "If he reacts badly" in text
    assert "Do not pitch again in the same breath" in text


# --- what NEITHER version may do -------------------------------------------


@pytest.mark.parametrize("version", WRITER_PROMPT_VERSIONS)
def test_no_manipulative_relationship_scripting_in_any_version(version):
    text = "\n".join(
        (
            voice_rules(version),
            emoji_rules(version),
            content_rules(version),
            response_instructions(version),
        )
    ).lower()

    # None of the tactics this pass explicitly refuses to build. Written as
    # techniques rather than bare words on purpose: the prompts legitimately
    # contain "promise to meet in person" and "confess love" inside their
    # PROHIBITIONS, and a bare substring check would fail on the rule that
    # forbids the very thing it is looking for.
    for banned in (
        "fractionation",
        "enslav",
        "make him dependent",
        "create dependency",
        "promise him a future together",
        "tell him you love him",
        "imply you will meet",
    ):
        assert banned not in text, banned

    # And the prohibitions themselves are present rather than merely absent.
    assert "never confess love or promise to meet in person" in text


def test_v2_states_the_relationship_boundary_positively():
    text = voice_rules(WRITER_V2)
    assert "No fake romantic future" in text
    assert "No dependency pressure, no guilt, no shame" in text
    # "Never say no" behaviour is refused explicitly rather than left implicit.
    assert "No pretending you can never say no" in text
    # Warmth is allowed. Deception is not.
    assert "girlfriend-ish closeness are fine. Deception is not" in text


@pytest.mark.parametrize("version", WRITER_PROMPT_VERSIONS)
def test_no_version_moves_pricing_or_inventory_into_the_writer(version):
    text = content_rules(version).lower()
    # The writer is told to express a decision, never to make one.
    assert "invent" in text or "express" in text
    assert "decide the price" not in text
    assert "choose which media" not in text


def test_v2_says_explicitly_that_the_decision_is_already_made():
    text = content_rules(WRITER_V2)
    assert "are already decided for you" in text
    assert "Express that decision" in text
    assert "never promise something the approved details do not contain" in text


@pytest.mark.parametrize("version", WRITER_PROMPT_VERSIONS)
def test_stop_words_and_the_em_dash_rule_survive_in_every_version(version):
    text = voice_rules(version)
    assert "baby, babe, daddy, mommy" in text
    assert "NEVER use an em dash" in text
    assert "Never confess love or promise to meet in person" in text


@pytest.mark.parametrize("version", WRITER_PROMPT_VERSIONS)
def test_the_ai_honesty_rule_survives_in_every_version(version):
    text = voice_rules(version)
    assert "whether you're a real person or an AI" in text
    assert "don't claim to be a real human" in text


# --- version selection ------------------------------------------------------


def test_an_unknown_version_resolves_to_the_frozen_default():
    assert normalize_writer_prompt_version("writer_v99") == WRITER_V1
    assert normalize_writer_prompt_version(None) == WRITER_V1
    assert normalize_writer_prompt_version("") == WRITER_V1


def test_the_two_versions_are_genuinely_different_text():
    assert v1_text() != v2_text()
    assert len(v2_text()) > 1000
