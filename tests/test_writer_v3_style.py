"""What writer_v3 says, what it refuses to say, and what it stopped saying.

V3 is defined by subtraction, so most of these assertions are absences: no
"favorite creator", no "mirrors energy", no bubble-count arithmetic, no demand
for three options, no proof-of-specificity checkbox. An absence is easy to
reintroduce by accident, which is exactly why each one is pinned here.

The other half of the file is the freeze. ``cleo_legacy_v1`` and ``cleo_v2``
exist so V2 and V3 can be compared against a fixed baseline, and a baseline that
drifts while V3 is being tuned is not one. The two hashes below are of the V1
and V2 blocks as they shipped, taken from the commit before the V3 pass.
"""
from __future__ import annotations

import hashlib

import pytest

from ai.writer_style import (
    DEFAULT_REPLY_MODE,
    MODE_ASSISTED,
    MODE_AUTO,
    WRITER_PROMPT_VERSIONS,
    WRITER_V1,
    WRITER_V2,
    WRITER_V3,
    candidate_count,
    content_rules,
    default_communication_style,
    emoji_rules,
    enforces_message_shape,
    layered_style_pressure,
    normalize_reply_mode,
    normalize_writer_prompt_version,
    output_format_instruction,
    persists_improvised_facts,
    response_instructions,
    role_framing,
    voice_rules,
)


def blocks(version: str, mode: str = DEFAULT_REPLY_MODE) -> str:
    return "\n\n".join(
        (
            voice_rules(version),
            emoji_rules(version),
            content_rules(version),
            response_instructions(version, mode),
        )
    )


# --- the freeze -------------------------------------------------------------

# sha256 of "\n\n".join(voice, emoji, content, response) for each frozen
# version, computed on the commit before writer_v3 existed. If one of these
# fails, a V1/V2 baseline moved and every recorded comparison against it is now
# meaningless — the fix is to restore the text, not to update the hash.
FROZEN_BLOCK_DIGESTS = {
    WRITER_V1: "3f79f520db34f179208c6f5a408366c2c9afed8901338d277ddeff78cd9439d6",
    WRITER_V2: "d53a1e9ee75d725e8d1eb196ec5e05f9c039e290ebdf7a4ef51363475d14c36e",
}


@pytest.mark.parametrize("version", sorted(FROZEN_BLOCK_DIGESTS))
def test_the_frozen_writer_versions_are_byte_for_byte_unchanged(version):
    digest = hashlib.sha256(blocks(version).encode("utf-8")).hexdigest()
    assert digest == FROZEN_BLOCK_DIGESTS[version], (
        f"{version} text changed; the V2-vs-V3 comparison baseline must not move"
    )


@pytest.mark.parametrize("version", (WRITER_V1, WRITER_V2))
@pytest.mark.parametrize("mode", (MODE_AUTO, MODE_ASSISTED))
def test_a_frozen_version_ignores_the_reply_mode(version, mode):
    """Auto and Assisted get the identical prompt under V1 and V2.

    The mode parameter exists for V3. A frozen profile that started asking for
    one reply in Auto would be a different profile.
    """
    assert response_instructions(version, mode) == response_instructions(version)
    assert candidate_count(version, mode) == 3
    assert "JSON array of 3 strings" in output_format_instruction(version, mode)


# --- what V3 no longer says -------------------------------------------------


def test_v3_does_not_claim_to_be_his_favorite_creator():
    framing = role_framing(
        WRITER_V3, fan_name="Marcus", creator_display_name="Sophia"
    )
    assert "favorite creator" not in framing.lower()
    assert "favourite creator" not in framing.lower()
    # It states the situation instead, and leaves the relationship to the
    # supplied buyer state.
    assert framing.startswith(
        "You are the creator replying to a fan in private messages on a paid "
        "creator platform."
    )
    assert "Sophia" in framing
    assert "Marcus" not in framing


def test_v1_and_v2_keep_the_favorite_creator_framing():
    """Frozen means frozen, including the part V3 exists to remove."""
    for version in (WRITER_V1, WRITER_V2):
        framing = role_framing(
            version, fan_name="Marcus", creator_display_name="Sophia"
        )
        assert framing == "You are Marcus's favorite creator. Your name is Sophia."


def test_v3_never_mirrors_how_the_fan_types():
    text = blocks(WRITER_V3, MODE_AUTO).lower()
    assert "mirror" not in text
    assert "match his energy" not in text
    assert "match his heat" not in text

    voice = voice_rules(WRITER_V3)
    # And says the distinction explicitly, so "adapt" is not read as "copy".
    assert "Do not adapt to the mechanics of how he types" in voice
    assert "his slang, his spelling, his punctuation, his emoji habits" in voice
    assert "warmth, seriousness, flirt intensity, sexual intensity, pace" in voice


def test_v3_default_persona_style_carries_no_hidden_mirroring_instruction():
    assert default_communication_style(WRITER_V1) == "Short casual texts, mirrors energy."
    assert default_communication_style(WRITER_V2) == "Short casual texts, mirrors energy."
    assert "mirror" not in default_communication_style(WRITER_V3).lower()


def test_v3_does_no_bubble_count_arithmetic():
    voice = voice_rules(WRITER_V3)
    for banned in (
        "One bubble is the default",
        "one bubble by default",
        "Two bubbles when",
        "Three should be rare",
        "exactly",
    ):
        assert banned.lower() not in voice.lower(), banned
    # What it says instead.
    assert "Use one or several message bubbles when it feels natural" in voice
    assert "Do not split a single thought unnaturally" in voice


def test_v3_drops_the_specificity_checkbox_and_the_forced_callback():
    text = blocks(WRITER_V3, MODE_AUTO)
    for banned in (
        "Every reply needs at least one thing",
        "Every reply must contain",
        "Silently run a specificity test",
        "REACT BEFORE YOU ADVANCE",
        "callback to something earlier",
    ):
        assert banned not in text, banned
    # An ordinary short reply is explicitly allowed.
    assert "yeah I get that" in voice_rules(WRITER_V3)


def test_v3_carries_no_stock_stop_word_list():
    voice = voice_rules(WRITER_V3)
    assert "baby, babe, daddy, mommy" not in voice
    assert "STOP WORDS" not in voice


def test_v3_is_much_smaller_than_v2():
    v2 = blocks(WRITER_V2)
    v3 = blocks(WRITER_V3, MODE_AUTO)
    assert len(v3) < len(v2) / 2, (
        f"writer_v3 is {len(v3)} chars against writer_v2's {len(v2)}; V3 is "
        "supposed to be meaningfully simpler, not V2 plus another layer"
    )


# --- what V3 still says -----------------------------------------------------


def test_v3_keeps_the_boundaries_that_are_product_rules_not_style():
    voice = voice_rules(WRITER_V3)
    assert "No promise of a real-life meeting" in voice
    assert "No guilt, no shame, no dependency pressure" in voice
    assert "You are allowed to say no." in voice
    assert "No em dashes" in voice
    assert "whether you're a real person or an AI" in voice


def test_v3_keeps_commercial_authority_outside_the_writer():
    content = content_rules(WRITER_V3)
    assert "decided outside this conversation and supplied to you separately" in content
    assert "Those instructions are authoritative" in content
    assert "never invent content that does not exist" in content
    assert "never name a price you were not given" in content
    assert "never resend something he already bought" in content.lower()


# --- improvisation, and what may never be improvised ------------------------


def test_v3_allows_ordinary_personal_facts_to_be_improvised():
    voice = voice_rules(WRITER_V3)
    assert "improvise a plausible one that fits your persona" in voice
    for example in ("favourite colour", "a food", "a drink", "music", "a small hobby"):
        assert example in voice, example
    assert 'Answering "I don\'t have one" merely because nobody wrote it down' in voice


def test_v3_says_an_improvised_fact_becomes_canon():
    voice = voice_rules(WRITER_V3)
    assert "becomes part of your canon" in voice
    assert "Established facts about you are canon. Never contradict them." in voice


def test_v3_refuses_to_improvise_identity():
    voice = voice_rules(WRITER_V3)
    assert (
        "Your name, age, where you are from, where you live, what you do, your "
        "background, and whether you can meet in person are identity facts"
        in voice
    )
    assert "do not invent it" in voice


# --- output cardinality -----------------------------------------------------


def test_v3_full_auto_asks_for_exactly_one_reply():
    instruction = response_instructions(WRITER_V3, MODE_AUTO)
    assert instruction.startswith("Write ONE reply.")
    assert "Do not write alternatives" in instruction
    assert "option" not in instruction.replace("not a draft and not an option", "")
    assert candidate_count(WRITER_V3, MODE_AUTO) == 1

    fmt = output_format_instruction(WRITER_V3, MODE_AUTO)
    assert "exactly one string" in fmt
    assert '["your reply"]' in fmt
    assert "3" not in fmt


def test_v3_assisted_still_offers_the_operator_a_list():
    instruction = response_instructions(WRITER_V3, MODE_ASSISTED)
    assert "Write 3 reply options" in instruction
    assert candidate_count(WRITER_V3, MODE_ASSISTED) == 3
    assert "JSON array of 3 strings" in output_format_instruction(
        WRITER_V3, MODE_ASSISTED
    )


def test_an_unspecified_mode_is_assisted():
    assert normalize_reply_mode(None) == MODE_ASSISTED
    assert normalize_reply_mode("nonsense") == MODE_ASSISTED
    assert normalize_reply_mode("AUTO") == MODE_AUTO
    assert candidate_count(WRITER_V3) == 3


# --- the capability table ---------------------------------------------------


def test_only_v3_opts_out_of_the_deterministic_shape_policy():
    assert enforces_message_shape(WRITER_V1) is True
    assert enforces_message_shape(WRITER_V2) is True
    assert enforces_message_shape(WRITER_V3) is False


def test_only_v3_persists_improvised_creator_facts():
    assert persists_improvised_facts(WRITER_V1) is False
    assert persists_improvised_facts(WRITER_V2) is False
    assert persists_improvised_facts(WRITER_V3) is True


def test_only_v3_removes_the_surrounding_style_layers():
    assert layered_style_pressure(WRITER_V1) is True
    assert layered_style_pressure(WRITER_V2) is True
    assert layered_style_pressure(WRITER_V3) is False


def test_v3_is_a_registered_version_and_the_default_is_still_frozen():
    assert WRITER_V3 in WRITER_PROMPT_VERSIONS
    assert normalize_writer_prompt_version("writer_v3") == WRITER_V3
    assert normalize_writer_prompt_version("writer_v99") == WRITER_V1
