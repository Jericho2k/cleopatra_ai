"""The AI stack is versioned, and the frozen version stays frozen.

``cleo_legacy_v1`` exists so old and new conversational behaviour can be
compared against the same fixed runtime. A comparison against a moving baseline
is not a comparison, so these tests pin exactly what the legacy profile
resolves to, and exactly how V2 differs from it.

They also pin the two things that make this a *versioning* system rather than a
config file: an unconfigured deployment keeps behaving as it did before this
shipped, and no client can name a provider or a model.
"""
from __future__ import annotations

import pytest

from ai import stack_profiles
from ai.stack_profiles import (
    CLEO_LEGACY_V1,
    CLEO_V2,
    DEFAULT_PROFILE_ID,
    PROFILE_IDS,
    STAGE_FAN_INTELLIGENCE,
    STAGE_FAN_SUMMARY,
    STAGE_ORDER,
    STAGE_SITUATION_ANALYZER,
    STAGE_WRITER_COMMERCIAL,
    STAGE_WRITER_DEFAULT,
    STAGE_WRITER_SAFETY,
    environment_profile_id,
    get_profile,
    normalize_profile_id,
)
from ai.writer_style import WRITER_V1, WRITER_V2


# --- the frozen legacy stack ------------------------------------------------


def test_legacy_profile_is_the_stack_that_shipped_on_main():
    """Every stage, exactly as it was before the V2 pass."""
    legacy = CLEO_LEGACY_V1

    analyzer = legacy.stage(STAGE_SITUATION_ANALYZER)
    assert analyzer.resolved_primary() == ("anthropic", "claude-haiku-4-5-20251001")
    assert analyzer.prompt_version == "analyzer_v1"
    assert analyzer.reasoning is False
    assert analyzer.temperature == 0.0
    assert analyzer.resolved_max_tokens() == 650

    writer = legacy.stage(STAGE_WRITER_DEFAULT)
    assert writer.resolved_primary() == ("openrouter", "moonshotai/kimi-k2.6")
    assert writer.resolved_fallback() == ("together", "Qwen/Qwen3.7-Plus")
    assert writer.prompt_version == WRITER_V1
    assert writer.reasoning is False

    commercial = legacy.stage(STAGE_WRITER_COMMERCIAL)
    # The legacy split: a sale arrives in a different model's voice.
    assert commercial.resolved_primary() == ("together", "Qwen/Qwen3.7-Plus")
    assert commercial.resolved_fallback() is None
    assert commercial.prompt_version == WRITER_V1

    safety = legacy.stage(STAGE_WRITER_SAFETY)
    assert safety.resolved_primary() == ("together", "Qwen/Qwen3.7-Plus")

    extractor = legacy.stage(STAGE_FAN_INTELLIGENCE)
    assert extractor.resolved_primary() == ("together", "openai/gpt-oss-120b")
    assert extractor.temperature == 0.0
    assert extractor.resolved_max_tokens() == 700

    summary = legacy.stage(STAGE_FAN_SUMMARY)
    assert summary.resolved_primary() == (
        "together",
        "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    )
    assert summary.temperature == 0.3


def test_legacy_still_honours_the_environment_escape_hatches(monkeypatch):
    """A deployment with WRITER_DEFAULT_MODEL set is running that model today."""
    monkeypatch.setenv("WRITER_DEFAULT_PROVIDER", "together")
    monkeypatch.setenv("WRITER_DEFAULT_MODEL", "moonshotai/Kimi-K3")

    assert CLEO_LEGACY_V1.stage(STAGE_WRITER_DEFAULT).resolved_primary() == (
        "together",
        "moonshotai/Kimi-K3",
    )
    # V2 is pinned: a variable set months ago for the old stack must not
    # silently re-point the new one.
    assert CLEO_V2.stage(STAGE_WRITER_DEFAULT).resolved_primary() == (
        "openrouter",
        "moonshotai/kimi-k2.6",
    )


# --- V2 ---------------------------------------------------------------------


def test_v2_puts_one_writer_in_front_of_conversation_and_commerce():
    ordinary = CLEO_V2.stage(STAGE_WRITER_DEFAULT)
    commercial = CLEO_V2.stage(STAGE_WRITER_COMMERCIAL)

    assert ordinary.resolved_primary() == ("openrouter", "moonshotai/kimi-k2.6")
    assert commercial.resolved_primary() == ("openrouter", "moonshotai/kimi-k2.6")
    # Same voice, so a sale does not read as a different person.
    assert ordinary.resolved_primary() == commercial.resolved_primary()
    # Qwen is the deterministic fallback for both, on a different provider, so
    # an OpenRouter incident cannot silence the writer.
    assert ordinary.resolved_fallback() == ("together", "Qwen/Qwen3.7-Plus")
    assert commercial.resolved_fallback() == ("together", "Qwen/Qwen3.7-Plus")
    assert ordinary.reasoning is False
    assert commercial.reasoning is False


def test_v2_changes_the_writer_prompt_and_nothing_else_downstream():
    for stage_name in (STAGE_WRITER_DEFAULT, STAGE_WRITER_COMMERCIAL, STAGE_WRITER_SAFETY):
        assert CLEO_V2.stage(stage_name).prompt_version == WRITER_V2
    assert CLEO_V2.writer_prompt_version() == WRITER_V2
    assert CLEO_LEGACY_V1.writer_prompt_version() == WRITER_V1


def test_v2_does_not_repoint_the_analyzer_extractor_or_summary():
    """Changing a model for symmetry is how a comparison stops being one."""
    for stage_name in (
        STAGE_SITUATION_ANALYZER,
        STAGE_FAN_INTELLIGENCE,
        STAGE_FAN_SUMMARY,
    ):
        assert (
            CLEO_V2.stage(stage_name).resolved_primary()
            == CLEO_LEGACY_V1.stage(stage_name).resolved_primary()
        )


def test_the_safety_writer_stays_off_openrouter_in_both_profiles():
    # A crisis turn is not commercial expression, and keeping it on a second
    # provider means one incident cannot take every writer down at once.
    assert CLEO_V2.stage(STAGE_WRITER_SAFETY).resolved_primary()[0] == "together"
    assert CLEO_LEGACY_V1.stage(STAGE_WRITER_SAFETY).resolved_primary()[0] == "together"


def test_reasoning_is_off_on_every_writer_target():
    """A reasoning writer returns content=null, which the parser sees as junk."""
    for profile in (CLEO_LEGACY_V1, CLEO_V2):
        for stage_name in (
            STAGE_WRITER_DEFAULT,
            STAGE_WRITER_COMMERCIAL,
            STAGE_WRITER_SAFETY,
        ):
            spec = profile.stage(stage_name)
            assert spec.reasoning is False
            assert spec.primary_target().metadata["reasoning_enabled"] is False
            fallback = spec.fallback_target()
            if fallback is not None:
                assert fallback.metadata["reasoning_enabled"] is False


# --- resolution -------------------------------------------------------------


def test_unset_environment_keeps_the_frozen_behaviour(monkeypatch):
    """Switching every fan onto a new brain because a variable is absent is
    the exact outcome a versioning system exists to prevent."""
    monkeypatch.delenv("AI_STACK_PROFILE", raising=False)

    assert environment_profile_id() == DEFAULT_PROFILE_ID == "cleo_legacy_v1"


def test_the_environment_variable_selects_the_production_default(monkeypatch):
    monkeypatch.setenv("AI_STACK_PROFILE", "cleo_v2")
    assert environment_profile_id() == "cleo_v2"


def test_an_unknown_environment_value_falls_back_rather_than_crashing(monkeypatch):
    monkeypatch.setenv("AI_STACK_PROFILE", "cleo_v99")
    assert environment_profile_id() == DEFAULT_PROFILE_ID


@pytest.mark.parametrize(
    "value",
    ["", None, "cleo_v99", "openrouter/moonshotai/kimi-k2.6", "../../etc/passwd", 42],
)
def test_arbitrary_profile_strings_are_rejected(value):
    """The client names a profile, never a provider or a model."""
    assert normalize_profile_id(value) is None
    assert stack_profiles.is_valid_profile_id(value) is False


def test_get_profile_never_loses_a_turn_over_a_stale_override(monkeypatch):
    """An unknown id on the reply path falls back; the endpoints reject it."""
    monkeypatch.setenv("AI_STACK_PROFILE", "cleo_v2")
    assert get_profile("cleo_v99").profile_id == "cleo_v2"
    assert get_profile(None).profile_id == "cleo_v2"


# --- the registry itself ----------------------------------------------------


def test_every_profile_describes_every_real_stage():
    for profile_id in PROFILE_IDS:
        profile = get_profile(profile_id)
        assert set(profile.stages) == set(STAGE_ORDER)
        described = profile.describe()
        assert described["profile_id"] == profile_id
        assert [row["stage"] for row in described["stages"]] == list(STAGE_ORDER)


def test_the_detail_view_never_leaks_an_api_key():
    for profile_id in PROFILE_IDS:
        for stage in get_profile(profile_id).describe()["stages"]:
            assert "api_key" not in stage
            assert not any("KEY" in str(value) for value in stage.values())
