"""Which writer model each route uses, per profile — and the prompt it writes with.

The routing CONDITIONS are shared application logic and are identical under
every profile: they describe the conversation, not the brain answering it. What
changes per profile is only which model each of the three routes points at, and
which writer voice it uses.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from ai.prompt_builder import build_prompt
from ai.writer_router import WriterRoute, select_writer_route
from ai.writer_style import WRITER_V1, WRITER_V2
from models.schemas import ConversationContext, Fan, Message, Persona, StageType


def ctx(**overrides):
    base = dict(
        situation={},
        commercial_decision={},
        buyer_lifecycle={},
        fan_profile=SimpleNamespace(
            needs_human_review=False, spend_tier="casual", total_spent=0
        ),
        conversation_stage=StageType.FLIRTING,
        active_session=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


COMMERCIAL = ctx(commercial_decision={"action": "PRESENT_SESSION_OPTIONS"})
ORDINARY = ctx()
CRISIS = ctx(situation={"crisis_signal": "self_harm"})


def test_ordinary_conversation_uses_kimi_under_both_profiles():
    for profile_id in ("cleo_legacy_v1", "cleo_v2"):
        decision = select_writer_route(ORDINARY, profile_id=profile_id)
        assert decision.route is WriterRoute.DEFAULT
        assert decision.primary_target.model == "moonshotai/kimi-k2.6"
        assert decision.primary_target.provider == "openrouter"


def test_v2_writes_commercial_turns_with_kimi_too():
    decision = select_writer_route(COMMERCIAL, profile_id="cleo_v2")

    assert decision.route is WriterRoute.COMMERCIAL_COMPLEX
    assert decision.primary_target.model == "moonshotai/kimi-k2.6"
    # No personality discontinuity: ordinary chat and a sale are the same voice.
    ordinary = select_writer_route(ORDINARY, profile_id="cleo_v2")
    assert decision.primary_target.model == ordinary.primary_target.model


def test_legacy_still_hands_commercial_turns_to_qwen():
    decision = select_writer_route(COMMERCIAL, profile_id="cleo_legacy_v1")

    assert decision.route is WriterRoute.COMMERCIAL_COMPLEX
    assert decision.primary_target.model == "Qwen/Qwen3.7-Plus"
    assert decision.primary_target.provider == "together"
    assert decision.fallback_target is None


def test_qwen_is_the_deterministic_fallback_for_every_v2_writer_turn():
    for conversation in (ORDINARY, COMMERCIAL):
        decision = select_writer_route(conversation, profile_id="cleo_v2")
        assert decision.fallback_target is not None
        assert decision.fallback_target.model == "Qwen/Qwen3.7-Plus"
        # Deliberately a different provider from the primary.
        assert decision.fallback_target.provider != decision.primary_target.provider


def test_a_crisis_turn_stays_on_together_under_both_profiles():
    for profile_id in ("cleo_legacy_v1", "cleo_v2"):
        decision = select_writer_route(CRISIS, profile_id=profile_id)
        assert decision.route is WriterRoute.SAFETY_SENSITIVE
        assert decision.primary_target.provider == "together"


@pytest.mark.parametrize(
    "conversation, expected_route",
    [
        (ORDINARY, WriterRoute.DEFAULT),
        (COMMERCIAL, WriterRoute.COMMERCIAL_COMPLEX),
        (CRISIS, WriterRoute.SAFETY_SENSITIVE),
        (ctx(active_session={"plan": []}), WriterRoute.COMMERCIAL_COMPLEX),
        (ctx(situation={"purchase_signal": "declined"}), WriterRoute.COMMERCIAL_COMPLEX),
        (ctx(buyer_lifecycle={"stage": "VIP"}), WriterRoute.COMMERCIAL_COMPLEX),
    ],
)
def test_the_route_a_turn_takes_is_the_same_under_every_profile(
    conversation, expected_route
):
    """Routing conditions are shared application logic, not a profile property."""
    legacy = select_writer_route(conversation, profile_id="cleo_legacy_v1")
    v2 = select_writer_route(conversation, profile_id="cleo_v2")

    assert legacy.route is expected_route
    assert v2.route is expected_route
    assert legacy.reason == v2.reason


def test_reasoning_is_off_on_every_selected_target():
    for profile_id in ("cleo_legacy_v1", "cleo_v2"):
        for conversation in (ORDINARY, COMMERCIAL, CRISIS):
            decision = select_writer_route(conversation, profile_id=profile_id)
            assert decision.primary_target.metadata["reasoning_enabled"] is False
            if decision.fallback_target is not None:
                assert decision.fallback_target.metadata["reasoning_enabled"] is False


def test_the_decision_carries_the_profile_and_its_writer_voice():
    legacy = select_writer_route(ORDINARY, profile_id="cleo_legacy_v1")
    v2 = select_writer_route(ORDINARY, profile_id="cleo_v2")

    assert legacy.ai_stack_profile == "cleo_legacy_v1"
    assert legacy.prompt_version == WRITER_V1
    assert v2.ai_stack_profile == "cleo_v2"
    assert v2.prompt_version == WRITER_V2

    telemetry = v2.telemetry_metadata()
    assert telemetry["ai_stack_profile"] == "cleo_v2"
    assert telemetry["writer_prompt_version"] == WRITER_V2


def test_the_route_reads_the_profile_off_the_context_when_not_told(monkeypatch):
    monkeypatch.setenv("AI_STACK_PROFILE", "cleo_legacy_v1")
    decision = select_writer_route(ctx(ai_stack_profile="cleo_v2"))

    assert decision.ai_stack_profile == "cleo_v2"


# --- the prompt the writer is actually handed -------------------------------


def _conversation_context(**overrides) -> ConversationContext:
    base = dict(
        fan_message="hey what are you up to",
        conversation_history=[Message(role="fan", content="hey what are you up to")],
        fan_profile=Fan(id="fan-1", display_name="Sam"),
        creator_persona=Persona(),
        similar_exchanges=[],
        conversation_stage=StageType.FLIRTING,
    )
    base.update(overrides)
    return ConversationContext(**base)


def test_each_prompt_version_produces_its_own_writer_voice():
    legacy = str(build_prompt(_conversation_context(), prompt_version=WRITER_V1))
    v2 = str(build_prompt(_conversation_context(), prompt_version=WRITER_V2))

    # V1's exact wording, unchanged.
    assert "SOUND HUMAN WITHOUT GOING FLAT" in legacy
    assert "SOUND HUMAN WITHOUT GOING FLAT" not in v2

    # V2's principles, present only in V2.
    for phrase in (
        "REACT BEFORE YOU ADVANCE",
        "BE SPECIFIC, NOT GENERIC",
        "MOST MESSAGES CAN JUST BE NORMAL",
        "DO NOT INVENT YOUR CURRENT LIFE",
        "YOU DO NOT KNOW EVERYTHING",
    ):
        assert phrase in v2, phrase
        assert phrase not in legacy, phrase


def test_the_default_prompt_version_is_the_frozen_one():
    """Every existing caller that passes nothing keeps writer_v1."""
    assert "SOUND HUMAN WITHOUT GOING FLAT" in str(build_prompt(_conversation_context()))


def test_an_unknown_prompt_version_falls_back_rather_than_losing_the_turn():
    prompt = str(build_prompt(_conversation_context(), prompt_version="writer_v99"))

    assert "SOUND HUMAN WITHOUT GOING FLAT" in prompt


def test_the_context_carries_the_version_when_the_caller_does_not():
    prompt = str(
        build_prompt(_conversation_context(writer_prompt_version=WRITER_V2))
    )

    assert "REACT BEFORE YOU ADVANCE" in prompt


def test_both_versions_keep_the_deterministic_commercial_authority():
    """The writer voice changes. What the writer is ALLOWED to do does not."""
    decision = {
        "action": "PRESENT_SESSION_OPTIONS",
        "goal": "offer the approved options",
        "must_not_send_media": True,
        "package_options": [
            {"label": "quick private session", "price_cents": 2500, "step_count": 2}
        ],
    }
    for version in (WRITER_V1, WRITER_V2):
        prompt = str(
            build_prompt(
                _conversation_context(commercial_decision=decision),
                prompt_version=version,
            )
        )
        assert "FINAL COMMERCIAL POLICY" in prompt
        assert "PRESENT_SESSION_OPTIONS" in prompt
        assert "Do NOT send media" in prompt
        assert "Do not invent another price or package" in prompt
