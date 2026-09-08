"""REL-001 — analyzer failure must be visible and must stop Full Auto.

_fallback_result returned a complete, well-formed, entirely neutral analysis —
purchase_signal "none", crisis_signal "none", resend_requested "false" — with no
marker distinguishing it from a real one. Downstream, _debounced_auto_reply used
those fields to set and clear decline locks, decide freezing, resend PPVs, and
route the writer. During any analyzer incident Full Auto did not stop; it
degraded to guessing and kept selling.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from ai import situation_analyzer
from ai.situation_analyzer import (
    DEGRADED_PARSE,
    DEGRADED_SHAPE,
    DEGRADED_TRANSPORT,
    analysis_is_degraded,
    analyze_situation,
    degraded_reason,
)
from models.model_runtime import ModelResult, ModelTarget, ModelUsage
from models.schemas import (
    ConversationContext,
    Fan,
    Message,
    Persona,
    StageType,
    SuggestionResponse,
)
from services import analyzer_telemetry


def _target():
    return ModelTarget(
        name="analyzer",
        provider="together",
        model="test-model",
        base_url="https://api.together.xyz/v1",
        api_key_env="TOGETHER_API_KEY",
    )


def _ctx(fan_message="hey there"):
    return ConversationContext(
        fan_message=fan_message,
        conversation_history=[Message(role="fan", content=fan_message)],
        fan_profile=Fan(id="fan-1", display_name="Fan"),
        creator_persona=Persona(),
        similar_exchanges=[],
        conversation_stage=StageType.WARMING_UP,
    )


VALID_ANALYSIS = {
    "fan_mood": "horny",
    "fan_intent": "wants to buy",
    "conversation_energy": "rising",
    "strategic_move": "push_for_ppv",
    "tone": "flirty",
    "personal_details_mentioned": [],
    "avoid_repeating": "",
    "purchase_signal": "ready_to_buy",
    "offer_response": "accepted",
    "crisis_signal": "none",
    "wants_media": "true",
}


@pytest.fixture
def analyzer(monkeypatch):
    """Drives analyze_situation's provider call; records telemetry writes."""
    state = {"behaviour": None}

    async def fake_complete(target, **_kwargs):
        behaviour = state["behaviour"]
        if isinstance(behaviour, Exception):
            raise behaviour
        return ModelResult(
            text=behaviour,
            target=target,
            usage=ModelUsage(input_tokens=10, output_tokens=10),
            latency_ms=5,
        )

    async def noop(*_args, **_kwargs):
        return None

    monkeypatch.setattr(situation_analyzer, "complete", fake_complete)
    monkeypatch.setattr(situation_analyzer, "get_runtime_target", lambda _p: _target())
    monkeypatch.setattr(situation_analyzer, "record_model_result", noop)
    monkeypatch.setattr(situation_analyzer, "record_model_failure", noop)
    analyzer_telemetry.reset_for_tests()
    return state


def _analyze(analyzer, behaviour, fan_message="hey there"):
    analyzer["behaviour"] = behaviour
    return asyncio.run(analyze_situation(_ctx(fan_message)))


# --- the four analyzer outcomes ---------------------------------------------


def test_valid_analysis_is_not_marked_degraded(analyzer):
    result = _analyze(analyzer, json.dumps(VALID_ANALYSIS))

    assert analysis_is_degraded(result) is False
    assert result["analysis_degraded"] is False
    assert result["purchase_signal"] == "ready_to_buy"


def test_valid_neutral_analysis_is_distinguishable_from_a_guess(analyzer):
    """The core REL-001 requirement: neutral-because-true vs neutral-because-failed."""
    neutral = {**VALID_ANALYSIS, "purchase_signal": "none", "offer_response": "none"}
    real = _analyze(analyzer, json.dumps(neutral))
    guessed = _analyze(analyzer, RuntimeError("provider exploded"))

    assert real["purchase_signal"] == guessed["purchase_signal"] == "none"
    assert analysis_is_degraded(real) is False
    assert analysis_is_degraded(guessed) is True


def test_transport_exception_is_degraded(analyzer):
    result = _analyze(analyzer, RuntimeError("connection reset"))

    assert analysis_is_degraded(result) is True
    assert degraded_reason(result) == DEGRADED_TRANSPORT


def test_timeout_is_degraded(analyzer):
    result = _analyze(analyzer, TimeoutError("analyzer timed out"))

    assert analysis_is_degraded(result) is True
    assert degraded_reason(result) == DEGRADED_TRANSPORT


def test_malformed_json_is_degraded(analyzer):
    result = _analyze(analyzer, "{not json at all")

    assert analysis_is_degraded(result) is True
    assert degraded_reason(result) == DEGRADED_PARSE


def test_valid_json_of_the_wrong_shape_is_degraded(analyzer):
    """A JSON list parses fine and used to sail through as a truthy non-dict."""
    result = _analyze(analyzer, json.dumps(["not", "a", "dict"]))

    assert analysis_is_degraded(result) is True
    assert degraded_reason(result) == DEGRADED_SHAPE


def test_degraded_reason_never_leaks_provider_detail(analyzer):
    secret = "sk-live-abcdef exploded while calling https://internal.host/v1"
    result = _analyze(analyzer, RuntimeError(secret))

    assert "sk-live" not in json.dumps(result)
    assert degraded_reason(result) == DEGRADED_TRANSPORT


# --- the safety backstop survives -------------------------------------------


def test_self_harm_backstop_still_fires_on_a_degraded_analysis(analyzer):
    """A failed analyzer must not weaken deterministic crisis detection."""
    result = _analyze(analyzer, RuntimeError("down"), fan_message="i want to cut my wrist")

    assert analysis_is_degraded(result) is True
    assert result["crisis_signal"] == "self_harm"


def test_self_harm_backstop_still_fires_on_a_good_analysis(analyzer):
    result = _analyze(
        analyzer, json.dumps(VALID_ANALYSIS), fan_message="i want to cut my wrist"
    )

    assert result["crisis_signal"] == "self_harm"


# --- telemetry ---------------------------------------------------------------


def test_degraded_outcomes_are_countable(analyzer):
    _analyze(analyzer, json.dumps(VALID_ANALYSIS))
    _analyze(analyzer, RuntimeError("down"))
    _analyze(analyzer, "{bad")

    health = analyzer_telemetry.analyzer_health(hours=1)

    assert health["analyses"] == 3
    assert health["degraded"] == 2
    assert health["degraded_by_reason"] == {DEGRADED_TRANSPORT: 1, DEGRADED_PARSE: 1}


def test_health_surface_exposes_analyzer_counts(analyzer):
    """Answerable without reading Railway logs."""
    _analyze(analyzer, RuntimeError("down"))

    from services.analyzer_telemetry import analyzer_health

    assert analyzer_health(hours=1)["degraded"] == 1


# --- Assisted ----------------------------------------------------------------


def test_assisted_response_can_carry_the_degraded_flag():
    flagged = SuggestionResponse(
        suggestions=["come closer"],
        analysis_degraded=True,
        analysis_degraded_reason=DEGRADED_TRANSPORT,
    )
    assert flagged.analysis_degraded is True
    assert flagged.analysis_degraded_reason == DEGRADED_TRANSPORT


def test_assisted_response_defaults_to_not_degraded():
    """A normal suggestion must never claim degradation, or the signal is noise."""
    assert SuggestionResponse(suggestions=["a", "b", "c"]).analysis_degraded is False


def test_assisted_path_marks_the_response(monkeypatch):
    """get_suggestions must not present a guessed analysis as a normal one."""
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1] / "services" / "suggestions.py"
    ).read_text(encoding="utf-8")

    assert "assisted_degraded = analysis_is_degraded(situation)" in source
    assert "analysis_degraded=assisted_degraded" in source
