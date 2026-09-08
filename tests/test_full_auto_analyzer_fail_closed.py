"""REL-001 — Full Auto must send nothing when the situation analysis is degraded.

The fabricated analysis is uniformly neutral, so acting on it means a fan saying
"I'll take the $50 one" reads as small talk, decline locks get set and cleared
from a guess, and a PPV can be resent. This drives _debounced_auto_reply far
enough to reach the analyzer and asserts that nothing downstream of it runs.
"""

from __future__ import annotations

import asyncio

import pytest

from ai.situation_analyzer import DEGRADED_TRANSPORT
from models.schemas import Fan, Message, Persona, StageType
from services import suggestions
from services.suggestions import AnalyzerDegradedError

NEUTRAL_DEGRADED = {
    "fan_mood": "curious",
    "fan_intent": "engaging with creator",
    "conversation_energy": "flat",
    "strategic_move": "mirror_warmth",
    "tone": "playful",
    "personal_details_mentioned": [],
    "avoid_repeating": "",
    "purchase_signal": "none",
    "offer_response": "none",
    "crisis_signal": "none",
    "wants_media": "false",
    "resend_requested": "false",
    "analysis_degraded": True,
    "degraded_reason": DEGRADED_TRANSPORT,
}

HEALTHY = {**NEUTRAL_DEGRADED, "analysis_degraded": False, "degraded_reason": ""}


@pytest.fixture
def auto(monkeypatch):
    """Enough of the world for _debounced_auto_reply to reach the analyzer."""
    calls: dict[str, int] = {}

    def count(name):
        calls[name] = calls.get(name, 0) + 1

    async def a(value=None, **_kwargs):
        return value

    fan = Fan(id="fan-1", display_name="Fan", auto_mode=True, needs_human_review=False)

    async def fake_history(_fan_id, limit=None):
        return [Message(role="fan", content="i'll take the $50 one")]

    async def fake_gather_fan(_fan_id):
        return {}

    async def fake_analyze(_ctx, **_kwargs):
        count("analyze")
        return dict(fake_analyze.situation)

    fake_analyze.situation = NEUTRAL_DEGRADED

    async def fake_affordability(**_kwargs):
        count("refresh_affordability")
        return {}

    async def fake_price_learning(**_kwargs):
        count("refresh_price_learning")
        return {}

    async def fake_generate(*_args, **_kwargs):
        count("generate_replies")
        return ["a reply"]

    async def fake_crisis(_creator_id, _fan_id, _situation):
        count("crisis_check")
        return False

    class _Table:
        def select(self, *_a, **_k):
            return self

        def update(self, *_a, **_k):
            return self

        def eq(self, *_a, **_k):
            return self

        def single(self):
            return self

        def execute(self):
            return type("R", (), {"data": {}})()

    monkeypatch.setattr(suggestions, "get_supabase", lambda: type("D", (), {"table": lambda self, n: _Table()})())
    monkeypatch.setattr(suggestions, "get_conversation_history", fake_history)
    monkeypatch.setattr(suggestions, "get_fan_by_id", lambda _f: a(fan))
    monkeypatch.setattr(suggestions, "get_fan_intelligence_context", fake_gather_fan)
    monkeypatch.setattr(suggestions, "get_fan_lifecycle_context", fake_gather_fan)
    monkeypatch.setattr(suggestions, "get_affordability_context", fake_gather_fan)
    monkeypatch.setattr(suggestions, "get_price_learning_context", fake_gather_fan)
    monkeypatch.setattr(suggestions, "get_creator_persona", lambda _c: a(Persona()))
    monkeypatch.setattr(suggestions, "get_ppv_offers", lambda _c: a([]))
    monkeypatch.setattr(suggestions, "get_sent_ppv", lambda _f: a([]))
    monkeypatch.setattr(suggestions, "get_fan_session", lambda _f: a(None))
    monkeypatch.setattr(
        suggestions, "find_similar_exchanges", lambda *_a, **_k: a([])
    )
    monkeypatch.setattr(
        suggestions, "classify_stage", lambda *_a, **_k: StageType.WARMING_UP
    )
    monkeypatch.setattr(suggestions, "analyze_situation", fake_analyze)
    monkeypatch.setattr(
        suggestions, "refresh_affordability_from_situation", fake_affordability
    )
    monkeypatch.setattr(suggestions, "refresh_price_learning", fake_price_learning)
    monkeypatch.setattr(suggestions, "generate_replies", fake_generate)
    monkeypatch.setattr(suggestions, "_crisis_freezes_chat", fake_crisis)

    return {"calls": calls, "analyze": fake_analyze}


def _run(skip_debounce=True):
    return asyncio.run(
        suggestions._debounced_auto_reply(
            "fan-1", "creator-1", skip_debounce=skip_debounce, skip_availability=True
        )
    )


def test_degraded_analysis_raises_and_sends_nothing(auto):
    auto["analyze"].situation = NEUTRAL_DEGRADED

    with pytest.raises(AnalyzerDegradedError) as excinfo:
        _run()

    assert DEGRADED_TRANSPORT in str(excinfo.value)
    assert auto["calls"].get("analyze") == 1
    assert auto["calls"].get("generate_replies") is None, "Full Auto generated a reply"


def test_degraded_analysis_executes_no_commercial_action(auto):
    """No decline lock, no price learning, no affordability write from a guess."""
    auto["analyze"].situation = NEUTRAL_DEGRADED

    with pytest.raises(AnalyzerDegradedError):
        _run()

    assert auto["calls"].get("refresh_affordability") is None
    assert auto["calls"].get("refresh_price_learning") is None


def test_the_error_reaches_the_durable_retry_with_a_useful_reason(auto):
    """The outer except Exception must not swallow it into a generic failure."""
    auto["analyze"].situation = NEUTRAL_DEGRADED

    with pytest.raises(AnalyzerDegradedError) as excinfo:
        _run()

    message = str(excinfo.value)
    assert "degraded" in message
    assert "sent nothing" in message


def test_crisis_backstop_still_freezes_on_a_degraded_analysis(auto, monkeypatch):
    """A positive crisis signal from the deterministic regex still wins."""
    auto["analyze"].situation = {**NEUTRAL_DEGRADED, "crisis_signal": "self_harm"}
    froze = {"called": False}

    async def fake_crisis(_creator_id, _fan_id, situation):
        froze["called"] = True
        return situation.get("crisis_signal") == "self_harm"

    monkeypatch.setattr(suggestions, "_crisis_freezes_chat", fake_crisis)

    # Freezing returns cleanly rather than raising: a human has it now.
    _run()

    assert froze["called"] is True
    assert auto["calls"].get("generate_replies") is None


def test_healthy_analysis_proceeds_normally(auto):
    """The counterpart — without this, the gate could be blocking everything."""
    auto["analyze"].situation = HEALTHY

    _run()

    assert auto["calls"].get("refresh_affordability") == 1
    assert auto["calls"].get("analyze") == 1
