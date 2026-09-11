from types import SimpleNamespace

from ai.writer_router import WriterRoute, select_writer_route


def _ctx(**overrides):
    values = {
        "situation": {},
        "commercial_decision": None,
        "conversation_stage": "WARMING_UP",
        "active_session": None,
        "fan_profile": SimpleNamespace(
            total_spent=0,
            spend_tier="cold",
            needs_human_review=False,
        ),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_default_route_uses_openrouter_kimi_k26_with_together_fallback(monkeypatch):
    monkeypatch.setenv("WRITER_ROUTING_ENABLED", "true")
    monkeypatch.delenv("WRITER_DEFAULT_PROVIDER", raising=False)
    monkeypatch.delenv("WRITER_DEFAULT_MODEL", raising=False)
    decision = select_writer_route(_ctx())

    assert decision.route == WriterRoute.DEFAULT
    assert decision.primary_target.provider == "openrouter"
    assert decision.primary_target.model == "moonshotai/kimi-k2.6"
    assert decision.fallback_target is not None
    assert decision.fallback_target.provider == "together"
    assert decision.fallback_target.model == "Qwen/Qwen3.7-Plus"


def test_deprecated_kimi_environment_setting_is_redirected_on_together(monkeypatch):
    monkeypatch.setenv("WRITER_ROUTING_ENABLED", "true")
    monkeypatch.setenv("WRITER_DEFAULT_PROVIDER", "together")
    monkeypatch.setenv("WRITER_DEFAULT_MODEL", "moonshotai/Kimi-K2.6")

    decision = select_writer_route(_ctx())

    assert decision.primary_target.model == "moonshotai/Kimi-K3"


def test_kimi_k26_is_kept_on_openrouter(monkeypatch):
    monkeypatch.setenv("WRITER_ROUTING_ENABLED", "true")
    monkeypatch.setenv("WRITER_DEFAULT_PROVIDER", "openrouter")
    monkeypatch.setenv("WRITER_DEFAULT_MODEL", "moonshotai/Kimi-K2.6")

    decision = select_writer_route(_ctx())

    assert decision.primary_target.model == "moonshotai/Kimi-K2.6"


def test_commercial_action_routes_to_complex_writer(monkeypatch):
    monkeypatch.setenv("WRITER_ROUTING_ENABLED", "true")
    decision = select_writer_route(
        _ctx(commercial_decision={"action": "PAUSE_UNTIL_PAYDAY"})
    )

    assert decision.route == WriterRoute.COMMERCIAL_COMPLEX
    assert decision.primary_target.model == "Qwen/Qwen3.7-Plus"
    assert decision.fallback_target is None


def test_active_session_routes_to_complex_writer(monkeypatch):
    monkeypatch.setenv("WRITER_ROUTING_ENABLED", "true")
    decision = select_writer_route(_ctx(active_session={"status": "active"}))

    assert decision.route == WriterRoute.COMMERCIAL_COMPLEX
    assert decision.reason == "active_paid_session"


def test_crisis_routes_to_complex_writer(monkeypatch):
    monkeypatch.setenv("WRITER_ROUTING_ENABLED", "true")
    decision = select_writer_route(
        _ctx(situation={"crisis_signal": "self_harm"})
    )

    assert decision.route == WriterRoute.SAFETY_SENSITIVE
    assert decision.primary_target.model == "Qwen/Qwen3.7-Plus"


def test_high_value_fan_routes_to_complex_writer(monkeypatch):
    monkeypatch.setenv("WRITER_ROUTING_ENABLED", "true")
    fan = SimpleNamespace(
        total_spent=150,
        spend_tier="active",
        needs_human_review=False,
    )
    decision = select_writer_route(_ctx(fan_profile=fan))

    assert decision.route == WriterRoute.COMMERCIAL_COMPLEX
    assert decision.reason == "high_value_fan"
