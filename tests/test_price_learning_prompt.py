from ai.prompt_builder import _render_price_learning


def test_price_learning_prompt_is_internal_and_approved_only():
    rendered = _render_price_learning(
        {
            "mode": "RANGE",
            "confidence": "MEDIUM",
            "recommended_floor_cents": 2500,
            "recommended_target_cents": 3500,
            "recommended_ceiling_cents": 4500,
            "reason_codes": ["evidence_weighted_anchor"],
        }
    )
    assert "target: $35" in rendered
    assert "approved packages only" in rendered.lower()
    assert "do not disclose" in rendered.lower()
