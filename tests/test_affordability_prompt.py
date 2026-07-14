from ai.prompt_builder import _render_affordability


def test_affordability_prompt_separates_now_history_and_future():
    rendered = _render_affordability(
        {
            "status": "LIMITED_NOW",
            "current_limit_cents": 2800,
            "latest_offer_selected_cents": 2800,
            "payday_raw": "Friday",
            "highest_confirmed_purchase_cents": 6000,
            "confirmed_purchase_count": 3,
        }
    )
    assert "current-session ceiling: $28" in rendered
    assert "selected offer awaiting purchase: $28" in rendered
    assert "future liquidity mentioned: Friday" in rendered
    assert "highest confirmed purchase: $60" in rendered
    assert "not estimated wealth" in rendered.lower()
