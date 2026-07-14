from ai.prompt_builder import _render_fan_intelligence


def test_writer_context_puts_hard_limits_first_and_formats_money():
    rendered = _render_fan_intelligence(
        {
            "hard_limits": ["no humiliation"],
            "facts": [
                {
                    "fact_key": "stated_budget_cents",
                    "value": 4000,
                    "status": "explicit",
                },
                {
                    "fact_key": "location",
                    "value": "Germany",
                    "status": "confirmed",
                },
            ],
            "conflicts": [],
        }
    )
    assert "Hard limits (never violate or negotiate): no humiliation" in rendered
    assert "stated budget: $40" in rendered
    assert "location: Germany" in rendered


def test_conflicts_are_not_rendered_as_truth():
    rendered = _render_fan_intelligence(
        {
            "facts": [],
            "hard_limits": [],
            "conflicts": [
                {"fact_key": "location", "values": ["Germany", "France"]}
            ],
        }
    )
    assert "Conflicted information" in rendered
    assert "location" in rendered
    assert "Germany" not in rendered
    assert "France" not in rendered
