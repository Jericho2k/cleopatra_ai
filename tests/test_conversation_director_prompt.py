from ai.prompt_builder import _render_conversation_director


def test_director_prompt_enforces_one_question_and_repetition_guard():
    rendered = _render_conversation_director(
        {
            "phase": "QUALIFY",
            "previous_phase": "FLIRT",
            "action": "DISCOVER_PREFERENCE",
            "turns_in_phase": 1,
            "recent_actions": ["PLAYFUL_FLIRT", "DISCOVER_PREFERENCE"],
            "question_due": True,
            "transition_reason": "flirt_ready_for_one_preference_question",
        }
    )

    lowered = rendered.lower()
    assert "exactly one natural" in lowered
    assert "all 3 reply options" in lowered
    assert "do not repeat" in lowered
