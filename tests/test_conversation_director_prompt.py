from ai.prompt_builder import _render_conversation_director


def test_director_prompt_states_the_discovery_objective_not_a_forced_question():
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
    # An objective. "MANDATORY: ask exactly one question" is what turned the
    # creator into an interviewer.
    assert "find out more about what he actually wants" in lowered
    assert "mandatory" not in lowered
    assert "do not repeat" in lowered
