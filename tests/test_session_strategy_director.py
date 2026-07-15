from models.session_strategy import (
    NextBestAction,
    SessionGoal,
    derive_session_strategy,
)


def test_director_discovery_controls_noncommercial_strategy():
    result = derive_session_strategy(
        commercial_decision={"action": "CONTINUE_NORMAL_CHAT"},
        conversation_director={
            "phase": "QUALIFY",
            "action": "DISCOVER_PREFERENCE",
            "transition_reason": "flirt_ready_for_one_preference_question",
        },
    )

    assert result.goal == SessionGoal.QUALIFY
    assert result.next_action == NextBestAction.ASK_ONE_QUESTION
    assert result.must_ask_question is True


def test_director_soft_offer_seeds_content_without_inventing_offer():
    result = derive_session_strategy(
        commercial_decision={"action": "CONTINUE_NORMAL_CHAT"},
        conversation_director={
            "phase": "SOFT_OFFER",
            "action": "SEED_PREMIUM_CONTENT",
            "transition_reason": "tension_ready_for_soft_commercial_bridge",
        },
    )

    assert result.goal == SessionGoal.WARM
    assert result.next_action == NextBestAction.SEED_PREMIUM_CONTENT
    assert "price" in result.writer_avoid
    assert result.approved_offer_ids == []
