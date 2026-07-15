from models.conversation_director import (
    ConversationPhase,
    DirectorAction,
    advance_conversation_director,
)


def test_flirt_progresses_to_one_qualification_move_then_tension():
    opening = advance_conversation_director(
        situation={"strategic_move": "acknowledge_compliment_and_redirect"},
        conversation_stage="FLIRTING",
        fan_turn_count=2,
    )
    assert opening.phase == ConversationPhase.FLIRT
    assert opening.action == DirectorAction.PLAYFUL_FLIRT

    qualify = advance_conversation_director(
        previous=opening,
        situation={"strategic_move": "build_tension"},
        conversation_stage="FLIRTING",
        fan_turn_count=3,
    )
    assert qualify.phase == ConversationPhase.QUALIFY
    assert qualify.action == DirectorAction.DISCOVER_PREFERENCE
    assert qualify.question_due is True

    tension = advance_conversation_director(
        previous=qualify,
        situation={"strategic_move": "build_tension"},
        conversation_stage="FLIRTING",
        fan_turn_count=4,
    )
    assert tension.phase == ConversationPhase.TENSION
    assert tension.action == DirectorAction.BUILD_TENSION
    assert tension.question_due is False


def test_tension_does_not_loop_forever():
    result = advance_conversation_director(
        previous={
            "phase": "TENSION",
            "action": "BUILD_TENSION",
            "turns_in_phase": 3,
            "same_action_streak": 3,
            "recent_actions": [
                "BUILD_TENSION",
                "BUILD_TENSION",
                "BUILD_TENSION",
            ],
            "qualification_complete": True,
        },
        situation={"strategic_move": "build_tension"},
        conversation_stage="FLIRTING",
        fan_turn_count=8,
    )

    assert result.action != DirectorAction.BUILD_TENSION


def test_commercial_offer_overrides_progression():
    result = advance_conversation_director(
        previous={"phase": "RAPPORT", "action": "DEEPEN_RAPPORT"},
        commercial_decision={"action": "PRESENT_SESSION_OPTIONS"},
        fan_turn_count=3,
    )

    assert result.phase == ConversationPhase.OFFER
    assert result.action == DirectorAction.PRESENT_APPROVED_OPTIONS
    assert result.offer_eligible is True


def test_decline_moves_to_objection_and_stops_questioning():
    result = advance_conversation_director(
        situation={"purchase_signal": "declined"},
        fan_turn_count=5,
    )

    assert result.phase == ConversationPhase.OBJECTION
    assert result.action == DirectorAction.HANDLE_OBJECTION
    assert result.must_not_ask_question is True
