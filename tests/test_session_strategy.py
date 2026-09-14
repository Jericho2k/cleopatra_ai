from models.session_strategy import NextBestAction, SessionGoal, derive_session_strategy


def test_crisis_always_hands_off_and_suppresses_selling():
    result = derive_session_strategy(
        situation={"crisis_signal": "self_harm"},
        commercial_decision={"action": "PRESENT_SESSION_OPTIONS"},
    )
    assert result.goal == SessionGoal.CARE
    assert result.next_action == NextBestAction.HAND_OFF
    assert "selling" in result.writer_avoid


def test_accepted_approved_offer_closes_exactly():
    result = derive_session_strategy(
        commercial_decision={
            "action": "SEND_NEXT_PPV_STEP",
            "new_status": "OFFER_SELECTED",
            "session_budget_cents": 2800,
            "next_offer": {"offer_id": "offer:1", "price_cents": 2800},
        }
    )
    assert result.goal == SessionGoal.CLOSE
    assert result.next_action == NextBestAction.SEND_NEXT_STEP
    assert result.selected_offer_price_cents == 2800
    assert result.approved_offer_ids == ["offer:1"]


def test_affordability_pause_never_counteroffers():
    result = derive_session_strategy(
        affordability={"temporary_constraint": True},
        price_learning={"mode": "NO_OFFER"},
    )
    assert result.goal == SessionGoal.HOLD
    assert result.next_action == NextBestAction.PAUSE_SELLING
    assert "counteroffer" in result.writer_avoid


def test_new_prospect_qualifies_once():
    result = derive_session_strategy(
        lifecycle={"stage": "PROSPECT"},
        conversation_stage="WARMING_UP",
    )
    assert result.goal == SessionGoal.QUALIFY
    # An objective, not a forced sentence: discovery no longer demands a question.
    assert result.must_ask_question is False
    assert result.next_action == NextBestAction.ASK_ONE_QUESTION
    assert result.next_action == NextBestAction.ASK_ONE_QUESTION
