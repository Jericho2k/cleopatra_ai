from models.session_strategy import NextBestAction, SessionGoal, derive_session_strategy


def test_crisis_always_hands_off_and_suppresses_selling():
    result = derive_session_strategy(
        situation={"crisis_signal": "self_harm"},
        commercial_decision={"action": "PRESENT_SESSION_OPTIONS"},
    )
    assert result.goal == SessionGoal.CARE
    assert result.next_action == NextBestAction.HAND_OFF
    assert "selling" in result.writer_avoid


def test_selected_approved_offer_closes_exactly():
    result = derive_session_strategy(
        commercial_decision={
            "action": "CREATE_PAID_SESSION",
            "session_budget_cents": 2800,
            "package_options": [{"package_id": "package:1", "price_cents": 2800}],
        }
    )
    assert result.goal == SessionGoal.CLOSE
    assert result.next_action == NextBestAction.CREATE_PAID_SESSION
    assert result.selected_offer_price_cents == 2800
    assert result.approved_offer_ids == ["package:1"]


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
    assert result.must_ask_question is True
    assert result.next_action == NextBestAction.ASK_ONE_QUESTION
