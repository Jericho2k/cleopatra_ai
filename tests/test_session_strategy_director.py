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


# ---------------------------------------------------------------------------
# The opening turn, and the class of bug it belonged to
# ---------------------------------------------------------------------------
#
# A fresh-fan smoke test hit this in production:
#
#   [CONVERSATION DIRECTOR] phase=OPENING action=RESPOND_AND_OPEN
#   AttributeError: type object 'NextBestAction' has no attribute
#                   'RESPOND_AND_OPEN'
#
# derive_session_strategy mapped the director's RESPOND_AND_OPEN to
# NextBestAction.RESPOND_AND_OPEN, which has never existed. Enum attribute
# access is not checked until it runs, and the only turn that reaches this line
# is the first exchange with a brand new fan — so every existing test passed
# while the product's very first message was an exception.
#
# derive_session_strategy is called OUTSIDE the persistence try/except in
# services/adaptive_session_planner.plan_next_action, so the AttributeError
# propagated out of the whole Auto turn: no strategy, no writer call, no reply.

import pytest

from models.conversation_director import ConversationPhase, DirectorAction


def _strategy_for(action: str, phase: str = "OPENING"):
    return derive_session_strategy(
        commercial_decision={"action": "CONTINUE_NORMAL_CHAT"},
        conversation_director={
            "phase": phase,
            "action": action,
            "transition_reason": "first_contact",
        },
    )


def test_opening_turn_produces_a_valid_strategy():
    """The exact production trace: phase=OPENING action=RESPOND_AND_OPEN."""
    result = _strategy_for("RESPOND_AND_OPEN")

    assert result.goal == SessionGoal.RAPPORT
    assert result.phase == "OPENING"
    # Not a seventeenth enum member: the director's vocabulary describes WHY the
    # conversation moves, NextBestAction describes WHAT the writer does next,
    # and an opening beat is ordinary conversation. There is no rapport to
    # DEEPEN on a first exchange.
    assert result.next_action == NextBestAction.CONTINUE_CHAT
    assert "respond specifically to what he said" in result.writer_goal
    assert result.must_ask_question is False


def test_opening_turn_is_serialisable_for_persistence_and_the_prompt():
    """to_context() is what reaches the audit table and the writer prompt."""
    context = _strategy_for("RESPOND_AND_OPEN").to_context()

    assert context["next_action"] == "CONTINUE_CHAT"
    assert context["phase"] == "OPENING"
    assert isinstance(context["writer_avoid"], list)


def test_deepen_rapport_still_has_its_own_action():
    """The sibling branch is unchanged; only the impossible label moved."""
    result = _strategy_for("DEEPEN_RAPPORT", phase="RAPPORT")

    assert result.next_action == NextBestAction.DEEPEN_RAPPORT
    assert result.goal == SessionGoal.RAPPORT


# --- the audit: no other impossible enum reference of this kind -------------


@pytest.mark.parametrize("director_action", [member.value for member in DirectorAction])
def test_every_director_action_produces_a_usable_strategy(director_action):
    """Drive derive_session_strategy with every value the director can emit.

    A mapping table checked by eye is how the original bug survived review. This
    executes the branch instead, so an action added to DirectorAction that names
    a NextBestAction member which does not exist fails here rather than on a
    fan's first message.
    """
    result = _strategy_for(director_action)

    assert isinstance(result.next_action, NextBestAction)
    assert isinstance(result.goal, SessionGoal)
    assert result.writer_goal.strip()
    # Serialisation is part of "usable": it is persisted and rendered.
    assert result.to_context()["next_action"] == result.next_action.value


@pytest.mark.parametrize("phase", [member.value for member in ConversationPhase])
def test_every_director_phase_produces_a_usable_strategy(phase):
    """Same audit across phases, since phase also flows straight through."""
    result = _strategy_for("RESPOND_AND_OPEN", phase=phase)

    assert isinstance(result.next_action, NextBestAction)
    assert result.to_context()["phase"] == result.phase


def test_no_enum_attribute_in_the_planner_is_impossible():
    """Static backstop for the whole module, not just the director branch.

    Enum attribute access raises only when reached, and several branches here
    are reachable on rare turns. Parsing the source catches a typo in any of
    them at the same moment the typo is written.
    """
    import ast
    from pathlib import Path

    from models import session_strategy as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    enums = {"NextBestAction": NextBestAction, "SessionGoal": SessionGoal}

    impossible = [
        f"{node.value.id}.{node.attr} (line {node.lineno})"
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id in enums
        and not hasattr(enums[node.value.id], node.attr)
    ]

    assert impossible == [], f"references to enum members that do not exist: {impossible}"
