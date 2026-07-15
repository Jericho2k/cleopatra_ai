"""Persistent conversation progression for non-random multi-turn behavior."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ConversationPhase(str, Enum):
    OPENING = "OPENING"
    RAPPORT = "RAPPORT"
    FLIRT = "FLIRT"
    QUALIFY = "QUALIFY"
    TENSION = "TENSION"
    SOFT_OFFER = "SOFT_OFFER"
    OFFER = "OFFER"
    OBJECTION = "OBJECTION"
    PAID_SESSION = "PAID_SESSION"
    FOLLOW_UP = "FOLLOW_UP"
    PAUSED = "PAUSED"
    SAFETY = "SAFETY"


class DirectorAction(str, Enum):
    RESPOND_AND_OPEN = "RESPOND_AND_OPEN"
    DEEPEN_RAPPORT = "DEEPEN_RAPPORT"
    PLAYFUL_FLIRT = "PLAYFUL_FLIRT"
    DISCOVER_PREFERENCE = "DISCOVER_PREFERENCE"
    BUILD_TENSION = "BUILD_TENSION"
    SEED_PREMIUM_CONTENT = "SEED_PREMIUM_CONTENT"
    PIVOT_ENERGY = "PIVOT_ENERGY"
    PRESENT_APPROVED_OPTIONS = "PRESENT_APPROVED_OPTIONS"
    HANDLE_OBJECTION = "HANDLE_OBJECTION"
    CONTINUE_PAID_SESSION = "CONTINUE_PAID_SESSION"
    POST_SESSION_FOLLOWUP = "POST_SESSION_FOLLOWUP"
    PAUSE_SELLING = "PAUSE_SELLING"
    HAND_OFF = "HAND_OFF"


class ConversationDirectorState(BaseModel):
    phase: ConversationPhase = ConversationPhase.OPENING
    previous_phase: ConversationPhase | None = None
    action: DirectorAction = DirectorAction.RESPOND_AND_OPEN
    fan_turn_count: int = 0
    creator_turn_count: int = 0
    turns_in_phase: int = 1
    same_action_streak: int = 1
    recent_actions: list[str] = Field(default_factory=list)
    engagement_score: int = 0
    qualification_complete: bool = False
    offer_eligible: bool = False
    question_due: bool = False
    must_not_ask_question: bool = False
    transition_reason: str = "new_conversation"
    director_version: int = 1
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_context(self) -> dict[str, Any]:
        return {
            "phase": self.phase.value,
            "previous_phase": self.previous_phase.value if self.previous_phase else None,
            "action": self.action.value,
            "fan_turn_count": self.fan_turn_count,
            "creator_turn_count": self.creator_turn_count,
            "turns_in_phase": self.turns_in_phase,
            "same_action_streak": self.same_action_streak,
            "recent_actions": self.recent_actions,
            "engagement_score": self.engagement_score,
            "qualification_complete": self.qualification_complete,
            "offer_eligible": self.offer_eligible,
            "question_due": self.question_due,
            "must_not_ask_question": self.must_not_ask_question,
            "transition_reason": self.transition_reason,
            "director_version": self.director_version,
            "updated_at": self.updated_at.isoformat(),
        }


def advance_conversation_director(
    *,
    previous: dict[str, Any] | ConversationDirectorState | None = None,
    situation: dict[str, Any] | None = None,
    commercial_decision: dict[str, Any] | None = None,
    lifecycle: dict[str, Any] | None = None,
    active_session: dict[str, Any] | None = None,
    conversation_stage: str | None = None,
    fan_turn_count: int = 0,
    creator_turn_count: int = 0,
) -> ConversationDirectorState:
    """Advance one deterministic relationship/conversation phase."""

    prev = _mapping(previous)
    situation = situation or {}
    decision = commercial_decision or {}
    lifecycle = lifecycle or {}
    active_session = active_session or {}

    prev_phase = _phase(prev.get("phase"))
    prev_action = _action(prev.get("action"))
    prev_turns = max(0, _int(prev.get("turns_in_phase")))
    prev_streak = max(0, _int(prev.get("same_action_streak")))
    recent_actions = [str(v) for v in (prev.get("recent_actions") or []) if str(v)][-4:]

    commercial_action = _enum_value(decision.get("action"))
    purchase_signal = str(situation.get("purchase_signal") or "none").lower()
    crisis = str(situation.get("crisis_signal") or "none").lower()
    strategic_move = str(situation.get("strategic_move") or "").lower()
    interest_signal = str(situation.get("commercial_interest_signal") or "none").lower()
    stage = str(conversation_stage or "").upper()
    lifecycle_stage = str(lifecycle.get("stage") or "PROSPECT").upper()

    qualification_complete = _has_preferences(situation) or bool(
        prev.get("qualification_complete")
    )
    warm_signal = bool(
        interest_signal == "warm_compliment"
        or strategic_move in {
            "acknowledge_compliment_and_redirect",
            "build_tension",
            "tease_and_deflect",
            "flirt",
            "match_energy",
        }
        or stage in {"FLIRTING", "PRE_UPSELL"}
    )
    direct_interest = bool(
        _truthy(situation.get("wants_explicit"))
        or _truthy(situation.get("wants_media"))
        or purchase_signal in {"ready_to_buy", "selected", "money_available"}
    )
    engagement_score = min(
        100,
        fan_turn_count * 6
        + (18 if warm_signal else 0)
        + (25 if direct_interest else 0)
        + (10 if lifecycle_stage in {"FIRST_TIME_BUYER", "REPEAT_BUYER", "VIP"} else 0),
    )

    authoritative = _authoritative_state(
        crisis=crisis,
        commercial_action=commercial_action,
        purchase_signal=purchase_signal,
        active_session=active_session,
        stage=stage,
        lifecycle_stage=lifecycle_stage,
    )
    if authoritative:
        phase, action, reason, must_not_ask = authoritative
    else:
        phase, action, reason = _choose_noncommercial_move(
            prev_phase=prev_phase,
            prev_turns=prev_turns,
            fan_turn_count=fan_turn_count,
            warm_signal=warm_signal,
            direct_interest=direct_interest,
            qualification_complete=qualification_complete,
            engagement_score=engagement_score,
        )
        must_not_ask = action in {
            DirectorAction.SEED_PREMIUM_CONTENT,
            DirectorAction.PIVOT_ENERGY,
        }

        if action == prev_action and prev_streak >= 2:
            phase, action, reason = _break_repetition(
                action=action,
                qualification_complete=qualification_complete,
            )
            must_not_ask = action in {
                DirectorAction.SEED_PREMIUM_CONTENT,
                DirectorAction.PIVOT_ENERGY,
            }

    return _build_state(
        phase=phase,
        action=action,
        reason=reason,
        previous_phase=prev_phase,
        previous_action=prev_action,
        previous_turns=prev_turns,
        previous_streak=prev_streak,
        recent_actions=recent_actions,
        fan_turn_count=fan_turn_count,
        creator_turn_count=creator_turn_count,
        engagement_score=engagement_score,
        qualification_complete=qualification_complete,
        offer_eligible=phase in {
            ConversationPhase.OFFER,
            ConversationPhase.PAID_SESSION,
        }
        or (
            phase == ConversationPhase.SOFT_OFFER
            and engagement_score >= 55
            and fan_turn_count >= 5
        ),
        question_due=action == DirectorAction.DISCOVER_PREFERENCE,
        must_not_ask_question=must_not_ask,
    )


def _choose_noncommercial_move(
    *,
    prev_phase: ConversationPhase | None,
    prev_turns: int,
    fan_turn_count: int,
    warm_signal: bool,
    direct_interest: bool,
    qualification_complete: bool,
    engagement_score: int,
) -> tuple[ConversationPhase, DirectorAction, str]:
    if fan_turn_count <= 1:
        return ConversationPhase.OPENING, DirectorAction.RESPOND_AND_OPEN, "opening_turn"

    if direct_interest and fan_turn_count >= 3:
        return (
            ConversationPhase.SOFT_OFFER,
            DirectorAction.SEED_PREMIUM_CONTENT,
            "active_interest_without_authorized_offer",
        )

    if prev_phase in {None, ConversationPhase.OPENING}:
        if warm_signal:
            return ConversationPhase.FLIRT, DirectorAction.PLAYFUL_FLIRT, "warm_opening"
        return ConversationPhase.RAPPORT, DirectorAction.DEEPEN_RAPPORT, "opening_to_rapport"

    if prev_phase == ConversationPhase.RAPPORT:
        if warm_signal:
            return (
                ConversationPhase.FLIRT,
                DirectorAction.PLAYFUL_FLIRT,
                "rapport_became_flirty",
            )
        if fan_turn_count >= 3:
            return (
                ConversationPhase.QUALIFY,
                DirectorAction.DISCOVER_PREFERENCE,
                "rapport_ready_for_discovery",
            )
        return ConversationPhase.RAPPORT, DirectorAction.DEEPEN_RAPPORT, "continue_rapport"

    if prev_phase == ConversationPhase.FLIRT:
        if prev_turns >= 1 and fan_turn_count >= 3:
            return (
                ConversationPhase.QUALIFY,
                DirectorAction.DISCOVER_PREFERENCE,
                "flirt_ready_for_one_preference_question",
            )
        return ConversationPhase.FLIRT, DirectorAction.PLAYFUL_FLIRT, "continue_initial_flirt"

    if prev_phase == ConversationPhase.QUALIFY:
        if qualification_complete or prev_turns >= 1:
            return (
                ConversationPhase.TENSION,
                DirectorAction.BUILD_TENSION,
                "qualification_complete_or_attempted",
            )
        return (
            ConversationPhase.QUALIFY,
            DirectorAction.DISCOVER_PREFERENCE,
            "complete_single_discovery_move",
        )

    if prev_phase == ConversationPhase.TENSION:
        if prev_turns >= 2 and fan_turn_count >= 6 and engagement_score >= 55:
            return (
                ConversationPhase.SOFT_OFFER,
                DirectorAction.SEED_PREMIUM_CONTENT,
                "tension_ready_for_soft_commercial_bridge",
            )
        return (
            ConversationPhase.TENSION,
            DirectorAction.BUILD_TENSION,
            "continue_tension_briefly",
        )

    if prev_phase == ConversationPhase.SOFT_OFFER:
        if prev_turns >= 1:
            return (
                ConversationPhase.RAPPORT,
                DirectorAction.PIVOT_ENERGY,
                "soft_offer_not_pursued_reset_pressure",
            )
        return (
            ConversationPhase.SOFT_OFFER,
            DirectorAction.SEED_PREMIUM_CONTENT,
            "continue_single_soft_bridge",
        )

    if prev_phase in {
        ConversationPhase.OBJECTION,
        ConversationPhase.FOLLOW_UP,
        ConversationPhase.PAUSED,
    }:
        return ConversationPhase.RAPPORT, DirectorAction.DEEPEN_RAPPORT, "relationship_reset"

    return ConversationPhase.RAPPORT, DirectorAction.DEEPEN_RAPPORT, "default_rapport"


def _authoritative_state(
    *,
    crisis: str,
    commercial_action: str,
    purchase_signal: str,
    active_session: dict[str, Any],
    stage: str,
    lifecycle_stage: str,
) -> tuple[ConversationPhase, DirectorAction, str, bool] | None:
    if crisis != "none" or commercial_action == "HAND_OFF_TO_HUMAN":
        return ConversationPhase.SAFETY, DirectorAction.HAND_OFF, "crisis_or_handoff", True
    if commercial_action in {"PAUSE_NO_BUDGET", "PAUSE_UNTIL_PAYDAY"}:
        return ConversationPhase.PAUSED, DirectorAction.PAUSE_SELLING, "commercial_pause", True
    if active_session.get("status") == "active" or commercial_action in {
        "CREATE_PAID_SESSION",
        "SEND_NEXT_PPV_STEP",
    }:
        return (
            ConversationPhase.PAID_SESSION,
            DirectorAction.CONTINUE_PAID_SESSION,
            "active_or_confirmed_paid_session",
            False,
        )
    if commercial_action in {
        "PRESENT_SESSION_OPTIONS",
        "END_TEASER_AND_OFFER",
        "RESUME_PREVIOUS_OFFER",
    }:
        return (
            ConversationPhase.OFFER,
            DirectorAction.PRESENT_APPROVED_OPTIONS,
            "commercial_policy_authorized_offer",
            False,
        )
    if purchase_signal == "declined" or stage == "OBJECTION":
        return (
            ConversationPhase.OBJECTION,
            DirectorAction.HANDLE_OBJECTION,
            "decline_or_objection",
            True,
        )
    if stage == "RETENTION":
        return (
            ConversationPhase.FOLLOW_UP,
            DirectorAction.POST_SESSION_FOLLOWUP,
            "buyer_continuity",
            False,
        )
    return None


def _break_repetition(
    *,
    action: DirectorAction,
    qualification_complete: bool,
) -> tuple[ConversationPhase, DirectorAction, str]:
    if action == DirectorAction.BUILD_TENSION:
        if qualification_complete:
            return (
                ConversationPhase.SOFT_OFFER,
                DirectorAction.SEED_PREMIUM_CONTENT,
                "repetition_guard_tension_to_soft_bridge",
            )
        return (
            ConversationPhase.QUALIFY,
            DirectorAction.DISCOVER_PREFERENCE,
            "repetition_guard_tension_to_discovery",
        )
    if action == DirectorAction.DISCOVER_PREFERENCE:
        return (
            ConversationPhase.FLIRT,
            DirectorAction.PLAYFUL_FLIRT,
            "repetition_guard_stop_interview",
        )
    if action == DirectorAction.PLAYFUL_FLIRT:
        return (
            ConversationPhase.RAPPORT,
            DirectorAction.DEEPEN_RAPPORT,
            "repetition_guard_change_texture",
        )
    if action == DirectorAction.SEED_PREMIUM_CONTENT:
        return (
            ConversationPhase.RAPPORT,
            DirectorAction.PIVOT_ENERGY,
            "repetition_guard_remove_sales_pressure",
        )
    return (
        ConversationPhase.RAPPORT,
        DirectorAction.PIVOT_ENERGY,
        "repetition_guard_generic_pivot",
    )


def _build_state(
    *,
    phase: ConversationPhase,
    action: DirectorAction,
    reason: str,
    previous_phase: ConversationPhase | None,
    previous_action: DirectorAction | None,
    previous_turns: int,
    previous_streak: int,
    recent_actions: list[str],
    fan_turn_count: int,
    creator_turn_count: int,
    engagement_score: int,
    qualification_complete: bool,
    offer_eligible: bool,
    question_due: bool,
    must_not_ask_question: bool,
) -> ConversationDirectorState:
    return ConversationDirectorState(
        phase=phase,
        previous_phase=previous_phase,
        action=action,
        fan_turn_count=max(0, fan_turn_count),
        creator_turn_count=max(0, creator_turn_count),
        turns_in_phase=previous_turns + 1 if phase == previous_phase else 1,
        same_action_streak=previous_streak + 1 if action == previous_action else 1,
        recent_actions=[*recent_actions, action.value][-5:],
        engagement_score=max(0, min(100, engagement_score)),
        qualification_complete=qualification_complete,
        offer_eligible=offer_eligible,
        question_due=question_due,
        must_not_ask_question=must_not_ask_question,
        transition_reason=reason,
    )


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        dumped = value.model_dump(mode="json")
        return dumped if isinstance(dumped, dict) else {}
    return {}


def _phase(value: Any) -> ConversationPhase | None:
    try:
        return ConversationPhase(str(getattr(value, "value", value)))
    except (TypeError, ValueError):
        return None


def _action(value: Any) -> DirectorAction | None:
    try:
        return DirectorAction(str(getattr(value, "value", value)))
    except (TypeError, ValueError):
        return None


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value) or "").upper()


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _has_preferences(situation: dict[str, Any]) -> bool:
    if str(situation.get("desired_experience") or "").strip():
        return True
    intelligence = situation.get("learned_fan_intelligence") or {}
    facts = intelligence.get("facts") or []
    keys = {
        "content_preference",
        "preferred_dynamic",
        "preferred_format",
        "explicit_interest",
        "kink",
    }
    return any(
        isinstance(fact, dict)
        and str(fact.get("fact_key") or "").lower() in keys
        and fact.get("status") != "contradicted"
        for fact in facts
    )
