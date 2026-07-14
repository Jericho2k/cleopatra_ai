"""Deterministic next-best-action strategy for creator conversations.

The commercial policy remains authoritative. This module only translates the
current state into an execution strategy for the writer and operator UI.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SessionGoal(str, Enum):
    CARE = "CARE"
    RAPPORT = "RAPPORT"
    QUALIFY = "QUALIFY"
    WARM = "WARM"
    TEASE = "TEASE"
    PRESENT_OFFER = "PRESENT_OFFER"
    HANDLE_OBJECTION = "HANDLE_OBJECTION"
    CLOSE = "CLOSE"
    DELIVER = "DELIVER"
    FOLLOW_UP = "FOLLOW_UP"
    HOLD = "HOLD"


class NextBestAction(str, Enum):
    HAND_OFF = "HAND_OFF"
    CONTINUE_CHAT = "CONTINUE_CHAT"
    ASK_ONE_QUESTION = "ASK_ONE_QUESTION"
    BUILD_TENSION = "BUILD_TENSION"
    PRESENT_APPROVED_OPTIONS = "PRESENT_APPROVED_OPTIONS"
    ACCEPT_NO_AND_RESET = "ACCEPT_NO_AND_RESET"
    HOLD_PRICE = "HOLD_PRICE"
    PAUSE_SELLING = "PAUSE_SELLING"
    RESUME_PREVIOUS_OFFER = "RESUME_PREVIOUS_OFFER"
    CREATE_PAID_SESSION = "CREATE_PAID_SESSION"
    SEND_NEXT_STEP = "SEND_NEXT_STEP"
    POST_SESSION_FOLLOWUP = "POST_SESSION_FOLLOWUP"


class SessionStrategy(BaseModel):
    goal: SessionGoal = SessionGoal.RAPPORT
    phase: str = "RAPPORT"
    next_action: NextBestAction = NextBestAction.CONTINUE_CHAT
    writer_goal: str = "continue naturally and preserve rapport"
    writer_avoid: list[str] = Field(default_factory=list)
    must_ask_question: bool = False
    must_not_ask_question: bool = False
    max_messages: int | None = None
    approved_offer_ids: list[str] = Field(default_factory=list)
    approved_offer_prices_cents: list[int] = Field(default_factory=list)
    selected_offer_price_cents: int | None = None
    route_hint: str = "default"
    reason_codes: list[str] = Field(default_factory=list)
    planner_version: int = 1
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_context(self) -> dict[str, Any]:
        return {
            "goal": self.goal.value,
            "phase": self.phase,
            "next_action": self.next_action.value,
            "writer_goal": self.writer_goal,
            "writer_avoid": self.writer_avoid,
            "must_ask_question": self.must_ask_question,
            "must_not_ask_question": self.must_not_ask_question,
            "max_messages": self.max_messages,
            "approved_offer_ids": self.approved_offer_ids,
            "approved_offer_prices_cents": self.approved_offer_prices_cents,
            "selected_offer_price_cents": self.selected_offer_price_cents,
            "route_hint": self.route_hint,
            "reason_codes": self.reason_codes,
            "planner_version": self.planner_version,
            "updated_at": self.updated_at.isoformat(),
        }


def derive_session_strategy(
    *,
    situation: dict[str, Any] | None = None,
    commercial_decision: dict[str, Any] | None = None,
    lifecycle: dict[str, Any] | None = None,
    affordability: dict[str, Any] | None = None,
    price_learning: dict[str, Any] | None = None,
    active_session: dict[str, Any] | None = None,
    conversation_stage: str | None = None,
) -> SessionStrategy:
    """Build one deterministic execution strategy.

    Commercial decisions win over every heuristic below. The planner never
    invents an offer or authorizes a price.
    """

    situation = situation or {}
    decision = commercial_decision or {}
    lifecycle = lifecycle or {}
    affordability = affordability or {}
    price_learning = price_learning or {}
    active_session = active_session or {}

    action = _enum_value(decision.get("action"))
    stage = str(conversation_stage or "").upper()
    purchase_signal = str(situation.get("purchase_signal") or "none").lower()
    crisis = str(situation.get("crisis_signal") or "none").lower()
    lifecycle_stage = str(lifecycle.get("stage") or "PROSPECT").upper()
    price_mode = str(price_learning.get("mode") or "DISCOVERY").upper()

    offer_ids, offer_prices = _approved_offers(decision)
    selected_price = _int_or_none(decision.get("session_budget_cents"))
    if selected_price is None:
        mention_price = _int_or_none(decision.get("mention_price"))
        selected_price = mention_price * 100 if mention_price is not None else None

    shared = {
        "approved_offer_ids": offer_ids,
        "approved_offer_prices_cents": offer_prices,
        "selected_offer_price_cents": selected_price,
    }

    if crisis != "none" or action == "HAND_OFF_TO_HUMAN":
        return SessionStrategy(
            goal=SessionGoal.CARE,
            phase="SAFETY",
            next_action=NextBestAction.HAND_OFF,
            writer_goal="respond with genuine care and stop all commercial activity",
            writer_avoid=["selling", "flirting", "pricing", "media", "pressure"],
            route_hint="safety_sensitive",
            reason_codes=["crisis_or_handoff", "commercial_policy_authoritative"],
            **shared,
        )

    if action in {"PAUSE_NO_BUDGET", "PAUSE_UNTIL_PAYDAY"} or affordability.get("temporary_constraint"):
        return SessionStrategy(
            goal=SessionGoal.HOLD,
            phase="PAUSED",
            next_action=NextBestAction.PAUSE_SELLING,
            writer_goal="close the commercial exchange warmly and preserve future rapport",
            writer_avoid=["counteroffer", "cheaper pitch", "guilt", "urgency", "new PPV"],
            must_not_ask_question=True,
            max_messages=1,
            route_hint="commercial_complex",
            reason_codes=["temporary_or_explicit_affordability_pause"],
            **shared,
        )

    if action == "SEND_NEXT_PPV_STEP" or (
        active_session.get("status") == "active" and active_session.get("awaiting_purchase_index") is None
    ):
        return SessionStrategy(
            goal=SessionGoal.DELIVER,
            phase="PAID_SESSION",
            next_action=NextBestAction.SEND_NEXT_STEP,
            writer_goal="continue the approved paid progression without changing price or content",
            writer_avoid=["new package", "discount", "unapproved media", "price change"],
            must_not_ask_question=True,
            route_hint="commercial_complex",
            reason_codes=["active_paid_session"],
            **shared,
        )

    if action == "CREATE_PAID_SESSION":
        return SessionStrategy(
            goal=SessionGoal.CLOSE,
            phase="PURCHASE_CONFIRMED",
            next_action=NextBestAction.CREATE_PAID_SESSION,
            writer_goal="confirm the exact approved selection and move into delivery",
            writer_avoid=["renegotiation", "different package", "different price", "extra qualification"],
            must_not_ask_question=True,
            max_messages=2,
            route_hint="commercial_complex",
            reason_codes=["approved_offer_selected"],
            **shared,
        )

    if action in {"PRESENT_SESSION_OPTIONS", "END_TEASER_AND_OFFER"}:
        return SessionStrategy(
            goal=SessionGoal.PRESENT_OFFER,
            phase="OFFER",
            next_action=NextBestAction.PRESENT_APPROVED_OPTIONS,
            writer_goal="present only the approved options clearly and let the fan choose",
            writer_avoid=["invented bundle", "invented price", "hidden discount", "third option"],
            must_ask_question=True,
            max_messages=2,
            route_hint="commercial_complex",
            reason_codes=["commercial_policy_requests_approved_options"],
            **shared,
        )

    if action == "ASK_ONE_QUALIFYING_QUESTION":
        return SessionStrategy(
            goal=SessionGoal.QUALIFY,
            phase="QUALIFICATION",
            next_action=NextBestAction.ASK_ONE_QUESTION,
            writer_goal="ask one natural question that reveals the desired experience",
            writer_avoid=["multiple questions", "price pitch", "premature PPV", "interview tone"],
            must_ask_question=True,
            max_messages=2,
            reason_codes=["commercial_policy_requests_qualification"],
            **shared,
        )

    if action in {"START_FREE_TEASER", "CONTINUE_FREE_TEXT"}:
        return SessionStrategy(
            goal=SessionGoal.TEASE,
            phase="TEASER",
            next_action=NextBestAction.BUILD_TENSION,
            writer_goal="build tension within the configured free-text boundary",
            writer_avoid=["free media", "unapproved promise", "immediate discount", "overlong scene"],
            route_hint="default",
            reason_codes=["limited_teaser_or_free_text"],
            **shared,
        )

    if action == "RESUME_PREVIOUS_OFFER":
        return SessionStrategy(
            goal=SessionGoal.CLOSE,
            phase="REENGAGEMENT",
            next_action=NextBestAction.RESUME_PREVIOUS_OFFER,
            writer_goal="warmly resume the exact prior approved interest without pressure",
            writer_avoid=["new package", "new price", "guilt", "claiming a promise"],
            max_messages=1,
            route_hint="commercial_complex",
            reason_codes=["money_available_or_payday_return"],
            **shared,
        )

    if purchase_signal == "declined" or stage == "OBJECTION":
        return SessionStrategy(
            goal=SessionGoal.HANDLE_OBJECTION,
            phase="OBJECTION",
            next_action=NextBestAction.ACCEPT_NO_AND_RESET,
            writer_goal="accept the resistance naturally and preserve the relationship",
            writer_avoid=["instant cheaper offer", "arguing", "guilt", "repeating the same pitch"],
            must_not_ask_question=True,
            max_messages=1,
            route_hint="commercial_complex",
            reason_codes=["decline_or_objection"],
            **shared,
        )

    if purchase_signal in {"ready_to_buy", "selected"} or price_mode == "EXACT":
        return SessionStrategy(
            goal=SessionGoal.CLOSE,
            phase="CLOSE",
            next_action=NextBestAction.HOLD_PRICE,
            writer_goal="keep the exact approved offer clear and remove unnecessary friction",
            writer_avoid=["new price", "new package", "more qualification", "pressure"],
            route_hint="commercial_complex",
            reason_codes=["high_purchase_intent_or_exact_price"],
            **shared,
        )

    if lifecycle_stage in {"PROSPECT", "FIRST_PURCHASE_PROSPECT"} and not _has_preferences(situation):
        return SessionStrategy(
            goal=SessionGoal.QUALIFY,
            phase="DISCOVERY",
            next_action=NextBestAction.ASK_ONE_QUESTION,
            writer_goal="learn one useful preference while keeping the exchange playful",
            writer_avoid=["interview", "price pitch", "generic menu", "multiple questions"],
            must_ask_question=True,
            reason_codes=["first_purchase_discovery"],
            **shared,
        )

    if stage in {"FLIRTING", "PRE_UPSELL", "UPSELL_ACTIVE", "HIGH_VALUE"}:
        return SessionStrategy(
            goal=SessionGoal.WARM,
            phase="WARMING",
            next_action=NextBestAction.BUILD_TENSION,
            writer_goal="match the current energy and build anticipation without forcing an offer",
            writer_avoid=["abrupt sale", "generic menu", "invented promise"],
            reason_codes=["conversation_stage_warm_or_flirty"],
            **shared,
        )

    if stage == "RETENTION" or lifecycle_stage in {"FIRST_TIME_BUYER", "REPEAT_BUYER", "VIP"}:
        return SessionStrategy(
            goal=SessionGoal.FOLLOW_UP,
            phase="RELATIONSHIP",
            next_action=NextBestAction.POST_SESSION_FOLLOWUP,
            writer_goal="reinforce continuity and respond to the fan before considering another offer",
            writer_avoid=["immediate repitch", "generic thank-you", "recycled content"],
            reason_codes=["buyer_continuity_or_retention"],
            **shared,
        )

    return SessionStrategy(
        goal=SessionGoal.RAPPORT,
        phase="RAPPORT",
        next_action=NextBestAction.CONTINUE_CHAT,
        writer_goal="continue naturally and gather context without forcing a commercial move",
        writer_avoid=["premature offer", "generic script", "invented backstory"],
        reason_codes=["default_rapport"],
        **shared,
    )


def _approved_offers(decision: dict[str, Any]) -> tuple[list[str], list[int]]:
    ids: list[str] = []
    prices: list[int] = []
    for raw in decision.get("package_options") or []:
        if hasattr(raw, "model_dump"):
            raw = raw.model_dump(mode="json")
        if not isinstance(raw, dict):
            continue
        package_id = str(raw.get("package_id") or "")
        if package_id:
            ids.append(package_id)
        price = _int_or_none(raw.get("price_cents"))
        if price is not None:
            prices.append(price)
    return ids, prices


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value) or "").upper()


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value) if value is not None and value != "" else None
    except (TypeError, ValueError):
        return None


def _has_preferences(situation: dict[str, Any]) -> bool:
    desired = str(situation.get("desired_experience") or "").strip()
    intelligence = situation.get("learned_fan_intelligence") or {}
    facts = intelligence.get("facts") or []
    return bool(desired or facts)
