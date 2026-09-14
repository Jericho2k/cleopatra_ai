"""Deterministic commercial policy engine.

``decide_next_action`` is pure: policy + persisted state + typed observations +
read-only runtime facts -> one commercial decision. The LLM only phrases that
decision.

THE SHAPE OF A SALE
-------------------
One offer at a time. The fan is shown the next thing and its price; he is never
shown a menu, never told how many further steps might exist, and never quoted an
eventual session total. When he accepts, the next move is DELIVERY — not a
confirmation, not a restatement, and not a question about which part he wants
first. After the purchase the conversation goes back to being a conversation for
a beat before anything else is offered.

Everything about how far the progression could go is internal (see
``services/media_packages.plan_progression``). This module only ever emits the
single next unlock.
"""
from datetime import datetime, timedelta, timezone

from models.commercial import (
    ActionType,
    CommercialDecision,
    CommercialEvent,
    CreatorPolicy,
    EventType,
    FanCommercialState,
    FanStatus,
    Offer,
    SextingMode,
)


# What "he has told us what he wants" is worth. A fan who arrives explicitly
# asking to see more is not in the same conversation as one who said "hey", and
# making him serve a fixed number of rapport turns before anything can be
# offered is a state machine talking to itself.
OFFER_READINESS_THRESHOLD = 5

# Explicit interest in content is, on its own, enough to reach that threshold.
# This is the high-intent fast path: it is not a shortcut around inventory,
# pricing, caps or purchase gating, every one of which still applies below.
HIGH_INTENT_EVENTS = (
    EventType.WANTS_EXPLICIT,
    EventType.READY_TO_BUY,
    EventType.BUDGET_STATED,
    EventType.COUNTEROFFER_STATED,
)


class CommercialContext:
    def __init__(
        self,
        fan_has_bought_before: bool = False,
        approved_sets_available: bool = True,
        within_daily_caps: bool = True,
        frozen_for_review: bool = False,
        fan_repeats_interest: bool = False,
        next_offer: Offer | None = None,
        now: datetime | None = None,
        session_exists: bool = False,
        paused_session_available: bool = False,
        session_has_pending_purchase: bool = False,
        session_has_remaining_steps: bool = False,
        session_cooldown_active: bool = False,
    ):
        self.fan_has_bought_before = fan_has_bought_before
        self.approved_sets_available = approved_sets_available
        self.within_daily_caps = within_daily_caps
        self.frozen_for_review = frozen_for_review
        self.fan_repeats_interest = fan_repeats_interest
        self.next_offer = next_offer
        self.now = now or datetime.now(timezone.utc)
        self.session_exists = session_exists
        self.paused_session_available = paused_session_available
        self.session_has_pending_purchase = session_has_pending_purchase
        self.session_has_remaining_steps = session_has_remaining_steps
        self.session_cooldown_active = session_cooldown_active


def _has(events: list[CommercialEvent], event_type: EventType) -> bool:
    return any(event.type == event_type for event in events)


def _get(events: list[CommercialEvent], event_type: EventType) -> CommercialEvent | None:
    return next((event for event in events if event.type == event_type), None)


def free_mode_on_cooldown(
    policy: CreatorPolicy,
    state: FanCommercialState,
    now: datetime,
) -> bool:
    if not state.free_session_ended_at:
        return False
    end = state.free_session_ended_at
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    return now < end + timedelta(hours=max(0, policy.free_session_cooldown_hours))


def compute_readiness(
    events: list[CommercialEvent],
    state: FanCommercialState,
    ctx: CommercialContext,
) -> int:
    score = 0
    if _has(events, EventType.WANTS_EXPLICIT):
        score += 5
    if _has(events, EventType.WANTS_MEDIA):
        score += 3
    if ctx.fan_has_bought_before:
        score += 1
    if ctx.fan_repeats_interest:
        score += 1
    if _has(events, EventType.BUDGET_STATED):
        score += 3
    if _has(events, EventType.COUNTEROFFER_STATED):
        score += 3
    if _has(events, EventType.READY_TO_BUY):
        score += 5
    if _has(events, EventType.OFFER_ACCEPTED):
        score += 5
    if _has(events, EventType.MONEY_UNAVAILABLE):
        score -= 5
    return score


def high_intent(events: list[CommercialEvent]) -> bool:
    """Whether he has already said, plainly, that he wants paid content.

    Nothing about conversation length is consulted, on purpose. "your bikini
    post made me hard, I want to see what's underneath" is a different opening
    from "hey", and treating them the same is what produced twenty ceremonial
    messages before an offer a fan had already asked for.
    """
    return any(_has(events, event_type) for event_type in HIGH_INTENT_EVENTS)


def _offer_next_unlock(
    offer: Offer | None,
    *,
    goal: str,
    reason: str,
    may_be_explicit: bool = True,
    replacement: bool = False,
) -> CommercialDecision:
    """The one forward commercial move: put the next unlock on the table."""
    return CommercialDecision(
        action=ActionType.OFFER_NEXT_UNLOCK,
        goal=goal,
        next_offer=offer,
        mention_price=(offer.price_cents // 100 if offer else None),
        must_not_send_media=True,
        may_be_explicit=may_be_explicit,
        new_status=FanStatus.OFFER_PENDING,
        max_messages=2,
        conversation_continuation="optional",
        replacement_for_unavailable=replacement,
        reason=reason,
    )


def decide_next_action(
    policy: CreatorPolicy,
    state: FanCommercialState,
    events: list[CommercialEvent],
    ctx: CommercialContext,
) -> CommercialDecision:
    if _has(events, EventType.CRISIS) or ctx.frozen_for_review:
        return CommercialDecision(
            action=ActionType.HAND_OFF_TO_HUMAN,
            goal="respond with genuine care; no selling of any kind",
            must_not_send_media=True,
            new_status=FanStatus.HUMAN_REVIEW,
            reason="crisis or frozen",
        )

    if _has(events, EventType.MONEY_AVAILABLE) and state.status in {
        FanStatus.PAUSED_UNTIL_PAYDAY,
        FanStatus.PAUSED_NO_BUDGET,
    }:
        return CommercialDecision(
            action=ActionType.RESUME_PREVIOUS_OFFER,
            goal=(
                "money is available again; warmly resume the exact thing he "
                "wanted before without pressure"
            ),
            next_offer=state.pending_offer,
            must_not_send_media=True,
            may_be_explicit=policy.sexting_mode != SextingMode.PAID_ONLY,
            mention_previous_interest=True,
            new_status=(FanStatus.OFFER_SELECTED if ctx.paused_session_available else FanStatus.IDLE),
            max_messages=1,
            conversation_continuation="optional",
            reason="money available lifts pause",
        )

    # Acceptance outranks a simultaneous statement that he cannot spend more.
    # Example: "yeah send it; more money comes Friday."
    accepted = _get(events, EventType.OFFER_ACCEPTED)
    if accepted:
        accepted_set_id = accepted.metadata.get("set_id")
        if not accepted_set_id:
            # He said yes to something we cannot pin to an approved set. State
            # the one offer again rather than guessing which thing he meant.
            return _offer_next_unlock(
                ctx.next_offer or state.pending_offer,
                goal=(
                    "he sounds ready; say plainly what the next thing is and "
                    "what it costs, once"
                ),
                reason="acceptance could not be matched to an approved set",
            )
        if not ctx.approved_sets_available or not ctx.within_daily_caps:
            return CommercialDecision(
                action=ActionType.CONTINUE_NORMAL_CHAT,
                goal="acknowledge him without promising unavailable content",
                must_not_send_media=True,
                must_not_ask_question=True,
                max_messages=1,
                conversation_continuation="none",
                reason="accepted offer unavailable or capped",
            )
        # One meaningful acceptance is enough. Send it.
        return CommercialDecision(
            action=ActionType.SEND_NEXT_PPV_STEP,
            goal=(
                "he said yes, so this is the send: write the message that goes "
                "with it and nothing else. Do not ask again whether he wants it, "
                "do not restate the price, do not describe what comes after"
            ),
            must_not_send_media=False,
            may_be_explicit=True,
            mention_price=(accepted.amount_cents // 100 if accepted.amount_cents else None),
            new_status=FanStatus.OFFER_SELECTED,
            session_budget_cents=accepted.amount_cents,
            accepted_offer_set_id=str(accepted_set_id),
            must_not_ask_question=True,
            max_messages=2,
            conversation_continuation="none",
            reason="fan accepted the offer on the table",
        )

    if _has(events, EventType.MONEY_UNAVAILABLE):
        payday = _get(events, EventType.PAYDAY_MENTIONED)
        if payday and policy.payday_reengagement_enabled:
            return CommercialDecision(
                action=ActionType.PAUSE_UNTIL_PAYDAY,
                goal="he cannot buy now; be warm, apply zero pressure and close cleanly",
                must_not_send_media=True,
                new_status=FanStatus.PAUSED_UNTIL_PAYDAY,
                schedule_payday_followup=True,
                must_not_ask_question=True,
                max_messages=1,
                conversation_continuation="none",
                reason=f"cannot buy now + payday '{payday.raw_expression}'",
            )
        return CommercialDecision(
            action=ActionType.PAUSE_NO_BUDGET,
            goal="he cannot buy now; stay warm and stop selling",
            must_not_send_media=True,
            new_status=FanStatus.PAUSED_NO_BUDGET,
            must_not_ask_question=True,
            max_messages=1,
            conversation_continuation="none",
            reason="cannot buy now, no payday",
        )

    counteroffer = _get(events, EventType.COUNTEROFFER_STATED)
    if counteroffer:
        # His number does not buy an unapproved discount. It DOES narrow what
        # can be offered next, which the caller has already applied as a hard
        # ceiling when building ctx.next_offer.
        return _offer_next_unlock(
            ctx.next_offer or state.pending_offer,
            goal=(
                "take his number seriously without inventing a discount; say "
                "what you can actually send him at its real price"
            ),
            reason="counteroffer does not match the approved price",
        )

    if _has(events, EventType.OFFER_DECLINED):
        return CommercialDecision(
            action=ActionType.CONTINUE_NORMAL_CHAT,
            goal="accept the no gracefully; no counter-pitch and no guilt",
            must_not_send_media=True,
            new_status=FanStatus.IDLE,
            must_not_ask_question=True,
            max_messages=1,
            conversation_continuation="none",
            reason="offer declined without affordability pause",
        )

    if state.status == FanStatus.OFFER_PENDING and state.pending_offer:
        if _has(events, EventType.OFFER_DETAILS_REQUESTED):
            return CommercialDecision(
                action=ActionType.RESUME_PREVIOUS_OFFER,
                goal=(
                    "answer what he asked about the exact thing on the table; "
                    "same content, same price, no new offer"
                ),
                next_offer=state.pending_offer,
                mention_price=state.pending_offer.price_cents // 100,
                must_not_send_media=True,
                may_be_explicit=True,
                new_status=FanStatus.OFFER_PENDING,
                max_messages=2,
                conversation_continuation="optional",
                reason="pending offer detail request resumes exact snapshot",
            )

    if state.status == FanStatus.OFFER_SELECTED:
        if ctx.session_has_pending_purchase:
            return CommercialDecision(
                action=ActionType.CONTINUE_NORMAL_CHAT,
                goal=(
                    "the PPV is locked and awaiting confirmation; do not imply it "
                    "was purchased, opened, seen, or enjoyed"
                ),
                must_not_send_media=True,
                may_be_explicit=False,
                new_status=FanStatus.PAYMENT_PENDING,
                must_not_ask_question=True,
                max_messages=1,
                conversation_continuation="none",
                reason="selected offer self-healed to payment pending",
            )
        if ctx.session_exists and ctx.session_has_remaining_steps:
            return CommercialDecision(
                action=ActionType.SEND_NEXT_PPV_STEP,
                goal=(
                    "send the exact accepted unlock as a locked purchase-gated PPV; "
                    "do not call it purchased or change its content or price"
                ),
                must_not_send_media=False,
                may_be_explicit=True,
                new_status=FanStatus.OFFER_SELECTED,
                must_not_ask_question=True,
                max_messages=1,
                conversation_continuation="none",
                reason="recover unsent accepted PPV",
            )

    if state.status == FanStatus.PAYMENT_PENDING:
        return CommercialDecision(
            action=ActionType.CONTINUE_NORMAL_CHAT,
            goal=(
                "acknowledge his latest message while the PPV is still locked; "
                "do not imply he purchased it, opened it, saw it, or reacted to unseen content"
            ),
            must_not_send_media=True,
            may_be_explicit=False,
            new_status=FanStatus.PAYMENT_PENDING,
            must_not_ask_question=True,
            max_messages=1,
            conversation_continuation="none",
            reason="PPV is awaiting confirmed unlock",
        )

    if state.status in {FanStatus.PAUSED_NO_BUDGET, FanStatus.PAUSED_UNTIL_PAYDAY}:
        if policy.sexting_mode == SextingMode.FREE_TEXT_ALLOWED and not free_mode_on_cooldown(policy, state, ctx.now):
            return CommercialDecision(
                action=ActionType.CONTINUE_FREE_TEXT,
                goal="keep him engaged with text only; no media and no price",
                must_not_send_media=True,
                may_be_explicit=True,
                new_status=FanStatus.FREE_TEXT_SESSION,
                reason="paused but free text is allowed",
            )
        return CommercialDecision(
            action=ActionType.CONTINUE_NORMAL_CHAT,
            goal="keep it pleasant without selling or giving a paid service away",
            must_not_send_media=True,
            reason="paused, selling suppressed",
        )

    if state.status == FanStatus.PAID_SESSION_ACTIVE:
        if ctx.session_has_pending_purchase:
            return CommercialDecision(
                action=ActionType.CONTINUE_NORMAL_CHAT,
                goal=(
                    "the PPV is still locked; do not imply it was purchased, "
                    "opened, seen, or enjoyed"
                ),
                must_not_send_media=True,
                may_be_explicit=False,
                new_status=FanStatus.PAYMENT_PENDING,
                must_not_ask_question=True,
                max_messages=1,
                conversation_continuation="none",
                reason="legacy paid-session state self-healed to payment pending",
            )
        if ctx.session_cooldown_active:
            return CommercialDecision(
                action=ActionType.CONTINUE_NORMAL_CHAT,
                goal=(
                    "react to the exact piece he just unlocked and stay in it with "
                    "him — no media, no price, and nothing about what comes next"
                ),
                must_not_send_media=True,
                may_be_explicit=True,
                must_not_ask_question=True,
                reason="post-purchase cooldown",
            )
        if ctx.session_has_remaining_steps:
            return CommercialDecision(
                action=ActionType.SEND_NEXT_PPV_STEP,
                goal="deliver the accepted unlock he has not received yet",
                must_not_send_media=False,
                may_be_explicit=True,
                reason="accepted unlock still undelivered",
            )
        return CommercialDecision(
            action=ActionType.CONTINUE_NORMAL_CHAT,
            goal="stay in the moment with him; the last piece is delivered",
            must_not_send_media=True,
            may_be_explicit=True,
            new_status=FanStatus.IDLE,
            max_messages=1,
            reason="delivered unlock complete",
        )

    readiness = compute_readiness(events, state, ctx)
    wants = _has(events, EventType.WANTS_EXPLICIT) or _has(events, EventType.WANTS_MEDIA)
    sellable = ctx.approved_sets_available and ctx.within_daily_caps and bool(ctx.next_offer)

    # The high-intent fast path. He has said what he wants; the only remaining
    # questions are inventory, caps and price, all of which are answered above.
    if wants and (readiness >= OFFER_READINESS_THRESHOLD or high_intent(events)) and sellable:
        return _offer_next_unlock(
            ctx.next_offer,
            goal=(
                "stay in the moment and offer him the one next thing plainly, "
                "with its price, once"
            ),
            reason=f"readiness={readiness}, intent is explicit",
        )

    cooldown = free_mode_on_cooldown(policy, state, ctx.now)
    if wants:
        if policy.sexting_mode == SextingMode.FREE_TEXT_ALLOWED:
            if not cooldown and state.teaser_messages_used < policy.free_text_max_messages:
                return CommercialDecision(
                    action=ActionType.CONTINUE_FREE_TEXT,
                    goal="give a genuine text-only experience; media remains paid",
                    must_not_send_media=policy.media_always_paid,
                    may_be_explicit=True,
                    new_status=FanStatus.FREE_TEXT_SESSION,
                    reason="free text mode",
                )
            if sellable:
                return _offer_next_unlock(
                    ctx.next_offer,
                    goal="the free allowance is used up; offer the next thing plainly",
                    reason="free text allowance exhausted",
                )
            return CommercialDecision(
                action=ActionType.CONTINUE_NORMAL_CHAT,
                goal="the free-session allowance is exhausted or cooling down; stay friendly without continuing the service",
                must_not_send_media=True,
                may_be_explicit=False,
                new_status=FanStatus.IDLE,
                reason="free text allowance unavailable",
            )

        if policy.sexting_mode == SextingMode.HYBRID_TEASER:
            if not cooldown and state.teaser_messages_used < policy.teaser_max_messages:
                return CommercialDecision(
                    action=ActionType.START_FREE_TEASER,
                    goal="give a limited text-only preview without media",
                    must_not_send_media=True,
                    may_be_explicit=True,
                    new_status=FanStatus.FREE_TEASER,
                    reason=f"teaser {state.teaser_messages_used}/{policy.teaser_max_messages}",
                )
            if sellable:
                return _offer_next_unlock(
                    ctx.next_offer,
                    goal="the preview is over; offer him the next thing plainly",
                    reason="teaser exhausted or cooling down",
                )
            return CommercialDecision(
                action=ActionType.CONTINUE_NORMAL_CHAT,
                goal="keep it good without promising content that does not exist",
                must_not_send_media=True,
                may_be_explicit=True,
                reason="teaser exhausted with nothing sellable",
            )

        if readiness >= 3:
            return CommercialDecision(
                action=ActionType.DISCOVER_DESIRED_EXPERIENCE,
                goal=(
                    "find out what he actually wants. A question is one way to do "
                    "that and is not required: an observation, a tease or an "
                    "opinion that invites him to say more does the same job"
                ),
                must_not_send_media=True,
                reason=f"paid only, readiness={readiness}",
            )

    return CommercialDecision(
        action=ActionType.CONTINUE_NORMAL_CHAT,
        goal="keep the conversation going naturally",
        must_not_send_media=True,
        reason=f"default, readiness={readiness}",
    )
