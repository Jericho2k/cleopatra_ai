"""Deterministic commercial policy engine.

``decide_next_action`` is pure: policy + persisted state + typed observations +
read-only runtime facts -> one commercial decision. The LLM only phrases that
decision.
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
    PackageOption,
    SextingMode,
)


class CommercialContext:
    def __init__(
        self,
        fan_has_bought_before: bool = False,
        approved_sets_available: bool = True,
        within_daily_caps: bool = True,
        frozen_for_review: bool = False,
        fan_repeats_interest: bool = False,
        package_options: list[PackageOption] | None = None,
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
        self.package_options = package_options or []
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
        score += 3
    if _has(events, EventType.WANTS_MEDIA):
        score += 2
    if ctx.fan_has_bought_before:
        score += 1
    if ctx.fan_repeats_interest:
        score += 1
    if _has(events, EventType.BUDGET_STATED):
        score += 3
    if _has(events, EventType.COUNTEROFFER_STATED):
        score += 3
    if _has(events, EventType.READY_TO_BUY):
        score += 3
    if _has(events, EventType.PACKAGE_SELECTED):
        score += 5
    if _has(events, EventType.MONEY_UNAVAILABLE):
        score -= 5
    return score


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
                "money is available again; warmly resume the exact experience "
                "he wanted before without pressure"
            ),
            must_not_send_media=True,
            may_be_explicit=policy.sexting_mode != SextingMode.PAID_ONLY,
            mention_previous_interest=True,
            new_status=(FanStatus.PAID_SESSION_ACTIVE if ctx.paused_session_available else FanStatus.IDLE),
            max_messages=1,
            conversation_continuation="optional",
            reason="money available lifts pause",
        )

    # A package acceptance outranks a simultaneous statement that he cannot spend
    # more. Example: "I'll take the $28 one; more money comes Friday."
    selected = _get(events, EventType.PACKAGE_SELECTED)
    if selected:
        selected_set_ids = list(selected.metadata.get("set_ids") or [])
        selected_set_id = selected.metadata.get("set_id")
        if selected_set_id and not selected_set_ids:
            selected_set_ids = [selected_set_id]
        if not selected_set_ids:
            return CommercialDecision(
                action=ActionType.PRESENT_SESSION_OPTIONS,
                goal="restate the exact available options once because the choice was ambiguous",
                package_options=ctx.package_options or state.offered_packages,
                must_not_send_media=True,
                new_status=FanStatus.OFFER_PENDING,
                max_messages=2,
                conversation_continuation="required",
                reason="package choice could not be matched",
            )
        if not ctx.approved_sets_available or not ctx.within_daily_caps:
            return CommercialDecision(
                action=ActionType.CONTINUE_NORMAL_CHAT,
                goal="acknowledge his choice without promising unavailable content",
                must_not_send_media=True,
                must_not_ask_question=True,
                max_messages=1,
                conversation_continuation="none",
                reason="selected package unavailable or capped",
            )
        return CommercialDecision(
            action=ActionType.CREATE_PAID_SESSION,
            goal="confirm the package he selected and begin the matching paid experience",
            must_not_send_media=False,
            may_be_explicit=True,
            mention_price=(selected.amount_cents // 100 if selected.amount_cents else None),
            new_status=FanStatus.PAID_SESSION_ACTIVE,
            session_budget_cents=selected.amount_cents,
            selected_package_set_id=selected_set_ids[0],
            selected_package_set_ids=selected_set_ids,
            must_not_ask_question=True,
            max_messages=2,
            conversation_continuation="none",
            reason="fan selected an offered package",
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
        return CommercialDecision(
            action=ActionType.PRESENT_SESSION_OPTIONS,
            goal=(
                "acknowledge his amount without promising an unapproved discount; "
                "offer only the exact available packages"
            ),
            package_options=ctx.package_options or state.offered_packages,
            must_not_send_media=True,
            may_be_explicit=True,
            new_status=FanStatus.OFFER_PENDING,
            max_messages=2,
            conversation_continuation="required",
            reason="counteroffer does not match an approved package",
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
                goal="stay in the paid-session mood while waiting for the current PPV purchase; do not send another step",
                must_not_send_media=True,
                may_be_explicit=True,
                reason="awaiting purchase of current step",
            )
        if ctx.session_cooldown_active:
            return CommercialDecision(
                action=ActionType.CONTINUE_NORMAL_CHAT,
                goal="continue the fantasy briefly between purchased PPV steps; no new media yet",
                must_not_send_media=True,
                may_be_explicit=True,
                reason="post-purchase cooldown",
            )
        if not ctx.session_has_remaining_steps:
            return CommercialDecision(
                action=ActionType.CONTINUE_NORMAL_CHAT,
                goal="close the completed paid experience warmly and return to natural chat",
                must_not_send_media=True,
                may_be_explicit=False,
                new_status=FanStatus.IDLE,
                max_messages=1,
                reason="session has no remaining steps",
            )
        if not ctx.within_daily_caps:
            return CommercialDecision(
                action=ActionType.CONTINUE_NORMAL_CHAT,
                goal="stay in the moment; daily limits are reached, so send nothing else",
                must_not_send_media=True,
                may_be_explicit=True,
                reason="daily cap",
            )
        return CommercialDecision(
            action=ActionType.SEND_NEXT_PPV_STEP,
            goal="continue the paid experience and deliver the next planned step",
            must_not_send_media=False,
            may_be_explicit=True,
            reason="paid session active and ready for next step",
        )

    readiness = compute_readiness(events, state, ctx)
    wants = _has(events, EventType.WANTS_EXPLICIT) or _has(events, EventType.WANTS_MEDIA)

    if wants and readiness >= 5 and ctx.approved_sets_available and ctx.within_daily_caps:
        if policy.offer_two_packages and ctx.package_options and state.confirmed_budget_cents is None:
            return CommercialDecision(
                action=ActionType.PRESENT_SESSION_OPTIONS,
                goal="offer the exact available packages and let him choose",
                package_options=ctx.package_options,
                must_not_send_media=True,
                may_be_explicit=True,
                new_status=FanStatus.OFFER_PENDING,
                max_messages=2,
                conversation_continuation="required",
                reason=f"readiness={readiness}, no confirmed package",
            )
        if state.confirmed_budget_cents:
            return CommercialDecision(
                action=ActionType.CREATE_PAID_SESSION,
                goal="use the confirmed amount to build the paid experience",
                must_not_send_media=False,
                may_be_explicit=True,
                new_status=FanStatus.PAID_SESSION_ACTIVE,
                session_budget_cents=state.confirmed_budget_cents,
                selected_package_set_ids=state.selected_package_set_ids,
                selected_package_set_id=state.selected_package_set_id,
                reason=f"readiness={readiness}, confirmed budget",
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
            return CommercialDecision(
                action=ActionType.END_TEASER_AND_OFFER,
                goal="the preview is over; transition to the exact paid options",
                package_options=ctx.package_options,
                must_not_send_media=True,
                may_be_explicit=True,
                new_status=FanStatus.OFFER_PENDING,
                reason="teaser exhausted or cooling down",
            )

        if readiness >= 3:
            return CommercialDecision(
                action=ActionType.ASK_ONE_QUALIFYING_QUESTION,
                goal="ask one natural question that clarifies what experience he wants",
                must_not_send_media=True,
                reason=f"paid only, readiness={readiness}",
            )

    return CommercialDecision(
        action=ActionType.CONTINUE_NORMAL_CHAT,
        goal="keep the conversation going naturally",
        must_not_send_media=True,
        reason=f"default, readiness={readiness}",
    )
