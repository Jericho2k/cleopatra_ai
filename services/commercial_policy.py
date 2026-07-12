"""Deterministic commercial policy engine.

`decide_next_action` is pure: (policy, state, events, context) -> decision.
The writer receives the resulting decision and only phrases it.
"""
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
    """Read-only facts the policy needs that are not durable fan state."""

    def __init__(
        self,
        fan_has_bought_before: bool = False,
        approved_sets_available: bool = True,
        within_daily_caps: bool = True,
        frozen_for_review: bool = False,
        fan_repeats_interest: bool = False,
        package_options: list[PackageOption] | None = None,
    ):
        self.fan_has_bought_before = fan_has_bought_before
        self.approved_sets_available = approved_sets_available
        self.within_daily_caps = within_daily_caps
        self.frozen_for_review = frozen_for_review
        self.fan_repeats_interest = fan_repeats_interest
        self.package_options = package_options or []


def _has(events: list[CommercialEvent], event_type: EventType) -> bool:
    return any(event.type == event_type for event in events)


def _get(events: list[CommercialEvent], event_type: EventType) -> CommercialEvent | None:
    return next((event for event in events if event.type == event_type), None)


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
    """The single place that decides what the business does next."""

    if _has(events, EventType.CRISIS) or ctx.frozen_for_review:
        return CommercialDecision(
            action=ActionType.HAND_OFF_TO_HUMAN,
            goal="respond with genuine care; no selling of any kind",
            must_not_send_media=True,
            must_not_ask_question=False,
            new_status=FanStatus.HUMAN_REVIEW,
            reason="crisis or frozen",
        )

    if _has(events, EventType.MONEY_AVAILABLE):
        if state.status in (FanStatus.PAUSED_UNTIL_PAYDAY, FanStatus.PAUSED_NO_BUDGET):
            return CommercialDecision(
                action=ActionType.RESUME_PREVIOUS_OFFER,
                goal=(
                    "money is available again; warmly resume the exact experience "
                    "he wanted before without pressure"
                ),
                must_not_send_media=True,
                may_be_explicit=policy.sexting_mode != SextingMode.PAID_ONLY,
                mention_previous_interest=True,
                new_status=FanStatus.IDLE,
                max_messages=1,
                conversation_continuation="optional",
                reason="money_available lifts pause",
            )

    # Acceptance of a real package is stronger than a simultaneous statement
    # that he cannot spend *more*. This branch deliberately precedes affordability.
    selected = _get(events, EventType.PACKAGE_SELECTED)
    if selected:
        if not selected.metadata.get("set_id"):
            return CommercialDecision(
                action=ActionType.PRESENT_SESSION_OPTIONS,
                goal=(
                    "the fan appears to have chosen an option, but it could not be "
                    "matched safely; restate the exact available options once"
                ),
                package_options=ctx.package_options or state.offered_packages,
                must_not_send_media=True,
                may_be_explicit=False,
                new_status=FanStatus.OFFER_PENDING,
                must_not_ask_question=False,
                max_messages=2,
                conversation_continuation="required",
                reason="package selection could not be matched to an offered set",
            )
        if not ctx.approved_sets_available or not ctx.within_daily_caps:
            return CommercialDecision(
                action=ActionType.CONTINUE_NORMAL_CHAT,
                goal="acknowledge his choice, but do not promise or send unavailable content",
                must_not_send_media=True,
                must_not_ask_question=True,
                max_messages=1,
                conversation_continuation="none",
                reason="package selected but unavailable/capped",
            )
        return CommercialDecision(
            action=ActionType.CREATE_PAID_SESSION,
            goal="confirm the package he selected and deliver the matching paid experience",
            must_not_send_media=False,
            may_be_explicit=True,
            mention_price=(selected.amount_cents // 100 if selected.amount_cents else None),
            new_status=FanStatus.PAID_SESSION_ACTIVE,
            session_budget_cents=selected.amount_cents,
            selected_package_set_id=selected.metadata.get("set_id"),
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
                goal=(
                    "he cannot buy any available option now but said when money "
                    "arrives; be warm, apply zero pressure, and close the exchange cleanly"
                ),
                must_not_send_media=True,
                may_be_explicit=False,
                new_status=FanStatus.PAUSED_UNTIL_PAYDAY,
                schedule_payday_followup=True,
                must_not_ask_question=True,
                max_messages=1,
                conversation_continuation="none",
                reason=f"cannot buy now + payday '{payday.raw_expression}'",
            )
        return CommercialDecision(
            action=ActionType.PAUSE_NO_BUDGET,
            goal="he cannot buy any available option now; stay warm and stop selling",
            must_not_send_media=True,
            may_be_explicit=False,
            new_status=FanStatus.PAUSED_NO_BUDGET,
            must_not_ask_question=True,
            max_messages=1,
            conversation_continuation="none",
            reason="cannot buy now, no payday",
        )

    if _has(events, EventType.OFFER_DECLINED):
        return CommercialDecision(
            action=ActionType.CONTINUE_NORMAL_CHAT,
            goal="accept the no gracefully; no counter-pitch and no guilt",
            must_not_send_media=True,
            may_be_explicit=False,
            new_status=FanStatus.IDLE,
            must_not_ask_question=True,
            max_messages=1,
            conversation_continuation="none",
            reason="offer declined without affordability pause",
        )

    if state.status in (FanStatus.PAUSED_NO_BUDGET, FanStatus.PAUSED_UNTIL_PAYDAY):
        if policy.sexting_mode == SextingMode.FREE_TEXT_ALLOWED:
            return CommercialDecision(
                action=ActionType.CONTINUE_FREE_TEXT,
                goal="keep him engaged with text only; no media and no price",
                must_not_send_media=True,
                may_be_explicit=True,
                reason="paused but free text allowed",
            )
        return CommercialDecision(
            action=ActionType.CONTINUE_NORMAL_CHAT,
            goal="keep it pleasant without selling or giving a paid service away",
            must_not_send_media=True,
            may_be_explicit=False,
            reason="paused, selling suppressed",
        )

    if state.status == FanStatus.PAID_SESSION_ACTIVE:
        if not ctx.within_daily_caps:
            return CommercialDecision(
                action=ActionType.CONTINUE_NORMAL_CHAT,
                goal="stay in the moment; daily limits are reached, so send nothing else",
                must_not_send_media=True,
                reason="daily cap",
            )
        return CommercialDecision(
            action=ActionType.SEND_NEXT_PPV_STEP,
            goal="continue the paid experience and deliver the next planned step",
            must_not_send_media=False,
            may_be_explicit=True,
            reason="paid session active",
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
                must_not_ask_question=False,
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
                reason=f"readiness={readiness}, confirmed budget",
            )

    if wants:
        if policy.sexting_mode == SextingMode.FREE_TEXT_ALLOWED:
            if state.teaser_messages_used < policy.free_text_max_messages:
                return CommercialDecision(
                    action=ActionType.CONTINUE_FREE_TEXT,
                    goal="give a genuine text-only experience; media remains paid",
                    must_not_send_media=policy.media_always_paid,
                    may_be_explicit=True,
                    new_status=FanStatus.FREE_TEXT_SESSION,
                    reason="free text mode",
                )

        if policy.sexting_mode == SextingMode.HYBRID_TEASER:
            if state.teaser_messages_used < policy.teaser_max_messages:
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
                reason="teaser exhausted",
            )

        if readiness >= 3:
            return CommercialDecision(
                action=ActionType.ASK_ONE_QUALIFYING_QUESTION,
                goal="ask one natural question that clarifies what experience he wants",
                must_not_send_media=True,
                may_be_explicit=False,
                reason=f"paid_only, readiness={readiness}",
            )

    return CommercialDecision(
        action=ActionType.CONTINUE_NORMAL_CHAT,
        goal="keep the conversation going naturally",
        must_not_send_media=True,
        may_be_explicit=False,
        reason=f"default, readiness={readiness}",
    )
