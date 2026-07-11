"""Deterministic commercial policy engine.

`decide_next_action` is a PURE function: (policy, state, events, context) -> decision.
No I/O, no LLM, no side effects — so it is fully testable and the business rules
stop depending on whether the model follows a paragraph of prompt text.

The generator receives the resulting CommercialDecision and only phrases it.
"""
from models.commercial import (
    ActionType,
    CommercialDecision,
    CommercialEvent,
    CreatorPolicy,
    EventType,
    FanCommercialState,
    FanStatus,
    SextingMode,
)


class CommercialContext:
    """Read-only facts the policy needs that aren't in state/events."""
    def __init__(
        self,
        fan_has_bought_before: bool = False,
        approved_sets_available: bool = True,
        within_daily_caps: bool = True,
        frozen_for_review: bool = False,
        fan_repeats_interest: bool = False,
    ):
        self.fan_has_bought_before = fan_has_bought_before
        self.approved_sets_available = approved_sets_available
        self.within_daily_caps = within_daily_caps
        self.frozen_for_review = frozen_for_review
        self.fan_repeats_interest = fan_repeats_interest


def _has(events: list[CommercialEvent], t: EventType) -> bool:
    return any(e.type == t for e in events)


def _get(events: list[CommercialEvent], t: EventType) -> CommercialEvent | None:
    for e in events:
        if e.type == t:
            return e
    return None


def compute_readiness(
    events: list[CommercialEvent],
    state: FanCommercialState,
    ctx: CommercialContext,
) -> int:
    """How ready is this fan to be sold to, right now? Replaces 'the model felt
    like asking about budget', which produced the awkward mid-flirt qualification.

    0-2  -> flirt, understand intent, do not qualify
    3-4  -> may ask ONE natural preference question
    5+   -> may qualify budget / present session options
    <0   -> do not sell at all
    """
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

    # ---- hard stops, in priority order -------------------------------------
    if _has(events, EventType.CRISIS) or ctx.frozen_for_review:
        return CommercialDecision(
            action=ActionType.HAND_OFF_TO_HUMAN,
            goal="respond with genuine care; no selling of any kind",
            must_not_send_media=True,
            new_status=FanStatus.HUMAN_REVIEW,
            reason="crisis or frozen",
        )

    # ---- money became available: lift any pause, resume what he wanted ------
    if _has(events, EventType.MONEY_AVAILABLE):
        if state.status in (FanStatus.PAUSED_UNTIL_PAYDAY, FanStatus.PAUSED_NO_BUDGET):
            return CommercialDecision(
                action=ActionType.RESUME_PREVIOUS_OFFER,
                goal="he says money is available; warmly pick up the experience he "
                     "wanted before, without pressure",
                must_not_send_media=True,
                may_be_explicit=policy.sexting_mode != SextingMode.PAID_ONLY,
                mention_previous_interest=True,
                new_status=FanStatus.IDLE,
                reason="money_available lifts pause",
            )

    # ---- he can't pay ------------------------------------------------------
    if _has(events, EventType.MONEY_UNAVAILABLE):
        payday = _get(events, EventType.PAYDAY_MENTIONED)
        if payday and policy.payday_reengagement_enabled:
            return CommercialDecision(
                action=ActionType.PAUSE_UNTIL_PAYDAY,
                goal="he can't pay yet but told you when he can. Be warm, zero guilt, "
                     "no re-pitching. Make it clear it'll still be here for him.",
                must_not_send_media=True,
                may_be_explicit=False,
                new_status=FanStatus.PAUSED_UNTIL_PAYDAY,
                schedule_payday_followup=True,
                reason=f"money_unavailable + payday '{payday.raw_expression}'",
            )
        return CommercialDecision(
            action=ActionType.PAUSE_NO_BUDGET,
            goal="he can't pay and didn't say when he can. Stay warm, stop selling. "
                 "If it comes up naturally, find out when payday is.",
            must_not_send_media=True,
            may_be_explicit=False,
            new_status=FanStatus.PAUSED_NO_BUDGET,
            reason="money_unavailable, no payday",
        )

    # ---- already paused: do NOT sell, regardless of what he asks for --------
    if state.status in (FanStatus.PAUSED_NO_BUDGET, FanStatus.PAUSED_UNTIL_PAYDAY):
        # FREE_TEXT_ALLOWED creators may still give him a text experience.
        if policy.sexting_mode == SextingMode.FREE_TEXT_ALLOWED:
            return CommercialDecision(
                action=ActionType.CONTINUE_FREE_TEXT,
                goal="keep him engaged with a text-only experience; no media, no prices",
                must_not_send_media=True,
                may_be_explicit=True,
                reason="paused but free text allowed",
            )
        return CommercialDecision(
            action=ActionType.CONTINUE_NORMAL_CHAT,
            goal="he still can't pay. Keep it warm and pleasant but do NOT sell, "
                 "pitch, tease paid content, or mention prices — even if he asks.",
            must_not_send_media=True,
            may_be_explicit=False,
            reason="paused, selling suppressed",
        )

    # ---- a paid session is running ----------------------------------------
    if state.status == FanStatus.PAID_SESSION_ACTIVE:
        if not ctx.within_daily_caps:
            return CommercialDecision(
                action=ActionType.CONTINUE_NORMAL_CHAT,
                goal="stay in the moment; daily limits reached, no more sending",
                reason="daily cap",
            )
        return CommercialDecision(
            action=ActionType.SEND_NEXT_PPV_STEP,
            goal="continue the paid experience and deliver the next step when the "
                 "moment is right",
            must_not_send_media=False,
            may_be_explicit=True,
            reason="paid session active",
        )

    # ---- he wants the experience -------------------------------------------
    readiness = compute_readiness(events, state, ctx)
    wants = _has(events, EventType.WANTS_EXPLICIT) or _has(events, EventType.WANTS_MEDIA)

    if wants and readiness >= 5 and ctx.approved_sets_available and ctx.within_daily_caps:
        if policy.offer_two_packages and state.confirmed_budget_cents is None:
            return CommercialDecision(
                action=ActionType.PRESENT_SESSION_OPTIONS,
                goal="offer him a choice of two experiences so he picks his own level",
                package_options=[25, 60],
                must_not_send_media=True,
                may_be_explicit=True,
                new_status=FanStatus.OFFER_PENDING,
                reason=f"readiness={readiness}, no confirmed budget",
            )
        return CommercialDecision(
            action=ActionType.CREATE_PAID_SESSION,
            goal="he's ready — build the session and move toward the first send",
            must_not_send_media=False,
            may_be_explicit=True,
            new_status=FanStatus.PAID_SESSION_ACTIVE,
            reason=f"readiness={readiness}",
        )

    if wants:
        mode = policy.sexting_mode

        if mode == SextingMode.FREE_TEXT_ALLOWED:
            if state.teaser_messages_used < policy.free_text_max_messages:
                return CommercialDecision(
                    action=ActionType.CONTINUE_FREE_TEXT,
                    goal="give him a real text-only experience; media stays paid",
                    must_not_send_media=policy.media_always_paid,
                    may_be_explicit=True,
                    new_status=FanStatus.FREE_TEXT_SESSION,
                    reason="free text mode",
                )

        if mode == SextingMode.HYBRID_TEASER:
            if state.teaser_messages_used < policy.teaser_max_messages:
                return CommercialDecision(
                    action=ActionType.START_FREE_TEASER,
                    goal="give him a taste — a few genuinely explicit exchanges, no media",
                    must_not_send_media=True,
                    may_be_explicit=True,
                    new_status=FanStatus.FREE_TEASER,
                    reason=f"teaser {state.teaser_messages_used}/{policy.teaser_max_messages}",
                )
            return CommercialDecision(
                action=ActionType.END_TEASER_AND_OFFER,
                goal="the preview is over — transition him toward the full paid experience",
                package_options=[25, 60] if policy.offer_two_packages else [],
                must_not_send_media=True,
                may_be_explicit=True,
                new_status=FanStatus.OFFER_PENDING,
                reason="teaser exhausted",
            )

        # PAID_ONLY: build a little tension, then qualify and offer.
        if readiness >= 3:
            return CommercialDecision(
                action=ActionType.ASK_ONE_QUALIFYING_QUESTION,
                goal="find out what he actually wants — ONE natural question, not a form",
                must_not_send_media=True,
                may_be_explicit=False,
                reason=f"paid_only, readiness={readiness}",
            )

    # ---- default -----------------------------------------------------------
    return CommercialDecision(
        action=ActionType.CONTINUE_NORMAL_CHAT,
        goal="keep the conversation going naturally",
        must_not_send_media=True,
        may_be_explicit=False,
        reason=f"default, readiness={readiness}",
    )