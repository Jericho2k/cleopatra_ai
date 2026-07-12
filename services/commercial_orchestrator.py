"""Commercial orchestrator.

One call per fan message. It:
  1. converts the analyzer's observations into typed events
  2. loads the creator's policy and the fan's durable commercial state
  3. asks the (pure, tested) policy engine what to do
  4. persists the new state
  5. schedules any future action the decision implies (e.g. the payday follow-up)
  6. returns the decision, which the prompt builder expresses

The point: the LLM no longer decides whether to sell, pause, tease or follow up.
"""
from datetime import datetime, timezone

from db.commercial_queries import (
    cancel_actions_for_fan,
    get_creator_policy,
    get_fan_state,
    save_fan_state,
    schedule_action,
)
from models.commercial import ActionType, CommercialDecision, EventType, FanStatus
from services.commercial_events import extract_events, stated_budget_cents
from services.commercial_policy import CommercialContext, decide_next_action
from services.payday import resolve_payday


async def orchestrate(
    creator_id: str,
    fan_id: str,
    situation: dict,
    *,
    fan_has_bought_before: bool = False,
    approved_sets_available: bool = True,
    within_daily_caps: bool = True,
    frozen_for_review: bool = False,
) -> CommercialDecision:
    events = extract_events(situation)
    policy = await get_creator_policy(creator_id)
    state = await get_fan_state(fan_id)

    ctx = CommercialContext(
        fan_has_bought_before=fan_has_bought_before,
        approved_sets_available=approved_sets_available,
        within_daily_caps=within_daily_caps,
        frozen_for_review=frozen_for_review,
    )

    decision = decide_next_action(policy, state, events, ctx)

    # ---- fold the decision + observations back into durable state -----------
    if decision.new_status:
        state.status = decision.new_status

    # confirmed budget only — never inferred
    cents = stated_budget_cents(events)
    if cents:
        state.confirmed_budget_cents = cents
        state.budget_source = "fan_explicit"

    desired = (situation.get("desired_experience") or "").strip()
    if desired:
        state.desired_experience = desired

    # count teaser/free messages so the limits actually bite
    if decision.action in (ActionType.START_FREE_TEASER, ActionType.CONTINUE_FREE_TEXT):
        state.teaser_messages_used += 1
        if state.free_session_started_at is None:
            state.free_session_started_at = datetime.now(timezone.utc)

    # ---- payday: turn a promise into an executable future action ------------
    if decision.schedule_payday_followup:
        payday_ev = next((e for e in events if e.type == EventType.PAYDAY_MENTIONED), None)
        raw = payday_ev.raw_expression if payday_ev else ""
        when, conf = resolve_payday(raw, send_hour=policy.payday_send_hour_local)
        state.payday_raw = raw or None
        state.payday_at = when
        state.payday_confidence = conf

        if when and conf >= 0.6:
            await schedule_action(
                creator_id=creator_id,
                fan_id=fan_id,
                action_type="PAYDAY_REENGAGEMENT",
                execute_at=when,
                payload={
                    "desired_experience": state.desired_experience or "",
                    "last_offer_price_cents": state.last_declined_price_cents,
                    "payday_raw": raw,
                },
                # upsert key: if he later says "actually Monday", this REPLACES the
                # Friday action instead of creating a second one.
                dedupe_key=f"payday:{fan_id}:{when.date().isoformat()}",
            )
            print(f"[COMMERCIAL] fan={fan_id} PAUSED_UNTIL_PAYDAY, follow-up {when.isoformat()}")
        else:
            # Couldn't resolve the date confidently — don't guess a day.
            state.status = FanStatus.PAUSED_NO_BUDGET
            print(f"[COMMERCIAL] fan={fan_id} payday '{raw}' unresolved — no follow-up scheduled")

    # ---- he resolved it himself: cancel any pending follow-up ---------------
    if any(e.type in (EventType.MONEY_AVAILABLE, EventType.PURCHASED) for e in events):
        try:
            await cancel_actions_for_fan(fan_id, "PAYDAY_REENGAGEMENT")
        except Exception as e:
            print(f"[COMMERCIAL] cancel follow-up failed fan={fan_id}: {e}")

    await save_fan_state(fan_id, creator_id, state)
    print(f"[COMMERCIAL] fan={fan_id} action={decision.action.value} "
          f"status={state.status.value} ({decision.reason})")
    return decision