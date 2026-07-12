"""Commercial orchestrator: observations -> policy -> durable state/actions."""
from datetime import datetime, timezone

from db.commercial_queries import (
    cancel_actions_for_fan,
    get_creator_policy,
    get_fan_state,
    get_offerable_packages,
    merge_fan_ai_summary,
    save_fan_state,
    schedule_action,
)
from db.queries import clear_fan_decline_lock
from models.commercial import (
    ActionType,
    CommercialDecision,
    CommercialEvent,
    EventType,
    FanStatus,
    PackageOption,
)
from services.commercial_events import extract_events, selected_package_event, stated_budget_cents
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

    package_options = await get_offerable_packages(creator_id, fan_id, policy)
    _resolve_selected_package(events, state.offered_packages or package_options)

    ctx = CommercialContext(
        fan_has_bought_before=fan_has_bought_before,
        approved_sets_available=approved_sets_available and bool(package_options or state.offered_packages),
        within_daily_caps=within_daily_caps,
        frozen_for_review=frozen_for_review,
        package_options=package_options,
    )
    decision = decide_next_action(policy, state, events, ctx)

    now = datetime.now(timezone.utc)
    if decision.new_status:
        state.status = decision.new_status

    desired = str(situation.get("desired_experience") or "").strip()
    if desired:
        state.desired_experience = desired

    if decision.action in (ActionType.PRESENT_SESSION_OPTIONS, ActionType.END_TEASER_AND_OFFER):
        state.offered_packages = decision.package_options
        state.last_offer_at = now
        state.selected_package_id = None
        state.selected_package_set_id = None
        state.selected_package_label = None
        state.selected_package_price_cents = None

    selected = selected_package_event(events)
    if selected:
        package = _package_from_event(selected, state.offered_packages or package_options)
        cents = selected.amount_cents or (package.price_cents if package else None)

        # Persist the explicit amount even if package matching failed, but do not
        # activate or plan a session unless it resolves to an offered set.
        if cents:
            state.confirmed_budget_cents = cents
            state.budget_source = "package_selected" if package else "fan_explicit"

        if package:
            state.selected_package_id = package.package_id
            state.selected_package_set_id = package.set_id
            state.selected_package_label = package.label
            state.selected_package_price_cents = package.price_cents
            decision.selected_package_set_id = package.set_id
            decision.session_budget_cents = package.price_cents
            decision.mention_price = package.price_cents // 100

        if decision.action == ActionType.CREATE_PAID_SESSION and package:
            state.status = FanStatus.PAID_SESSION_ACTIVE
            try:
                await clear_fan_decline_lock(fan_id)
            except Exception as exc:
                print(f"[COMMERCIAL] clear legacy decline lock failed fan={fan_id}: {exc}")

    elif (cents := stated_budget_cents(events)):
        state.confirmed_budget_cents = cents
        state.budget_source = "fan_explicit"

    if decision.action in (ActionType.START_FREE_TEASER, ActionType.CONTINUE_FREE_TEXT):
        state.teaser_messages_used += 1
        if state.free_session_started_at is None:
            state.free_session_started_at = now

    payday_event = next((e for e in events if e.type == EventType.PAYDAY_MENTIONED), None)
    if payday_event:
        raw = payday_event.raw_expression
        when, confidence = resolve_payday(
            raw,
            send_hour=policy.payday_send_hour_local,
            timezone_name=policy.timezone,
        )
        state.payday_raw = raw or None
        state.payday_at = when
        state.payday_confidence = confidence
        try:
            await merge_fan_ai_summary(fan_id, {
                "payday": raw,
                "payday_at": when.isoformat() if when else None,
            })
        except Exception as exc:
            print(f"[COMMERCIAL] ai_summary payday merge failed fan={fan_id}: {exc}")

        if decision.schedule_payday_followup:
            if when and confidence >= 0.6:
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
                    # One logical payday action per fan. A corrected date replaces it.
                    dedupe_key=f"payday:{fan_id}",
                )
                print(
                    f"[COMMERCIAL] fan={fan_id} PAUSED_UNTIL_PAYDAY, "
                    f"follow-up {when.isoformat()}"
                )
            else:
                state.status = FanStatus.PAUSED_NO_BUDGET
                print(
                    f"[COMMERCIAL] fan={fan_id} payday '{raw}' unresolved — "
                    "no follow-up scheduled"
                )

    # Selecting/buying now means a payday mention is CRM knowledge only, not a
    # reason to schedule a second sales message.
    selected_resolved_now = bool(
        selected
        and selected.metadata.get("package_id")
        and decision.action == ActionType.CREATE_PAID_SESSION
    )
    resolved_now = selected_resolved_now or any(
        e.type in (EventType.MONEY_AVAILABLE, EventType.PURCHASED)
        for e in events
    )
    if resolved_now:
        try:
            await cancel_actions_for_fan(fan_id, "PAYDAY_REENGAGEMENT")
        except Exception as exc:
            print(f"[COMMERCIAL] cancel follow-up failed fan={fan_id}: {exc}")

    await save_fan_state(fan_id, creator_id, state)
    print(
        f"[COMMERCIAL] fan={fan_id} action={decision.action.value} "
        f"status={state.status.value} ({decision.reason})"
    )
    return decision


def _resolve_selected_package(
    events: list[CommercialEvent],
    offered_packages: list[PackageOption],
) -> None:
    event = selected_package_event(events)
    if not event or not offered_packages:
        return

    package: PackageOption | None = None
    if event.amount_cents is not None:
        package = min(
            offered_packages,
            key=lambda candidate: abs(candidate.price_cents - event.amount_cents),
        )
        # Only accept a price match within one dollar. Otherwise preserve the raw
        # event and let the policy avoid inventing a package.
        if abs(package.price_cents - event.amount_cents) > 100:
            package = None
    elif event.package_position == "first":
        package = offered_packages[0]
    elif event.package_position == "second" and len(offered_packages) >= 2:
        package = offered_packages[1]

    if package:
        event.amount_cents = package.price_cents
        event.metadata.update({
            "package_id": package.package_id,
            "set_id": package.set_id,
            "label": package.label,
        })


def _package_from_event(
    event: CommercialEvent,
    offered_packages: list[PackageOption],
) -> PackageOption | None:
    package_id = event.metadata.get("package_id")
    if package_id:
        return next((p for p in offered_packages if p.package_id == package_id), None)
    if event.amount_cents is not None:
        return next((p for p in offered_packages if p.price_cents == event.amount_cents), None)
    return None
