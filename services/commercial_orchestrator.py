"""Commercial orchestrator: observations -> policy -> durable state/actions."""
from datetime import datetime, timedelta, timezone

from db.commercial_queries import (
    cancel_actions_for_fan,
    get_creator_policy,
    get_fan_state,
    get_offerable_packages,
    merge_fan_ai_summary,
    save_fan_state,
    schedule_action,
)
from db.queries import clear_fan_decline_lock, save_fan_session
from models.commercial import (
    ActionType,
    CommercialDecision,
    CommercialEvent,
    EventType,
    FanStatus,
    PackageOption,
)
from services.commercial_events import (
    augment_pending_offer_events,
    extract_events,
    selected_package_event,
    stated_budget_cents,
)
from services.commercial_policy import CommercialContext, decide_next_action
from services.followup_lifecycle import (
    complete_session_state,
    pending_offer_expiry_obligation,
)
from services.price_learning import select_recommended_packages
from services.payday import resolve_payday
from services.session_lifecycle import (
    has_pending_purchase,
    has_remaining_steps,
    is_cooldown_active,
    normalize_session,
    resume_session,
)


def _learned_explicit_value(situation: dict, fact_key: str):
    intelligence = situation.get("learned_fan_intelligence") or {}
    for fact in intelligence.get("facts") or []:
        if fact.get("fact_key") != fact_key:
            continue
        if fact.get("status") not in {"explicit", "confirmed"}:
            continue
        return fact.get("value")
    return None


def _augment_events_with_safe_learned_context(
    events: list[CommercialEvent],
    situation: dict,
) -> None:
    """Use durable facts only where persistence is genuinely safe.

    A prior payday can complete an affordability pause when the current message says
    money is unavailable but omits the already-known date. Historical budgets are
    deliberately *not* converted into a current spend ceiling; price learning is a
    later Phase 2 concern and purchases remain authoritative.
    """

    types = {event.type for event in events}
    if EventType.MONEY_UNAVAILABLE not in types or EventType.PAYDAY_MENTIONED in types:
        return
    payday = _learned_explicit_value(situation, "payday")
    if payday:
        events.append(
            CommercialEvent(
                type=EventType.PAYDAY_MENTIONED,
                raw_expression=str(payday),
                confidence=0.85,
                metadata={"source": "passive_fan_intelligence"},
            )
        )


def _current_hard_ceiling(
    situation: dict,
    events: list[CommercialEvent],
) -> int | None:
    """Return only an explicit, current affordability ceiling.

    Historical purchases and price-learning estimates are intentionally ignored.
    """
    affordability = situation.get("affordability") or {}
    values = [
        affordability.get("current_limit_cents"),
        affordability.get("current_available_cents"),
    ]
    for event in events:
        if event.type in {
            EventType.BUDGET_STATED,
            EventType.BUDGET_LIMIT_STATED,
            EventType.COUNTEROFFER_STATED,
        }:
            values.append(event.amount_cents)
    parsed: list[int] = []
    for value in values:
        try:
            cents = int(value)
        except (TypeError, ValueError):
            continue
        if cents > 0:
            parsed.append(cents)
    return min(parsed) if parsed else None


def _clear_followup_obligation(state) -> None:
    state.next_followup_at = None
    state.next_followup_type = None
    state.next_followup_payload = {}
    state.next_followup_dedupe_key = None


async def _sync_pending_offer_expiry(
    *,
    creator_id: str,
    fan_id: str,
    state,
    policy,
    anchor: datetime,
) -> None:
    """Make fan state and the durable queue agree about one pending offer."""
    if state.status != FanStatus.OFFER_PENDING or not state.offered_packages:
        try:
            await cancel_actions_for_fan(fan_id, "OFFER_EXPIRY")
        except Exception as exc:
            print(f"[OFFER EXPIRY] cancellation failed fan={fan_id}: {exc}")
        if state.next_followup_type == "OFFER_EXPIRY":
            _clear_followup_obligation(state)
        return

    previous_type = state.next_followup_type
    state.last_offer_at = anchor
    obligation = pending_offer_expiry_obligation(
        state,
        policy=policy,
        fan_id=fan_id,
    )
    if obligation is None:
        return

    if previous_type and previous_type != "OFFER_EXPIRY":
        try:
            await cancel_actions_for_fan(fan_id, previous_type)
        except Exception as exc:
            print(
                f"[OFFER EXPIRY] superseded action cancellation failed "
                f"fan={fan_id} type={previous_type}: {exc}"
            )
    try:
        await cancel_actions_for_fan(fan_id, "OFFER_EXPIRY")
        await schedule_action(
            creator_id=creator_id,
            fan_id=fan_id,
            action_type=obligation.action_type,
            execute_at=obligation.execute_at,
            payload=obligation.payload,
            dedupe_key=obligation.dedupe_key,
        )
    except Exception as exc:
        # The state obligation is persisted by the caller and repaired by the
        # worker, so a queue write failure cannot lose the expiry promise.
        print(f"[OFFER EXPIRY] scheduling repair needed fan={fan_id}: {exc}")
    state.next_followup_at = obligation.execute_at
    state.next_followup_type = obligation.action_type
    state.next_followup_payload = obligation.payload
    state.next_followup_dedupe_key = obligation.dedupe_key


async def acknowledge_fan_return(
    creator_id: str,
    fan_id: str,
    *,
    now: datetime | None = None,
) -> None:
    """Cancel a due abandoned-offer nudge or refresh a still-live offer.

    This runs as soon as a fan message is persisted, including when auto mode is
    off, so a scheduled proactive message can never race an active conversation.
    """
    state = await get_fan_state(fan_id)
    changed = False
    if state.next_followup_type == "ABANDONED_OFFER_FOLLOWUP":
        await cancel_actions_for_fan(fan_id, "ABANDONED_OFFER_FOLLOWUP")
        _clear_followup_obligation(state)
        changed = True

    if state.next_followup_type == "INACTIVITY_REENGAGEMENT":
        await cancel_actions_for_fan(fan_id, "INACTIVITY_REENGAGEMENT")
        _clear_followup_obligation(state)
        changed = True

    if state.status == FanStatus.OFFER_PENDING and state.offered_packages:
        policy = await get_creator_policy(creator_id)
        await _sync_pending_offer_expiry(
            creator_id=creator_id,
            fan_id=fan_id,
            state=state,
            policy=policy,
            anchor=now or datetime.now(timezone.utc),
        )
        changed = True

    if changed:
        await save_fan_state(fan_id, creator_id, state)


async def orchestrate(
    creator_id: str,
    fan_id: str,
    situation: dict,
    *,
    fan_has_bought_before: bool = False,
    approved_sets_available: bool = True,
    within_daily_caps: bool = True,
    frozen_for_review: bool = False,
    active_session: dict | None = None,
) -> CommercialDecision:
    events = extract_events(situation)
    _augment_events_with_safe_learned_context(events, situation)
    policy = await get_creator_policy(creator_id)
    state = await get_fan_state(fan_id)
    now = datetime.now(timezone.utc)

    # Orchestrate is called because the fan is actively talking. Any proactive
    # abandoned-offer nudge is obsolete even through an alternate ingestion path.
    if state.next_followup_type == "ABANDONED_OFFER_FOLLOWUP":
        try:
            await cancel_actions_for_fan(fan_id, "ABANDONED_OFFER_FOLLOWUP")
        except Exception as exc:
            print(f"[OFFER FOLLOWUP] cancellation failed fan={fan_id}: {exc}")
        else:
            _clear_followup_obligation(state)

    if state.status == FanStatus.OFFER_PENDING and state.offered_packages:
        augment_pending_offer_events(
            events,
            str(situation.get("_latest_fan_message") or ""),
            state.offered_packages,
        )

    # Reset a consumed free allowance only after the configured cooldown has
    # genuinely elapsed. The next qualifying message can then start a new window.
    if state.free_session_ended_at:
        ended = state.free_session_ended_at
        if ended.tzinfo is None:
            ended = ended.replace(tzinfo=timezone.utc)
        if now >= ended + timedelta(hours=max(0, policy.free_session_cooldown_hours)):
            state.teaser_messages_used = 0
            state.free_session_started_at = None
            state.free_session_ended_at = None

    price_learning = situation.get("price_learning") or {}
    current_desired = str(situation.get("desired_experience") or "").strip()
    desired_experience = current_desired or str(state.desired_experience or "").strip()
    hard_ceiling_cents = _current_hard_ceiling(situation, events)
    package_options = await get_offerable_packages(
        creator_id,
        fan_id,
        policy,
        price_learning=price_learning,
        desired_experience=desired_experience or None,
        hard_ceiling_cents=hard_ceiling_cents,
    )
    package_options = select_recommended_packages(
        package_options,
        price_learning,
        max_options=2 if policy.offer_two_packages else 1,
    )
    active_offer_options = (
        state.offered_packages
        if state.status == FanStatus.OFFER_PENDING and state.offered_packages
        else package_options
    )
    _resolve_selected_package(events, active_offer_options)

    session = normalize_session(active_session)
    ctx = CommercialContext(
        fan_has_bought_before=fan_has_bought_before,
        approved_sets_available=approved_sets_available and bool(active_offer_options),
        within_daily_caps=within_daily_caps,
        frozen_for_review=frozen_for_review,
        package_options=active_offer_options,
        now=now,
        session_exists=bool(session),
        paused_session_available=bool(session and session.get("status") == "paused"),
        session_has_pending_purchase=has_pending_purchase(session),
        session_has_remaining_steps=has_remaining_steps(session),
        session_cooldown_active=is_cooldown_active(session),
    )
    decision = decide_next_action(policy, state, events, ctx)

    if decision.new_status:
        state.status = decision.new_status

    if current_desired:
        state.desired_experience = current_desired

    if decision.action in {ActionType.PRESENT_SESSION_OPTIONS, ActionType.END_TEASER_AND_OFFER}:
        state.offered_packages = decision.package_options
        state.last_offer_at = now
        state.selected_package_id = None
        state.selected_package_set_id = None
        state.selected_package_set_ids = []
        state.selected_package_label = None
        state.selected_package_price_cents = None

    selected = selected_package_event(events)
    if selected:
        package = _package_from_event(selected, state.offered_packages or package_options)
        cents = selected.amount_cents or (package.price_cents if package else None)
        if cents:
            state.confirmed_budget_cents = cents
            state.budget_source = "package_selected" if package else "fan_explicit"

        if package:
            set_ids = list(package.set_ids or ([package.set_id] if package.set_id else []))
            state.selected_package_id = package.package_id
            state.selected_package_set_id = set_ids[0] if set_ids else None
            state.selected_package_set_ids = set_ids
            state.selected_package_label = package.label
            state.selected_package_price_cents = package.price_cents
            decision.selected_package_set_id = state.selected_package_set_id
            decision.selected_package_set_ids = set_ids
            decision.session_budget_cents = package.price_cents
            decision.mention_price = package.price_cents // 100
            # Give the writer the exact selected snapshot entry, including its
            # approved description. Selection still does not equal purchase.
            decision.package_options = [package]

        if decision.action == ActionType.CREATE_PAID_SESSION and package:
            # Selection authorizes creation of a locked PPV. It is not a paid
            # session until the platform confirms the unlock.
            state.status = FanStatus.OFFER_SELECTED
            state.free_session_ended_at = now if state.free_session_started_at else state.free_session_ended_at
            try:
                await clear_fan_decline_lock(fan_id)
            except Exception as exc:
                print(f"[COMMERCIAL] clear legacy decline lock failed fan={fan_id}: {exc}")

    elif (cents := stated_budget_cents(events)):
        state.confirmed_budget_cents = cents
        state.budget_source = "fan_explicit"

    if decision.action in {ActionType.START_FREE_TEASER, ActionType.CONTINUE_FREE_TEXT}:
        state.teaser_messages_used += 1
        if state.free_session_started_at is None:
            state.free_session_started_at = now
        limit = (
            policy.teaser_max_messages
            if decision.action == ActionType.START_FREE_TEASER
            else policy.free_text_max_messages
        )
        if state.teaser_messages_used >= max(1, limit):
            state.free_session_ended_at = now

    if decision.action == ActionType.END_TEASER_AND_OFFER and state.free_session_started_at:
        state.free_session_ended_at = state.free_session_ended_at or now

    if decision.action == ActionType.RESUME_PREVIOUS_OFFER and session and session.get("status") == "paused":
        resumed = resume_session(session)
        await save_fan_session(fan_id, resumed)
        state.status = FanStatus.OFFER_SELECTED

    # Self-heal legacy/stale sessions whose index already passed the final step.
    if (
        session
        and session.get("status") == "active"
        and not has_pending_purchase(session)
        and not has_remaining_steps(session)
    ):
        state, followup_obligation = complete_session_state(
            state,
            session,
            policy=policy,
            fan_id=fan_id,
            buyer_stage="UNKNOWN",
            now=now,
        )
        await save_fan_state(fan_id, creator_id, state)
        await save_fan_session(fan_id, None)
        if followup_obligation:
            try:
                await schedule_action(
                    creator_id=creator_id,
                    fan_id=fan_id,
                    action_type=followup_obligation.action_type,
                    execute_at=followup_obligation.execute_at,
                    payload=followup_obligation.payload,
                    dedupe_key=followup_obligation.dedupe_key,
                )
            except Exception as exc:
                print(
                    f"[FOLLOWUP REPAIR NEEDED] fan={fan_id} "
                    f"type=POST_SESSION_FOLLOWUP error={exc}"
                )

    payday_event = next((event for event in events if event.type == EventType.PAYDAY_MENTIONED), None)
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
                payday_payload = {
                    "desired_experience": state.desired_experience or "",
                    "last_offer_price_cents": state.last_declined_price_cents,
                    "selected_package_id": state.selected_package_id,
                    "selected_package": next(
                        (
                            package.model_dump(mode="json")
                            for package in state.offered_packages
                            if package.package_id == state.selected_package_id
                        ),
                        None,
                    ),
                    "payday_at": when.isoformat(),
                    "payday_raw": raw,
                }
                payday_dedupe_key = f"payday:{fan_id}"
                await schedule_action(
                    creator_id=creator_id,
                    fan_id=fan_id,
                    action_type="PAYDAY_REENGAGEMENT",
                    execute_at=when,
                    payload=payday_payload,
                    dedupe_key=payday_dedupe_key,
                )
                state.next_followup_at = when
                state.next_followup_type = "PAYDAY_REENGAGEMENT"
                state.next_followup_payload = payday_payload
                state.next_followup_dedupe_key = payday_dedupe_key
                print(f"[COMMERCIAL] fan={fan_id} payday follow-up {when.isoformat()}")
            else:
                state.status = FanStatus.PAUSED_NO_BUDGET
                state.next_followup_at = None
                state.next_followup_type = None
                state.next_followup_payload = {}
                state.next_followup_dedupe_key = None
                print(f"[COMMERCIAL] fan={fan_id} payday '{raw}' unresolved")

    selected_resolved_now = bool(
        selected
        and selected.metadata.get("package_id")
        and decision.action == ActionType.CREATE_PAID_SESSION
    )
    resolved_now = selected_resolved_now or any(
        event.type in {EventType.MONEY_AVAILABLE, EventType.PURCHASED}
        for event in events
    )
    if resolved_now:
        try:
            await cancel_actions_for_fan(fan_id, "PAYDAY_REENGAGEMENT")
            if state.next_followup_type == "PAYDAY_REENGAGEMENT":
                state.next_followup_at = None
                state.next_followup_type = None
                state.next_followup_payload = {}
                state.next_followup_dedupe_key = None
        except Exception as exc:
            print(f"[COMMERCIAL] cancel follow-up failed fan={fan_id}: {exc}")

    await _sync_pending_offer_expiry(
        creator_id=creator_id,
        fan_id=fan_id,
        state=state,
        policy=policy,
        anchor=now,
    )

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
    package_id = str(event.metadata.get("package_id") or "")
    if package_id:
        package = next(
            (item for item in offered_packages if item.package_id == package_id),
            None,
        )
    if package is None:
        if event.amount_cents is not None:
            package = min(offered_packages, key=lambda item: abs(item.price_cents - event.amount_cents))
            if abs(package.price_cents - event.amount_cents) > 100:
                package = None
        elif event.package_position == "first":
            package = offered_packages[0]
        elif event.package_position == "second" and len(offered_packages) >= 2:
            package = offered_packages[1]

    if package:
        set_ids = list(package.set_ids or ([package.set_id] if package.set_id else []))
        event.amount_cents = package.price_cents
        event.metadata.update({
            "package_id": package.package_id,
            "set_id": set_ids[0] if set_ids else None,
            "set_ids": set_ids,
            "label": package.label,
            "experience": package.experience,
            "legal_description": package.legal_description or package.experience,
        })


def _package_from_event(
    event: CommercialEvent,
    offered_packages: list[PackageOption],
) -> PackageOption | None:
    package_id = event.metadata.get("package_id")
    if package_id:
        return next((package for package in offered_packages if package.package_id == package_id), None)
    if event.amount_cents is not None:
        return next((package for package in offered_packages if package.price_cents == event.amount_cents), None)
    return None
