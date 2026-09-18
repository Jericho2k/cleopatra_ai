"""Commercial orchestrator: observations -> policy -> durable state/actions."""
from datetime import datetime, timedelta, timezone

from db.commercial_queries import (
    cancel_actions_for_fan,
    get_creator_policy,
    get_fan_state,
    get_next_offer_with_inventory,
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
    Offer,
    SextingMode,
)
from services.commercial_events import (
    accepted_offer_event,
    augment_pending_offer_events,
    extract_events,
    stated_budget_cents,
)
from services.commercial_policy import CommercialContext, decide_next_action
from services.experience_director import scene_allows_new_offer
from services.followup_lifecycle import complete_session_state
from services.offer_lifecycle import sync_pending_offer_expiry
from services.payday import resolve_payday
from services.session_lifecycle import (
    has_pending_purchase,
    has_remaining_steps,
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

    if state.status == FanStatus.OFFER_PENDING and state.pending_offer is not None:
        policy = await get_creator_policy(creator_id)
        await sync_pending_offer_expiry(
            creator_id=creator_id,
            fan_id=fan_id,
            state=state,
            policy=policy,
            anchor=now or datetime.now(timezone.utc),
            cancel_action=cancel_actions_for_fan,
            schedule=schedule_action,
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
    scene: dict | None = None,
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

    if state.status == FanStatus.OFFER_PENDING and state.pending_offer is not None:
        augment_pending_offer_events(
            events,
            str(situation.get("_latest_fan_message") or ""),
            state.pending_offer,
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
    next_offer, vault_asset_types = await get_next_offer_with_inventory(
        creator_id,
        fan_id,
        policy,
        price_learning=price_learning,
        desired_experience=desired_experience or None,
        hard_ceiling_cents=hard_ceiling_cents,
        # Content selection weighs scene continuity, explicitness AND whether
        # the candidate actually advances the interaction he is in.
        scene=scene,
    )
    # A live offer is held to exactly as presented until it is resolved; only
    # when nothing is pending does the freshly built next unlock apply.
    active_offer = (
        state.pending_offer
        if state.status == FanStatus.OFFER_PENDING and state.pending_offer is not None
        else next_offer
    )
    _resolve_accepted_offer(events, active_offer)

    session = normalize_session(active_session)
    ctx = CommercialContext(
        fan_has_bought_before=fan_has_bought_before,
        approved_sets_available=approved_sets_available and active_offer is not None,
        within_daily_caps=within_daily_caps,
        frozen_for_review=frozen_for_review,
        next_offer=active_offer,
        now=now,
        session_exists=bool(session),
        paused_session_available=bool(session and session.get("status") == "paused"),
        session_has_pending_purchase=has_pending_purchase(session),
        session_has_remaining_steps=has_remaining_steps(session),
        # Choreography's only input to policy, and it can only narrow: the
        # scene may withhold the DISCOVERY of a new offer after an unlock it
        # has not been talked about yet. Everything else here is commercial.
        experience_allows_new_offer=scene_allows_new_offer(scene),
    )
    decision = decide_next_action(policy, state, events, ctx)

    # The authoritative inventory statement travels WITH the decision, built
    # from the rows this very call loaded. Downstream nothing has to re-derive
    # what exists, so the writer's statement and the planner's rows cannot drift.
    _attach_media_inventory(
        decision,
        active_offer=active_offer,
        vault_asset_types=vault_asset_types,
        session=session,
        desired_experience=desired_experience,
        latest_fan_message=str(situation.get("_latest_fan_message") or ""),
    )

    if decision.new_status:
        state.status = decision.new_status

    if current_desired:
        state.desired_experience = current_desired

    if decision.action == ActionType.OFFER_NEXT_UNLOCK and decision.next_offer:
        state.pending_offer = decision.next_offer
        state.last_offer_at = now
        state.accepted_offer_id = None
        state.accepted_offer_set_id = None
        state.accepted_offer_label = None
        state.accepted_offer_price_cents = None

    accepted = accepted_offer_event(events)
    if accepted:
        offer = _offer_from_event(accepted, state.pending_offer or active_offer)
        cents = accepted.amount_cents or (offer.price_cents if offer else None)
        if cents:
            state.confirmed_budget_cents = cents
            state.budget_source = "offer_accepted" if offer else "fan_explicit"

        if offer:
            state.pending_offer = offer
            state.accepted_offer_id = offer.offer_id
            state.accepted_offer_set_id = offer.set_id
            state.accepted_offer_label = offer.label
            state.accepted_offer_price_cents = offer.price_cents
            decision.accepted_offer_set_id = offer.set_id
            decision.session_budget_cents = offer.price_cents
            decision.mention_price = offer.price_cents // 100
            # Narrow the authorised inventory to the one thing he accepted.
            decision.authorized_asset_types = list(offer.asset_types)
            if (
                decision.unavailable_asset_type_requested
                in decision.authorized_asset_types
            ):
                decision.unavailable_asset_type_requested = None

        if decision.action == ActionType.SEND_NEXT_PPV_STEP and offer:
            # Acceptance authorizes creation of a locked PPV. It is not a paid
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

    if (
        decision.action == ActionType.OFFER_NEXT_UNLOCK
        and state.free_session_started_at
        and state.teaser_messages_used
    ):
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
                    "accepted_offer_id": state.accepted_offer_id,
                    "accepted_offer": (
                        state.pending_offer.model_dump(mode="json")
                        if state.pending_offer is not None
                        and state.pending_offer.offer_id == state.accepted_offer_id
                        else None
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

    accepted_resolved_now = bool(
        accepted
        and accepted.metadata.get("offer_id")
        and decision.action == ActionType.SEND_NEXT_PPV_STEP
    )
    resolved_now = accepted_resolved_now or any(
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

    await sync_pending_offer_expiry(
        creator_id=creator_id,
        fan_id=fan_id,
        state=state,
        policy=policy,
        anchor=now,
        cancel_action=cancel_actions_for_fan,
        schedule=schedule_action,
    )

    await save_fan_state(fan_id, creator_id, state)
    print(
        f"[COMMERCIAL] fan={fan_id} action={decision.action.value} "
        f"status={state.status.value} ({decision.reason})"
    )
    return decision


def _attach_media_inventory(
    decision: CommercialDecision,
    *,
    active_offer: Offer | None,
    vault_asset_types: tuple[str, ...],
    session: dict | None,
    desired_experience: str,
    latest_fan_message: str,
) -> None:
    """Record what media this decision may actually promise.

    Deliberately derived here rather than in the writer layer: this is the only
    place that has the approved rows, the offer snapshot and the active session
    in one scope at the same instant.
    """
    from services.inventory_authority import (
        ASSET_VIDEO,
        asset_types_from_session,
    )
    from services.media_packages import wants_video

    session_types = asset_types_from_session(session)
    decision_types = tuple(decision.next_offer.asset_types) if decision.next_offer else ()
    offer_types = tuple(active_offer.asset_types) if active_offer else ()

    decision.vault_asset_types = list(vault_asset_types)
    decision.authorized_asset_types = list(
        decision_types or session_types or offer_types
    )
    promisable = set(decision.authorized_asset_types)
    asked_for_video = bool(
        wants_video(desired_experience) or wants_video(latest_fan_message)
    )
    decision.unavailable_asset_type_requested = (
        ASSET_VIDEO if asked_for_video and ASSET_VIDEO not in promisable else None
    )


def _resolve_accepted_offer(
    events: list[CommercialEvent],
    active_offer: Offer | None,
) -> None:
    """Bind an acceptance to the exact offer that was on the table.

    There is one offer, so there is nothing to match against except it. A price
    the fan named that is not its price is not acceptance of it and is left
    alone for the counteroffer path.
    """
    event = accepted_offer_event(events)
    if not event or active_offer is None:
        return
    if (
        event.amount_cents is not None
        and abs(active_offer.price_cents - event.amount_cents) > 100
    ):
        return
    event.amount_cents = active_offer.price_cents
    event.metadata.update(
        {
            "offer_id": active_offer.offer_id,
            "set_id": active_offer.set_id,
            "label": active_offer.label,
            "experience": active_offer.experience,
            "legal_description": active_offer.legal_description or active_offer.experience,
        }
    )


def _offer_from_event(
    event: CommercialEvent,
    active_offer: Offer | None,
) -> Offer | None:
    if active_offer is None:
        return None
    offer_id = str(event.metadata.get("offer_id") or "")
    if offer_id and offer_id != active_offer.offer_id:
        return None
    if not offer_id and event.amount_cents is not None:
        return active_offer if active_offer.price_cents == event.amount_cents else None
    return active_offer


async def consume_free_text_allowance(creator_id: str, fan_id: str) -> None:
    """Spend one of the creator's configured free explicit-text messages.

    Decoupling sexual text from the commercial decision created a real risk the
    brief called out by name: an explicit reply on a ``CONTINUE_NORMAL_CHAT``
    turn is not a ``CONTINUE_FREE_TEXT`` action, so nothing would have counted
    it, and a HYBRID_TEASER creator configured for four free messages would
    have been giving away an unbounded sexting service.

    This is what makes ``TextIntimacyDecision.consumes_free_allowance`` real. It
    writes the same counter the commercial free-text actions already write, so
    one budget governs both routes and ``free_mode_on_cooldown`` still ends the
    window.
    """
    policy = await get_creator_policy(creator_id)
    state = await get_fan_state(fan_id)
    now = datetime.now(timezone.utc)

    state.teaser_messages_used += 1
    if state.free_session_started_at is None:
        state.free_session_started_at = now
    limit = (
        policy.free_text_max_messages
        if policy.sexting_mode == SextingMode.FREE_TEXT_ALLOWED
        else policy.teaser_max_messages
    )
    if state.teaser_messages_used >= max(1, limit):
        state.free_session_ended_at = now
    await save_fan_state(fan_id, creator_id, state)
    print(
        f"[TEXT INTIMACY] fan={fan_id} free allowance "
        f"{state.teaser_messages_used}/{limit}"
    )
