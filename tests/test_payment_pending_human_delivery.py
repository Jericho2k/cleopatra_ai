from __future__ import annotations

import inspect
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ai import prompt_builder
from models.commercial import (
    ActionType,
    CommercialEvent,
    CreatorPolicy,
    EventType,
    FanCommercialState,
    FanStatus,
    PackageOption,
)
from models.conversation_director import (
    ConversationPhase,
    DirectorAction,
    advance_conversation_director,
)
from models.session_strategy import SessionGoal, derive_session_strategy
from services import suggestions
from services.commercial_policy import CommercialContext, decide_next_action
from services.human_delivery import AvailabilityMode, build_delivery_schedule


def selected_event() -> CommercialEvent:
    return CommercialEvent(
        type=EventType.PACKAGE_SELECTED,
        amount_cents=3000,
        metadata={"package_id": "pkg", "set_id": "set-1", "set_ids": ["set-1"]},
    )


def package() -> PackageOption:
    return PackageOption(
        package_id="pkg",
        label="quick private session",
        price_cents=3000,
        set_id="set-1",
        set_ids=["set-1"],
        experience="shower",
        legal_description="shower",
    )


def test_selection_enters_offer_selected_not_paid_session():
    decision = decide_next_action(
        CreatorPolicy(),
        FanCommercialState(status=FanStatus.OFFER_PENDING, offered_packages=[package()]),
        [selected_event()],
        CommercialContext(package_options=[package()]),
    )
    assert decision.action == ActionType.CREATE_PAID_SESSION
    assert decision.new_status == FanStatus.OFFER_SELECTED


def test_payment_pending_holds_without_claiming_content_was_seen():
    decision = decide_next_action(
        CreatorPolicy(),
        FanCommercialState(status=FanStatus.PAYMENT_PENDING),
        [],
        CommercialContext(
            session_exists=True,
            session_has_pending_purchase=True,
            session_has_remaining_steps=True,
        ),
    )
    assert decision.action == ActionType.CONTINUE_NORMAL_CHAT
    assert decision.new_status == FanStatus.PAYMENT_PENDING
    assert decision.must_not_send_media is True
    assert "not imply" in decision.goal


def test_legacy_paid_session_with_pending_unlock_self_heals_to_payment_pending():
    decision = decide_next_action(
        CreatorPolicy(),
        FanCommercialState(status=FanStatus.PAID_SESSION_ACTIVE),
        [],
        CommercialContext(
            session_exists=True,
            session_has_pending_purchase=True,
            session_has_remaining_steps=True,
        ),
    )
    assert decision.action == ActionType.CONTINUE_NORMAL_CHAT
    assert decision.new_status == FanStatus.PAYMENT_PENDING
    assert decision.must_not_send_media is True


def test_offer_selected_with_unsent_session_recovers_exact_locked_step():
    decision = decide_next_action(
        CreatorPolicy(),
        FanCommercialState(status=FanStatus.OFFER_SELECTED),
        [],
        CommercialContext(
            session_exists=True,
            session_has_pending_purchase=False,
            session_has_remaining_steps=True,
        ),
    )
    assert decision.action == ActionType.SEND_NEXT_PPV_STEP
    assert decision.new_status == FanStatus.OFFER_SELECTED
    assert decision.must_not_send_media is False


def test_create_session_strategy_is_not_misclassified_as_delivery():
    strategy = derive_session_strategy(
        commercial_decision={"action": "CREATE_PAID_SESSION", "package_options": [package().model_dump()]},
        active_session={"status": "active", "current_index": 0, "awaiting_purchase_index": None, "plan": [{}]},
    )
    assert strategy.phase == "OFFER_SELECTED"
    assert strategy.goal == SessionGoal.CLOSE


def test_recovered_first_locked_step_is_not_called_paid_delivery():
    strategy = derive_session_strategy(
        commercial_decision={
            "action": "SEND_NEXT_PPV_STEP",
            "new_status": "OFFER_SELECTED",
        },
        active_session={
            "status": "active",
            "current_index": 0,
            "awaiting_purchase_index": None,
            "plan": [{}],
        },
    )
    assert strategy.phase == "OFFER_SELECTED"
    assert strategy.goal == SessionGoal.CLOSE


def test_awaiting_unlock_strategy_is_hold_not_delivery():
    strategy = derive_session_strategy(
        commercial_decision={"action": "CONTINUE_NORMAL_CHAT"},
        active_session={"status": "active", "current_index": 0, "awaiting_purchase_index": 0, "plan": [{}]},
    )
    assert strategy.phase == "PAYMENT_PENDING"
    assert strategy.goal == SessionGoal.HOLD
    assert "unseen" in " ".join(strategy.writer_avoid)


def test_director_calls_selection_and_pending_unlock_payment_pending():
    selection = advance_conversation_director(
        commercial_decision={"action": "CREATE_PAID_SESSION"},
        active_session={"status": "active", "awaiting_purchase_index": None},
        fan_turn_count=4,
        creator_turn_count=3,
    )
    assert selection.phase == ConversationPhase.PAYMENT_PENDING
    assert selection.action == DirectorAction.WAIT_FOR_PAYMENT

    waiting = advance_conversation_director(
        commercial_decision={"action": "CONTINUE_NORMAL_CHAT"},
        active_session={"status": "active", "awaiting_purchase_index": 0},
        fan_turn_count=5,
        creator_turn_count=4,
    )
    assert waiting.phase == ConversationPhase.PAYMENT_PENDING
    assert waiting.action == DirectorAction.WAIT_FOR_PAYMENT


def _timed_history(creator_gap: timedelta, *, creator_messages: int = 1):
    now = datetime(2026, 8, 4, 12, 0, tzinfo=timezone.utc)
    history = []
    for offset in range(creator_messages):
        history.append(
            {
                "role": "creator",
                "content": "previous",
                "sent_at": now - creator_gap - timedelta(minutes=offset),
            }
        )
    history.append({"role": "fan", "content": "hey", "sent_at": now})
    return history


def test_live_delivery_timing_is_jittered_bounded_and_length_aware():
    history = _timed_history(timedelta(minutes=2))
    short = build_delivery_schedule(
        "hey", ["hey :)"], conversation_history=history, rng=random.Random(7)
    )
    long = build_delivery_schedule(
        "I had a really long day and wanted to tell you what happened",
        ["aw okay tell me, what happened? I wanna hear it"],
        conversation_history=history,
        rng=random.Random(7),
    )
    assert short.availability_mode == AvailabilityMode.LIVE
    assert 0.5 <= short.availability_delay_seconds <= 8.0
    assert 2.5 <= short.composition_delay_seconds <= 22.0
    assert long.initial_delay_seconds > short.initial_delay_seconds

    split = build_delivery_schedule(
        "hi",
        ["first bubble", "a much longer second bubble"],
        conversation_history=history,
        rng=random.Random(3),
    )
    assert len(split.inter_part_delays_seconds) == 1
    assert 1.5 <= split.inter_part_delays_seconds[0] <= 14.0


def test_timing_modes_follow_real_conversation_cadence():
    assert build_delivery_schedule(
        "hey",
        ["hi"],
        conversation_history=_timed_history(timedelta(minutes=10)),
        conversation_phase="TENSION",
        rng=random.Random(1),
    ).availability_mode == AvailabilityMode.INTIMATE
    assert build_delivery_schedule(
        "hey",
        ["hi"],
        conversation_history=_timed_history(timedelta(minutes=10)),
        rng=random.Random(1),
    ).availability_mode == AvailabilityMode.WARM
    assert build_delivery_schedule(
        "hey",
        ["hi"],
        conversation_history=_timed_history(timedelta(hours=2)),
        rng=random.Random(1),
    ).availability_mode == AvailabilityMode.CASUAL
    assert build_delivery_schedule(
        "hey",
        ["hi"],
        conversation_history=_timed_history(timedelta(days=1)),
        rng=random.Random(1),
    ).availability_mode == AvailabilityMode.RETURNING
    assert build_delivery_schedule(
        "hey",
        ["hi"],
        conversation_history=[
            {
                "role": "fan",
                "content": "first message",
                "sent_at": datetime(2026, 8, 4, 12, 0, tzinfo=timezone.utc),
            }
        ],
        rng=random.Random(1),
    ).availability_mode == AvailabilityMode.NEW


def test_ordinary_chat_can_take_minutes_but_live_chat_stays_responsive():
    live = build_delivery_schedule(
        "hey",
        ["hi"],
        conversation_history=_timed_history(timedelta(minutes=1)),
        rng=random.Random(2),
    )
    casual = build_delivery_schedule(
        "hey",
        ["hi"],
        conversation_history=_timed_history(timedelta(hours=2)),
        rng=random.Random(2),
    )
    returning = build_delivery_schedule(
        "hey",
        ["hi"],
        conversation_history=_timed_history(timedelta(days=1)),
        rng=random.Random(2),
    )

    assert live.availability_delay_seconds <= 8.0
    assert 600.0 <= casual.availability_delay_seconds <= 1200.0
    assert 1800.0 <= returning.availability_delay_seconds <= 2400.0


def test_typing_indicator_starts_after_availability_pause():
    source = inspect.getsource(suggestions._debounced_auto_reply)
    availability_wait = source.index('phase="availability"')
    typing_request = source.index('/typing"')
    composition_wait = source.index('phase="before_part_1"')
    assert availability_wait < typing_request < composition_wait


def test_stale_auto_reply_cannot_clear_newer_task(monkeypatch):
    stale_task = object()
    newer_task = object()
    suggestions._pending_auto_replies["fan-1"] = newer_task
    monkeypatch.setattr(suggestions.asyncio, "current_task", lambda: stale_task)

    assert suggestions._release_auto_reply_slot("fan-1") is False
    assert suggestions._pending_auto_replies["fan-1"] is newer_task

    monkeypatch.setattr(suggestions.asyncio, "current_task", lambda: newer_task)
    assert suggestions._release_auto_reply_slot("fan-1") is True
    assert "fan-1" not in suggestions._pending_auto_replies


def test_prompt_has_narrow_anti_witty_and_bounded_knowledge_rules():
    source = inspect.getsource(prompt_builder.build_prompt)
    assert "Do not try to land a clever line" in source
    assert "ordinary young woman with uneven knowledge" in source
    assert "ask him to explain" in source


def test_reaction_prompt_is_purchase_gated_in_send_orchestration():
    source = (
        Path(__file__).resolve().parents[1] / "services" / "suggestions.py"
    ).read_text(encoding="utf-8")
    assert "_send_reaction_fishing" not in source
    assert 'action_type="POST_PURCHASE_REACTION"' in source
    assert '"_delivery": {"text": line}' in source
    assert "if creator_id and not already_recorded:" in source
    persistence = (
        Path(__file__).resolve().parents[1] / "services" / "ppv_persistence.py"
    ).read_text(encoding="utf-8")
    assert "persist_ppv_reconciliation(" in source
    assert "state.status = FanStatus.PAYMENT_PENDING" in persistence
    assert 'action_type="PPV_RECONCILE"' in persistence
    assert "response_body = await send_apifansly_message(" in source
    assert "platform accepted PPV but did not return a message ID" in source
    assert source.index("response_body = await send_apifansly_message(") < source.index(
        "persist_ppv_reconciliation("
    )
