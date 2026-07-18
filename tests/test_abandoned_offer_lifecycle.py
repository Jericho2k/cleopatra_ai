from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from models.commercial import CreatorPolicy, FanCommercialState, FanStatus, PackageOption
from services import commercial_orchestrator
from services.followup_lifecycle import (
    expire_pending_offer_state,
    pending_offer_expiry_obligation,
)
from workers import scheduled_actions as worker


OFFERED_AT = datetime(2026, 7, 18, 12, 0, tzinfo=timezone.utc)


def run(coro):
    return asyncio.run(coro)


def pending_state() -> FanCommercialState:
    return FanCommercialState(
        status=FanStatus.OFFER_PENDING,
        desired_experience="shower",
        last_offer_at=OFFERED_AT,
        offered_packages=[
            PackageOption(
                package_id="pkg-quick",
                label="quick shower",
                price_cents=4500,
                set_ids=["set-shower-1"],
                experience="shower, wet, teasing",
            ),
            PackageOption(
                package_id="pkg-full",
                label="full shower",
                price_cents=7000,
                set_ids=["set-shower-1", "set-shower-2"],
                experience="shower, wet, explicit progression",
            ),
        ],
    )


def test_pending_offer_expiry_preserves_exact_approved_snapshot():
    obligation = pending_offer_expiry_obligation(
        pending_state(),
        policy=CreatorPolicy(pending_offer_expiry_hours=24),
        fan_id="fan-1",
    )

    assert obligation is not None
    assert obligation.action_type == "OFFER_EXPIRY"
    assert obligation.execute_at == datetime(2026, 7, 19, 12, 0, tzinfo=timezone.utc)
    assert [p["package_id"] for p in obligation.payload["offered_packages"]] == [
        "pkg-quick",
        "pkg-full",
    ]
    assert obligation.payload["primary_experience"] == "shower, wet, teasing"
    assert "set-shower-1" in obligation.payload["offered_packages"][0]["set_ids"]


def test_exact_pending_offer_expires_then_schedules_followup():
    state = pending_state()
    expiry = pending_offer_expiry_obligation(
        state,
        policy=CreatorPolicy(),
        fan_id="fan-1",
    )
    expired_at = datetime(2026, 7, 19, 12, 0, tzinfo=timezone.utc)

    expired, followup, changed = expire_pending_offer_state(
        state,
        payload=expiry.payload,
        policy=CreatorPolicy(abandoned_offer_followup_delay_hours=18),
        fan_id="fan-1",
        now=expired_at,
    )

    assert changed is True
    assert expired.status == FanStatus.IDLE
    assert expired.offered_packages == []
    assert expired.last_offer_at == OFFERED_AT
    assert expired.next_followup_type == "ABANDONED_OFFER_FOLLOWUP"
    assert followup is not None
    assert followup.execute_at == datetime(2026, 7, 20, 6, 0, tzinfo=timezone.utc)
    assert followup.payload["primary_experience"] == "shower, wet, teasing"
    assert [p["package_id"] for p in followup.payload["offered_packages"]] == [
        "pkg-quick",
        "pkg-full",
    ]


def test_stale_expiry_cannot_clear_a_newer_offer():
    old = pending_state()
    expiry = pending_offer_expiry_obligation(old, policy=CreatorPolicy(), fan_id="fan-1")
    newer = pending_state()
    newer.last_offer_at = datetime(2026, 7, 18, 14, 0, tzinfo=timezone.utc)

    output, followup, changed = expire_pending_offer_state(
        newer,
        payload=expiry.payload,
        policy=CreatorPolicy(),
        fan_id="fan-1",
        now=datetime(2026, 7, 19, 12, 0, tzinfo=timezone.utc),
    )

    assert changed is False
    assert followup is None
    assert output.status == FanStatus.OFFER_PENDING
    assert len(output.offered_packages) == 2


def test_fan_return_cancels_abandoned_offer_followup(monkeypatch):
    state = FanCommercialState(
        status=FanStatus.IDLE,
        last_offer_at=OFFERED_AT,
        next_followup_at=datetime(2026, 7, 20, 6, 0, tzinfo=timezone.utc),
        next_followup_type="ABANDONED_OFFER_FOLLOWUP",
        next_followup_payload={"primary_experience": "shower"},
        next_followup_dedupe_key="abandoned-offer:fan-1:ref",
    )
    cancelled = []
    saved = []
    monkeypatch.setattr(
        commercial_orchestrator,
        "get_fan_state",
        lambda _fan_id: async_value(state),
    )
    monkeypatch.setattr(
        commercial_orchestrator,
        "cancel_actions_for_fan",
        lambda fan_id, action_type=None: async_append(cancelled, (fan_id, action_type)),
    )
    monkeypatch.setattr(
        commercial_orchestrator,
        "save_fan_state",
        lambda fan_id, creator_id, value: async_append(saved, (fan_id, creator_id, value)),
    )

    run(commercial_orchestrator.acknowledge_fan_return("creator-1", "fan-1"))

    assert cancelled == [("fan-1", "ABANDONED_OFFER_FOLLOWUP")]
    assert saved[0][2].next_followup_type is None


def test_fan_return_refreshes_still_pending_offer_expiry(monkeypatch):
    state = pending_state()
    old_expiry = pending_offer_expiry_obligation(state, policy=CreatorPolicy(), fan_id="fan-1")
    state.next_followup_at = old_expiry.execute_at
    state.next_followup_type = old_expiry.action_type
    state.next_followup_payload = old_expiry.payload
    state.next_followup_dedupe_key = old_expiry.dedupe_key
    returned_at = datetime(2026, 7, 18, 18, 0, tzinfo=timezone.utc)
    cancelled = []
    scheduled = []
    saved = []
    monkeypatch.setattr(commercial_orchestrator, "get_fan_state", lambda _fan_id: async_value(state))
    monkeypatch.setattr(
        commercial_orchestrator,
        "get_creator_policy",
        lambda _creator_id: async_value(CreatorPolicy(pending_offer_expiry_hours=24)),
    )
    monkeypatch.setattr(
        commercial_orchestrator,
        "cancel_actions_for_fan",
        lambda fan_id, action_type=None: async_append(cancelled, (fan_id, action_type)),
    )
    monkeypatch.setattr(
        commercial_orchestrator,
        "schedule_action",
        lambda **kwargs: async_append(scheduled, kwargs),
    )
    monkeypatch.setattr(
        commercial_orchestrator,
        "save_fan_state",
        lambda fan_id, creator_id, value: async_append(saved, (fan_id, creator_id, value)),
    )

    run(commercial_orchestrator.acknowledge_fan_return(
        "creator-1",
        "fan-1",
        now=returned_at,
    ))

    assert cancelled == [("fan-1", "OFFER_EXPIRY")]
    assert scheduled[0]["action_type"] == "OFFER_EXPIRY"
    assert scheduled[0]["execute_at"] == datetime(2026, 7, 19, 18, 0, tzinfo=timezone.utc)
    assert saved[0][2].last_offer_at == returned_at
    assert saved[0][2].offered_packages[0].package_id == "pkg-quick"


def test_offer_expiry_worker_persists_followup_before_queue_repair(monkeypatch):
    state = pending_state()
    expiry = pending_offer_expiry_obligation(state, policy=CreatorPolicy(), fan_id="fan-1")
    action = {
        "creator_id": "creator-1",
        "fan_id": "fan-1",
        "payload": expiry.payload,
    }
    saved = []
    scheduled = []
    monkeypatch.setattr(worker, "get_fan_state", lambda _fan_id: async_value(state))
    monkeypatch.setattr(worker, "get_creator_policy", lambda _creator_id: async_value(CreatorPolicy()))
    monkeypatch.setattr(
        worker,
        "save_fan_state",
        lambda fan_id, creator_id, value: async_append(saved, (fan_id, creator_id, value)),
    )
    monkeypatch.setattr(worker, "schedule_action", lambda **kwargs: async_append(scheduled, kwargs))

    result = run(worker._run_offer_expiry(action))

    assert result.sent_message is False
    assert saved[0][2].status == FanStatus.IDLE
    assert saved[0][2].next_followup_type == "ABANDONED_OFFER_FOLLOWUP"
    assert scheduled[0]["action_type"] == "ABANDONED_OFFER_FOLLOWUP"


def test_abandoned_offer_goal_cannot_invent_or_claim_selection(monkeypatch):
    captured = []
    monkeypatch.setattr(
        worker,
        "_send_goal",
        lambda _action, goal: capture_handler_result(captured, goal),
    )
    result = run(
        worker._run_abandoned_offer_followup(
            {"payload": {"primary_experience": "shower, wet, teasing"}}
        )
    )

    assert result.sent_message is True
    assert "shower, wet, teasing" in captured[0]
    assert "without claiming he selected it" in captured[0]
    assert "reference only that approved" in captured[0]


async def async_value(value):
    return value


async def async_append(target, value):
    target.append(value)


async def capture_handler_result(target, value):
    target.append(value)
    return worker.HandlerResult(sent_message=True)
