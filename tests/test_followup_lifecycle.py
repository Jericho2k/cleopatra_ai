from datetime import datetime, timezone

from models.commercial import CreatorPolicy, FanCommercialState, FanStatus, PackageOption
from services.followup_lifecycle import (
    abandoned_ppv_payload,
    complete_session_state,
    followup_at,
    next_awake_time,
    next_reconcile_at,
    payment_expires_at,
    pending_reference,
    post_session_payload,
)


NOW = datetime(2026, 7, 18, 12, 0, tzinfo=timezone.utc)


def pending(**overrides):
    value = {
        "media_id": "media-1",
        "set_id": "set-1",
        "step_index": 0,
        "price": 45,
        "sent_at": NOW.isoformat(),
    }
    value.update(overrides)
    return value


def test_followup_times_are_deterministic_and_utc():
    assert followup_at(NOW, 18) == datetime(2026, 7, 19, 6, 0, tzinfo=timezone.utc)
    assert followup_at("2026-07-18T12:00:00", 1).tzinfo == timezone.utc


def test_followup_moves_out_of_overnight_sleep_window():
    during_sleep = datetime(2026, 7, 18, 23, 30, tzinfo=timezone.utc)
    assert next_awake_time(
        during_sleep,
        sleep_start_hour=23,
        sleep_end_hour=7,
        timezone_name="UTC",
    ) == datetime(2026, 7, 19, 7, 0, tzinfo=timezone.utc)
    awake = datetime(2026, 7, 18, 12, 0, tzinfo=timezone.utc)
    assert next_awake_time(
        awake,
        sleep_start_hour=23,
        sleep_end_hour=7,
        timezone_name="UTC",
    ) == awake


def test_pending_payment_uses_explicit_expiry_or_policy_window():
    explicit = pending(expires_at="2026-07-20T12:00:00+00:00")
    assert payment_expires_at(explicit, payment_window_hours=24) == datetime(
        2026, 7, 20, 12, 0, tzinfo=timezone.utc
    )
    assert payment_expires_at(pending(), payment_window_hours=24) == datetime(
        2026, 7, 19, 12, 0, tzinfo=timezone.utc
    )


def test_reconcile_never_schedules_after_expiry():
    expiry = datetime(2026, 7, 18, 12, 12, tzinfo=timezone.utc)
    assert next_reconcile_at(NOW, expires_at=expiry, recheck_minutes=20) == expiry


def test_pending_reference_is_stable_for_legacy_records():
    first = pending_reference(pending())
    assert first == pending_reference(pending())
    assert first != pending_reference(pending(media_id="media-2"))
    assert pending_reference(pending(reference="platform-ref")) == "platform-ref"


def test_followup_payloads_preserve_only_the_authoritative_snapshot():
    session_payload = post_session_payload(
        {
            "completed_at": NOW.isoformat(),
            "commercial_package_id": "pkg-1",
            "set_ids": ["set-1", "set-2"],
            "scene_key": "shower",
            "revenue_cents": 7000,
            "plan": [{}, {}],
        },
        buyer_stage="REPEAT_BUYER",
    )
    assert session_payload["experience"] == "shower"
    assert session_payload["revenue_cents"] == 7000
    assert session_payload["buyer_stage"] == "REPEAT_BUYER"

    abandoned = abandoned_ppv_payload(
        pending(),
        desired_experience="shower",
        selected_package_id="pkg-1",
    )
    assert abandoned["price_cents"] == 4500
    assert abandoned["desired_experience"] == "shower"


def test_completed_session_clears_active_offer_and_persists_followup_obligation():
    state = FanCommercialState(
        status=FanStatus.PAID_SESSION_ACTIVE,
        desired_experience="shower",
        confirmed_budget_cents=7000,
        selected_package_id="pkg-1",
        selected_package_set_ids=["set-1", "set-2"],
        offered_packages=[
            PackageOption(package_id="pkg-1", label="full", price_cents=7000)
        ],
    )
    completed, obligation = complete_session_state(
        state,
        {
            "completed_at": NOW.isoformat(),
            "commercial_package_id": "pkg-1",
            "set_ids": ["set-1", "set-2"],
            "scene_key": "ignored because desired experience is authoritative",
            "revenue_cents": 7000,
            "plan": [{}, {}],
        },
        policy=CreatorPolicy(post_session_followup_delay_hours=18),
        fan_id="fan-1",
        buyer_stage="FIRST_TIME_BUYER",
    )
    assert completed.status == FanStatus.IDLE
    assert completed.confirmed_budget_cents is None
    assert completed.offered_packages == []
    assert completed.last_session_experience == "shower"
    assert completed.next_followup_type == "POST_SESSION_FOLLOWUP"
    assert obligation is not None
    assert obligation.execute_at == datetime(2026, 7, 19, 6, 0, tzinfo=timezone.utc)
    assert obligation.payload["buyer_stage"] == "FIRST_TIME_BUYER"


def test_completed_session_can_disable_followup_without_losing_history():
    completed, obligation = complete_session_state(
        FanCommercialState(status=FanStatus.PAID_SESSION_ACTIVE),
        {"completed_at": NOW.isoformat(), "revenue_cents": 2500},
        policy=CreatorPolicy(post_session_followup_enabled=False),
        fan_id="fan-1",
        buyer_stage="FIRST_TIME_BUYER",
    )
    assert obligation is None
    assert completed.next_followup_at is None
    assert completed.last_session_revenue_cents == 2500
