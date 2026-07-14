from datetime import datetime, timedelta, timezone

from models.affordability import (
    AffordabilityEvent,
    AffordabilityEventType,
    AffordabilityState,
    AffordabilityStatus,
    apply_affordability_event,
)


def test_selected_28_plus_payday_is_not_unavailable():
    now = datetime(2026, 7, 14, 12, tzinfo=timezone.utc)
    state = AffordabilityState()
    state = apply_affordability_event(
        state,
        AffordabilityEvent(
            event_type=AffordabilityEventType.OFFER_SELECTED,
            amount_cents=2800,
            occurred_at=now,
        ),
    )
    state = apply_affordability_event(
        state,
        AffordabilityEvent(
            event_type=AffordabilityEventType.CURRENT_LIMIT_STATED,
            amount_cents=2800,
            occurred_at=now,
        ),
    )
    state = apply_affordability_event(
        state,
        AffordabilityEvent(
            event_type=AffordabilityEventType.PAYDAY_MENTIONED,
            raw_expression="Friday",
            occurred_at=now,
            metadata={"payday_at": "2026-07-17T18:00:00+00:00"},
        ),
    )

    assert state.status == AffordabilityStatus.LIMITED_NOW
    assert state.latest_offer_selected_cents == 2800
    assert state.current_limit_cents == 2800
    assert state.temporary_constraint is False
    assert state.payday_raw == "Friday"


def test_cannot_afford_until_payday_is_temporary_constraint():
    now = datetime(2026, 7, 14, 12, tzinfo=timezone.utc)
    state = apply_affordability_event(
        AffordabilityState(),
        AffordabilityEvent(
            event_type=AffordabilityEventType.MONEY_UNAVAILABLE,
            occurred_at=now,
        ),
    )
    state = apply_affordability_event(
        state,
        AffordabilityEvent(
            event_type=AffordabilityEventType.PAYDAY_MENTIONED,
            raw_expression="Friday",
            occurred_at=now,
            metadata={"payday_at": "2026-07-17T18:00:00+00:00"},
        ),
    )

    assert state.status == AffordabilityStatus.TEMPORARILY_UNAVAILABLE
    assert state.constraint_until == datetime(2026, 7, 17, 18, tzinfo=timezone.utc)


def test_plain_decline_does_not_create_poverty_signal():
    state = apply_affordability_event(
        AffordabilityState(),
        AffordabilityEvent(
            event_type=AffordabilityEventType.OFFER_DECLINED,
            amount_cents=6000,
        ),
    )
    assert state.status == AffordabilityStatus.UNKNOWN
    assert state.temporary_constraint is False
    assert state.latest_rejected_price_cents == 6000


def test_confirmed_purchase_is_history_not_budget():
    state = apply_affordability_event(
        AffordabilityState(),
        AffordabilityEvent(
            event_type=AffordabilityEventType.PURCHASE_CONFIRMED,
            amount_cents=4000,
        ),
    )
    assert state.confirmed_purchase_count == 1
    assert state.highest_confirmed_purchase_cents == 4000
    assert state.current_limit_cents is None
    assert state.current_available_cents is None


def test_current_signal_expires_without_deleting_history():
    now = datetime(2026, 7, 14, 12, tzinfo=timezone.utc)
    state = apply_affordability_event(
        AffordabilityState(),
        AffordabilityEvent(
            event_type=AffordabilityEventType.COUNTEROFFER_STATED,
            amount_cents=2500,
            occurred_at=now,
        ),
        current_signal_ttl_hours=24,
    )
    normalized = state.normalized(now=now + timedelta(hours=25))
    assert normalized.current_limit_cents is None
    assert normalized.latest_counteroffer_cents == 2500
