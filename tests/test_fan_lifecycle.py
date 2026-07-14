from datetime import datetime, timedelta, timezone

from models.fan_lifecycle import (
    BuyerLifecycleStage,
    LifecycleInputs,
    LifecyclePolicy,
    derive_lifecycle,
)


NOW = datetime(2026, 7, 14, 12, 0, tzinfo=timezone.utc)
POLICY = LifecyclePolicy(
    vip_spend_cents=50_000,
    vip_purchase_count=5,
    repeat_buyer_purchase_count=2,
    first_purchase_intent_ttl_hours=72,
)


def test_no_purchase_is_prospect():
    result = derive_lifecycle(LifecycleInputs(now=NOW), POLICY)
    assert result.stage == BuyerLifecycleStage.PROSPECT
    assert result.reason_codes == ["no_confirmed_purchase"]


def test_explicit_purchase_intent_creates_first_purchase_prospect():
    result = derive_lifecycle(
        LifecycleInputs(now=NOW, purchase_intent_signal=True),
        POLICY,
    )
    assert result.stage == BuyerLifecycleStage.FIRST_PURCHASE_PROSPECT
    assert result.intent_expires_at == NOW + timedelta(hours=72)


def test_intent_persists_until_ttl_without_requiring_repeat_signal():
    result = derive_lifecycle(
        LifecycleInputs(
            now=NOW,
            existing_intent_expires_at=NOW + timedelta(hours=2),
        ),
        POLICY,
    )
    assert result.stage == BuyerLifecycleStage.FIRST_PURCHASE_PROSPECT


def test_expired_intent_returns_to_prospect():
    result = derive_lifecycle(
        LifecycleInputs(
            now=NOW,
            existing_intent_expires_at=NOW - timedelta(seconds=1),
        ),
        POLICY,
    )
    assert result.stage == BuyerLifecycleStage.PROSPECT
    assert result.intent_expires_at is None


def test_one_confirmed_purchase_is_first_time_buyer():
    result = derive_lifecycle(
        LifecycleInputs(
            now=NOW,
            purchase_count=1,
            purchase_revenue_cents=2_500,
        ),
        POLICY,
    )
    assert result.stage == BuyerLifecycleStage.FIRST_TIME_BUYER
    assert result.intent_expires_at is None


def test_two_confirmed_purchases_is_repeat_buyer():
    result = derive_lifecycle(
        LifecycleInputs(
            now=NOW,
            purchase_count=2,
            purchase_revenue_cents=5_000,
        ),
        POLICY,
    )
    assert result.stage == BuyerLifecycleStage.REPEAT_BUYER


def test_vip_by_total_spend_overrides_purchase_stage():
    result = derive_lifecycle(
        LifecycleInputs(
            now=NOW,
            purchase_count=1,
            purchase_revenue_cents=2_500,
            fan_total_spent_cents=70_000,
        ),
        POLICY,
    )
    assert result.stage == BuyerLifecycleStage.VIP
    assert "vip_spend_threshold" in result.reason_codes


def test_vip_by_frequency():
    result = derive_lifecycle(
        LifecycleInputs(
            now=NOW,
            purchase_count=5,
            purchase_revenue_cents=20_000,
        ),
        POLICY,
    )
    assert result.stage == BuyerLifecycleStage.VIP
    assert "vip_purchase_frequency" in result.reason_codes


def test_operational_flags_do_not_corrupt_buyer_stage():
    result = derive_lifecycle(
        LifecycleInputs(
            now=NOW,
            purchase_count=2,
            sales_paused=True,
            needs_human_review=True,
            active_paid_session=True,
        ),
        POLICY,
    )
    assert result.stage == BuyerLifecycleStage.REPEAT_BUYER
    assert result.flags == {
        "active_paid_session": True,
        "sales_paused": True,
        "needs_human_review": True,
    }
