from datetime import datetime, timedelta, timezone

from models.price_learning import (
    PriceLearningConfidence,
    PriceRecommendationMode,
    derive_price_learning_profile,
)


def event(event_type: str, cents: int, *, days_ago: int = 0) -> dict:
    now = datetime(2026, 7, 14, tzinfo=timezone.utc)
    return {
        "event_type": event_type,
        "amount_cents": cents,
        "occurred_at": (now - timedelta(days=days_ago)).isoformat(),
    }


def test_selected_offer_is_exact_and_authoritative():
    profile = derive_price_learning_profile(
        [event("PURCHASE_CONFIRMED", 6000)],
        affordability={"latest_offer_selected_cents": 2800},
        lifecycle={"stage": "REPEAT_BUYER"},
        now=datetime(2026, 7, 14, tzinfo=timezone.utc),
    )
    assert profile.mode == PriceRecommendationMode.EXACT
    assert profile.recommended_target_cents == 2800
    assert profile.recommended_floor_cents == profile.recommended_ceiling_cents


def test_temporary_constraint_suppresses_offer():
    profile = derive_price_learning_profile(
        [event("PURCHASE_CONFIRMED", 6000)],
        affordability={"temporary_constraint": True},
        lifecycle={"stage": "VIP"},
        now=datetime(2026, 7, 14, tzinfo=timezone.utc),
    )
    assert profile.mode == PriceRecommendationMode.NO_OFFER
    assert profile.recommended_target_cents is None


def test_confirmed_purchases_outweigh_soft_decline():
    profile = derive_price_learning_profile(
        [
            event("PURCHASE_CONFIRMED", 4000),
            event("PURCHASE_CONFIRMED", 5000),
            event("OFFER_DECLINED", 6000),
        ],
        affordability={},
        lifecycle={"stage": "REPEAT_BUYER"},
        now=datetime(2026, 7, 14, tzinfo=timezone.utc),
    )
    assert profile.mode == PriceRecommendationMode.RANGE
    assert profile.recommended_target_cents >= 4000
    assert "declines_are_soft_resistance_not_budget_ceiling" in profile.reason_codes
    assert profile.confidence in {
        PriceLearningConfidence.MEDIUM,
        PriceLearningConfidence.HIGH,
    }


def test_current_limit_caps_recommendation_without_becoming_permanent_wealth():
    profile = derive_price_learning_profile(
        [event("PURCHASE_CONFIRMED", 6000)],
        affordability={"current_limit_cents": 2800},
        lifecycle={"stage": "VIP"},
        now=datetime(2026, 7, 14, tzinfo=timezone.utc),
    )
    assert profile.recommended_target_cents <= 2800
    assert profile.recommended_ceiling_cents <= 2800
    assert "current_explicit_cap_respected" in profile.reason_codes


def test_no_evidence_uses_discovery_not_fake_high_confidence():
    profile = derive_price_learning_profile(
        [],
        affordability={},
        lifecycle={"stage": "PROSPECT"},
        now=datetime(2026, 7, 14, tzinfo=timezone.utc),
    )
    assert profile.mode == PriceRecommendationMode.DISCOVERY
    assert profile.confidence == PriceLearningConfidence.NONE


def test_purchase_after_selection_clears_exact_pending_mode():
    now = datetime(2026, 7, 14, tzinfo=timezone.utc)
    profile = derive_price_learning_profile(
        [
            {
                "event_type": "OFFER_SELECTED",
                "amount_cents": 2800,
                "occurred_at": (now - timedelta(minutes=5)).isoformat(),
            },
            {
                "event_type": "PURCHASE_CONFIRMED",
                "amount_cents": 2800,
                "occurred_at": now.isoformat(),
            },
        ],
        affordability={
            "latest_offer_selected_cents": 2800,
            "confirmed_purchase_count": 1,
        },
        lifecycle={"stage": "FIRST_TIME_BUYER"},
        now=now,
    )
    assert profile.mode != PriceRecommendationMode.EXACT
    assert profile.confirmed_purchase_count == 1
