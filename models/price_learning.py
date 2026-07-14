"""Deterministic, evidence-backed per-fan price learning.

The engine learns from the immutable affordability ledger. It never estimates
wealth, never invents a discount, and never treats one declined offer as a
permanent budget ceiling. The output is a recommendation over approved package
prices, not permission to create a new price.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Iterable

from pydantic import BaseModel, Field


class PriceRecommendationMode(str, Enum):
    NO_OFFER = "NO_OFFER"
    DISCOVERY = "DISCOVERY"
    RANGE = "RANGE"
    EXACT = "EXACT"


class PriceLearningConfidence(str, Enum):
    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class PriceLearningPolicy(BaseModel):
    min_offer_cents: int = Field(default=500, ge=0)
    max_offer_cents: int = Field(default=50_000, ge=100)
    first_purchase_target_cents: int = Field(default=2_500, ge=0)
    repeat_buyer_uplift_bps: int = Field(default=1_000, ge=0, le=5_000)
    vip_uplift_bps: int = Field(default=1_500, ge=0, le=7_500)
    max_step_up_bps: int = Field(default=2_500, ge=0, le=10_000)
    range_width_bps: int = Field(default=2_000, ge=0, le=7_500)
    price_step_cents: int = Field(default=500, ge=1)
    evidence_lookback_days: int = Field(default=365, ge=1, le=3650)


class PriceLearningProfile(BaseModel):
    mode: PriceRecommendationMode = PriceRecommendationMode.DISCOVERY
    confidence: PriceLearningConfidence = PriceLearningConfidence.NONE
    lifecycle_stage: str = "PROSPECT"
    recommended_floor_cents: int | None = Field(default=None, ge=0)
    recommended_target_cents: int | None = Field(default=None, ge=0)
    recommended_ceiling_cents: int | None = Field(default=None, ge=0)
    anchor_cents: int | None = Field(default=None, ge=0)
    confirmed_purchase_count: int = Field(default=0, ge=0)
    positive_signal_count: int = Field(default=0, ge=0)
    resistance_signal_count: int = Field(default=0, ge=0)
    evidence_score: float = Field(default=0.0, ge=0.0)
    evidence_summary: dict[str, Any] = Field(default_factory=dict)
    reason_codes: list[str] = Field(default_factory=list)
    state_version: int = 1
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_context(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "confidence": self.confidence.value,
            "lifecycle_stage": self.lifecycle_stage,
            "recommended_floor_cents": self.recommended_floor_cents,
            "recommended_target_cents": self.recommended_target_cents,
            "recommended_ceiling_cents": self.recommended_ceiling_cents,
            "anchor_cents": self.anchor_cents,
            "confirmed_purchase_count": self.confirmed_purchase_count,
            "positive_signal_count": self.positive_signal_count,
            "resistance_signal_count": self.resistance_signal_count,
            "evidence_score": round(self.evidence_score, 4),
            "evidence_summary": self.evidence_summary,
            "reason_codes": self.reason_codes,
            "updated_at": _as_utc(self.updated_at).isoformat(),
        }


_POSITIVE_WEIGHTS = {
    "PURCHASE_CONFIRMED": 3.0,
    "OFFER_SELECTED": 1.4,
    "COUNTEROFFER_STATED": 1.2,
    "CURRENT_LIMIT_STATED": 1.1,
    "CURRENT_AMOUNT_STATED": 1.0,
}


def derive_price_learning_profile(
    events: Iterable[dict[str, Any]],
    *,
    affordability: dict[str, Any] | None = None,
    lifecycle: dict[str, Any] | None = None,
    policy: PriceLearningPolicy | None = None,
    now: datetime | None = None,
) -> PriceLearningProfile:
    """Derive one conservative recommendation from immutable evidence."""

    policy = policy or PriceLearningPolicy()
    current = _as_utc(now) if now else datetime.now(timezone.utc)
    affordability = affordability or {}
    lifecycle = lifecycle or {}
    event_rows = list(events)
    stage = str(lifecycle.get("stage") or "PROSPECT").upper()
    reasons: list[str] = []

    if affordability.get("temporary_constraint"):
        return PriceLearningProfile(
            mode=PriceRecommendationMode.NO_OFFER,
            confidence=PriceLearningConfidence.MEDIUM,
            lifecycle_stage=stage,
            confirmed_purchase_count=int(
                affordability.get("confirmed_purchase_count") or 0
            ),
            reason_codes=["temporary_cash_constraint", "suppress_offer_now"],
            evidence_summary={"source": "affordability_snapshot"},
            updated_at=current,
        )

    selected = _money(affordability.get("latest_offer_selected_cents"))
    latest_selected_at = _latest_event_time(event_rows, "OFFER_SELECTED")
    latest_purchase_at = _latest_event_time(event_rows, "PURCHASE_CONFIRMED")
    selected_is_pending = selected is not None and (
        latest_selected_at is None
        or latest_purchase_at is None
        or latest_selected_at > latest_purchase_at
    )
    if selected_is_pending:
        exact = _clamp(selected, policy)
        return PriceLearningProfile(
            mode=PriceRecommendationMode.EXACT,
            confidence=PriceLearningConfidence.HIGH,
            lifecycle_stage=stage,
            recommended_floor_cents=exact,
            recommended_target_cents=exact,
            recommended_ceiling_cents=exact,
            anchor_cents=exact,
            confirmed_purchase_count=int(
                affordability.get("confirmed_purchase_count") or 0
            ),
            positive_signal_count=1,
            evidence_score=3.0,
            reason_codes=["selected_offer_is_authoritative", "approved_price_only"],
            evidence_summary={"selected_offer_cents": exact},
            updated_at=current,
        )

    weighted: list[tuple[int, float]] = []
    purchases: list[int] = []
    resistance: list[int] = []
    counts: dict[str, int] = {}

    for raw in event_rows:
        event_type = str(raw.get("event_type") or "").upper()
        amount = _money(raw.get("amount_cents"))
        occurred = _as_utc(raw.get("occurred_at"))
        if not event_type or amount is None or amount <= 0 or occurred is None:
            continue
        age_days = max(0.0, (current - occurred).total_seconds() / 86400)
        if age_days > min(policy.evidence_lookback_days, _event_max_age_days(event_type)):
            continue
        recency = _recency_multiplier(age_days)
        counts[event_type] = counts.get(event_type, 0) + 1
        if event_type in _POSITIVE_WEIGHTS:
            weight = _POSITIVE_WEIGHTS[event_type] * recency
            weighted.append((amount, weight))
            if event_type == "PURCHASE_CONFIRMED":
                purchases.append(amount)
        elif event_type == "OFFER_DECLINED":
            resistance.append(amount)

    current_cap_candidates = [
        _money(affordability.get("current_available_cents")),
        _money(affordability.get("current_limit_cents")),
    ]
    current_caps = [value for value in current_cap_candidates if value is not None]
    current_cap = min(current_caps) if current_caps else None

    if weighted:
        anchor = _weighted_median(weighted)
        reasons.append("evidence_weighted_anchor")
    else:
        anchor = policy.first_purchase_target_cents
        reasons.append("creator_cold_start_target")

    target = anchor
    if stage == "REPEAT_BUYER":
        target = _apply_bps(target, policy.repeat_buyer_uplift_bps)
        reasons.append("repeat_buyer_uplift")
    elif stage == "VIP":
        target = _apply_bps(target, policy.vip_uplift_bps)
        reasons.append("vip_uplift")
    elif stage in {"PROSPECT", "FIRST_PURCHASE_PROSPECT"} and not purchases:
        target = min(target, policy.first_purchase_target_cents)
        reasons.append("first_purchase_friction_control")

    if purchases:
        highest_paid = max(purchases)
        max_step = _apply_bps(highest_paid, policy.max_step_up_bps)
        target = min(target, max_step)
        reasons.append("step_up_capped_from_confirmed_purchase")

    if current_cap is not None:
        target = min(target, current_cap)
        reasons.append("current_explicit_cap_respected")

    target = _clamp_round(target, policy)
    if current_cap is not None:
        target = min(target, _clamp(current_cap, policy))
    width = policy.range_width_bps
    floor = _clamp_round(_apply_bps(target, -width), policy)
    ceiling = _clamp_round(_apply_bps(target, width), policy)

    if current_cap is not None:
        ceiling = min(ceiling, _clamp(current_cap, policy))
        target = min(target, ceiling)
        floor = min(floor, target)

    score = sum(weight for _, weight in weighted)
    confidence = _confidence(score)
    mode = (
        PriceRecommendationMode.RANGE
        if weighted or current_cap is not None
        else PriceRecommendationMode.DISCOVERY
    )

    if resistance:
        reasons.append("declines_are_soft_resistance_not_budget_ceiling")

    return PriceLearningProfile(
        mode=mode,
        confidence=confidence,
        lifecycle_stage=stage,
        recommended_floor_cents=floor,
        recommended_target_cents=target,
        recommended_ceiling_cents=max(target, ceiling),
        anchor_cents=_clamp_round(anchor, policy),
        confirmed_purchase_count=max(
            len(purchases), int(affordability.get("confirmed_purchase_count") or 0)
        ),
        positive_signal_count=len(weighted),
        resistance_signal_count=len(resistance),
        evidence_score=score,
        evidence_summary={
            "event_counts": counts,
            "current_explicit_cap_cents": current_cap,
            "highest_confirmed_purchase_cents": max(purchases) if purchases else None,
            "latest_soft_resistance_cents": resistance[-1] if resistance else None,
        },
        reason_codes=reasons,
        updated_at=current,
    )


def select_recommended_packages(
    package_options: list[Any],
    price_learning: dict[str, Any] | None,
    *,
    max_options: int = 2,
) -> list[Any]:
    """Choose the best approved packages; never create or alter a price."""

    options = list(package_options or [])
    if not options:
        return []
    context = price_learning or {}
    mode = str(context.get("mode") or "").upper()
    target = _money(context.get("recommended_target_cents"))
    floor = _money(context.get("recommended_floor_cents"))
    ceiling = _money(context.get("recommended_ceiling_cents"))
    limit = max(1, int(max_options))

    if target is None or mode in {"", "NO_OFFER"}:
        return options

    priced = [(option, _option_price(option)) for option in options]
    priced = [(option, price) for option, price in priced if price is not None]
    if not priced:
        return options[:limit]

    if mode == "EXACT":
        return [min(priced, key=lambda pair: abs(pair[1] - target))[0]]

    eligible = [
        (option, price)
        for option, price in priced
        if (floor is None or price >= floor) and (ceiling is None or price <= ceiling)
    ]
    pool = eligible or priced
    lower = sorted(
        (pair for pair in pool if pair[1] <= target),
        key=lambda pair: (target - pair[1], -pair[1]),
    )
    upper = sorted(
        (pair for pair in pool if pair[1] > target),
        key=lambda pair: (pair[1] - target, pair[1]),
    )
    selected: list[Any] = []
    for bucket in (lower, upper):
        if bucket and bucket[0][0] not in selected:
            selected.append(bucket[0][0])
    if len(selected) < limit:
        for option, _ in sorted(pool, key=lambda pair: abs(pair[1] - target)):
            if option not in selected:
                selected.append(option)
            if len(selected) >= limit:
                break
    return selected[:limit]


def profile_from_row(row: dict[str, Any] | None) -> PriceLearningProfile:
    if not row:
        return PriceLearningProfile()
    payload = dict(row)
    for key in ("fan_id", "creator_id", "created_at"):
        payload.pop(key, None)
    return PriceLearningProfile.model_validate(payload)


def _option_price(option: Any) -> int | None:
    if isinstance(option, dict):
        return _money(option.get("price_cents"))
    return _money(getattr(option, "price_cents", None))


def _latest_event_time(events: list[dict[str, Any]], event_type: str) -> datetime | None:
    values = [
        parsed
        for row in events
        if str(row.get("event_type") or "").upper() == event_type
        if (parsed := _as_utc(row.get("occurred_at"))) is not None
    ]
    return max(values) if values else None


def _event_max_age_days(event_type: str) -> int:
    return {
        "CURRENT_AMOUNT_STATED": 7,
        "CURRENT_LIMIT_STATED": 30,
        "OFFER_SELECTED": 30,
        "COUNTEROFFER_STATED": 90,
        "OFFER_DECLINED": 90,
        "PURCHASE_CONFIRMED": 3650,
    }.get(event_type, 365)


def _weighted_median(values: list[tuple[int, float]]) -> int:
    ordered = sorted(values, key=lambda item: item[0])
    total = sum(weight for _, weight in ordered)
    midpoint = total / 2
    running = 0.0
    for amount, weight in ordered:
        running += weight
        if running >= midpoint:
            return amount
    return ordered[-1][0]


def _confidence(score: float) -> PriceLearningConfidence:
    if score >= 7.0:
        return PriceLearningConfidence.HIGH
    if score >= 3.0:
        return PriceLearningConfidence.MEDIUM
    if score > 0:
        return PriceLearningConfidence.LOW
    return PriceLearningConfidence.NONE


def _recency_multiplier(age_days: float) -> float:
    if age_days <= 30:
        return 1.0
    if age_days <= 90:
        return 0.85
    if age_days <= 180:
        return 0.7
    return 0.5


def _apply_bps(value: int, bps: int) -> int:
    return max(0, int(round(value * (10_000 + bps) / 10_000)))


def _clamp(value: int, policy: PriceLearningPolicy) -> int:
    return max(policy.min_offer_cents, min(policy.max_offer_cents, int(value)))


def _clamp_round(value: int, policy: PriceLearningPolicy) -> int:
    clamped = _clamp(value, policy)
    step = max(1, policy.price_step_cents)
    rounded = int(round(clamped / step) * step)
    return _clamp(rounded, policy)


def _money(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return None


def _as_utc(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
