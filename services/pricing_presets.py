"""Agency pricing strategy, as three understandable presets over existing policy.

The backend already has a price-policy hierarchy — environment defaults, then an
agency scope, then a creator scope (db/pricing_policy_queries.py). What it did
not have was a way for an agency to express intent without knowing what a basis
point is.

A preset is a NAMED SET OF VALUES for fields that already exist. It is not a new
pricing mode, it changes no formula, and it can express nothing the policy could
not already express. Three of them:

Conservative
    A lower opening probe inside an approved content range, and slower upward
    movement once evidence exists.

Balanced
    The shipped defaults. Selecting it writes the same numbers an unconfigured
    deployment already runs on.

Aggressive
    A higher opening probe and faster evidence-backed upward movement.

What a preset cannot do
-----------------------
None of them touch the things that bound a price. Aggressive still cannot exceed
the approved content maximum for the category, cannot exceed a fan's explicitly
stated budget ceiling, and cannot move outside the deterministic rules in
models/price_learning.py and models/content_pricing.py. "Aggressive" means
"probe higher inside what is already allowed", never "allowed more".

Recognising a preset
--------------------
``preset_for_policy`` reports which preset a stored policy matches, so the UI can
show the operator's choice rather than making them re-pick it. A policy that has
been hand-tuned matches none of them and is reported as ``custom``, which is the
honest answer and the reason the Advanced controls exist.
"""

from __future__ import annotations

from typing import Any

from models.price_learning import PriceLearningPolicy


CONSERVATIVE = "conservative"
BALANCED = "balanced"
AGGRESSIVE = "aggressive"
CUSTOM = "custom"

PRESET_IDS: tuple[str, ...] = (CONSERVATIVE, BALANCED, AGGRESSIVE)

# The fields a preset owns. Everything else in PriceLearningPolicy — the
# absolute offer bounds, the evidence window, the customer price grid — is
# deployment or agency configuration that a strategy choice must not silently
# rewrite.
#
# cold_start_probe_bps  : where in an approved range a fan with no evidence is
#                         probed. 2,500 bps on a $15-$80 set is about $31.
# max_step_up_bps       : the most one confirmed step may raise the next ask.
# repeat_buyer_uplift_bps / vip_uplift_bps : lifecycle-based uplift.
# first_purchase_target_cents : where a first purchase is aimed when nothing
#                         else is known.
# effortless_purchase_streak : confirmed effortless purchases before extra
#                         uplift applies. A HIGHER number is more conservative.
PRESET_FIELDS: tuple[str, ...] = (
    "cold_start_probe_bps",
    "max_step_up_bps",
    "repeat_buyer_uplift_bps",
    "vip_uplift_bps",
    "first_purchase_target_cents",
    "effortless_purchase_streak",
)

PRESETS: dict[str, dict[str, int]] = {
    CONSERVATIVE: {
        "cold_start_probe_bps": 1_500,
        "max_step_up_bps": 1_500,
        "repeat_buyer_uplift_bps": 500,
        "vip_uplift_bps": 1_000,
        "first_purchase_target_cents": 2_000,
        "effortless_purchase_streak": 3,
    },
    # Byte-for-byte the shipped defaults, so "Balanced" is a real statement
    # about behaviour rather than a fourth set of numbers.
    BALANCED: {
        "cold_start_probe_bps": 2_500,
        "max_step_up_bps": 2_500,
        "repeat_buyer_uplift_bps": 1_000,
        "vip_uplift_bps": 1_500,
        "first_purchase_target_cents": 2_500,
        "effortless_purchase_streak": 2,
    },
    AGGRESSIVE: {
        "cold_start_probe_bps": 4_000,
        "max_step_up_bps": 4_000,
        "repeat_buyer_uplift_bps": 1_500,
        "vip_uplift_bps": 2_500,
        "first_purchase_target_cents": 3_500,
        "effortless_purchase_streak": 1,
    },
}

PRESET_LABELS: dict[str, str] = {
    CONSERVATIVE: "Conservative",
    BALANCED: "Balanced",
    AGGRESSIVE: "Aggressive",
    CUSTOM: "Custom",
}

PRESET_DESCRIPTIONS: dict[str, str] = {
    CONSERVATIVE: (
        "Opens low inside each item's approved range and raises the ask slowly, "
        "only after confirmed purchases."
    ),
    BALANCED: "The shipped default. A middling opening ask and normal upward movement.",
    AGGRESSIVE: (
        "Opens higher inside each item's approved range and raises the ask faster "
        "once a fan has bought. Still capped by approved content prices and by any "
        "budget the fan has stated."
    ),
    CUSTOM: "Hand-tuned values that do not match any preset.",
}


def is_valid_preset(value: Any) -> bool:
    return str(value or "").strip().lower() in PRESETS


def normalize_preset(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    return text if text in PRESETS else None


def preset_settings(preset: str) -> dict[str, int]:
    """The policy fields one preset writes. Raises on an unknown preset."""
    known = normalize_preset(preset)
    if known is None:
        raise ValueError(
            f"Unknown pricing preset. Expected one of: {', '.join(PRESET_IDS)}"
        )
    return dict(PRESETS[known])


def apply_preset(
    existing: dict[str, Any] | None,
    preset: str,
) -> dict[str, Any]:
    """Merge a preset over stored settings, leaving every other field alone.

    An agency that has deliberately narrowed ``max_offer_cents`` keeps that when
    it switches strategy; a preset is a statement about probing, not a reset.
    """
    merged = {
        key: value
        for key, value in (existing or {}).items()
        if key in set(PriceLearningPolicy.model_fields)
    }
    merged.update(preset_settings(preset))
    return merged


def preset_for_policy(settings: Any) -> str:
    """Which preset a stored settings dict matches, or ``custom``.

    A field the settings do not carry is inherited, so it is compared against
    the policy defaults rather than treated as a mismatch. That is what lets an
    agency that has never configured anything be correctly reported as Balanced.
    """
    if not isinstance(settings, dict):
        settings = {}
    defaults = PriceLearningPolicy()
    effective = {
        field: settings.get(field, getattr(defaults, field))
        for field in PRESET_FIELDS
    }
    for preset_id in PRESET_IDS:
        candidate = PRESETS[preset_id]
        if all(
            _as_int(effective.get(field)) == candidate[field] for field in PRESET_FIELDS
        ):
            return preset_id
    return CUSTOM


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def describe_presets() -> list[dict[str, Any]]:
    """Read-only catalog for the dashboard's strategy picker."""
    return [
        {
            "preset": preset_id,
            "label": PRESET_LABELS[preset_id],
            "description": PRESET_DESCRIPTIONS[preset_id],
            "settings": dict(PRESETS[preset_id]),
        }
        for preset_id in PRESET_IDS
    ]
