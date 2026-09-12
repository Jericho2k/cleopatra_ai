"""Pricing strategy: named presets over the policy hierarchy that already exists.

A preset is a set of values for fields the policy already has. It introduces no
pricing mode and changes no formula, and — the thing worth asserting — nothing
it can express lets an offer exceed what the content is approved for or what the
fan has said he can spend.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from db import pricing_policy_queries
from db.pricing_policy_queries import (
    environment_price_learning_policy,
    get_effective_price_learning_policy,
    save_policy_scope_settings,
    validate_policy_settings,
)
from models.content_pricing import category_range_cents
from models.price_learning import PriceLearningPolicy
from services.pricing_presets import (
    AGGRESSIVE,
    BALANCED,
    CONSERVATIVE,
    CUSTOM,
    PRESET_FIELDS,
    PRESETS,
    apply_preset,
    describe_presets,
    normalize_preset,
    preset_for_policy,
    preset_settings,
)


# --- the presets themselves -------------------------------------------------


def test_balanced_is_exactly_the_shipped_defaults():
    """Otherwise "Balanced" is a fourth set of numbers rather than a statement
    about the behaviour an unconfigured deployment already has."""
    defaults = PriceLearningPolicy()

    for field in PRESET_FIELDS:
        assert PRESETS[BALANCED][field] == getattr(defaults, field), field


def test_conservative_probes_lower_and_moves_up_more_slowly():
    conservative = PRESETS[CONSERVATIVE]
    balanced = PRESETS[BALANCED]

    assert conservative["cold_start_probe_bps"] < balanced["cold_start_probe_bps"]
    assert conservative["max_step_up_bps"] < balanced["max_step_up_bps"]
    assert conservative["repeat_buyer_uplift_bps"] < balanced["repeat_buyer_uplift_bps"]
    # A HIGHER streak is more conservative: more evidence before extra uplift.
    assert (
        conservative["effortless_purchase_streak"]
        > balanced["effortless_purchase_streak"]
    )


def test_aggressive_probes_higher_and_moves_up_faster():
    aggressive = PRESETS[AGGRESSIVE]
    balanced = PRESETS[BALANCED]

    assert aggressive["cold_start_probe_bps"] > balanced["cold_start_probe_bps"]
    assert aggressive["max_step_up_bps"] > balanced["max_step_up_bps"]
    assert aggressive["vip_uplift_bps"] > balanced["vip_uplift_bps"]
    assert (
        aggressive["effortless_purchase_streak"]
        <= balanced["effortless_purchase_streak"]
    )


def test_every_preset_is_a_valid_policy():
    for preset_id in PRESETS:
        PriceLearningPolicy.model_validate(
            {**PriceLearningPolicy().model_dump(), **PRESETS[preset_id]}
        )


def test_a_preset_touches_only_probing_fields():
    """An agency that narrowed max_offer_cents keeps that when strategy changes."""
    existing = {"max_offer_cents": 9_000, "customer_price_step_cents": 100}

    merged = apply_preset(existing, AGGRESSIVE)

    assert merged["max_offer_cents"] == 9_000
    assert merged["customer_price_step_cents"] == 100
    assert merged["cold_start_probe_bps"] == PRESETS[AGGRESSIVE]["cold_start_probe_bps"]


def test_aggressive_cannot_raise_the_approved_content_ceiling():
    """The bound that makes "Aggressive" safe: it changes WHERE inside a range a
    fan is probed, never what the range is."""
    floor, ceiling = category_range_cents("nude_photo")

    for preset_id in PRESETS:
        policy = PriceLearningPolicy(
            **{**PriceLearningPolicy().model_dump(), **PRESETS[preset_id]}
        )
        probe = floor + int((ceiling - floor) * policy.cold_start_probe_bps / 10_000)
        assert floor <= probe <= ceiling, preset_id


# --- recognising a stored policy -------------------------------------------


def test_an_unconfigured_agency_reads_as_balanced():
    assert preset_for_policy({}) == BALANCED


def test_a_stored_preset_is_recognised_rather_than_re_asked():
    assert preset_for_policy(apply_preset({}, AGGRESSIVE)) == AGGRESSIVE
    assert preset_for_policy(apply_preset({}, CONSERVATIVE)) == CONSERVATIVE


def test_hand_tuned_values_report_themselves_as_custom():
    """The honest answer, and the reason Advanced controls exist."""
    tuned = {**PRESETS[BALANCED], "cold_start_probe_bps": 3_142}

    assert preset_for_policy(tuned) == CUSTOM


@pytest.mark.parametrize("value", ["", None, "AGGRESSIVE ", "custom", "maximal"])
def test_only_the_three_presets_are_accepted(value):
    if value == "AGGRESSIVE ":
        # Whitespace and case are tolerated; the identifier is still one of three.
        assert normalize_preset(value) == AGGRESSIVE
        return
    assert normalize_preset(value) is None
    with pytest.raises(ValueError):
        preset_settings(value)


def test_the_catalog_describes_each_preset_for_the_picker():
    described = describe_presets()

    assert [row["preset"] for row in described] == [CONSERVATIVE, BALANCED, AGGRESSIVE]
    for row in described:
        assert row["label"]
        assert row["description"]
        assert set(row["settings"]) == set(PRESET_FIELDS)


# --- persistence and precedence --------------------------------------------


class _Scopes:
    def __init__(self) -> None:
        self.policies: dict[tuple[str, str], dict] = {}
        self.memberships: dict[str, str] = {}

    def table(self, name):
        return _ScopeTable(self, name)


class _ScopeTable:
    def __init__(self, store: _Scopes, name: str) -> None:
        self.store = store
        self.name = name
        self.filters: dict[str, str] = {}
        self.payload: dict | None = None

    def select(self, *_a, **_k):
        return self

    def upsert(self, payload, **_k):
        self.payload = payload
        return self

    def eq(self, column, value):
        self.filters[column] = str(value)
        return self

    def limit(self, *_a, **_k):
        return self

    def execute(self):
        if self.payload is not None:
            key = (self.payload["scope_type"], self.payload["scope_id"])
            self.store.policies[key] = dict(self.payload["settings"])
            return SimpleNamespace(data=[dict(self.payload)])
        if self.name == "creator_pricing_scope_memberships":
            scope = self.store.memberships.get(self.filters.get("creator_id", ""))
            return SimpleNamespace(
                data=[{"agency_scope_id": scope}] if scope else []
            )
        key = (self.filters.get("scope_type", ""), self.filters.get("scope_id", ""))
        settings = self.store.policies.get(key)
        return SimpleNamespace(data=[{"settings": settings}] if settings else [])


@pytest.fixture
def scopes(monkeypatch):
    monkeypatch.setenv("PRICE_LEARNING_POLICY_CACHE_SECONDS", "0")
    pricing_policy_queries.clear_price_learning_policy_cache()
    store = _Scopes()
    monkeypatch.setattr(pricing_policy_queries, "get_supabase", lambda: store)
    return store


def test_railway_defaults_apply_when_nothing_is_configured(scopes, monkeypatch):
    monkeypatch.setenv("PRICE_LEARNING_COLD_START_PROBE_BPS", "1800")
    pricing_policy_queries.clear_price_learning_policy_cache()

    policy = asyncio.run(get_effective_price_learning_policy("creator-1"))

    assert policy.cold_start_probe_bps == 1_800
    assert environment_price_learning_policy().cold_start_probe_bps == 1_800


def test_agency_policy_beats_the_railway_default(scopes):
    scopes.memberships["creator-1"] = "agency-1"
    asyncio.run(
        save_policy_scope_settings("AGENCY", "agency-1", apply_preset({}, AGGRESSIVE))
    )
    pricing_policy_queries.clear_price_learning_policy_cache()

    policy = asyncio.run(get_effective_price_learning_policy("creator-1"))

    assert policy.cold_start_probe_bps == PRESETS[AGGRESSIVE]["cold_start_probe_bps"]


def test_a_creator_override_beats_the_agency(scopes):
    scopes.memberships["creator-1"] = "agency-1"
    asyncio.run(
        save_policy_scope_settings("AGENCY", "agency-1", apply_preset({}, AGGRESSIVE))
    )
    asyncio.run(
        save_policy_scope_settings("CREATOR", "creator-1", apply_preset({}, CONSERVATIVE))
    )
    pricing_policy_queries.clear_price_learning_policy_cache()

    policy = asyncio.run(get_effective_price_learning_policy("creator-1"))

    assert policy.cold_start_probe_bps == PRESETS[CONSERVATIVE]["cold_start_probe_bps"]


def test_an_agency_policy_does_not_leak_to_another_agencys_creator(scopes):
    scopes.memberships["creator-1"] = "agency-1"
    asyncio.run(
        save_policy_scope_settings("AGENCY", "agency-1", apply_preset({}, AGGRESSIVE))
    )
    pricing_policy_queries.clear_price_learning_policy_cache()

    unaffiliated = asyncio.run(get_effective_price_learning_policy("creator-2"))

    assert unaffiliated.cold_start_probe_bps == (
        environment_price_learning_policy().cold_start_probe_bps
    )


# --- what may be written ----------------------------------------------------


def test_unknown_keys_are_dropped_rather_than_stored():
    assert validate_policy_settings(
        {"cold_start_probe_bps": 2_000, "sql": "drop table fans"}
    ) == {"cold_start_probe_bps": 2_000}


@pytest.mark.parametrize(
    "settings",
    [
        {"cold_start_probe_bps": 99_999},
        {"customer_price_step_cents": 0},
        {"max_offer_cents": 10},
        {"min_offer_cents": 900_000},
    ],
)
def test_an_invalid_policy_is_refused_before_it_can_misprice_an_offer(settings):
    with pytest.raises(Exception):
        validate_policy_settings(settings)


def test_an_inverted_offer_range_is_refused():
    """The one constraint the model cannot express field by field, and the one
    that would otherwise fail at the moment an offer is built."""
    with pytest.raises(ValueError):
        validate_policy_settings({"min_offer_cents": 8_000, "max_offer_cents": 5_000})


def test_a_stored_settings_row_is_always_a_valid_policy(scopes):
    stored = asyncio.run(
        save_policy_scope_settings(
            "CREATOR", "creator-1", {**apply_preset({}, AGGRESSIVE), "junk": 1}
        )
    )

    assert "junk" not in stored
    PriceLearningPolicy.model_validate(
        {**PriceLearningPolicy().model_dump(), **stored}
    )
