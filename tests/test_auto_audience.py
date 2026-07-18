import pytest
from pathlib import Path

from services.auto_audience import AutoAudiencePolicy, evaluate_auto_eligibility


def eligibility(**overrides):
    params = {
        "creator_auto": True,
        "fan_auto_override": None,
        "needs_human_review": False,
        "policy": AutoAudiencePolicy(),
        "fan_list_ids": set(),
        "total_spent": 0,
        "spend_tier": "cold",
        "is_new_fan": False,
    }
    params.update(overrides)
    return evaluate_auto_eligibility(**params)


def test_human_review_and_explicit_off_have_priority():
    assert eligibility(needs_human_review=True, fan_auto_override=True).eligible is False
    assert eligibility(fan_auto_override=False).reason == "fan_override_off"


def test_explicit_fan_on_overrides_creator_and_targeting():
    result = eligibility(
        creator_auto=False,
        fan_auto_override=True,
        policy=AutoAudiencePolicy(scope="new_only"),
    )
    assert result.eligible is True
    assert result.reason == "fan_override_on"


def test_new_only_and_list_exclusions_are_deterministic():
    policy = AutoAudiencePolicy(scope="new_only", exclude_list_ids=["blocked"])
    assert eligibility(policy=policy, is_new_fan=True).eligible is True
    assert eligibility(policy=policy, is_new_fan=False).eligible is False
    assert eligibility(policy=policy, is_new_fan=True, fan_list_ids={"blocked"}).eligible is False


def test_matching_rules_support_any_or_all_combinations():
    any_policy = AutoAudiencePolicy(
        scope="matching",
        match_mode="any",
        include_list_ids=["vip"],
        spend_tiers=["whale"],
    )
    assert eligibility(policy=any_policy, spend_tier="whale").eligible is True

    all_policy = any_policy.model_copy(update={"match_mode": "all"})
    assert eligibility(policy=all_policy, spend_tier="whale").eligible is False
    assert eligibility(policy=all_policy, spend_tier="whale", fan_list_ids={"vip"}).eligible is True


def test_invalid_spend_range_is_rejected():
    with pytest.raises(ValueError):
        AutoAudiencePolicy(scope="matching", min_total_spend=100, max_total_spend=10)


def test_chat_reconciliation_uses_a_durable_claim_and_incremental_stop():
    root = Path(__file__).resolve().parents[1]
    migration = (root / "db" / "agency_operability_v1.sql").read_text(encoding="utf-8")
    source = (root / "main.py").read_text(encoding="utf-8")
    assert "claim_chat_reconciliation" in migration
    assert "last_chat_reconcile_at" in migration
    assert 'incremental=True' in source
    assert "page_ids.issubset(existing_platform_ids)" in source
