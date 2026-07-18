"""Deterministic creator-level audience rules for Full Auto."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class AutoAudiencePolicy(BaseModel):
    scope: Literal["all", "new_only", "matching"] = "all"
    match_mode: Literal["any", "all"] = "any"
    include_list_ids: list[str] = Field(default_factory=list)
    exclude_list_ids: list[str] = Field(default_factory=list)
    spend_tiers: list[str] = Field(default_factory=list)
    include_new_fans: bool = False
    min_total_spend: int | None = Field(default=None, ge=0)
    max_total_spend: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def normalize(self) -> "AutoAudiencePolicy":
        self.include_list_ids = _unique(self.include_list_ids)
        self.exclude_list_ids = _unique(self.exclude_list_ids)
        self.spend_tiers = _unique(self.spend_tiers)
        if (
            self.min_total_spend is not None
            and self.max_total_spend is not None
            and self.min_total_spend > self.max_total_spend
        ):
            raise ValueError("minimum spend cannot exceed maximum spend")
        return self


class AutoEligibility(BaseModel):
    eligible: bool
    reason: str


def evaluate_auto_eligibility(
    *,
    creator_auto: bool,
    fan_auto_override: bool | None,
    needs_human_review: bool,
    policy: AutoAudiencePolicy,
    fan_list_ids: set[str] | None = None,
    total_spent: int = 0,
    spend_tier: str = "cold",
    is_new_fan: bool = False,
) -> AutoEligibility:
    lists = {str(value) for value in (fan_list_ids or set())}
    if needs_human_review:
        return AutoEligibility(eligible=False, reason="needs_human_review")
    if fan_auto_override is False:
        return AutoEligibility(eligible=False, reason="fan_override_off")
    if fan_auto_override is True:
        return AutoEligibility(eligible=True, reason="fan_override_on")
    if not creator_auto:
        return AutoEligibility(eligible=False, reason="creator_auto_off")
    if lists.intersection(policy.exclude_list_ids):
        return AutoEligibility(eligible=False, reason="excluded_list")
    if policy.scope == "all":
        return AutoEligibility(eligible=True, reason="all_fans")
    if policy.scope == "new_only":
        return AutoEligibility(eligible=is_new_fan, reason="new_fan" if is_new_fan else "not_new")

    criteria: list[bool] = []
    if policy.include_list_ids:
        criteria.append(bool(lists.intersection(policy.include_list_ids)))
    if policy.spend_tiers:
        criteria.append(str(spend_tier) in policy.spend_tiers)
    if policy.min_total_spend is not None or policy.max_total_spend is not None:
        criteria.append(
            (policy.min_total_spend is None or total_spent >= policy.min_total_spend)
            and (policy.max_total_spend is None or total_spent <= policy.max_total_spend)
        )
    if policy.include_new_fans:
        criteria.append(is_new_fan)
    if not criteria:
        return AutoEligibility(eligible=False, reason="no_matching_rules")
    matched = all(criteria) if policy.match_mode == "all" else any(criteria)
    return AutoEligibility(eligible=matched, reason="rules_matched" if matched else "rules_not_matched")


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))

