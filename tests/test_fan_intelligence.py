from models.fan_intelligence import (
    FactCategory,
    FactCertainty,
    MergeAction,
    ProposedObservation,
)
from services.fan_intelligence import (
    parse_extraction_payload,
    plan_fact_merge,
    validate_observation,
)


def _proposal(**overrides):
    values = {
        "category": FactCategory.COMMERCIAL,
        "fact_key": "payday",
        "value": "Friday",
        "certainty": FactCertainty.EXPLICIT,
        "confidence": 0.98,
        "evidence": "I get paid Friday",
    }
    values.update(overrides)
    return ProposedObservation(**values)


def test_parser_is_strict_json_and_accepts_empty_observations():
    envelope = parse_extraction_payload('{"observations": []}')
    assert envelope.observations == []


def test_explicit_payday_is_validated():
    observation = validate_observation(
        _proposal(),
        fan_message="not today, I get paid Friday",
    )
    assert observation is not None
    assert observation.fact_key == "payday"
    assert observation.value_json == "Friday"


def test_money_is_normalized_from_exact_evidence():
    observation = validate_observation(
        _proposal(
            fact_key="stated_budget_cents",
            value=40,
            evidence="I only have $40",
        ),
        fan_message="I only have $40 rn",
    )
    assert observation is not None
    assert observation.value_json == 4000


def test_hard_limit_cannot_be_inferred():
    observation = validate_observation(
        _proposal(
            category=FactCategory.BOUNDARY,
            fact_key="hard_limit",
            value="no humiliation",
            certainty=FactCertainty.STRONG_INFERENCE,
            confidence=0.99,
            evidence="not really into humiliation",
        ),
        fan_message="not really into humiliation",
    )
    assert observation is None


def test_evidence_must_be_exact_substring_of_latest_message():
    observation = validate_observation(
        _proposal(evidence="gets paid every Friday"),
        fan_message="I get paid Friday",
    )
    assert observation is None


def test_same_value_reinforces_existing_fact():
    validated = validate_observation(_proposal(), fan_message="I get paid Friday")
    assert validated is not None
    plan = plan_fact_merge(
        [
            {
                "id": "fact-1",
                "normalized_value": validated.normalized_value,
                "status": "explicit",
                "is_active": True,
            }
        ],
        validated,
    )
    assert plan.action == MergeAction.REINFORCE
    assert plan.matched_fact_id == "fact-1"


def test_conflicting_explicit_singleton_is_not_silently_replaced():
    validated = validate_observation(_proposal(), fan_message="I get paid Friday")
    assert validated is not None
    plan = plan_fact_merge(
        [
            {
                "id": "fact-1",
                "normalized_value": '"thursday"',
                "status": "confirmed",
                "is_active": True,
            }
        ],
        validated,
    )
    assert plan.action == MergeAction.CONFLICT
    assert plan.conflicting_fact_ids == ["fact-1"]


def test_explicit_value_supersedes_only_inferred_singleton():
    validated = validate_observation(_proposal(), fan_message="I get paid Friday")
    assert validated is not None
    plan = plan_fact_merge(
        [
            {
                "id": "fact-1",
                "normalized_value": '"sometime this week"',
                "status": "inferred",
                "is_active": True,
            }
        ],
        validated,
    )
    assert plan.action == MergeAction.REPLACE_INFERRED
