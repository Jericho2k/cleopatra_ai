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


# ---------------------------------------------------------------------------
# Malformed extractor output does not silently lose the turn's state
# ---------------------------------------------------------------------------
#
# The observed line::
#
#     [FAN INTELLIGENCE] invalid extraction ... JSON parse error
#
# One malformed response dropped everything the fan said that turn. Three
# deterministic repairs run first — fences, a JSON object wrapped in prose, and
# a bare list — and only a response past all three costs the one bounded retry.


def test_a_code_fenced_object_is_read():
    envelope = parse_extraction_payload(
        '```json\n{"observations": []}\n```'
    )
    assert envelope.observations == []


def test_an_object_wrapped_in_prose_is_read():
    envelope = parse_extraction_payload(
        'Sure, here you go:\n{"observations": []}\nHope that helps!'
    )
    assert envelope.observations == []


def test_a_bare_list_is_read_as_observations():
    envelope = parse_extraction_payload(
        '[{"category":"commercial","fact_key":"payday","value":"Friday",'
        '"certainty":"explicit","confidence":0.98,"evidence":"I get paid Friday"}]'
    )
    assert len(envelope.observations) == 1
    assert envelope.observations[0].fact_key == "payday"


def test_braces_inside_a_string_do_not_confuse_the_extractor():
    envelope = parse_extraction_payload(
        'note: he wrote "{not json}"\n'
        '{"observations": [{"category":"identity","fact_key":"location",'
        '"value":"Berlin {area}","certainty":"explicit","confidence":0.9,'
        '"evidence":"I live in Berlin"}]}'
    )
    assert envelope.observations[0].value == "Berlin {area}"


def test_genuinely_malformed_output_still_raises_so_the_caller_can_retry():
    import pytest

    for text in ("", "   ", "no json here at all", "{unclosed"):
        with pytest.raises(Exception):
            parse_extraction_payload(text)


def test_one_bounded_repair_runs_and_the_reply_is_never_blocked(monkeypatch):
    """Best-effort: a second malformed answer ends the attempt, not the turn."""
    import asyncio

    from services import fan_intelligence

    calls: list[dict] = []

    class _Result:
        def __init__(self, text):
            self.text = text
            self.usage = type("U", (), {"input_tokens": 1, "output_tokens": 1})()
            self.target = None
            self.latency_ms = 1

    async def flaky_complete(_target, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return _Result("I could not do that.")
        return _Result('{"observations": []}')

    async def noop(*_a, **_k):
        return None

    monkeypatch.setenv("FAN_INTELLIGENCE_ENABLED", "true")
    monkeypatch.setattr(fan_intelligence, "complete", flaky_complete)
    monkeypatch.setattr(fan_intelligence, "record_model_result", noop)
    monkeypatch.setattr(fan_intelligence, "record_model_failure", noop)

    asyncio.run(
        fan_intelligence.learn_from_fan_message(
            creator_id="creator-1",
            fan_id="fan-1",
            fan_message="I get paid Friday",
        )
    )

    assert len(calls) == 2, "exactly one repair attempt, against the same model"
    # The repair says what was wrong, which is the whole point of retrying.
    repair_messages = calls[1]["messages"]
    assert repair_messages[-1]["content"].startswith("That was not valid JSON")


def test_an_extraction_that_cannot_be_repaired_gives_up_quietly(monkeypatch):
    import asyncio

    from services import fan_intelligence

    calls: list[dict] = []

    class _Result:
        def __init__(self, text):
            self.text = text
            self.usage = type("U", (), {"input_tokens": 1, "output_tokens": 1})()
            self.target = None
            self.latency_ms = 1

    async def always_bad(_target, **kwargs):
        calls.append(kwargs)
        return _Result("still not json")

    async def noop(*_a, **_k):
        return None

    monkeypatch.setenv("FAN_INTELLIGENCE_ENABLED", "true")
    monkeypatch.setattr(fan_intelligence, "complete", always_bad)
    monkeypatch.setattr(fan_intelligence, "record_model_result", noop)
    monkeypatch.setattr(fan_intelligence, "record_model_failure", noop)

    # No exception reaches the caller: enrichment never blocks a reply.
    asyncio.run(
        fan_intelligence.learn_from_fan_message(
            creator_id="creator-1",
            fan_id="fan-1",
            fan_message="I get paid Friday",
        )
    )
    assert len(calls) == 2, "bounded: it does not keep retrying"
