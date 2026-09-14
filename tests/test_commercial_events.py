"""Regression tests for mixed commercial language."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.commercial_events import normalize_commercial_facts  # noqa: E402
from models.commercial import EventType  # noqa: E402
from services.commercial_events import extract_events  # noqa: E402


def test_accepts_28_now_and_only_remembers_friday():
    normalized = normalize_commercial_facts(
        {
            "payday_raw": "Friday",
            "payday_confidence": 0.9,
            "purchase_signal": "declined",  # simulate the LLM's old mistake
        },
        "can we do $28 one cause I dont have more money rn, I get a paycheck on Friday",
        ["the lingerie set is $28"],
    )
    assert normalized["offer_response"] == "accepted"
    assert normalized["selected_offer_price_usd"] == "28"
    assert normalized["current_budget_limit_usd"] == "28"
    assert normalized["cannot_afford_any_offer_now"] == "false"
    assert normalized["purchase_signal"] == "ready_to_buy"

    events = extract_events(normalized)
    types = {event.type for event in events}
    assert EventType.OFFER_ACCEPTED in types
    assert EventType.BUDGET_LIMIT_STATED in types
    assert EventType.PAYDAY_MENTIONED in types
    assert EventType.MONEY_UNAVAILABLE not in types


def test_cannot_afford_it_is_a_real_pause():
    normalized = normalize_commercial_facts(
        {},
        "I can't afford it right now, I get paid Friday",
        ["the lingerie set is $28"],
    )
    assert normalized["cannot_afford_any_offer_now"] == "true"
    events = extract_events(normalized)
    assert EventType.MONEY_UNAVAILABLE in {event.type for event in events}


def test_a_bare_yes_to_the_one_named_price_is_acceptance():
    normalized = normalize_commercial_facts(
        {},
        "yeah send it",
        ["it's $28 if you want it"],
    )
    assert normalized["offer_response"] == "accepted"
    assert normalized["selected_offer_price_usd"] == "28"
    assert EventType.OFFER_ACCEPTED in {
        event.type for event in extract_events(normalized)
    }


def test_no_event_carries_an_ordinal_position_any_more():
    normalized = normalize_commercial_facts(
        {},
        "I'll take the first one",
        ["it's $28 if you want it"],
    )
    assert "selected_offer_position" not in normalized
    for event in extract_events(normalized):
        assert not hasattr(event, "package_position")
