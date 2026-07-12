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
        ["I have a lingerie set at $28, or a toy play video for $60"],
    )
    assert normalized["offer_response"] == "accepted"
    assert normalized["selected_offer_price_usd"] == "28"
    assert normalized["current_budget_limit_usd"] == "28"
    assert normalized["cannot_afford_any_offer_now"] == "false"
    assert normalized["purchase_signal"] == "ready_to_buy"

    events = extract_events(normalized)
    types = {event.type for event in events}
    assert EventType.PACKAGE_SELECTED in types
    assert EventType.BUDGET_LIMIT_STATED in types
    assert EventType.PAYDAY_MENTIONED in types
    assert EventType.MONEY_UNAVAILABLE not in types


def test_cannot_afford_either_is_a_real_pause():
    normalized = normalize_commercial_facts(
        {},
        "I can't afford either right now, I get paid Friday",
        ["quick set $28 or full session $60"],
    )
    assert normalized["cannot_afford_any_offer_now"] == "true"
    events = extract_events(normalized)
    assert EventType.MONEY_UNAVAILABLE in {event.type for event in events}
