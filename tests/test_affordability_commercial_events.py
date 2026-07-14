from models.commercial import EventType
from services.commercial_events import extract_events, normalize_commercial_facts


def test_non_matching_offer_amount_becomes_counteroffer():
    result = normalize_commercial_facts(
        {},
        "can you do $25 instead?",
        ["I can do a quick one for $28 or the full one for $60"],
    )
    events = extract_events(result)
    assert any(
        event.type == EventType.COUNTEROFFER_STATED
        and event.amount_cents == 2500
        for event in events
    )
    assert not any(event.type == EventType.PACKAGE_SELECTED for event in events)


def test_matching_offer_amount_remains_package_selection():
    result = normalize_commercial_facts(
        {},
        "can we do the $28 one?",
        ["I can do a quick one for $28 or the full one for $60"],
    )
    events = extract_events(result)
    assert any(
        event.type == EventType.PACKAGE_SELECTED
        and event.amount_cents == 2800
        for event in events
    )
