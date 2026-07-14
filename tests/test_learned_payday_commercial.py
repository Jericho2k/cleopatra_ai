from models.commercial import CommercialEvent, EventType
from services.commercial_orchestrator import _augment_events_with_safe_learned_context


def test_known_explicit_payday_can_complete_affordability_pause():
    events = [CommercialEvent(type=EventType.MONEY_UNAVAILABLE)]
    situation = {
        "learned_fan_intelligence": {
            "facts": [
                {
                    "fact_key": "payday",
                    "value": "Friday",
                    "status": "confirmed",
                }
            ]
        }
    }
    _augment_events_with_safe_learned_context(events, situation)
    payday = next(event for event in events if event.type == EventType.PAYDAY_MENTIONED)
    assert payday.raw_expression == "Friday"
    assert payday.metadata["source"] == "passive_fan_intelligence"


def test_inferred_payday_is_not_used_commercially():
    events = [CommercialEvent(type=EventType.MONEY_UNAVAILABLE)]
    situation = {
        "learned_fan_intelligence": {
            "facts": [
                {
                    "fact_key": "payday",
                    "value": "Friday",
                    "status": "inferred",
                }
            ]
        }
    }
    _augment_events_with_safe_learned_context(events, situation)
    assert EventType.PAYDAY_MENTIONED not in {event.type for event in events}


def test_learned_payday_does_not_duplicate_current_message_event():
    events = [
        CommercialEvent(type=EventType.MONEY_UNAVAILABLE),
        CommercialEvent(type=EventType.PAYDAY_MENTIONED, raw_expression="Monday"),
    ]
    situation = {
        "learned_fan_intelligence": {
            "facts": [
                {
                    "fact_key": "payday",
                    "value": "Friday",
                    "status": "confirmed",
                }
            ]
        }
    }
    _augment_events_with_safe_learned_context(events, situation)
    payday_events = [event for event in events if event.type == EventType.PAYDAY_MENTIONED]
    assert len(payday_events) == 1
    assert payday_events[0].raw_expression == "Monday"
