"""Turn the situation analyzer's raw JSON into typed CommercialEvents.

The LLM's job is to say what the fan MEANT (observations). This module converts
those observations into the typed events the policy engine consumes. No business
decisions happen here — just translation.
"""
import re

from models.commercial import CommercialEvent, EventType


def _truthy(v) -> bool:
    return str(v).strip().lower() in ("true", "yes", "1")


def extract_events(situation: dict) -> list[CommercialEvent]:
    """situation = raw dict from analyze_situation()."""
    events: list[CommercialEvent] = []
    if not situation:
        return events

    # Crisis outranks everything downstream; the policy engine hard-stops on it.
    if (situation.get("crisis_signal") or "none") != "none":
        events.append(CommercialEvent(
            type=EventType.CRISIS,
            raw_expression=str(situation.get("crisis_signal")),
        ))

    signal = (situation.get("purchase_signal") or "none").lower()
    if signal == "declined":
        events.append(CommercialEvent(type=EventType.MONEY_UNAVAILABLE))
    elif signal == "money_available":
        events.append(CommercialEvent(type=EventType.MONEY_AVAILABLE))
    elif signal == "ready_to_buy":
        events.append(CommercialEvent(type=EventType.READY_TO_BUY))
    elif signal == "bought":
        events.append(CommercialEvent(type=EventType.PURCHASED))

    if _truthy(situation.get("wants_explicit")):
        events.append(CommercialEvent(type=EventType.WANTS_EXPLICIT))
    if _truthy(situation.get("wants_media")):
        events.append(CommercialEvent(type=EventType.WANTS_MEDIA))

    payday_raw = (situation.get("payday_raw") or "").strip()
    if payday_raw:
        events.append(CommercialEvent(
            type=EventType.PAYDAY_MENTIONED,
            raw_expression=payday_raw,
            confidence=0.9,
        ))

    budget = str(situation.get("budget_stated_usd") or "").strip()
    m = re.search(r"\d+(?:\.\d+)?", budget)
    if m:
        events.append(CommercialEvent(
            type=EventType.BUDGET_STATED,
            raw_expression=m.group(0),
        ))

    return events


def stated_budget_cents(events: list[CommercialEvent]) -> int | None:
    for e in events:
        if e.type == EventType.BUDGET_STATED:
            try:
                return int(round(float(e.raw_expression) * 100))
            except (TypeError, ValueError):
                return None
    return None