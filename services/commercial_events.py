"""Translate situation-analyzer JSON into typed commercial observations.

The crucial rule is that acceptance, present budget and future payday are
independent facts. A message such as "I'll take the $28 one; I get paid Friday"
means PACKAGE_SELECTED($28) + BUDGET_LIMIT_STATED($28) + PAYDAY_MENTIONED,
not MONEY_UNAVAILABLE.
"""
import re

from models.commercial import CommercialEvent, EventType


def _truthy(value) -> bool:
    return str(value).strip().lower() in {"true", "yes", "1"}


def _money_cents(value) -> int | None:
    if value is None:
        return None
    match = re.search(r"\d+(?:\.\d+)?", str(value))
    if not match:
        return None
    try:
        return int(round(float(match.group(0)) * 100))
    except (TypeError, ValueError):
        return None


def normalize_commercial_facts(
    result: dict,
    latest_message: str,
    recent_creator_messages: list[str] | None = None,
) -> dict:
    """Deterministic backstop for high-value commercial facts.

    This intentionally handles only narrow, high-confidence patterns. It keeps
    package selection, current affordability, and future payday as independent
    observations so a mixed sentence cannot be collapsed into a generic decline.
    """
    out = {**_fallback_situation(), **(result or {})}
    text = (latest_message or "").strip().lower()
    creator_text = "\n".join(recent_creator_messages or []).lower()

    offered_amounts = _extract_money_values(creator_text)
    message_amounts = _extract_money_values(text)

    selected_price: int | None = None
    selected_position = ""

    acceptance_words = re.search(
        r"\b(can we do|i(?:'| a)?ll take|i want|go with|choose|give me|send me|do the|take the)\b",
        text,
    )
    if message_amounts and acceptance_words:
        selected_price = message_amounts[0]
    elif re.search(r"\b(first|cheaper|smaller|lower)\s+(?:one|option)\b", text):
        selected_position = "first"
        if offered_amounts:
            selected_price = offered_amounts[0]
    elif re.search(r"\b(second|full|bigger|higher)\s+(?:one|option)\b", text):
        selected_position = "second"
        if len(offered_amounts) >= 2:
            selected_price = offered_amounts[1]

    # "$28 one" / "the 28 one" is acceptance when that amount appeared in a
    # recent creator offer, even without an explicit acceptance verb.
    if selected_price is None and message_amounts and offered_amounts:
        matching = next((value for value in message_amounts if value in offered_amounts), None)
        if matching is not None and re.search(r"\b(one|option|that|this)\b", text):
            selected_price = matching

    if selected_price is not None or selected_position:
        out["offer_response"] = "accepted"
        out["selected_offer_price_usd"] = str(selected_price or "")
        out["selected_offer_position"] = selected_position
        out["purchase_signal"] = "ready_to_buy"
        out["cannot_afford_any_offer_now"] = "false"
        out["deferred_purchase_intent"] = "false"

        if re.search(
            r"\b(don'?t have more|can'?t spend more|only have|all i have|my limit|maximum|max)\b",
            text,
        ):
            out["current_budget_limit_usd"] = str(selected_price or "")

    # A negotiated amount that does not match an offered package is a
    # counteroffer, not package acceptance. Exact offered prices remain
    # authoritative; price learning is handled later.
    negotiation = re.search(
        r"\b(can (?:you|u) do|would you do|what about|how about|for|instead|i can do|i'll do)\b",
        text,
    )
    if message_amounts and offered_amounts and negotiation:
        proposed = message_amounts[0]
        if proposed not in offered_amounts:
            out["counteroffer_usd"] = str(proposed)
            out["selected_offer_price_usd"] = ""
            out["selected_offer_position"] = ""
            out["offer_response"] = "none"
            out["purchase_signal"] = "none"
            selected_price = None
            selected_position = ""

    cannot_buy_any = bool(re.search(
        r"\b(can'?t afford (?:either|any|it|that)|can'?t pay (?:right now|today|yet)|"
        r"don'?t have (?:any )?money|no money|broke|not enough for (?:either|any))\b",
        text,
    ))
    if cannot_buy_any and selected_price is None and not selected_position:
        payday = _find_payday(text)
        out["cannot_afford_any_offer_now"] = "true"
        out["offer_response"] = "deferred" if payday else "declined"
        out["deferred_purchase_intent"] = "true" if payday else "false"
        out["purchase_signal"] = "declined"

    budget_match = re.search(
        r"\b(?:i have|i've got|i can spend|my budget is|only have|max(?:imum)? is)\s*\$?\s*(\d+(?:\.\d+)?)",
        text,
    )
    if budget_match:
        amount = budget_match.group(1)
        out["budget_stated_usd"] = amount
        if re.search(r"\b(only|max|maximum|limit)\b", budget_match.group(0)):
            out["current_budget_limit_usd"] = amount

    if not str(out.get("payday_raw") or "").strip():
        payday = _find_payday(text)
        if payday:
            out["payday_raw"] = payday
            out["payday_confidence"] = 0.95

    return out


def _fallback_situation() -> dict:
    return {
        "fan_mood": "curious",
        "fan_intent": "engaging with creator",
        "conversation_energy": "flat",
        "strategic_move": "mirror_warmth",
        "tone": "playful",
        "personal_details_mentioned": [],
        "avoid_repeating": "",
        "purchase_signal": "none",
        "offer_response": "none",
        "selected_offer_price_usd": "",
        "selected_offer_position": "",
        "current_budget_limit_usd": "",
        "counteroffer_usd": "",
        "cannot_afford_any_offer_now": "false",
        "deferred_purchase_intent": "false",
        "resend_requested": "false",
        "crisis_signal": "none",
        "wants_explicit": "false",
        "wants_media": "false",
        "payday_raw": "",
        "payday_confidence": 0.0,
        "budget_stated_usd": "",
        "desired_experience": "",
    }


def _extract_money_values(text: str) -> list[int]:
    values: list[int] = []
    for match in re.finditer(r"\$\s*(\d+(?:\.\d+)?)", text or ""):
        value = int(round(float(match.group(1))))
        if value not in values:
            values.append(value)
    return values


def _find_payday(text: str) -> str:
    weekday = re.search(
        r"\b(?:this |next |on )?(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        text,
    )
    if weekday and re.search(r"\b(pay|paid|paycheck|payday|money|salary|wage)\b", text):
        return weekday.group(0).strip()
    relative = re.search(
        r"\b(tomorrow|next week|in \d{1,2} days?|the \d{1,2}(?:st|nd|rd|th))\b",
        text,
    )
    if relative and re.search(r"\b(pay|paid|paycheck|payday|money|salary|wage)\b", text):
        return relative.group(1)
    return ""


def extract_events(situation: dict) -> list[CommercialEvent]:
    """Convert the raw analyzer result into independent typed facts."""
    events: list[CommercialEvent] = []
    if not situation:
        return events

    if (situation.get("crisis_signal") or "none") != "none":
        events.append(CommercialEvent(
            type=EventType.CRISIS,
            raw_expression=str(situation.get("crisis_signal")),
        ))

    if _truthy(situation.get("wants_explicit")):
        events.append(CommercialEvent(type=EventType.WANTS_EXPLICIT))
    if _truthy(situation.get("wants_media")):
        events.append(CommercialEvent(type=EventType.WANTS_MEDIA))

    payday_raw = str(situation.get("payday_raw") or "").strip()
    if payday_raw:
        events.append(CommercialEvent(
            type=EventType.PAYDAY_MENTIONED,
            raw_expression=payday_raw,
            confidence=float(situation.get("payday_confidence") or 0.9),
        ))

    selected_cents = _money_cents(situation.get("selected_offer_price_usd"))
    selected_position_raw = str(situation.get("selected_offer_position") or "").lower()
    selected_position = (
        selected_position_raw
        if selected_position_raw in {"first", "second"}
        else None
    )
    offer_response = str(situation.get("offer_response") or "none").lower()

    if offer_response == "accepted" or selected_cents is not None or selected_position:
        events.append(CommercialEvent(
            type=EventType.PACKAGE_SELECTED,
            raw_expression=(
                str(situation.get("selected_offer_price_usd") or "")
                or selected_position_raw
            ),
            amount_cents=selected_cents,
            package_position=selected_position,
            confidence=0.98,
        ))
    elif offer_response == "declined":
        events.append(CommercialEvent(type=EventType.OFFER_DECLINED))
    elif offer_response == "deferred":
        events.append(CommercialEvent(type=EventType.DEFERRED_PURCHASE))

    counteroffer_cents = _money_cents(situation.get("counteroffer_usd"))
    if counteroffer_cents is not None:
        events.append(CommercialEvent(
            type=EventType.COUNTEROFFER_STATED,
            raw_expression=str(situation.get("counteroffer_usd")),
            amount_cents=counteroffer_cents,
            confidence=0.98,
        ))

    current_limit_cents = _money_cents(situation.get("current_budget_limit_usd"))
    if current_limit_cents is not None:
        events.append(CommercialEvent(
            type=EventType.BUDGET_LIMIT_STATED,
            raw_expression=str(situation.get("current_budget_limit_usd")),
            amount_cents=current_limit_cents,
        ))

    stated_cents = _money_cents(situation.get("budget_stated_usd"))
    if stated_cents is not None:
        events.append(CommercialEvent(
            type=EventType.BUDGET_STATED,
            raw_expression=str(situation.get("budget_stated_usd")),
            amount_cents=stated_cents,
        ))

    # This means he cannot purchase ANY currently offered option. It is not the
    # same as "I can't spend more than the cheaper option".
    if _truthy(situation.get("cannot_afford_any_offer_now")):
        events.append(CommercialEvent(type=EventType.MONEY_UNAVAILABLE))

    if _truthy(situation.get("deferred_purchase_intent")):
        events.append(CommercialEvent(type=EventType.DEFERRED_PURCHASE))

    signal = str(situation.get("purchase_signal") or "none").lower()
    if signal == "money_available":
        events.append(CommercialEvent(type=EventType.MONEY_AVAILABLE))
    elif signal == "ready_to_buy":
        events.append(CommercialEvent(type=EventType.READY_TO_BUY))
    elif signal == "bought":
        events.append(CommercialEvent(type=EventType.PURCHASED))
    elif signal == "declined":
        # Legacy fallback only. Structured package acceptance always wins and an
        # affordability pause requires cannot_afford_any_offer_now.
        if not any(e.type == EventType.PACKAGE_SELECTED for e in events):
            events.append(CommercialEvent(type=EventType.OFFER_DECLINED))

    return _dedupe(events)


def _dedupe(events: list[CommercialEvent]) -> list[CommercialEvent]:
    seen: set[tuple] = set()
    output: list[CommercialEvent] = []
    for event in events:
        key = (
            event.type,
            event.raw_expression,
            event.amount_cents,
            event.package_position,
        )
        if key not in seen:
            seen.add(key)
            output.append(event)
    return output


def stated_budget_cents(events: list[CommercialEvent]) -> int | None:
    for preferred_type in (EventType.PACKAGE_SELECTED, EventType.BUDGET_STATED, EventType.BUDGET_LIMIT_STATED):
        for event in events:
            if event.type == preferred_type and event.amount_cents is not None:
                return event.amount_cents
    return None


def selected_package_event(events: list[CommercialEvent]) -> CommercialEvent | None:
    return next((e for e in events if e.type == EventType.PACKAGE_SELECTED), None)
