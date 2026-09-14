"""Translate situation-analyzer JSON into typed commercial observations.

The crucial rule is that acceptance, present budget and future payday are
independent facts. A message such as "yeah send it; I get paid Friday" means
OFFER_ACCEPTED + PAYDAY_MENTIONED, not MONEY_UNAVAILABLE.

There is exactly one offer on the table at a time, so acceptance is a yes/no
fact about that offer. The ordinal machinery this module used to carry — "the
first one", "the second one", "the cheaper option", the ambiguity event when
neither could be resolved — existed only because the fan was being shown a menu.
"""
import re

from models.commercial import CommercialEvent, EventType, Offer


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


_PASSIVE_COMPLIMENT_RE = re.compile(
    r"\b(cute|sexy|hot|gorgeous|beautiful|pretty|stunning|adorable|fine|"
    r"attractive|good\s+in|look(?:s|ing)?\s+(?:so\s+)?(?:good|cute|sexy|hot))\b",
    re.IGNORECASE,
)

_DIRECT_COMMERCIAL_INTENT_RE = re.compile(
    r"\b("
    r"show\s+me|send\s+me|give\s+me|let\s+me\s+see|"
    r"can\s+i\s+(?:see|get|have|buy)|"
    r"i\s+(?:want|wanna|need)(?:\s+to)?\b|"
    r"want\s+(?:to\s+)?(?:see|watch|hear|buy|unlock|get)|"
    r"more\s+(?:pics?|photos?|videos?|content)|"
    r"(?:pics?|photos?|videos?|content|set|session|custom)\s+(?:please|now|more)|"
    r"how\s+much|what(?:'s|\s+is)\s+the\s+price|price|unlock|buy|purchase|"
    r"what\s+would\s+you\s+do|tell\s+me\s+what|make\s+me|"
    r"do\s+you\s+have\s+(?:more|a\s+video|pics?|photos?|content)"
    r")\b",
    re.IGNORECASE,
)


_OFFER_DETAIL_RE = re.compile(
    r"\b("
    r"what(?:'s|\s+is)\s+(?:in\s+)?(?:it|that|this)|"
    r"what\s+do\s+i\s+get|what(?:'s|\s+is)\s+included|what\s+does\s+(?:it|that)\s+include|"
    r"how\s+many\s+(?:pics?|photos?|pictures?|videos?)|"
    r"how\s+long\s+is\s+(?:it|the\s+video)|"
    r"tell\s+me\s+more(?:\s+about\s+(?:it|that|this))?(?:\s*[?.!])?$|"
    r"more\s+about\s+(?:it|that|this)|"
    r"explain\s+(?:it|that)|"
    r"i(?:'m|\s+am)\s+all\s+ears|go\s+on"
    r")\b",
    re.IGNORECASE,
)

_GENERIC_OFFER_ACCEPT_RE = re.compile(
    r"\b(?:yes|yeah|yep|yup|ok|okay|sure|please|"
    r"i(?:'m|\s+am)\s+in|let(?:'s|\s+us)\s+do\s+it|sounds\s+good|"
    r"i(?:'ll|\s+will)\s+take\s+it|i\s+want\s+it|i\s+need\s+it|deal|"
    r"send\s+it|send\s+me|show\s+me|do\s+it|go\s+ahead|unlock|buy\s+it|"
    r"that\s+one|this\s+one)\b",
    re.IGNORECASE,
)

_OFFER_TOKEN_STOPWORDS = {
    "a", "an", "and", "at", "for", "from", "i", "in", "is", "it", "me",
    "my", "of", "on", "one", "private", "session", "set",
    "that", "the", "this", "to", "want", "with", "you", "your",
}


def _offer_tokens(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", str(value or "").lower())
        if len(token) > 1 and token not in _OFFER_TOKEN_STOPWORDS
    }


def _acceptance_event(
    offer: Offer,
    raw_expression: str,
    *,
    reason: str,
) -> CommercialEvent:
    return CommercialEvent(
        type=EventType.OFFER_ACCEPTED,
        raw_expression=raw_expression,
        amount_cents=offer.price_cents,
        confidence=0.99,
        metadata={
            "offer_id": offer.offer_id,
            "set_id": offer.set_id,
            "label": offer.label,
            "experience": offer.experience,
            "legal_description": offer.legal_description or offer.experience,
            "acceptance_reason": reason,
        },
    )


def resolve_pending_offer_reference(
    latest_message: str,
    pending_offer: Offer | None,
) -> tuple[Offer | None, str]:
    """Did he accept the ONE offer on the table? Returns ``(offer, reason)``.

    There is nothing to disambiguate, because there is nothing to choose
    between. Either this message is a yes to the pending offer or it is not.
    """
    text = str(latest_message or "").strip().lower()
    if not text or pending_offer is None:
        return None, "no_active_offer"

    # A price reference is acceptance only when it is THE price.
    mentioned_cents = [value * 100 for value in _extract_money_values(text)]
    if mentioned_cents:
        if pending_offer.price_cents in mentioned_cents:
            return pending_offer, "exact_price"
        # He named a different number. That is a counteroffer, handled
        # separately; it is never acceptance of this offer.
        return None, "different_price_named"

    if _GENERIC_OFFER_ACCEPT_RE.search(text):
        return pending_offer, "acceptance"

    offer_tokens = _offer_tokens(
        " ".join(
            value
            for value in (
                pending_offer.label,
                pending_offer.legal_description or "",
                pending_offer.experience or "",
            )
            if value
        )
    )
    if offer_tokens & _offer_tokens(text):
        return pending_offer, "named_the_content"

    return None, "no_acceptance_reference"


def is_pending_offer_detail_request(latest_message: str) -> bool:
    text = str(latest_message or "").strip().lower()
    return bool(_OFFER_DETAIL_RE.search(text))


def augment_pending_offer_events(
    events: list[CommercialEvent],
    latest_message: str,
    pending_offer: Offer | None,
) -> None:
    """Attach the latest message to the exact persisted offer snapshot.

    A question about the offer outranks an analyzer that read the question as a
    yes. Otherwise acceptance is resolved against the single pending offer, and
    an unresolved message simply is not acceptance.
    """
    if pending_offer is None:
        return

    if is_pending_offer_detail_request(latest_message):
        events[:] = [event for event in events if event.type != EventType.OFFER_ACCEPTED]
        if not any(event.type == EventType.OFFER_DETAILS_REQUESTED for event in events):
            events.append(
                CommercialEvent(
                    type=EventType.OFFER_DETAILS_REQUESTED,
                    raw_expression=str(latest_message or ""),
                    confidence=0.99,
                    metadata={"offer_id": pending_offer.offer_id},
                )
            )
        return

    offer, reason = resolve_pending_offer_reference(latest_message, pending_offer)
    existing = accepted_offer_event(events)
    if offer:
        replacement = _acceptance_event(offer, str(latest_message or ""), reason=reason)
        if existing:
            events[events.index(existing)] = replacement
        else:
            events.append(replacement)
        return

    if reason == "different_price_named" and existing:
        # The analyzer called a different number acceptance. It is not.
        events.remove(existing)


def _has_direct_commercial_intent(text: str) -> bool:
    return bool(_DIRECT_COMMERCIAL_INTENT_RE.search(text or ""))


def _is_passive_compliment(text: str) -> bool:
    # A compliment can show warmth without asking to buy or receive anything.
    return bool(_PASSIVE_COMPLIMENT_RE.search(text or ""))


def _has_structured_commercial_response(out: dict) -> bool:
    offer_response = str(out.get("offer_response") or "none").lower()
    purchase_signal = str(out.get("purchase_signal") or "none").lower()
    return bool(
        offer_response not in {"", "none"}
        or str(out.get("selected_offer_price_usd") or "").strip()
        or str(out.get("counteroffer_usd") or "").strip()
        or str(out.get("budget_stated_usd") or "").strip()
        or str(out.get("current_budget_limit_usd") or "").strip()
        or _truthy(out.get("cannot_afford_any_offer_now"))
        or _truthy(out.get("deferred_purchase_intent"))
        or _truthy(out.get("resend_requested"))
        or purchase_signal in {"bought", "money_available", "declined"}
    )


# Unmistakably sexual escalation, as opposed to a compliment. Paired with a
# direct request below, this is what makes "your bikini post made me hard, I
# want to see what's underneath" reach the offer path on the turn he says it.
_EXPLICIT_ESCALATION_RE = re.compile(
    r"\b("
    r"hard|horny|turned\s+on|throbbing|aching|stiff|"
    r"underneath|under\s+(?:it|that|the)|"
    r"naked|nude|nudes|topless|bare|"
    r"take\s+(?:it|them)\s+off|took\s+(?:it|them)\s+off|"
    r"cum|cumming|jerk|jerking|stroking|touch\s+myself|"
    r"pussy|tits|ass|body"
    r")\b",
    re.IGNORECASE,
)


def _promote_direct_intent(out: dict, text: str) -> None:
    """Make an unmistakable request for content a fact, not a model's opinion.

    The deterministic layer could already SUPPRESS interest the model
    over-read from a compliment. It could not establish interest the model
    under-read, which is why a fan who opened with "I want to see what's
    underneath" could still be walked through a rapport ladder: every layer
    downstream was waiting for wants_media, and nothing deterministic ever set
    it.

    Deliberately narrow. It needs an explicit request — "show me", "send me",
    "I want to see", "how much" — and it never fires over an affordability
    pause or a decline, because those are the fan saying the opposite.
    """
    if not _has_direct_commercial_intent(text):
        return
    if _truthy(out.get("cannot_afford_any_offer_now")):
        return
    if str(out.get("purchase_signal") or "").lower() == "declined":
        return
    if str(out.get("offer_response") or "").lower() in {"declined", "deferred"}:
        return

    out["wants_media"] = "true"
    if _EXPLICIT_ESCALATION_RE.search(text or ""):
        out["wants_explicit"] = "true"


def _normalize_compliment_only_interest(out: dict, text: str) -> None:
    # Prevent a sexual compliment from becoming an immediate PPV request.
    # Direct requests, accepted offers, counteroffers, budget statements,
    # purchases, and affordability events remain untouched.
    if not _is_passive_compliment(text):
        return
    if _has_direct_commercial_intent(text):
        return
    if _has_structured_commercial_response(out):
        return

    out["wants_explicit"] = "false"
    out["wants_media"] = "false"
    if str(out.get("purchase_signal") or "").lower() in {
        "ready_to_buy",
        "selected",
        "uncertain",
    }:
        out["purchase_signal"] = "none"
    if str(out.get("strategic_move") or "").lower() in {
        "push_for_ppv",
        "hint_at_content",
    }:
        out["strategic_move"] = "acknowledge_compliment_and_redirect"
    out["commercial_interest_signal"] = "warm_compliment"


# Words that make a number a CEILING rather than a price he agreed to. Without
# one of these in his own message, no amount may become current_budget_limit_usd:
# "that's all I have" is a limit, "yeah send it" is a purchase.
_CURRENT_LIMIT_RE = re.compile(
    r"\b(don'?t have more|can'?t spend more|can'?t do more|no more than|"
    r"only have|only got|that'?s all i have|all i have|all i(?:'| a)?ve got|"
    r"my limit|limit is|maximum|max)\b",
    re.IGNORECASE,
)


def _states_a_current_limit(text: str) -> bool:
    """Whether he actually said a ceiling, in words, in this message."""
    return bool(_CURRENT_LIMIT_RE.search(text or ""))


def _echoes_without_limit_language(
    reported_limit: object,
    selected_price: int,
    text: str,
) -> bool:
    """A reported limit that is just the accepted price, with nothing backing it.

    Only strips the echo. A genuinely different number the analyzer extracted
    from somewhere else in the message is left alone, because that is evidence
    this function has no basis to overrule.
    """
    raw = str(reported_limit or "").strip()
    if not raw:
        return False
    try:
        value = int(round(float(raw.replace("$", ""))))
    except (TypeError, ValueError):
        return False
    return value == int(selected_price) and not _states_a_current_limit(text)


def _echoes_the_accepted_price(out: dict, text: str) -> bool:
    """A reported limit that is simply the price he accepted, with no limit words."""
    limit = _money_cents(out.get("current_budget_limit_usd"))
    accepted = _money_cents(out.get("selected_offer_price_usd"))
    if limit is None or accepted is None or limit != accepted:
        return False
    return not _states_a_current_limit(text)


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

    acceptance_words = re.search(
        r"\b(can we do|i(?:'| a)?ll take|i want|go with|give me|send me|do the|take the|"
        r"send it|yes|yeah|yep|do it|go ahead)\b",
        text,
    )
    if message_amounts and acceptance_words:
        selected_price = message_amounts[0]

    # "$28" / "the 28 one" is acceptance when that exact amount is the price the
    # creator just named, even without an explicit acceptance verb.
    if selected_price is None and message_amounts and offered_amounts:
        matching = next((value for value in message_amounts if value in offered_amounts), None)
        if matching is not None and re.search(r"\b(one|that|this)\b", text):
            selected_price = matching

    # A bare yes to the single offer the creator just named is acceptance of it.
    if selected_price is None and offered_amounts and acceptance_words and not message_amounts:
        selected_price = offered_amounts[-1]

    if selected_price is not None:
        out["offer_response"] = "accepted"
        out["selected_offer_price_usd"] = str(selected_price)
        out["purchase_signal"] = "ready_to_buy"
        out["cannot_afford_any_offer_now"] = "false"
        out["deferred_purchase_intent"] = "false"

        # A limit needs limit WORDS. Saying yes to $30 is willingness to pay
        # $30; it is not "$30 is all I have", and the two are stored as
        # completely different facts (models/affordability.py).
        if _states_a_current_limit(text):
            out["current_budget_limit_usd"] = str(selected_price)
        elif _echoes_without_limit_language(
            out.get("current_budget_limit_usd"), selected_price, text
        ):
            # The analyzer is an LLM and its prompt's worked example pairs an
            # acceptance with a limit, so it generalises: a plain "yeah send
            # it" comes back with current_budget_limit_usd set to the offered
            # price. Downstream that is a HARD CEILING — it becomes
            # affordability.current_limit_cents, then price learning's
            # current_explicit_cap_cents, and every later offer to this fan is
            # capped at the first price he ever paid. This is the Terry
            # regression (AFFORDABILITY status=LIMITED_NOW limit=3000 after a
            # $30 purchase), and it is stripped deterministically rather than
            # trusted to prompt wording.
            print(
                "[COMMERCIAL EVENTS] dropped an echoed budget limit of "
                f"${out['current_budget_limit_usd']}: acceptance is not a ceiling"
            )
            out["current_budget_limit_usd"] = ""

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
            out["offer_response"] = "none"
            out["purchase_signal"] = "none"
            selected_price = None

    cannot_buy_any = bool(re.search(
        r"\b(can'?t afford (?:it|that|this)|can'?t pay (?:right now|today|yet)|"
        r"don'?t have (?:any )?money|no money|broke|not enough for (?:it|that))\b",
        text,
    ))
    if cannot_buy_any and selected_price is None:
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

    # The same echo, when the ANALYZER reported the acceptance and the regex
    # backstop did not. "ok" is not an acceptance word, so a bare "ok" after a
    # $30 offer reaches here with the model's accepted/limit pair intact and
    # would otherwise slip past the check above.
    if _echoes_the_accepted_price(out, text):
        print(
            "[COMMERCIAL EVENTS] dropped an echoed budget limit of "
            f"${out['current_budget_limit_usd']}: acceptance is not a ceiling"
        )
        out["current_budget_limit_usd"] = ""

    if not str(out.get("payday_raw") or "").strip():
        payday = _find_payday(text)
        if payday:
            out["payday_raw"] = payday
            out["payday_confidence"] = 0.95

    _normalize_compliment_only_interest(out, text)
    _promote_direct_intent(out, text)
    out["_latest_fan_message"] = latest_message
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
        "commercial_interest_signal": "none",
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
    offer_response = str(situation.get("offer_response") or "none").lower()

    if offer_response == "accepted" or selected_cents is not None:
        events.append(CommercialEvent(
            type=EventType.OFFER_ACCEPTED,
            raw_expression=str(situation.get("selected_offer_price_usd") or ""),
            amount_cents=selected_cents,
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
        # Legacy fallback only. Structured acceptance always wins and an
        # affordability pause requires cannot_afford_any_offer_now.
        if not any(e.type == EventType.OFFER_ACCEPTED for e in events):
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
        )
        if key not in seen:
            seen.add(key)
            output.append(event)
    return output


def stated_budget_cents(events: list[CommercialEvent]) -> int | None:
    for preferred_type in (
        EventType.OFFER_ACCEPTED,
        EventType.BUDGET_STATED,
        EventType.BUDGET_LIMIT_STATED,
    ):
        for event in events:
            if event.type == preferred_type and event.amount_cents is not None:
                return event.amount_cents
    return None


def accepted_offer_event(events: list[CommercialEvent]) -> CommercialEvent | None:
    return next((e for e in events if e.type == EventType.OFFER_ACCEPTED), None)
