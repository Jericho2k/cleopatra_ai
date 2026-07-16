"""Translate situation-analyzer JSON into typed commercial observations.

The crucial rule is that acceptance, present budget and future payday are
independent facts. A message such as "I'll take the $28 one; I get paid Friday"
means PACKAGE_SELECTED($28) + BUDGET_LIMIT_STATED($28) + PAYDAY_MENTIONED,
not MONEY_UNAVAILABLE.
"""
import re

from models.commercial import CommercialEvent, EventType, PackageOption


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
    r"what\s+are\s+(?:those|they|my\s+options|the\s+(?:two\s+)?options)|"
    r"what(?:'s|\s+is)\s+(?:the\s+)?difference|"
    r"how\s+are\s+they\s+different|"
    r"tell\s+me\s+more(?:\s+about\s+(?:those|them|the\s+options|the\s+(?:first|second|quick|full)\s+(?:one|option|session)))?(?:\s*[?.!])?$|"
    r"more\s+about\s+(?:those|them|the\s+options|the\s+(?:first|second|quick|full)\s+(?:one|option|session))|"
    r"what\s+do\s+i\s+get|what\s+do\s+(?:those|they)\s+include|what(?:'s|\s+is)\s+included|"
    r"explain\s+(?:them|those|the\s+options)|"
    r"i(?:'m|\s+am)\s+all\s+ears|go\s+on"
    r")\b",
    re.IGNORECASE,
)

_OFFER_DETAIL_QUESTION_RE = re.compile(
    r"\b(?:what|which|how)\b.*\b(?:first|second|1st|2nd|options?|quick|full|session)\b",
    re.IGNORECASE,
)

_GENERIC_OFFER_ACCEPT_RE = re.compile(
    r"\b(?:i(?:'m|\s+am)\s+in|let(?:'s|\s+us)\s+do\s+it|sounds\s+good|"
    r"i(?:'ll|\s+will)\s+take\s+it|i\s+want\s+it|deal|that\s+one|this\s+one)\b",
    re.IGNORECASE,
)

_SELECTION_WORD_RE = re.compile(
    r"\b(?:take|choose|pick|want|go\s+with|give\s+me|send\s+me|"
    r"option|one|session|package|set|quick|full|cheaper|lower|higher|second|first)\b",
    re.IGNORECASE,
)

_OFFER_TOKEN_STOPWORDS = {
    "a", "an", "and", "at", "for", "from", "i", "in", "is", "it", "me",
    "my", "of", "on", "one", "option", "package", "private", "session", "set",
    "that", "the", "this", "to", "want", "with", "you", "your",
}


def _offer_tokens(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", str(value or "").lower())
        if len(token) > 1 and token not in _OFFER_TOKEN_STOPWORDS
    }


def _event_for_package(package: PackageOption, raw_expression: str, *, reason: str) -> CommercialEvent:
    set_ids = list(package.set_ids or ([package.set_id] if package.set_id else []))
    return CommercialEvent(
        type=EventType.PACKAGE_SELECTED,
        raw_expression=raw_expression,
        amount_cents=package.price_cents,
        confidence=0.99,
        metadata={
            "package_id": package.package_id,
            "set_id": set_ids[0] if set_ids else None,
            "set_ids": set_ids,
            "label": package.label,
            "experience": package.experience,
            "legal_description": package.legal_description or package.experience,
            "selection_reason": reason,
        },
    )


def _unique_price_package(packages: list[PackageOption], *, lowest: bool) -> PackageOption | None:
    if not packages:
        return None
    target = (min if lowest else max)(package.price_cents for package in packages)
    matches = [package for package in packages if package.price_cents == target]
    return matches[0] if len(matches) == 1 else None


def _semantic_offer_match(
    text: str,
    packages: list[PackageOption],
) -> tuple[PackageOption | None, bool]:
    text_tokens = _offer_tokens(text)
    if not text_tokens:
        return None, False
    scored: list[tuple[int, PackageOption]] = []
    for package in packages:
        package_tokens = _offer_tokens(
            " ".join(
                value
                for value in (
                    package.label,
                    package.legal_description or "",
                    package.experience or "",
                )
                if value
            )
        )
        score = len(text_tokens & package_tokens)
        if score:
            scored.append((score, package))
    if not scored:
        return None, False
    best_score = max(score for score, _ in scored)
    best = [package for score, package in scored if score == best_score]
    return (best[0], False) if len(best) == 1 else (None, True)


def resolve_pending_offer_reference(
    latest_message: str,
    offered_packages: list[PackageOption],
) -> tuple[PackageOption | None, bool, str]:
    """Resolve a fan reference only against the persisted ordered snapshot.

    Returns ``(package, ambiguous, reason)``. No package outside the snapshot can
    ever be selected here.
    """
    text = str(latest_message or "").strip().lower()
    if not text or not offered_packages:
        return None, False, "no_active_snapshot"

    # Exact price references are authoritative only when they uniquely match the
    # active snapshot.
    mentioned_cents = [value * 100 for value in _extract_money_values(text)]
    for cents in mentioned_cents:
        matches = [package for package in offered_packages if package.price_cents == cents]
        if len(matches) == 1 and _SELECTION_WORD_RE.search(text):
            return matches[0], False, "exact_price"
        if len(matches) > 1:
            return None, True, "duplicate_price"

    if re.search(r"\b(?:first|1st)\b", text):
        return offered_packages[0], False, "ordinal_first"
    if re.search(r"\b(?:second|2nd)\b", text):
        if len(offered_packages) >= 2:
            return offered_packages[1], False, "ordinal_second"
        return None, True, "missing_second_option"

    if re.search(r"\b(?:cheaper|cheapest|lower|lowest|smaller)\b", text):
        package = _unique_price_package(offered_packages, lowest=True)
        return (package, package is None, "price_rank_low")
    if re.search(r"\b(?:pricier|expensive|higher|highest|bigger|premium)\b", text):
        package = _unique_price_package(offered_packages, lowest=False)
        return (package, package is None, "price_rank_high")

    semantic, tied = _semantic_offer_match(text, offered_packages)
    selectionish = bool(_SELECTION_WORD_RE.search(text) or len(_offer_tokens(text)) <= 3)
    if semantic and selectionish:
        return semantic, False, "label_or_experience"
    if tied and selectionish:
        return None, True, "ambiguous_label_or_experience"

    if _GENERIC_OFFER_ACCEPT_RE.search(text):
        if len(offered_packages) == 1:
            return offered_packages[0], False, "single_option_acceptance"
        return None, True, "generic_reference"

    return None, False, "no_selection_reference"


def is_pending_offer_detail_request(latest_message: str) -> bool:
    text = str(latest_message or "").strip().lower()
    return bool(_OFFER_DETAIL_RE.search(text) or _OFFER_DETAIL_QUESTION_RE.search(text))


def augment_pending_offer_events(
    events: list[CommercialEvent],
    latest_message: str,
    offered_packages: list[PackageOption],
) -> None:
    """Attach the latest message to the exact persisted offer snapshot.

    Detail questions outrank accidental analyzer selection. Selection references
    are resolved deterministically by index, exact price, unique label, or unique
    approved experience. Ambiguity is explicit and never guessed.
    """
    if not offered_packages:
        return

    if is_pending_offer_detail_request(latest_message):
        events[:] = [event for event in events if event.type != EventType.PACKAGE_SELECTED]
        if not any(event.type == EventType.OFFER_DETAILS_REQUESTED for event in events):
            events.append(
                CommercialEvent(
                    type=EventType.OFFER_DETAILS_REQUESTED,
                    raw_expression=str(latest_message or ""),
                    confidence=0.99,
                    metadata={"snapshot_package_ids": [p.package_id for p in offered_packages]},
                )
            )
        return

    package, ambiguous, reason = resolve_pending_offer_reference(
        latest_message,
        offered_packages,
    )
    existing = selected_package_event(events)
    if package:
        replacement = _event_for_package(package, str(latest_message or ""), reason=reason)
        if existing:
            events[events.index(existing)] = replacement
        else:
            events.append(replacement)
        return

    if ambiguous:
        events[:] = [event for event in events if event.type != EventType.PACKAGE_SELECTED]
        events.append(
            CommercialEvent(
                type=EventType.OFFER_SELECTION_AMBIGUOUS,
                raw_expression=str(latest_message or ""),
                confidence=0.99,
                metadata={
                    "reason": reason,
                    "snapshot_package_ids": [p.package_id for p in offered_packages],
                },
            )
        )


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
        or str(out.get("selected_offer_position") or "").strip()
        or str(out.get("counteroffer_usd") or "").strip()
        or str(out.get("budget_stated_usd") or "").strip()
        or str(out.get("current_budget_limit_usd") or "").strip()
        or _truthy(out.get("cannot_afford_any_offer_now"))
        or _truthy(out.get("deferred_purchase_intent"))
        or _truthy(out.get("resend_requested"))
        or purchase_signal in {"bought", "money_available", "declined"}
    )


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

    _normalize_compliment_only_interest(out, text)
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
