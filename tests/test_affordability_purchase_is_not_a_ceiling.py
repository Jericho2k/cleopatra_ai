"""The Terry regression: a $30 purchase became a $30 ceiling, forever.

The logs were::

    AFFORDABILITY status=LIMITED_NOW limit=3000
    PRICE LEARNING mode=EXACT target=3000 confidence=HIGH

after Terry was offered and bought a $30 unlock. He never said "$30 max" and
never said "that's all I have". Two independent defects produced those lines,
and both of them turned willingness to pay into an inability to pay more:

1. ``current_limit_cents=3000``. The analyzer's prompt paired an acceptance
   with ``current_budget_limit_usd=<that price>`` in its worked example (the
   example's fan HAD said "I don't have more"), and the model generalised the
   pattern to a bare "yeah send it". ``current_limit_cents`` is read as a HARD
   ceiling by ``_current_hard_ceiling`` and by ``probe_price_cents``, so the
   first price he ever paid capped every offer that followed.

2. ``mode=EXACT target=3000``. ``latest_offer_selected_cents`` was never
   cleared by the purchase it led to, and a pending selection is authoritative
   in ``derive_price_learning_profile`` — so the resolved selection kept
   pinning the recommendation at exactly the amount he had already paid.

A purchase is a LOWER BOUND on willingness. Only an explicit current statement
may create a ceiling.
"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.affordability import (  # noqa: E402
    AffordabilityAuthority,
    AffordabilityEvent,
    AffordabilityEventType,
    AffordabilityState,
    AffordabilityStatus,
    apply_affordability_event,
)
from models.commercial import CreatorPolicy  # noqa: E402
from models.price_learning import (  # noqa: E402
    PriceLearningPolicy,
    PriceRecommendationMode,
    derive_price_learning_profile,
    probe_price_cents,
)
from services.commercial_events import extract_events, normalize_commercial_facts  # noqa: E402
from services.media_packages import build_next_offer  # noqa: E402

NOW = datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)


def _event(event_type, cents=None, *, at=NOW, authority=AffordabilityAuthority.CHAT_EXPLICIT):
    return AffordabilityEvent(
        event_type=event_type,
        authority=authority,
        amount_cents=cents,
        occurred_at=at,
    )


def _accept_then_buy(cents=3000):
    """The exact Terry sequence: offered, accepted, paid."""
    state = apply_affordability_event(
        AffordabilityState(),
        _event(AffordabilityEventType.OFFER_SELECTED, cents),
    )
    return apply_affordability_event(
        state,
        _event(
            AffordabilityEventType.PURCHASE_CONFIRMED,
            cents,
            at=NOW + timedelta(minutes=2),
            authority=AffordabilityAuthority.PAYMENT_CONFIRMED,
        ),
    )


# --- 1. the analyzer's echoed limit ------------------------------------------


def test_a_bare_acceptance_does_not_report_a_budget_limit():
    facts = normalize_commercial_facts(
        {"current_budget_limit_usd": "30"},
        "yeah send it",
        ["it's $30 for that one"],
    )
    assert facts["selected_offer_price_usd"] == "30"
    assert facts["current_budget_limit_usd"] == ""

    events = extract_events(facts)
    assert not [
        event
        for event in events
        if event.type.value in {"BUDGET_LIMIT_STATED", "COUNTEROFFER_STATED"}
    ]


def test_an_actual_stated_ceiling_still_survives():
    """The fix strips an echo, not a fact. "$30 max" has to keep working."""
    facts = normalize_commercial_facts(
        {},
        "yeah send it, that's all i have right now",
        ["it's $30 for that one"],
    )
    assert facts["current_budget_limit_usd"] == "30"


def test_a_limit_the_analyzer_read_somewhere_else_is_not_overruled():
    facts = normalize_commercial_facts(
        {"current_budget_limit_usd": "50"},
        "yeah send it",
        ["it's $30 for that one"],
    )
    assert facts["current_budget_limit_usd"] == "50"


def test_a_purchase_alone_never_reaches_limited_now():
    state = _accept_then_buy()
    context = state.to_context(now=NOW + timedelta(minutes=5))
    assert context["current_limit_cents"] is None
    assert context["status"] != AffordabilityStatus.LIMITED_NOW.value
    assert context["highest_confirmed_purchase_cents"] == 3000


# --- 2. the resolved selection that kept pinning EXACT -----------------------


def test_a_purchase_resolves_the_selection_it_came_from():
    state = _accept_then_buy()
    assert state.latest_offer_selected_cents is None, (
        "a selection that has been PAID is resolved; leaving it pending is what "
        "kept price learning pinned at the price he already paid"
    )


def test_price_learning_is_no_longer_pinned_to_exactly_what_he_paid():
    state = _accept_then_buy()
    profile = derive_price_learning_profile(
        [
            {
                "event_type": "PURCHASE_CONFIRMED",
                "amount_cents": 3000,
                "occurred_at": NOW + timedelta(minutes=2),
            }
        ],
        affordability=state.to_context(now=NOW + timedelta(minutes=5)),
        lifecycle={"stage": "REPEAT_BUYER"},
        now=NOW + timedelta(minutes=5),
    )
    assert profile.mode is not PriceRecommendationMode.EXACT
    assert profile.evidence_summary["current_explicit_cap_cents"] is None
    assert profile.evidence_summary["demonstrated_willingness_cents"] == 3000


# --- 3. and so the next probe can actually go up -----------------------------


def test_the_probe_after_a_purchase_rises_inside_approved_bounds():
    state = _accept_then_buy()
    context = {
        "mode": "RANGE",
        "confirmed_purchase_count": 1,
        "evidence_summary": {
            "demonstrated_willingness_cents": 3000,
            "confirmed_purchase_count": 1,
            "current_explicit_cap_cents": state.current_limit_cents,
        },
    }
    probe = probe_price_cents(2000, 8000, price_learning=context)
    assert probe is not None
    assert probe.price_cents > 3000, "a purchase is a floor to probe up from"
    assert probe.price_cents <= 8000, "and never past the approved ceiling"


def test_sequential_purchases_probe_upward_and_stay_inside_the_range():
    """Three successful unlocks in a row must climb, and must stop at the top."""
    prices: list[int] = []
    demonstrated = 0
    for _ in range(3):
        probe = probe_price_cents(
            2000,
            8000,
            price_learning={
                "mode": "RANGE",
                "confirmed_purchase_count": len(prices),
                "evidence_summary": {
                    "demonstrated_willingness_cents": demonstrated or None,
                    "confirmed_purchase_count": len(prices),
                    "current_explicit_cap_cents": None,
                },
            },
        )
        assert probe is not None
        prices.append(probe.price_cents)
        demonstrated = probe.price_cents

    assert prices == sorted(prices), f"prices must not go backwards: {prices}"
    assert prices[-1] > prices[0], f"sequential purchases must probe upward: {prices}"
    assert max(prices) <= 8000, "approved content bounds are absolute"


def test_an_explicitly_stated_ceiling_does_still_cap_the_probe():
    """The control. Without this, the fix above could be "caps never apply"."""
    probe = probe_price_cents(
        2000,
        8000,
        price_learning={
            "mode": "RANGE",
            "evidence_summary": {
                "demonstrated_willingness_cents": 3000,
                "current_explicit_cap_cents": 3000,
            },
        },
    )
    assert probe is not None
    assert probe.price_cents <= 3000


def test_the_second_offer_costs_more_than_the_first_and_stays_approved():
    """End to end through the real offer builder, not just the probe."""
    rows = [
        {
            "id": "set-a",
            "media_ids": ["a1", "a2", "a3"],
            "title": "bathroom lingerie",
            "tags": ["lingerie_photo"],
            "explicit_min": 2,
            "explicit_max": 3,
            "base_price_cents": 2000,
            "min_price_cents": 1000,
            "max_price_cents": 8000,
        },
    ]
    first = build_next_offer(rows, CreatorPolicy(), confirmed_purchase_count=0)
    second = build_next_offer(rows, CreatorPolicy(), confirmed_purchase_count=2)
    assert first is not None and second is not None
    assert second.price_cents > first.price_cents
    assert second.price_cents <= second.content_ceiling_cents


def test_a_purchase_is_not_evidence_of_a_ceiling_in_the_policy_defaults():
    """Guards the constant that makes upward probing possible at all."""
    assert PriceLearningPolicy().max_step_up_bps > 0


def test_an_analyzer_reported_acceptance_echo_is_stripped_too():
    """The regex backstop does not see every acceptance.

    "ok" is not one of the acceptance words, so a bare "ok" after a $30 offer
    arrives with the MODEL's accepted/limit pair intact and never touches the
    selected-price branch. The echo has to be caught on the reported facts as
    well as on the ones the backstop derived.
    """
    facts = normalize_commercial_facts(
        {
            "offer_response": "accepted",
            "selected_offer_price_usd": "30",
            "current_budget_limit_usd": "30",
        },
        "ok",
        ["it's $30 for that one"],
    )
    assert facts["current_budget_limit_usd"] == ""
    assert facts["selected_offer_price_usd"] == "30", "the acceptance survives"


def test_the_same_message_with_limit_words_keeps_the_ceiling():
    facts = normalize_commercial_facts(
        {
            "offer_response": "accepted",
            "selected_offer_price_usd": "30",
            "current_budget_limit_usd": "30",
        },
        "ok thats all i have though",
        ["it's $30 for that one"],
    )
    assert facts["current_budget_limit_usd"] == "30"
