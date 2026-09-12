"""Internal cents, human dollars — and never the two mixed up.

Money is stored, compared and allocated in integer cents everywhere inside the
backend, and stays that way: floats cannot represent a price and a rounding
error in a split is a wrong charge. What is enforced here is the OTHER side of
that boundary.

Cents are an internal representation, and the writer's prompt is customer-facing
copy in waiting. A model that reads ``price_cents: 3000`` will eventually write
"3000", and a model that reads ``$30.27`` will offer $30.27. So exactly one
renderer crosses the boundary, and the contract it keeps is narrow enough to
assert by scanning the built prompt:

* ``3000`` renders as ``$30``, never ``$30.00`` and never ``3000``;
* no ``*_cents`` key name and no raw cent magnitude reaches writer context;
* the default customer grid is $5, so cent-level prices do not arise by accident.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.content_pricing import DEFAULT_PRICE_STEP_CENTS, human_price_cents
from models.money import (
    customer_dollars,
    customer_dollars_or_none,
    is_on_price_grid,
    is_whole_dollars,
    whole_dollars,
)
from models.price_learning import PriceLearningPolicy


# ---------------------------------------------------------------------------
# 1. The renderer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cents", "expected"),
    [
        (3000, "$30"),
        (3500, "$35"),
        (500, "$5"),
        (0, "$0"),
        (12000, "$120"),
        # An agency that deliberately configures a cent grid gets an honest
        # rendering rather than a silent round.
        (3050, "$30.50"),
        (3027, "$30.27"),
        (3005, "$30.05"),
    ],
)
def test_internal_cents_render_as_the_amount_a_person_is_shown(cents, expected):
    assert customer_dollars(cents) == expected


def test_a_whole_dollar_amount_never_grows_a_decimal_tail():
    """Nobody writes "$30.00" in a text message."""
    for cents in (500, 1000, 2500, 3000, 9900):
        rendered = customer_dollars(cents)
        assert not rendered.endswith(".00")
        assert re.fullmatch(r"\$\d+", rendered)


def test_a_partial_amount_keeps_both_decimal_places():
    """"$30.5" is not a price anyone has ever seen."""
    assert customer_dollars(3050) == "$30.50"
    assert customer_dollars(3005) == "$30.05"


def test_unrenderable_values_produce_nothing_rather_than_a_wrong_number():
    assert customer_dollars(None) == ""
    assert customer_dollars("not money") == ""
    assert customer_dollars(None, default="—") == "—"
    assert customer_dollars_or_none(None) is None
    assert customer_dollars_or_none(3000) == "$30"


def test_whole_dollar_helpers():
    assert whole_dollars(3000) == 30
    assert whole_dollars(3050) == 30
    assert whole_dollars(None) == 0
    assert is_whole_dollars(3000) is True
    assert is_whole_dollars(3050) is False


# ---------------------------------------------------------------------------
# 2. The grid
# ---------------------------------------------------------------------------


def test_the_default_customer_grid_is_five_dollars():
    assert DEFAULT_PRICE_STEP_CENTS == 500
    assert PriceLearningPolicy().customer_price_step_cents == DEFAULT_PRICE_STEP_CENTS


@pytest.mark.parametrize("target", [1234, 1700, 2399, 3051, 4444])
def test_a_probed_price_lands_on_the_configured_grid(target):
    snapped = human_price_cents(target, 1000, 8000)
    assert is_on_price_grid(snapped, DEFAULT_PRICE_STEP_CENTS), (
        f"{target} snapped to {snapped}, which is not on the $5 grid"
    )
    assert customer_dollars(snapped).count(".") == 0


def test_a_deliberately_narrow_band_may_keep_its_exact_approved_price():
    """An approved fixed price of $17 stays $17: the grid is a default, not a
    licence to reprice content outside its approved band."""
    assert human_price_cents(1700, 1700, 1700) == 1700


def test_is_on_price_grid_rejects_a_nonsense_step():
    assert is_on_price_grid(3000, 0) is False
    assert is_on_price_grid(3000, -5) is False
    assert is_on_price_grid("x", 500) is False


# ---------------------------------------------------------------------------
# 3. Nothing cent-shaped reaches the writer
# ---------------------------------------------------------------------------

# Key names that mean "this number is in cents". Any of them appearing in copy
# handed to a model is a leak, because the model has no way to know the unit.
_CENT_KEY_RE = re.compile(r"\b[a-z_]*_cents\b", re.IGNORECASE)
# "3000 cents", "price in cents", etc.
_CENT_WORD_RE = re.compile(r"\bcents?\b", re.IGNORECASE)
# A dollar amount with a pointless zero tail, or a cent-level customer price.
_UGLY_MONEY_RE = re.compile(r"\$\d+\.00\b")


def _prompt_text(**context_overrides) -> str:
    from ai.prompt_builder import build_prompt
    from models.commercial import ActionType, CommercialDecision, PackageOption
    from models.schemas import ConversationContext, Fan, Message, Persona, StageType

    decision = CommercialDecision(
        action=ActionType.PRESENT_SESSION_OPTIONS,
        goal="present the approved options",
        mention_price=30,
        package_options=[
            PackageOption(
                package_id="package:quick:a",
                label="quick private session",
                price_cents=3000,
                set_ids=["a", "b"],
                experience="bedroom, black lingerie",
                legal_description="bedroom, black lingerie",
                step_count=2,
                media_count=5,
                asset_types=["photo_set", "photo_set"],
                content_floor_cents=1500,
                content_ceiling_cents=8000,
            ),
            PackageOption(
                package_id="package:full:a",
                label="full private session",
                price_cents=6500,
                set_ids=["a", "b", "c"],
                experience="bedroom, black lingerie, toy",
                legal_description="bedroom, black lingerie, toy",
                step_count=3,
                media_count=9,
                asset_types=["photo_set", "photo_set", "photo_set"],
                content_floor_cents=2000,
                content_ceiling_cents=12000,
            ),
        ],
    )

    context = dict(
        fan_message="how much?",
        conversation_history=[Message(role="fan", content="how much?")],
        fan_profile=Fan(id="fan-1", display_name="Jostar", total_spent=120),
        creator_persona=Persona(),
        similar_exchanges=[],
        conversation_stage=StageType.PRE_UPSELL,
        situation={"strategic_move": "present options", "purchase_signal": "none"},
        commercial_decision=decision.model_dump(mode="json"),
        affordability={
            "status": "CONSTRAINED",
            "current_limit_cents": 4000,
            "current_available_cents": 3500,
            "latest_offer_selected_cents": 3000,
            "latest_counteroffer_cents": 2500,
            "latest_rejected_price_cents": 5000,
            "last_confirmed_purchase_cents": 2000,
            "highest_confirmed_purchase_cents": 6000,
            "confirmed_purchase_count": 3,
        },
        price_learning={
            "mode": "CALIBRATED",
            "confidence": "MEDIUM",
            "recommended_floor_cents": 2000,
            "recommended_target_cents": 3000,
            "recommended_ceiling_cents": 5500,
            "reason_codes": ["repeat_buyer"],
        },
        session_strategy={
            "goal": "SELL",
            "phase": "OFFER",
            "next_action": "PRESENT_OPTIONS",
            "writer_goal": "present the options",
            "approved_offer_prices_cents": [3000, 6500],
        },
        buyer_lifecycle={
            "stage": "REPEAT",
            "confirmed_purchase_count": 3,
            "total_spent_cents": 12000,
        },
        fan_intelligence={
            "facts": [
                {"fact_key": "stated_budget_cents", "value": 4000, "status": "explicit"},
                {"fact_key": "accepted_price_cents", "value": 3000, "status": "explicit"},
                {"fact_key": "rejected_price_cents", "value": 5000, "status": "explicit"},
                {"fact_key": "counteroffer_cents", "value": 2500, "status": "inferred"},
            ]
        },
        media_inventory={
            "authorized_asset_types": ["photo_set"],
            "available_package_asset_types": ["photo_set"],
            "vault_asset_types": ["photo_set"],
            "known": True,
        },
    )
    context.update(context_overrides)
    messages = build_prompt(ConversationContext(**context))
    return "\n".join(
        block["text"] if isinstance(block, dict) and "text" in block else str(block)
        for message in messages
        for block in (
            message["content"]
            if isinstance(message["content"], list)
            else [message["content"]]
        )
    )


def test_no_cent_key_name_reaches_writer_context():
    leaked = sorted(set(_CENT_KEY_RE.findall(_prompt_text())))
    assert leaked == [], f"cent-denominated key names in writer context: {leaked}"


def test_the_word_cents_never_reaches_writer_context():
    assert not _CENT_WORD_RE.search(_prompt_text())


def test_no_raw_cent_magnitude_reaches_writer_context():
    """Every amount put into the prompt is one of the dollar renderings."""
    text = _prompt_text()
    for raw in ("3000", "6500", "4000", "5500", "12000", "2500"):
        assert raw not in text, f"raw cent value {raw} reached the writer"


def test_prices_are_rendered_as_whole_dollars():
    text = _prompt_text()
    assert "$30" in text
    assert "$65" in text
    assert not _UGLY_MONEY_RE.search(text), "no $30.00 in customer-facing copy"


def test_every_dollar_amount_in_the_prompt_is_on_a_human_grid():
    """A cent-level price in writer context means an internal value escaped."""
    text = _prompt_text()
    for amount in re.findall(r"\$(\d+(?:\.\d+)?)", text):
        if "." not in amount:
            continue
        pytest.fail(f"cent-level customer amount ${amount} reached the writer")


def test_the_prompt_builder_has_no_ad_hoc_money_formatting():
    """One renderer, so this contract has exactly one place to regress."""
    source = (
        Path(__file__).resolve().parents[1] / "ai" / "prompt_builder.py"
    ).read_text(encoding="utf-8")

    assert "customer_dollars" in source
    assert "/ 100" not in source, (
        "money must cross into writer context through models/money.py only"
    )


def test_internal_storage_is_still_integer_cents():
    """The fix is a rendering boundary, not a migration. Nothing here may turn
    into floats or decimal dollars."""
    from models.vault_pricing import allocate_step_prices

    rows = [
        {"id": "a", "base_price_cents": 1500, "min_price_cents": 1000, "max_price_cents": 4000},
        {"id": "b", "base_price_cents": 2000, "min_price_cents": 1000, "max_price_cents": 5000},
    ]
    allocations = allocate_step_prices(3500, rows)
    assert allocations is not None
    assert all(isinstance(value, int) for value in allocations)
    assert sum(allocations) == 3500
