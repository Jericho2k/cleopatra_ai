"""Tease inventory is not sellable inventory.

`tease` is represented in this codebase as the classifier categories
``teaser_clothed`` and ``teaser_bundle`` — the two entries of
``VAULT_CATEGORIES`` the agency priced $0-$0 — and, on hand-curated and legacy
rows, as a plain ``tease``/``teaser``/``free`` tag. Set generation copies a
media item's ``content_category`` into the SET's tags, so both forms show up on
``vault_sets``.

Three media queries in db/queries.py already excluded the categories. Nothing
excluded them from ``vault_sets``, which is what offer construction and session
planning actually read — so teaser content could be offered and delivered as a
paid unlock. One predicate now answers the question for all of them.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.content_pricing import (  # noqa: E402
    is_paid_sellable,
    paid_sellable_block_reason,
    paid_sellable_rows,
)
from services.media_packages import build_next_offer, usable_sets  # noqa: E402
from models.commercial import CreatorPolicy  # noqa: E402


def _set(set_id, **extra):
    row = {
        "id": set_id,
        "media_ids": [f"{set_id}-a", f"{set_id}-b"],
        "title": set_id,
        "suggested_price": 20,
        "base_price_cents": 2000,
        "min_price_cents": 1500,
        "max_price_cents": 8000,
    }
    row.update(extra)
    return row


# --- how tease is actually represented ---------------------------------------


def test_both_teaser_categories_are_blocked():
    for category in ("teaser_clothed", "teaser_bundle"):
        assert is_paid_sellable({"content_category": category}) is False
        assert paid_sellable_block_reason({"content_category": category}).startswith(
            "free_only_content"
        )


def test_the_category_carried_as_a_set_TAG_is_blocked_too():
    """Set generation writes content_category into tags; that is the live form."""
    assert is_paid_sellable({"tags": ["teaser_clothed", "bathroom", "blue"]}) is False


def test_a_bare_tease_tag_on_a_hand_curated_row_is_blocked():
    assert is_paid_sellable({"tags": ["tease"]}) is False
    assert is_paid_sellable({"tags": ["teaser"]}) is False
    assert is_paid_sellable({"tags": ["free_only"]}) is False


def test_striptease_is_not_caught_by_the_word_tease():
    """The category the agency prices at $15-$100 contains the substring.

    Matching on whole normalized tokens rather than substrings is the only
    reason this works, so it is asserted rather than assumed.
    """
    assert is_paid_sellable({"content_category": "striptease_video"}) is True
    assert is_paid_sellable({"tags": ["striptease", "lingerie_photo"]}) is True


def test_a_mixed_shoot_is_priced_by_its_strongest_content():
    """A nude set that happens to contain one clothed frame is a nude set.

    This mirrors row_category_range_cents, which already prices a mixed row by
    its most valuable category. Blocking it would quietly delete real inventory.
    """
    assert is_paid_sellable({"tags": ["teaser_clothed", "nude_photo"]}) is True


def test_a_hand_curated_set_with_no_category_evidence_is_left_alone():
    """Operators curate sets by hand. This boundary is about tease, not them."""
    assert is_paid_sellable({"title": "Valentines custom", "suggested_price": 40}) is True


def test_an_explicit_operator_decision_outranks_every_inference():
    assert is_paid_sellable({"content_category": "nude_video", "paid_sellable": False}) is False
    assert (
        paid_sellable_block_reason({"content_category": "nude_video", "paid_sellable": False})
        == "marked_not_paid_sellable"
    )


# --- and it is enforced where selling actually happens -----------------------


def test_teaser_sets_never_reach_the_planner_through_usable_sets():
    """usable_sets is the single chokepoint for offers AND for delivery."""
    rows = [
        _set("tease-1", content_category="teaser_clothed"),
        _set("tease-2", tags=["teaser_bundle"]),
        _set("real-1", content_category="nude_photo", tags=["nude_photo"]),
    ]
    assert [row["id"] for row in usable_sets(rows)] == ["real-1"]


def test_a_priced_teaser_set_is_still_not_sellable():
    """An operator typing a number into the Sets UI must not create a PPV.

    Before the predicate, price_bounds' final fall-through ("the approved price
    is exactly the approved price") meant a $0-$0 teaser with a price column
    read as a fixed-price sellable set.
    """
    priced_tease = _set(
        "tease-priced",
        content_category="teaser_clothed",
        base_price_cents=2500,
        min_price_cents=2500,
        max_price_cents=2500,
        dynamic_pricing_enabled=False,
    )
    assert usable_sets([priced_tease]) == []


def test_an_all_teaser_vault_produces_no_offer_at_all():
    """Refusing to offer is the correct outcome. Selling junk is not."""
    rows = usable_sets(
        [
            _set("tease-1", content_category="teaser_clothed"),
            _set("tease-2", content_category="teaser_bundle"),
        ]
    )
    assert build_next_offer(rows, CreatorPolicy()) is None


def test_the_offer_built_from_a_mixed_vault_is_the_real_content():
    rows = usable_sets(
        [
            _set("tease-1", content_category="teaser_clothed"),
            _set("real-1", content_category="nude_photo", tags=["nude_photo"]),
        ]
    )
    offer = build_next_offer(rows, CreatorPolicy())
    assert offer is not None
    assert offer.set_id == "real-1"


def test_tease_content_is_filtered_not_deleted():
    """It stays in the vault on purpose — a free reward is a real use for it."""
    rows = [_set("tease-1", content_category="teaser_clothed"), _set("real-1", content_category="nude_photo")]
    kept = paid_sellable_rows(rows)
    assert len(rows) == 2, "the caller's own list is untouched"
    assert [row["id"] for row in kept] == ["real-1"]
