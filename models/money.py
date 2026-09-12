"""The boundary between internal cents and human dollars.

Money is stored, compared and allocated in integer cents everywhere inside the
backend, and that does not change: floats cannot represent a price, and a
rounding error in a split is a wrong charge. ``models/vault_pricing.py`` and
``models/price_learning.py`` stay exactly as they are.

What this module fixes is the other side of that boundary. Cents are an internal
representation, and three audiences must never see one:

* the **writer**, whose prompt context is customer-facing copy in waiting — a
  model that reads ``price_cents: 3000`` can and eventually will write "3000";
* the **fan**, for the same reason one step later;
* the **dashboard**, where an operator reasons in the same units the fan does.

So exactly one function renders money for a person, and the contract it keeps is
narrow enough to test by scanning: ``$30``, never ``$30.00``, never ``$30.27`` on
the default grid, never ``3000``, never the word "cents".

Cent-level prices are not forbidden outright — an agency that deliberately
configures a one-cent grid gets ``$30.27`` rendered honestly rather than rounded
behind its back. They are simply not the default, and nothing produces one by
accident: ``models/content_pricing.DEFAULT_PRICE_STEP_CENTS`` is $5.
"""

from __future__ import annotations

from typing import Any

CENTS_PER_DOLLAR = 100


def customer_dollars(cents: Any, *, default: str = "") -> str:
    """Render internal cents as the amount a person is shown.

    ``3000 -> "$30"``, ``3500 -> "$35"``, ``3050 -> "$30.50"``. A whole-dollar
    amount never grows a ``.00`` tail, because no human writes one in a text
    message, and a partial amount keeps both decimal places, because ``$30.5``
    is not a price anyone has ever seen.
    """
    try:
        value = int(cents)
    except (TypeError, ValueError):
        return default
    sign = "-" if value < 0 else ""
    value = abs(value)
    whole, remainder = divmod(value, CENTS_PER_DOLLAR)
    if remainder == 0:
        return f"{sign}${whole}"
    return f"{sign}${whole}.{remainder:02d}"


def customer_dollars_or_none(cents: Any) -> str | None:
    """``customer_dollars`` for optional values: ``None`` in, ``None`` out."""
    if cents is None:
        return None
    rendered = customer_dollars(cents, default="")
    return rendered or None


def whole_dollars(cents: Any, *, default: int = 0) -> int:
    """The whole-dollar magnitude of an internal amount, truncated."""
    try:
        return int(cents) // CENTS_PER_DOLLAR
    except (TypeError, ValueError):
        return default


def is_on_price_grid(cents: Any, step_cents: int) -> bool:
    """Whether one customer-facing price sits on the agency's configured grid."""
    try:
        value = int(cents)
        step = int(step_cents)
    except (TypeError, ValueError):
        return False
    if step <= 0:
        return False
    return value >= 0 and value % step == 0


def is_whole_dollars(cents: Any) -> bool:
    """Whether an amount has no cent component at all."""
    try:
        return int(cents) % CENTS_PER_DOLLAR == 0
    except (TypeError, ValueError):
        return False
