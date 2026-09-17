"""Authoritative state a trajectory needs before its first turn.

WHY
---
The continuation brief asks for "authoritative purchase/delivery fixtures",
and names the reason: the paid-content trajectories contain claims of payment,
not seeded authoritative purchase events.

Two of review §5's rows depend on the difference. "He cannot open something he
paid for" is a test of what happens when the ledger DOES show a purchase and
the customer cannot reach it. A fixture where the customer merely types "i paid
for this" tests the opposite case — the one where nothing was bought — and the
correct behaviour there is to refuse, which is what
``services/content_access.py`` does:

    this customer has no confirmed purchase to resend. A complaint is not proof
    of payment [...]

So the trajectory passed while exercising none of the code it claimed to cover.
Seeding the ledger is what makes the row test the thing its label says.

THE GUARD
---------
This module writes rows that say a customer paid. That is fabricated payment
evidence, and against a real conversation it would be indefensible — it would
license a free resend of paid media, move commercial state, and corrupt the
only record this system treats as authoritative about money.

So every seed refuses unless the fan is a simulator test fan
(``fans.platform_fan_id`` starting ``test_``, via
``core.simulation.is_simulatable_fan``). The check reads the fan row from the
database rather than trusting the caller's id, because the caller is a CLI
argument and a typo is exactly how this would reach the wrong conversation.

References are namespaced ``eval:`` for the same reason: a seeded delivery is
identifiable as seeded forever after, by anyone reading the table, without
needing to know which run produced it.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import Any

from core import clock
from core.simulation import is_simulatable_fan
from core.supabase import get_supabase

#: Every seeded reference carries this. A row in ppv_deliveries is the system's
#: authority on whether money arrived, so one that was invented by an
#: evaluation must never be mistakable for one that was not.
SEED_PREFIX = "eval:"


class FixtureRefused(RuntimeError):
    """A fixture would have written authoritative state somewhere it must not."""


def is_seeded_reference(reference: object) -> bool:
    return str(reference or "").startswith(SEED_PREFIX)


async def _require_test_fan(creator_id: str, fan_id: str) -> dict[str, Any]:
    """The fan row, if and only if it is a simulator test fan."""

    def _load() -> dict[str, Any]:
        result = (
            get_supabase().table("fans")
            .select("id, creator_id, platform_fan_id")
            .eq("id", fan_id)
            .limit(1)
            .execute()
        )
        rows = result.data or []
        return dict(rows[0]) if rows else {}

    fan = await asyncio.to_thread(_load)
    if not fan:
        raise FixtureRefused(f"fan {fan_id} was not found")
    if str(fan.get("creator_id")) != str(creator_id):
        # A fan id that belongs to another creator is either a typo or a
        # cross-tenant mistake, and both end with authoritative state written
        # into a conversation nobody meant to touch.
        raise FixtureRefused(
            f"fan {fan_id} does not belong to creator {creator_id}"
        )
    if not is_simulatable_fan(fan.get("platform_fan_id")):
        raise FixtureRefused(
            f"fan {fan_id} is not a simulator test fan. Seeding a purchase "
            "writes evidence that this customer paid, which would license a "
            "free resend of paid media and move commercial state. It is "
            "allowed only against test_ fans."
        )
    return fan


async def seed_purchase(
    *,
    creator_id: str,
    fan_id: str,
    reference: str,
    media_ids: list[str],
    price_cents: int,
    purchased_days_ago: float = 0.0,
) -> dict[str, Any]:
    """Record a delivery the ledger shows as purchased.

    Written directly rather than through ``claim_delivery`` +
    ``transition_delivery``: those enforce the real lifecycle, which is correct
    for the product and wrong here — a fixture needs to establish a past state,
    not to re-enact reaching it, and re-enacting it would also send.

    ``purchased_days_ago`` is read off the evaluation clock, so a purchase can
    be placed before a simulated absence and the conversation can return to it
    a week later.
    """
    await _require_test_fan(creator_id, fan_id)

    if not str(reference).startswith(SEED_PREFIX):
        reference = f"{SEED_PREFIX}{reference}"
    when = clock.now() - timedelta(days=float(purchased_days_ago or 0.0))
    row = {
        "reference": reference,
        "creator_id": str(creator_id),
        "fan_id": str(fan_id),
        "status": "purchased",
        "media_ids": [str(value) for value in media_ids],
        "price_cents": int(price_cents),
        "source": "trajectory_fixture",
        "set_id": None,
        "step_index": None,
        "claimed_at": when.isoformat(),
        "purchased_at": when.isoformat(),
        "platform_message_id": f"{reference}:platform",
    }

    def _write() -> dict[str, Any]:
        result = (
            get_supabase().table("ppv_deliveries")
            .upsert(row, on_conflict="reference")
            .execute()
        )
        return (result.data or [row])[0]

    return await asyncio.to_thread(_write)


async def clear_seeded(creator_id: str, fan_id: str) -> int:
    """Remove this fan's seeded deliveries. Returns how many went.

    Scenario isolation: the brief asks for fresh state between independent
    scenarios, and a purchase left behind by the previous trajectory is state
    the next one would read as real.

    Only ``eval:`` references are removed, so a test fan that also carries
    genuine deliveries — one an operator made by hand while trying something —
    keeps them.
    """
    await _require_test_fan(creator_id, fan_id)

    def _delete() -> int:
        result = (
            get_supabase().table("ppv_deliveries")
            .delete()
            .eq("creator_id", str(creator_id))
            .eq("fan_id", str(fan_id))
            .like("reference", f"{SEED_PREFIX}%")
            .execute()
        )
        return len(result.data or [])

    return await asyncio.to_thread(_delete)


async def apply_seed(
    *, creator_id: str, fan_id: str, seed: dict[str, Any]
) -> list[dict[str, Any]]:
    """Establish everything one trajectory declares it needs beforehand.

    Returns the rows written, so a report can say what state the run started
    from rather than leaving a reader to infer it from the transcript.
    """
    if not isinstance(seed, dict):
        return []
    written: list[dict[str, Any]] = []
    for index, purchase in enumerate(seed.get("purchases") or []):
        if not isinstance(purchase, dict):
            continue
        written.append(
            await seed_purchase(
                creator_id=creator_id,
                fan_id=fan_id,
                reference=str(purchase.get("reference") or f"purchase-{index}"),
                media_ids=[
                    str(value) for value in (purchase.get("media_ids") or [])
                ],
                price_cents=int(purchase.get("price_cents") or 0),
                purchased_days_ago=float(purchase.get("purchased_days_ago") or 0.0),
            )
        )
    return written
