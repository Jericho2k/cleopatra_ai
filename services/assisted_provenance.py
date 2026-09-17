"""Assisted attribution that survives a restart, and says so when it does not.

THE PROBLEM
-----------
Full Auto generates and delivers inside one function, so its recorder is a
local variable. Assisted does not: ``get_suggestions`` produces candidates, a
person reads them, and some time later ``POST /reply`` sends one. Two HTTP
requests with a human in between.

``SUGGESTION_PROVENANCE`` bridged that with an in-process OrderedDict, and its
own comment said what that costs::

    The process-wide store. One per backend replica, which is why a miss is an
    ordinary outcome rather than an error.

So a deploy, a crash, an autoscale event, or the second request simply landing
on a different replica lost the record — and the message was then saved with NO
provenance at all. The reply became unattributable and nothing anywhere said
so, which is precisely the failure the provenance work exists to remove: a
reply that cannot name the commit, the flags or the model that produced it is
not evidence about any particular piece of code.

TWO CHANGES, AND THE SECOND MATTERS MORE
----------------------------------------
**Durable storage.** The record goes to ``public.assisted_provenance``, so it
survives a restart and is visible to every replica. The in-process store stays
in front of it as a cache: the overwhelmingly common case is the same replica
seconds later, and that case should not pay a round trip.

**An admitted absence.** When the record genuinely cannot be found, the message
is no longer saved with nothing. It carries a record saying attribution is
unavailable and why. The brief allows either durability or an explicit
surfaced gap, and the reason to do both is that durability can still fail —
the row can be swept, the write can have failed, the token can be from a build
before this existed — and a reply that is silently unattributable is
indistinguishable from one that was never checked.

NEVER BLOCKS A SEND
-------------------
Every function here swallows its own failures. An operator's message must not
fail to send because its evidence trail could not be completed, which is the
rule ``services/reply_provenance.py`` states and this inherits.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any

from core.supabase import get_supabase
from services.reply_provenance import (
    PROVENANCE_KEY,
    SUGGESTION_PROVENANCE,
    ReplyProvenance,
)

TABLE = "assisted_provenance"

#: How long a stored record is worth redeeming. Matches the in-process TTL:
#: a suggestion an operator has not sent within half an hour is one they are
#: not going to send, and the turn it came from no longer describes the
#: conversation.
TTL = timedelta(minutes=30)

#: Why a reply has no attribution. Recorded ON the message rather than left as
#: an absence, so "nobody checked" and "we looked and it was gone" stop reading
#: the same.
UNAVAILABLE_EXPIRED = "the record had expired when the reply was sent"
UNAVAILABLE_MISSING = "no record was found for this reply's token"
UNAVAILABLE_NO_TOKEN = "the reply was sent without a provenance token"


async def remember(provenance: ReplyProvenance) -> str:
    """Store a turn's recorder and return the token that redeems it.

    Written to both, in-process first. If the durable write fails the token is
    still valid on this replica, which is strictly better than failing the
    suggestion — and the miss it may later cause is now an admitted absence
    rather than a silent one.
    """
    token = SUGGESTION_PROVENANCE.put(provenance)
    row = {
        "token": token,
        "creator_id": str(provenance.creator_id),
        "fan_id": str(provenance.fan_id),
        "record": provenance.as_state(),
    }

    def _write() -> None:
        get_supabase().table(TABLE).upsert(row, on_conflict="token").execute()

    try:
        await asyncio.to_thread(_write)
    except Exception as exc:
        print(
            f"[PROVENANCE] could not store assisted record fan={provenance.fan_id}: "
            f"{type(exc).__name__}"
        )
    return token


async def redeem(
    token: object, *, creator_id: str = "", fan_id: str = ""
) -> tuple[ReplyProvenance | None, str]:
    """Take the record back, or say why there is none.

    Returns ``(provenance, unavailable_reason)``. Exactly one is meaningful:
    a record, or a sentence explaining its absence.

    Redemption deletes, in both stores. One generated turn becomes at most one
    sent message, and a token replayed against a second send would attribute
    that message to a turn it did not come from.
    """
    key = "" if token is None else str(token).strip()
    if not key:
        return None, UNAVAILABLE_NO_TOKEN

    cached = SUGGESTION_PROVENANCE.take(key, creator_id=creator_id, fan_id=fan_id)
    if cached is not None:
        # Still delete the durable row: the in-process hit means this replica
        # served both requests, and leaving the row would let a replay on
        # another replica redeem the same turn again.
        await forget(key)
        return cached, ""

    def _read() -> list[dict[str, Any]]:
        result = (
            get_supabase().table(TABLE)
            .select("token, creator_id, fan_id, record, created_at")
            .eq("token", key)
            .limit(1)
            .execute()
        )
        return list(result.data or [])

    try:
        rows = await asyncio.to_thread(_read)
    except Exception as exc:
        print(f"[PROVENANCE] could not read assisted record: {type(exc).__name__}")
        return None, UNAVAILABLE_MISSING

    if not rows:
        return None, UNAVAILABLE_MISSING
    row = rows[0]

    if creator_id and str(row.get("creator_id")) != str(creator_id):
        # The wrong record is worse than no record, so this is a miss rather
        # than a match. Same rule the in-process store applies.
        return None, UNAVAILABLE_MISSING
    if fan_id and str(row.get("fan_id")) != str(fan_id):
        return None, UNAVAILABLE_MISSING

    if _expired(row.get("created_at")):
        await forget(key)
        return None, UNAVAILABLE_EXPIRED

    restored = ReplyProvenance.from_state(row.get("record"))
    await forget(key)
    if restored is None:
        return None, UNAVAILABLE_MISSING
    return restored, ""


def _expired(created_at: Any) -> bool:
    if not created_at:
        return False
    try:
        stored = datetime.fromisoformat(str(created_at).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return False
    if stored.tzinfo is None:
        stored = stored.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) - stored > TTL


async def forget(token: str) -> None:
    """Remove a redeemed or expired record. Never raises."""

    def _delete() -> None:
        get_supabase().table(TABLE).delete().eq("token", str(token)).execute()

    try:
        await asyncio.to_thread(_delete)
    except Exception as exc:  # pragma: no cover - cleanup never blocks a send
        print(f"[PROVENANCE] could not clear assisted record: {type(exc).__name__}")


def unavailable_metadata(reason: str, *, creator_id: str, fan_id: str) -> dict[str, Any]:
    """A record saying this reply has no attribution, and why.

    The point of writing anything at all. Before this, a lost token meant the
    message was saved with no provenance key — identical, to anyone reading the
    row later, to a message from a build that never recorded provenance. An
    evaluation counting attributable replies would count both the same way.

    Deliberately the same key as a real record, with ``attribution_available``
    false, so a reader looking for provenance finds this instead of finding
    nothing.
    """
    return {
        PROVENANCE_KEY: {
            "mode": "assisted",
            "creator_id": str(creator_id),
            "fan_id": str(fan_id),
            "attribution_available": False,
            "attribution_unavailable_because": reason,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
    }
