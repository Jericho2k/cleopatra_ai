"""The durable conversation generation: one number that says "this is stale".

Full Auto used to decide "a newer message replaced this reply" from
``services.suggestions._pending_auto_replies`` — a dictionary in one Python
process, polled every 0.5 s by whichever coroutine happened to be sleeping.
That answer is only correct while every event for one fan lands in one process,
and it costs a live coroutine for the whole of a deliberate human-like pause.

The replacement is a monotonically increasing integer on the fan row:

  * a genuinely new FAN message bumps it;
  * a HUMAN (operator) creator reply bumps it;
  * an automated bubble from an already-authorized sequence does NOT — otherwise
    a reply would invalidate its own later parts;
  * a duplicate platform delivery does NOT, because the bump is driven by
    ``MessageWriteResult.inserted`` rather than by the arrival of a webhook.

Every planned outbound sequence records the generation it was produced from, and
every externally visible send boundary revalidates ``sequence_generation ==
current_conversation_generation``. Two workers, two processes and a restart all
read the same number, which is the whole point.
"""

from __future__ import annotations

import asyncio
from typing import Any

from core.supabase import get_supabase

#: Set once per process when the column or the RPC turns out to be absent, so a
#: deployment that reaches this code before the migration degrades to "no
#: supersession information" once rather than logging on every message.
_COLUMN_AVAILABLE = True
_RPC_AVAILABLE = True

#: What a caller sees when the durable generation cannot be read at all. It is
#: deliberately not -1 or None: an unknown generation must compare equal to
#: itself so a pre-migration deployment keeps delivering, and unequal to nothing.
UNKNOWN_GENERATION = 0


def _looks_missing(error: Exception, *names: str) -> bool:
    text = str(error).lower()
    if not any(name in text for name in names):
        return False
    return any(
        marker in text
        for marker in (
            "could not find",
            "does not exist",
            "undefined function",
            "undefined column",
            "unknown column",
            "pgrst202",
            "pgrst204",
            "42703",
            "42883",
        )
    )


async def current_generation(fan_id: str, *, client: Any = None) -> int:
    """Read the fan's current conversation generation. Never raises.

    ``client`` exists so the one caller that already holds a Supabase client —
    the message-write chokepoint in ``db.queries`` — bumps through the SAME
    client it just wrote with. Without it a test that substitutes that client
    would reach a real network for the bump alone.
    """
    global _COLUMN_AVAILABLE
    if not _COLUMN_AVAILABLE:
        return UNKNOWN_GENERATION
    db = client or get_supabase()

    def _read() -> int:
        response = (
            db
            .table("fans")
            .select("conversation_generation")
            .eq("id", str(fan_id))
            .limit(1)
            .execute()
        )
        rows = response.data or []
        if not rows:
            return UNKNOWN_GENERATION
        return int(rows[0].get("conversation_generation") or 0)

    try:
        return await asyncio.to_thread(_read)
    except Exception as error:  # noqa: BLE001 - staleness must never kill a turn
        if _looks_missing(error, "conversation_generation"):
            _COLUMN_AVAILABLE = False
            print(
                "[CONVERSATION GENERATION] fans.conversation_generation is not "
                "deployed; supersession falls back to the existing state "
                "revision. Apply db/conversation_supersession_v1.sql."
            )
        else:
            print(f"[CONVERSATION GENERATION READ ERROR] fan={fan_id}: {error}")
        return UNKNOWN_GENERATION


async def bump_generation(fan_id: str, *, reason: str, client: Any = None) -> int:
    """Advance the generation because the conversation genuinely moved on.

    Atomic through the RPC. The read-modify-write fallback exists only for a
    deployment whose migration has not been applied; two concurrent bumps
    collapsing into one there is harmless, because every consumer only ever asks
    "is this different from what I planned against", never "by how much".
    """
    global _COLUMN_AVAILABLE, _RPC_AVAILABLE
    if not _COLUMN_AVAILABLE:
        return UNKNOWN_GENERATION
    db = client or get_supabase()

    if _RPC_AVAILABLE:
        try:
            response = await asyncio.to_thread(
                lambda: db
                .rpc("bump_conversation_generation", {"p_fan_id": str(fan_id)})
                .execute()
            )
            value = response.data
            if isinstance(value, list):
                value = value[0] if value else None
            if isinstance(value, dict):
                value = value.get("bump_conversation_generation")
            generation = int(value or 0)
            print(
                f"[CONVERSATION GENERATION] fan={fan_id} generation={generation} "
                f"reason={reason}"
            )
            return generation
        except Exception as error:  # noqa: BLE001
            # No rpc() on the client at all is the in-memory PostgREST double
            # used in tests: fall back to the same read-modify-write a
            # pre-migration deployment would use rather than losing the bump.
            if isinstance(error, AttributeError) or _looks_missing(
                error, "bump_conversation_generation"
            ):
                _RPC_AVAILABLE = False
                print(
                    "[CONVERSATION GENERATION] bump_conversation_generation() is "
                    "not deployed; falling back to read-modify-write. Apply "
                    "db/conversation_supersession_v1.sql."
                )
            elif _looks_missing(error, "conversation_generation"):
                _COLUMN_AVAILABLE = False
                return UNKNOWN_GENERATION
            else:
                print(f"[CONVERSATION GENERATION BUMP ERROR] fan={fan_id}: {error}")
                return UNKNOWN_GENERATION

    current = await current_generation(fan_id, client=db)
    if not _COLUMN_AVAILABLE:
        return UNKNOWN_GENERATION
    nxt = int(current) + 1

    def _write() -> None:
        (
            db
            .table("fans")
            .update({"conversation_generation": nxt})
            .eq("id", str(fan_id))
            .execute()
        )

    try:
        await asyncio.to_thread(_write)
    except Exception as error:  # noqa: BLE001
        if _looks_missing(error, "conversation_generation"):
            _COLUMN_AVAILABLE = False
        else:
            print(f"[CONVERSATION GENERATION BUMP ERROR] fan={fan_id}: {error}")
        return UNKNOWN_GENERATION
    print(
        f"[CONVERSATION GENERATION] fan={fan_id} generation={nxt} reason={reason}"
    )
    return nxt


def bumps_generation(role: str, *, was_ai_suggested: bool, inserted: bool) -> bool:
    """Whether one persisted message moves the conversation on.

    The three rules this sprint depends on, in one place so a test can state
    them and a future caller cannot re-derive them differently.
    """
    if not inserted:
        # A duplicate platform delivery, or a redelivered webhook. The
        # conversation did not move; only our knowledge of it was repeated.
        return False
    normalized = str(role or "").lower()
    if normalized == "fan":
        return True
    if normalized == "creator":
        # A human took the conversation over. Automated bubbles carry
        # was_ai_suggested=True and must never invalidate their own siblings.
        return not was_ai_suggested
    return False
