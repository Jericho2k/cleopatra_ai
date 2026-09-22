"""Durable outbound sequences, their parts, and the per-fan execution lease.

Three properties this module exists to provide, none of which a Python
dictionary can:

  * a planned multi-bubble reply survives a restart between two bubbles;
  * every part carries its own due time, so a deliberate human pause is a row in
    an indexed queue rather than a sleeping coroutine holding a worker slot;
  * two worker processes cannot both be sending to one fan.

Everything here degrades rather than fails when the migration has not been
applied yet: a caller gets ``None``/``False`` and the orchestrator falls back to
inline delivery, which is exactly the previous behaviour.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from core.supabase import get_supabase

SEQUENCES = "outbound_sequences"
PARTS = "outbound_sequence_parts"
LEASES = "fan_execution_leases"

STATUS_PLANNED = "PLANNED"
STATUS_SENDING = "SENDING"
STATUS_COMPLETED = "COMPLETED"
STATUS_SUPERSEDED = "SUPERSEDED"
STATUS_CANCELLED = "CANCELLED"

PART_PENDING = "PENDING"
PART_SENT = "SENT"
PART_SUPERSEDED = "SUPERSEDED"
PART_FAILED = "FAILED"

ACTIVE_STATUSES = (STATUS_PLANNED, STATUS_SENDING)

#: Set once per process when the tables are absent, so a deployment that reaches
#: this code before the migration logs once instead of on every turn.
_TABLES_AVAILABLE = True
_LEASE_RPC_AVAILABLE = True


class OutboundStorageUnavailable(RuntimeError):
    """The durable outbound tables are not deployed in this database."""


@dataclass(frozen=True)
class SequencePart:
    part_index: int
    body: str
    due_at: datetime
    planned_delay_seconds: float = 0.0
    status: str = PART_PENDING
    id: str = ""
    platform_message_id: str = ""
    sent_at: datetime | None = None


@dataclass(frozen=True)
class OutboundSequence:
    id: str
    creator_id: str
    fan_id: str
    conversation_generation: int
    trigger_identity: str
    turn_id: str
    status: str
    cancel_reason: str = ""
    planned_timing: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    parts: tuple[SequencePart, ...] = ()

    def pending_parts(self) -> tuple[SequencePart, ...]:
        return tuple(part for part in self.parts if part.status == PART_PENDING)

    def part(self, index: int) -> SequencePart | None:
        return next((p for p in self.parts if p.part_index == int(index)), None)


def _looks_missing(error: Exception) -> bool:
    text = str(error).lower()
    return (
        any(name in text for name in (SEQUENCES, PARTS, LEASES))
        and any(
            marker in text
            for marker in (
                "could not find",
                "does not exist",
                "undefined table",
                "pgrst205",
                "42p01",
            )
        )
    )


def _mark_unavailable(error: Exception, what: str) -> None:
    global _TABLES_AVAILABLE
    _TABLES_AVAILABLE = False
    print(
        f"[OUTBOUND SEQUENCE] {what} is not deployed ({error}); durable timed "
        "delivery is disabled and replies send inline. Apply "
        "db/conversation_supersession_v1.sql."
    )


def storage_available() -> bool:
    return _TABLES_AVAILABLE


def _reset_availability_for_tests() -> None:  # pragma: no cover - test affordance
    global _TABLES_AVAILABLE, _LEASE_RPC_AVAILABLE
    _TABLES_AVAILABLE = True
    _LEASE_RPC_AVAILABLE = True


def _as_utc(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _sequence_from_rows(row: dict, part_rows: list[dict]) -> OutboundSequence:
    parts = tuple(
        sorted(
            (
                SequencePart(
                    part_index=int(part.get("part_index") or 0),
                    body=str(part.get("body") or ""),
                    due_at=_as_utc(part.get("due_at")) or datetime.now(timezone.utc),
                    planned_delay_seconds=float(
                        part.get("planned_delay_seconds") or 0.0
                    ),
                    status=str(part.get("status") or PART_PENDING),
                    id=str(part.get("id") or ""),
                    platform_message_id=str(part.get("platform_message_id") or ""),
                    sent_at=_as_utc(part.get("sent_at")),
                )
                for part in part_rows
            ),
            key=lambda part: part.part_index,
        )
    )
    return OutboundSequence(
        id=str(row.get("id") or ""),
        creator_id=str(row.get("creator_id") or ""),
        fan_id=str(row.get("fan_id") or ""),
        conversation_generation=int(row.get("conversation_generation") or 0),
        trigger_identity=str(row.get("trigger_identity") or ""),
        turn_id=str(row.get("turn_id") or ""),
        status=str(row.get("status") or STATUS_PLANNED),
        cancel_reason=str(row.get("cancel_reason") or ""),
        planned_timing=dict(row.get("planned_timing") or {}),
        metadata=dict(row.get("metadata") or {}),
        parts=parts,
    )


async def create_sequence(
    *,
    creator_id: str,
    fan_id: str,
    conversation_generation: int,
    trigger_identity: str,
    turn_id: str,
    parts: list[dict[str, Any]],
    planned_timing: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> OutboundSequence | None:
    """Persist one planned reply. Returns None when storage is unavailable.

    ``(fan_id, trigger_identity)`` is unique, so a retried action adopts the
    sequence it already planned rather than queueing a second copy of the same
    reply. That is the idempotency boundary for durable delivery.
    """
    if not _TABLES_AVAILABLE:
        return None
    # Idempotency is stated here as well as enforced by the unique index,
    # because a retried action adopting its own plan should be the ordinary
    # path rather than a caught exception.
    existing = await get_sequence_by_trigger(fan_id, trigger_identity)
    if existing is not None:
        return existing
    sequence_id = str(uuid.uuid4())
    header = {
        "id": sequence_id,
        "creator_id": str(creator_id),
        "fan_id": str(fan_id),
        "conversation_generation": int(conversation_generation),
        "trigger_identity": str(trigger_identity),
        "turn_id": str(turn_id),
        "status": STATUS_PLANNED,
        "planned_timing": planned_timing,
        "metadata": dict(metadata or {}),
    }
    part_rows = [
        {
            "sequence_id": sequence_id,
            "creator_id": str(creator_id),
            "fan_id": str(fan_id),
            "part_index": int(part["part_index"]),
            "body": str(part["body"]),
            "due_at": _as_utc(part["due_at"]).isoformat(),
            "planned_delay_seconds": float(part.get("planned_delay_seconds") or 0.0),
            "status": PART_PENDING,
        }
        for part in parts
    ]

    def _insert() -> list[dict]:
        db = get_supabase()
        db.table(SEQUENCES).insert(header).execute()
        if part_rows:
            db.table(PARTS).insert(part_rows).execute()
        return part_rows

    try:
        await asyncio.to_thread(_insert)
    except Exception as error:  # noqa: BLE001
        if _looks_missing(error):
            _mark_unavailable(error, "outbound_sequences")
            return None
        # A unique violation means this exact turn already planned a sequence.
        existing = await get_sequence_by_trigger(fan_id, trigger_identity)
        if existing is not None:
            return existing
        raise
    return _sequence_from_rows(header, part_rows)


async def get_sequence(sequence_id: str) -> OutboundSequence | None:
    if not _TABLES_AVAILABLE:
        return None

    def _read() -> tuple[dict | None, list[dict]]:
        db = get_supabase()
        rows = (
            db.table(SEQUENCES)
            .select("*")
            .eq("id", str(sequence_id))
            .limit(1)
            .execute()
        ).data or []
        if not rows:
            return None, []
        parts = (
            db.table(PARTS)
            .select("*")
            .eq("sequence_id", str(sequence_id))
            .execute()
        ).data or []
        return rows[0], list(parts)

    try:
        row, parts = await asyncio.to_thread(_read)
    except Exception as error:  # noqa: BLE001
        if _looks_missing(error):
            _mark_unavailable(error, "outbound_sequences")
            return None
        raise
    return _sequence_from_rows(row, parts) if row else None


async def get_sequence_by_trigger(
    fan_id: str, trigger_identity: str
) -> OutboundSequence | None:
    if not _TABLES_AVAILABLE:
        return None

    def _read() -> list[dict]:
        return (
            get_supabase()
            .table(SEQUENCES)
            .select("id")
            .eq("fan_id", str(fan_id))
            .eq("trigger_identity", str(trigger_identity))
            .limit(1)
            .execute()
        ).data or []

    try:
        rows = await asyncio.to_thread(_read)
    except Exception as error:  # noqa: BLE001
        if _looks_missing(error):
            _mark_unavailable(error, "outbound_sequences")
            return None
        raise
    if not rows:
        return None
    return await get_sequence(str(rows[0]["id"]))


async def active_sequences_for_fan(fan_id: str) -> list[OutboundSequence]:
    if not _TABLES_AVAILABLE:
        return []

    def _read() -> list[dict]:
        return (
            get_supabase()
            .table(SEQUENCES)
            .select("id")
            .eq("fan_id", str(fan_id))
            .in_("status", list(ACTIVE_STATUSES))
            .execute()
        ).data or []

    try:
        rows = await asyncio.to_thread(_read)
    except Exception as error:  # noqa: BLE001
        if _looks_missing(error):
            _mark_unavailable(error, "outbound_sequences")
            return []
        raise
    found = [await get_sequence(str(row["id"])) for row in rows]
    return [sequence for sequence in found if sequence is not None]


async def set_sequence_status(
    sequence_id: str, status: str, *, reason: str = ""
) -> None:
    if not _TABLES_AVAILABLE:
        return
    patch = {
        "status": status,
        "cancel_reason": reason[:500],
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    def _write() -> None:
        (
            get_supabase()
            .table(SEQUENCES)
            .update(patch)
            .eq("id", str(sequence_id))
            .execute()
        )

    try:
        await asyncio.to_thread(_write)
    except Exception as error:  # noqa: BLE001
        if _looks_missing(error):
            _mark_unavailable(error, "outbound_sequences")
            return
        raise


async def supersede_pending_parts(sequence_id: str, *, reason: str) -> None:
    """Retire every unsent bubble of one sequence. Already-sent parts stay canon."""
    if not _TABLES_AVAILABLE:
        return

    def _write() -> None:
        (
            get_supabase()
            .table(PARTS)
            .update({"status": PART_SUPERSEDED})
            .eq("sequence_id", str(sequence_id))
            .eq("status", PART_PENDING)
            .execute()
        )

    try:
        await asyncio.to_thread(_write)
    except Exception as error:  # noqa: BLE001
        if _looks_missing(error):
            _mark_unavailable(error, "outbound_sequences")
            return
        raise
    await set_sequence_status(sequence_id, STATUS_SUPERSEDED, reason=reason)


async def record_part_sent(
    *,
    sequence_id: str,
    part_index: int,
    platform_message_id: str,
    message_id: str | None,
) -> None:
    if not _TABLES_AVAILABLE:
        return
    patch = {
        "status": PART_SENT,
        "platform_message_id": str(platform_message_id or ""),
        "message_id": message_id,
        "sent_at": datetime.now(timezone.utc).isoformat(),
    }

    def _write() -> None:
        (
            get_supabase()
            .table(PARTS)
            .update(patch)
            .eq("sequence_id", str(sequence_id))
            .eq("part_index", int(part_index))
            .execute()
        )

    try:
        await asyncio.to_thread(_write)
    except Exception as error:  # noqa: BLE001
        if _looks_missing(error):
            _mark_unavailable(error, "outbound_sequences")
            return
        raise


# ---------------------------------------------------------------------------
# Per-fan execution lease
# ---------------------------------------------------------------------------


async def acquire_fan_lease(
    *,
    fan_id: str,
    creator_id: str,
    owner_token: str,
    ttl_seconds: int = 180,
    purpose: str = "",
) -> bool:
    """Take exclusive ownership of outbound work for one fan, or report False.

    Deliberately NOT held across a future scheduled action: a nine second
    inter-bubble pause releases it, and the next part re-acquires when it is
    actually due. Expiry is the crash-recovery path, so a worker killed
    mid-send blocks the fan for at most ``ttl_seconds`` rather than forever.
    """
    global _LEASE_RPC_AVAILABLE
    if _LEASE_RPC_AVAILABLE:
        try:
            response = await asyncio.to_thread(
                lambda: get_supabase()
                .rpc(
                    "acquire_fan_execution_lease",
                    {
                        "p_fan_id": str(fan_id),
                        "p_creator_id": str(creator_id),
                        "p_owner": str(owner_token),
                        "p_ttl_seconds": int(ttl_seconds),
                        "p_purpose": str(purpose)[:200],
                    },
                )
                .execute()
            )
            value = response.data
            if isinstance(value, list):
                value = value[0] if value else False
            if isinstance(value, dict):
                value = value.get("acquire_fan_execution_lease")
            return bool(value)
        except Exception as error:  # noqa: BLE001
            text = str(error).lower()
            # An AttributeError means this client has no rpc() at all, which is
            # the in-memory PostgREST double used in tests; anything else must
            # name the function or it is a real failure inside it.
            if not isinstance(error, AttributeError) and (
                "acquire_fan_execution_lease" not in text
            ):
                raise
            _LEASE_RPC_AVAILABLE = False
            print(
                "[FAN LEASE] acquire_fan_execution_lease() is not deployed; "
                "falling back to a conditional upsert. Apply "
                "db/conversation_supersession_v1.sql."
            )

    return await _acquire_fan_lease_by_upsert(
        fan_id=fan_id,
        creator_id=creator_id,
        owner_token=owner_token,
        ttl_seconds=ttl_seconds,
        purpose=purpose,
    )


async def _acquire_fan_lease_by_upsert(
    *,
    fan_id: str,
    creator_id: str,
    owner_token: str,
    ttl_seconds: int,
    purpose: str,
) -> bool:
    """Rollout fallback: insert-if-absent, else steal only an expired lease.

    Both writes are conditional at the database, so the window the RPC closes is
    narrowed to "two workers both find the lease expired in the same instant" —
    and that one is resolved by the unique primary key on ``fan_id``.
    """
    now = datetime.now(timezone.utc)
    expires_at = now.timestamp() + max(5, int(ttl_seconds))
    row = {
        "fan_id": str(fan_id),
        "creator_id": str(creator_id),
        "owner_token": str(owner_token),
        "purpose": str(purpose)[:200],
        "acquired_at": now.isoformat(),
        "expires_at": datetime.fromtimestamp(expires_at, tz=timezone.utc).isoformat(),
    }

    def _claim() -> bool:
        db = get_supabase()
        held = (
            db.table(LEASES)
            .select("fan_id, owner_token, expires_at")
            .eq("fan_id", str(fan_id))
            .limit(1)
            .execute()
        ).data or []
        if not held:
            try:
                inserted = db.table(LEASES).insert(row).execute()
                if inserted.data:
                    return True
            except Exception as error:  # noqa: BLE001
                if _looks_missing(error):
                    raise
                # Lost the insert race on the primary key. Somebody else owns
                # the fan now, so fall through to the conditional steal, which
                # will only succeed if their lease has already expired.
        # Expired: anyone may take it. This is the crash-recovery path, and it
        # is conditional AT THE DATABASE rather than decided from the read
        # above, so two workers reading "expired" in the same instant still
        # resolve to one winner.
        taken = (
            db.table(LEASES)
            .update(row)
            .eq("fan_id", str(fan_id))
            .lt("expires_at", now.isoformat())
            .execute()
        )
        if taken.data:
            return True
        # Not expired: only the current owner may renew it.
        mine = (
            db.table(LEASES)
            .update(row)
            .eq("fan_id", str(fan_id))
            .eq("owner_token", str(owner_token))
            .execute()
        )
        return bool(mine.data)

    try:
        return await asyncio.to_thread(_claim)
    except Exception as error:  # noqa: BLE001
        if _looks_missing(error):
            _mark_unavailable(error, "fan_execution_leases")
            # Without the table there is no cross-worker claim to make. The
            # in-process per-fan chaining in the worker still holds, which is
            # exactly the pre-migration guarantee.
            return True
        raise


async def release_fan_lease(*, fan_id: str, owner_token: str) -> None:
    def _release() -> None:
        (
            get_supabase()
            .table(LEASES)
            .delete()
            .eq("fan_id", str(fan_id))
            .eq("owner_token", str(owner_token))
            .execute()
        )

    try:
        await asyncio.to_thread(_release)
    except Exception as error:  # noqa: BLE001 - never fail delivery on a release
        if not _looks_missing(error):
            print(f"[FAN LEASE RELEASE ERROR] fan={fan_id}: {error}")
