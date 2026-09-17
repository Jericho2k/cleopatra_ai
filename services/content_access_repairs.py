"""The durable boundary around one free access repair.

WHAT THIS IS FOR
----------------
``services.content_access.resend_paid_content`` calls a platform adapter and
then writes a receipt. Without something durable between those two steps, three
things reproduce against the existing fixtures at backend 4a1683a:

* the platform accepts, ``save_message`` fails, the operator clicks again, and
  a second free copy goes out while the hold is still set;
* two operators click at the same moment and both send;
* a hold raised DURING the platform call is cleared by the repair on its way
  out, because ``clear_fan_review`` is an unconditional update by fan id.

A claim written and committed before the send fixes all three, and survives the
case a lock cannot: the process dying mid-send, which is the one where nobody
knows whether the customer received the media.

WHAT IS AND IS NOT PROMISED
---------------------------
Not exactly-once delivery. The platform offers no such guarantee, so promising
it would be a lie told in code. What is promised is exactly-once ATTEMPT — at
most one send per purchase per review case — plus an honest record of how each
attempt ended, including ``unknown``.

``unknown`` is the point of the design. A send whose outcome cannot be proven
is not a failure to retry and not a success to report; it is a question for a
person, with the evidence in front of them. Collapsing it into either is how a
customer gets two copies or how a repair that never arrived is reported as done.

NO LOCK IS HELD ACROSS THE PLATFORM CALL
----------------------------------------
``claim`` commits and returns. The caller then talks to the provider holding
nothing, and comes back to ``confirm``/``mark_failed``/``mark_unknown``. A
database connection is never pinned to a provider's latency.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from core.supabase import get_supabase

REPAIRS_TABLE = "content_access_repairs"

#: In flight. This process, or one that has since died, owns the attempt.
STATUS_CLAIMED = "claimed"
#: The platform returned a receipt and it is recorded. Nothing more to do.
STATUS_CONFIRMED = "confirmed"
#: The platform refused before sending anything. Safe to attempt again.
STATUS_FAILED = "failed"
#: The send may or may not have reached the customer and cannot be proven
#: either way. Never retried automatically.
STATUS_UNKNOWN = "unknown"

#: The conflict target of the claim index, in PostgREST's spelling.
_CLAIM_CONFLICT = "creator_id,fan_id,reference,review_case_id"

#: How long a claim may sit in flight before a retry is allowed to treat it as
#: abandoned. Long enough that a slow provider call is never mistaken for a
#: dead worker, short enough that a crashed process does not freeze a repair
#: until someone notices.
#:
#: An abandoned claim becomes ``unknown``, never ``failed``: the request may
#: well have reached the platform before the process died, and that is exactly
#: the case a person has to look at.
STALE_CLAIM_SECONDS = 300.0


class RepairInProgress(RuntimeError):
    """Another attempt owns this repair. Reconcile; do not send."""

    def __init__(self, repair: "Repair") -> None:
        super().__init__(repair.operator_message())
        self.repair = repair


@dataclass
class Repair:
    """One durable repair claim."""

    id: str
    creator_id: str
    fan_id: str
    reference: str
    review_case_id: str
    status: str
    media_ids: list[str]
    platform_message_id: str
    claimed_by: str
    detail: str
    claimed_at: str
    resolved_at: str

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "Repair":
        media = row.get("media_ids")
        return cls(
            id=str(row.get("id") or ""),
            creator_id=str(row.get("creator_id") or ""),
            fan_id=str(row.get("fan_id") or ""),
            reference=str(row.get("reference") or ""),
            review_case_id=str(row.get("review_case_id") or ""),
            status=str(row.get("status") or ""),
            media_ids=[str(value) for value in media] if isinstance(media, list) else [],
            platform_message_id=str(row.get("platform_message_id") or ""),
            claimed_by=str(row.get("claimed_by") or ""),
            detail=str(row.get("detail") or ""),
            claimed_at=str(row.get("claimed_at") or ""),
            resolved_at=str(row.get("resolved_at") or ""),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "reference": self.reference,
            "review_case_id": self.review_case_id,
            "status": self.status,
            "media_ids": list(self.media_ids),
            "platform_message_id": self.platform_message_id or None,
            "claimed_by": self.claimed_by,
            "detail": self.detail,
            "claimed_at": self.claimed_at,
            "resolved_at": self.resolved_at or None,
            # Said outright so a client never infers it from the status string.
            "settled": self.status in {STATUS_CONFIRMED, STATUS_FAILED},
            "needs_operator_decision": self.status == STATUS_UNKNOWN,
        }

    def operator_message(self) -> str:
        """What to tell the operator about an attempt they cannot repeat."""
        if self.status == STATUS_CONFIRMED:
            return (
                "this purchase was already resent for this review case, and the "
                "platform confirmed it. Sending again would give the customer a "
                "second copy of something they already have."
            )
        if self.status == STATUS_CLAIMED:
            return (
                "a repair for this purchase is already in progress. Wait for it "
                "to finish rather than starting a second one — both would send."
            )
        if self.status == STATUS_UNKNOWN:
            return (
                "an earlier repair for this purchase reached the platform but "
                "its outcome could not be confirmed, so it is not known whether "
                "the customer received it. Check the conversation before "
                "sending again: this is the one case where a retry may deliver "
                "a second copy."
            )
        return "this repair is not in a state that can be resumed."

    @property
    def is_stale(self) -> bool:
        """A claim old enough that the attempt owning it is presumed dead."""
        if self.status != STATUS_CLAIMED:
            return False
        started = _parse(self.claimed_at)
        if started is None:
            return False
        return (_now_dt() - started).total_seconds() > STALE_CLAIM_SECONDS


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _now_dt() -> datetime:
    return datetime.now(timezone.utc)


def _parse(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


async def find(
    *, creator_id: str, fan_id: str, reference: str, review_case_id: str
) -> Repair | None:
    """The existing claim for this exact purchase and review case, if any."""

    def _read() -> list[dict[str, Any]]:
        result = (
            get_supabase().table(REPAIRS_TABLE)
            .select("*")
            .eq("creator_id", creator_id)
            .eq("fan_id", fan_id)
            .eq("reference", reference)
            .eq("review_case_id", review_case_id)
            .limit(1)
            .execute()
        )
        return list(result.data or [])

    rows = await asyncio.to_thread(_read)
    return Repair.from_row(rows[0]) if rows else None


async def history(*, creator_id: str, fan_id: str, limit: int = 20) -> list[Repair]:
    """Every repair attempted for this customer, newest first.

    The operator panel needs this to show the actual state of the operation
    across a reload — a repair that is in flight, or whose outcome is unknown,
    must be visible as such rather than looking like nothing ever happened.
    """

    def _read() -> list[dict[str, Any]]:
        result = (
            get_supabase().table(REPAIRS_TABLE)
            .select("*")
            .eq("creator_id", creator_id)
            .eq("fan_id", fan_id)
            .order("claimed_at", desc=True)
            .limit(max(1, min(int(limit or 20), 100)))
            .execute()
        )
        return list(result.data or [])

    try:
        rows = await asyncio.to_thread(_read)
    except Exception as exc:
        print(f"[ACCESS REPAIR] history read failed fan={fan_id}: {type(exc).__name__}")
        return []
    return [Repair.from_row(row) for row in rows]


async def claim(
    *,
    creator_id: str,
    fan_id: str,
    reference: str,
    review_case_id: str,
    media_ids: list[str],
    claimed_by: str = "",
) -> Repair:
    """Take ownership of this repair, or raise because someone else has it.

    The insert is the mutual exclusion: the unique index in
    db/content_access_repair_v1.sql covers exactly this key, and PostgREST's
    ignore-duplicates returns nothing to the loser rather than overwriting the
    winner's row. Returns only to the caller that may now send.

    A ``failed`` claim is reusable — the platform refused before anything left,
    so nothing is owed. A ``claimed`` row older than STALE_CLAIM_SECONDS is
    taken over as ``unknown``, not as a fresh attempt: its send may have landed.
    """
    row = {
        "creator_id": str(creator_id),
        "fan_id": str(fan_id),
        "reference": str(reference),
        "review_case_id": str(review_case_id or ""),
        "status": STATUS_CLAIMED,
        "media_ids": [str(value) for value in media_ids],
        "claimed_by": str(claimed_by or ""),
        "claimed_at": _now(),
        "detail": "",
        "platform_message_id": None,
        "resolved_at": None,
    }

    def _insert() -> list[dict[str, Any]]:
        result = (
            get_supabase().table(REPAIRS_TABLE)
            .upsert(row, on_conflict=_CLAIM_CONFLICT, ignore_duplicates=True)
            .execute()
        )
        return list(result.data or [])

    written = await asyncio.to_thread(_insert)
    if written:
        return Repair.from_row(written[0])

    # Lost the race, or this is a retry. Either way: read what is there and let
    # its state decide. Never send on this path.
    existing = await find(
        creator_id=creator_id,
        fan_id=fan_id,
        reference=reference,
        review_case_id=review_case_id,
    )
    if existing is None:
        # The conflicting row vanished between the insert and the read — a
        # cascade delete, or a competing attempt rolled back. Reported as
        # unknown rather than retried: this is not a state we can reason about,
        # and guessing here costs a customer a duplicate.
        raise RepairInProgress(
            Repair.from_row(
                {
                    **row,
                    "status": STATUS_UNKNOWN,
                    "detail": "the claim conflicted but could not be read back",
                }
            )
        )

    if existing.status == STATUS_FAILED:
        return await _reopen(existing, media_ids=media_ids, claimed_by=claimed_by)
    if existing.is_stale:
        await mark_unknown(
            existing,
            detail=(
                "the attempt holding this claim stopped without reporting an "
                "outcome; whether the customer received the media is unknown"
            ),
        )
        existing.status = STATUS_UNKNOWN
    raise RepairInProgress(existing)


async def _reopen(repair: Repair, *, media_ids: list[str], claimed_by: str) -> Repair:
    """Re-claim a repair whose previous attempt provably sent nothing."""
    payload = {
        "status": STATUS_CLAIMED,
        "media_ids": [str(value) for value in media_ids],
        "claimed_by": str(claimed_by or ""),
        "claimed_at": _now(),
        "resolved_at": None,
        "detail": "",
        "platform_message_id": None,
    }

    def _update() -> list[dict[str, Any]]:
        result = (
            get_supabase().table(REPAIRS_TABLE)
            .update(payload)
            .eq("id", repair.id)
            # Compare-and-set: only the row still in `failed` is re-opened, so
            # two operators retrying a failed repair together cannot both win.
            .eq("status", STATUS_FAILED)
            .execute()
        )
        return list(result.data or [])

    written = await asyncio.to_thread(_update)
    if not written:
        refreshed = await find(
            creator_id=repair.creator_id,
            fan_id=repair.fan_id,
            reference=repair.reference,
            review_case_id=repair.review_case_id,
        )
        raise RepairInProgress(refreshed or repair)
    return Repair.from_row(written[0])


async def _settle(
    repair: Repair,
    *,
    status: str,
    platform_message_id: str = "",
    detail: str = "",
) -> Repair:
    payload = {
        "status": status,
        "detail": detail,
        "resolved_at": _now(),
    }
    if platform_message_id:
        payload["platform_message_id"] = platform_message_id

    def _update() -> list[dict[str, Any]]:
        result = (
            get_supabase().table(REPAIRS_TABLE)
            .update(payload)
            .eq("id", repair.id)
            .execute()
        )
        return list(result.data or [])

    try:
        written = await asyncio.to_thread(_update)
    except Exception as exc:
        # The outcome is now genuinely unrecorded. Say so in the log, because
        # the row still reads `claimed` and the next operator will be told a
        # repair is in progress — which is the safe direction to fail in.
        print(
            f"[ACCESS REPAIR] could not record outcome repair={repair.id} "
            f"status={status}: {type(exc).__name__}"
        )
        return repair
    return Repair.from_row(written[0]) if written else repair


async def confirm(repair: Repair, *, platform_message_id: str) -> Repair:
    """The platform returned a receipt and the local record is written."""
    return await _settle(
        repair, status=STATUS_CONFIRMED, platform_message_id=platform_message_id
    )


async def mark_failed(repair: Repair, *, detail: str) -> Repair:
    """Nothing left this process. Safe to attempt again."""
    return await _settle(repair, status=STATUS_FAILED, detail=detail)


async def mark_unknown(
    repair: Repair, *, detail: str, platform_message_id: str = ""
) -> Repair:
    """The send may have happened. Never retried without a person deciding."""
    return await _settle(
        repair,
        status=STATUS_UNKNOWN,
        platform_message_id=platform_message_id,
        detail=detail,
    )


def new_case_id() -> str:
    """An identity for one review hold, so clearing it can be conditional."""
    return uuid.uuid4().hex
