"""Durable, human-timed delivery of one creator reply.

The shape this implements, and why each arrow is a durable boundary rather than
an ``await asyncio.sleep``::

    Kimi returns the bubbles
        -> compute a DeliverySchedule (services/human_delivery.py, unchanged)
        -> persist an outbound sequence bound to the conversation generation
        -> persist one durable due action per bubble
        -> the worker exits; no coroutine, timer or dictionary survives it
        -> a bubble becomes due
        -> claim it, take the per-fan lease, revalidate the generation
        -> send, persist the receipt
        -> the next bubble becomes due

Two properties follow that the previous in-process design could not give:

  * a nine-second inter-bubble pause occupies an indexed row, not a model slot,
    a scheduled-action slot, or a per-fan polling coroutine;
  * a fan message that lands in ANOTHER process still stops bubble 2, because
    the check at the send boundary reads the database rather than this process.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from db import outbound_queries as store
from db.commercial_queries import cancel_action_by_dedupe_key, schedule_action
from db.queries import freeze_fan_for_review, save_message
from services import outbound_settlement as settlement
from services.conversation_generation import current_generation
from services.delivery_mode import is_immediate
from services.human_delivery import DeliverySchedule

DELIVER_PART_ACTION = "DELIVER_OUTBOUND_PART"

#: How long an unsent bubble may wait for its predecessor before the sequence is
#: abandoned. Queue wait, not deliberate delay: if bubble 1 has not gone out two
#: minutes after bubble 2 came due, something is wrong and sending bubble 2 out
#: of order would be worse than sending nothing.
MAX_ORDERING_WAIT_SECONDS = 120

#: A bubble that comes due while its predecessor is still in flight waits this
#: long and asks again. Short, and bounded by MAX_ORDERING_WAIT_SECONDS.
ORDERING_RETRY_SECONDS = 2


class OutboundDeliveryError(RuntimeError):
    """Delivery failed in a way the durable action should retry."""


@dataclass(frozen=True)
class PartOutcome:
    sent: bool
    reason: str = ""
    retry_at: datetime | None = None
    message_id: str | None = None


#: Where the orchestrator puts the message-row provenance for every bubble, so
#: a bubble delivered minutes later by another process carries the same record
#: the first one did.
MESSAGE_METADATA_KEY = "message_metadata"


def sequence_metadata(
    *,
    message_metadata: dict[str, Any] | None = None,
    post_send_operation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble what one planned sequence must remember about its own turn."""
    record: dict[str, Any] = {MESSAGE_METADATA_KEY: dict(message_metadata or {})}
    if post_send_operation:
        record[settlement.POST_SEND_KEY] = dict(post_send_operation)
    return record


def part_dedupe_key(sequence_id: str, part_index: int) -> str:
    return f"outbound-part:{sequence_id}:{int(part_index)}"


def plan_due_times(
    schedule: DeliverySchedule,
    part_count: int,
    *,
    now: datetime | None = None,
) -> list[tuple[float, datetime]]:
    """Turn the human-delivery schedule into (delay, absolute due time) pairs.

    The availability delay is deliberately absent: it is already spent before
    the turn runs, as the durable AUTO_REPLY action's own ``execute_at``
    (``services.suggestions.schedule_auto_reply``). Adding it here would charge
    a conversation for the same pause twice.
    """
    anchor = now or datetime.now(timezone.utc)
    due: list[tuple[float, datetime]] = []
    cursor = anchor
    for index in range(max(0, int(part_count))):
        if index == 0:
            delay = float(schedule.composition_delay_seconds)
        else:
            gaps = schedule.inter_part_delays_seconds
            delay = float(gaps[index - 1]) if index - 1 < len(gaps) else 1.5
        cursor = cursor + timedelta(seconds=max(0.0, delay))
        due.append((round(max(0.0, delay), 2), cursor))
    return due


def timing_plan(schedule: DeliverySchedule, part_count: int) -> dict[str, Any]:
    """What production WOULD have waited, reported even when it does not wait."""
    return {
        "availability_mode": schedule.availability_mode.value,
        "availability_delay_seconds": schedule.availability_delay_seconds,
        "composition_delay_seconds": schedule.composition_delay_seconds,
        "inter_part_delays_seconds": [
            float(value) for value in schedule.inter_part_delays_seconds
        ][: max(0, part_count - 1)],
        "part_count": int(part_count),
        "delivery_mode": "immediate" if is_immediate() else "durable",
    }


def _telemetry(event: str, **fields: Any) -> None:
    """Structural delivery telemetry. Identifiers and reasons only, never copy."""
    try:
        print(
            "[OUTBOUND SEQUENCE] "
            + json.dumps({"event": event, **fields}, sort_keys=True, default=str)
        )
    except Exception:  # pragma: no cover - telemetry never breaks delivery
        pass


async def schedule_outbound_sequence(
    *,
    creator_id: str,
    fan_id: str,
    trigger_identity: str,
    turn_id: str,
    parts: list[str],
    schedule: DeliverySchedule,
    conversation_generation: int,
    metadata: dict[str, Any] | None = None,
) -> store.OutboundSequence | None:
    """Persist the planned reply and queue each bubble at its own due time."""
    bodies = [str(part).strip() for part in parts if str(part).strip()]
    if not bodies:
        return None
    plan = plan_due_times(schedule, len(bodies))
    sequence = await store.create_sequence(
        creator_id=creator_id,
        fan_id=fan_id,
        conversation_generation=int(conversation_generation),
        trigger_identity=str(trigger_identity),
        turn_id=str(turn_id),
        parts=[
            {
                "part_index": index,
                "body": body,
                "due_at": plan[index][1],
                "planned_delay_seconds": plan[index][0],
            }
            for index, body in enumerate(bodies)
        ],
        planned_timing=timing_plan(schedule, len(bodies)),
        metadata=dict(metadata or {}),
    )
    if sequence is None:
        return None

    if not is_immediate():
        for part in sequence.parts:
            if part.status != store.PART_PENDING:
                continue
            await schedule_action(
                creator_id=creator_id,
                fan_id=fan_id,
                action_type=DELIVER_PART_ACTION,
                execute_at=part.due_at,
                payload={
                    "sequence_id": sequence.id,
                    "part_index": part.part_index,
                    "conversation_generation": int(conversation_generation),
                },
                dedupe_key=part_dedupe_key(sequence.id, part.part_index),
                replace_existing=False,
            )
        _notify_worker()

    _telemetry(
        "planned",
        sequence_id=sequence.id,
        fan_id=fan_id,
        creator_id=creator_id,
        generation=int(conversation_generation),
        parts=len(sequence.parts),
        timing=sequence.planned_timing,
    )
    return sequence


def _notify_worker() -> None:
    try:
        from workers.scheduled_actions import notify_work_available

        notify_work_available()
    except Exception:  # pragma: no cover - a nudge is never load bearing
        pass


async def _cancel_remaining_actions(sequence: store.OutboundSequence) -> None:
    for part in sequence.parts:
        if part.status != store.PART_PENDING:
            continue
        try:
            await cancel_action_by_dedupe_key(
                part_dedupe_key(sequence.id, part.part_index)
            )
        except Exception as exc:  # noqa: BLE001 - the send boundary is the guard
            print(
                f"[OUTBOUND SEQUENCE] could not cancel queued part "
                f"{part.part_index} of {sequence.id}: {exc}"
            )


async def supersede_sequence(
    sequence: store.OutboundSequence, *, reason: str
) -> None:
    """Retire every unsent bubble. Already-sent bubbles remain conversation canon."""
    await store.supersede_pending_parts(sequence.id, reason=reason)
    await _cancel_remaining_actions(sequence)
    _telemetry(
        "superseded",
        sequence_id=sequence.id,
        fan_id=sequence.fan_id,
        generation=sequence.conversation_generation,
        reason=reason,
        unsent_parts=len(sequence.pending_parts()),
    )


async def supersede_active_sequences(
    fan_id: str, *, reason: str, keep_sequence_id: str = ""
) -> int:
    """Retire any other planned reply for this fan. Used when a new turn wins."""
    superseded = 0
    for sequence in await store.active_sequences_for_fan(fan_id):
        if keep_sequence_id and sequence.id == keep_sequence_id:
            continue
        await supersede_sequence(sequence, reason=reason)
        superseded += 1
    return superseded


@dataclass(frozen=True)
class SendGate:
    """The authoritative answer to "may this bubble leave right now?"."""

    allowed: bool
    reason: str = ""
    retry_at: datetime | None = None
    terminal: bool = False


async def evaluate_send_gate(
    sequence: store.OutboundSequence,
    part: store.SequencePart,
    *,
    now: datetime | None = None,
) -> SendGate:
    """Revalidate one bubble against authoritative state, immediately before it sends.

    This is the ONLY place interruption is decided. It reads the database, so a
    fan message handled by another process, an operator reply typed in the
    dashboard, a review hold, or auto mode being switched off all stop the
    remaining bubbles without anything having to reach into this process.
    """
    from db.queries import get_fan_by_id

    moment = now or datetime.now(timezone.utc)

    if sequence.status not in store.ACTIVE_STATUSES:
        return SendGate(False, f"sequence_{sequence.status.lower()}", terminal=True)
    if part.status != store.PART_PENDING:
        return SendGate(False, f"part_{part.status.lower()}", terminal=True)

    earlier_unsent = [
        other
        for other in sequence.parts
        if other.part_index < part.part_index and other.status == store.PART_PENDING
    ]
    if earlier_unsent:
        waited = (moment - part.due_at).total_seconds()
        if waited > MAX_ORDERING_WAIT_SECONDS:
            return SendGate(
                False, "earlier_bubble_never_sent", terminal=True
            )
        return SendGate(
            False,
            "waiting_for_earlier_bubble",
            retry_at=moment + timedelta(seconds=ORDERING_RETRY_SECONDS),
        )

    generation = await current_generation(sequence.fan_id)
    if int(generation) != int(sequence.conversation_generation):
        return SendGate(False, "stale_generation", terminal=True)

    fan = await get_fan_by_id(sequence.fan_id)
    if fan is None:
        return SendGate(False, "fan_gone", terminal=True)
    if getattr(fan, "needs_human_review", False):
        return SendGate(False, "human_review_hold", terminal=True)
    if getattr(fan, "auto_mode", None) is False:
        return SendGate(False, "auto_mode_off", terminal=True)

    return SendGate(True)


async def _delivery_route(creator_id: str, fan_id: str) -> tuple[str, str, bool]:
    """Resolve (account_id, group_id, local_test) at send time, never at plan time."""
    from core.supabase import get_supabase
    from db.queries import get_fan_by_id

    fan = await get_fan_by_id(fan_id)
    if fan is None:
        raise OutboundDeliveryError("fan disappeared before delivery")
    local = str(getattr(fan, "platform_fan_id", "") or "").startswith("test_")
    creator_row = await asyncio.to_thread(
        lambda: (
            get_supabase()
            .table("creators")
            .select("apifansly_account_id")
            .eq("id", str(creator_id))
            .limit(1)
            .execute()
        )
    )
    rows = creator_row.data or []
    account_id = str((rows[0] if rows else {}).get("apifansly_account_id") or "")
    group_id = str(getattr(fan, "fansly_group_id", "") or "")
    if not local and (not account_id or not group_id):
        raise OutboundDeliveryError("no live delivery route for this fan")
    return account_id, group_id, local


async def send_part(
    sequence: store.OutboundSequence, part: store.SequencePart
) -> PartOutcome:
    """Send one bubble and persist its receipt. Assumes the gate already passed."""
    account_id, group_id, local = await _delivery_route(
        sequence.creator_id, sequence.fan_id
    )
    if local:
        platform_id = f"local-test:{sequence.id}:{part.part_index}"
    else:
        from main import send_fansly_message

        platform_id = await send_fansly_message(account_id, group_id, part.body)
        if not platform_id:
            raise OutboundDeliveryError("platform rejected an outbound bubble")

    metadata = dict((sequence.metadata or {}).get("message_metadata") or {})
    metadata.update(
        {
            "part": part.part_index,
            "parts": len(sequence.parts),
            "platform_message_id": platform_id,
            "outbound_sequence_id": sequence.id,
            "conversation_generation": sequence.conversation_generation,
        }
    )
    try:
        message_id = await save_message(
            sequence.fan_id,
            sequence.creator_id,
            "creator",
            part.body,
            was_ai_suggested=True,
            fansly_message_id=platform_id,
            media_context=metadata,
        )
    except Exception as exc:
        await freeze_fan_for_review(sequence.fan_id, "outbound_sent_but_not_persisted")
        raise OutboundDeliveryError(
            "bubble sent but its local receipt failed"
        ) from exc

    await store.record_part_sent(
        sequence_id=sequence.id,
        part_index=part.part_index,
        platform_message_id=str(platform_id),
        message_id=message_id,
    )
    _telemetry(
        "sent",
        sequence_id=sequence.id,
        fan_id=sequence.fan_id,
        generation=sequence.conversation_generation,
        part=part.part_index,
        parts=len(sequence.parts),
        planned_delay_seconds=part.planned_delay_seconds,
        late_by_seconds=round(
            max(0.0, (datetime.now(timezone.utc) - part.due_at).total_seconds()), 2
        ),
    )
    return PartOutcome(sent=True, message_id=message_id)


async def _settle_if_first(
    sequence: store.OutboundSequence, part: store.SequencePart
) -> None:
    """Run the commercial settlement the first delivered bubble owes.

    Bound to part 0 rather than to the turn, because a reply that never left
    must not leave a pending offer behind it, and a reply that did must record
    one exactly once however many bubbles follow.
    """
    if int(part.part_index) != 0:
        return
    instruction = dict(sequence.metadata or {}).get(settlement.POST_SEND_KEY)
    if not instruction:
        return
    try:
        result = await settlement.settle_after_first_part(
            creator_id=sequence.creator_id,
            fan_id=sequence.fan_id,
            instruction=instruction,
        )
    except Exception as exc:  # noqa: BLE001 - never resend over a settlement bug
        print(
            f"[OUTBOUND SETTLEMENT ERROR] sequence={sequence.id} "
            f"fan={sequence.fan_id}: {exc}"
        )
        return
    _telemetry(
        "settled",
        sequence_id=sequence.id,
        fan_id=sequence.fan_id,
        kind=str(instruction.get("kind") or ""),
        result=result,
    )


async def deliver_due_part(action: dict) -> PartOutcome:
    """Run one claimed ``DELIVER_OUTBOUND_PART`` obligation to a decision."""
    payload = action.get("payload") or {}
    sequence_id = str(payload.get("sequence_id") or "")
    part_index = int(payload.get("part_index") or 0)
    sequence = await store.get_sequence(sequence_id)
    if sequence is None:
        return PartOutcome(False, reason="sequence_gone")
    part = sequence.part(part_index)
    if part is None:
        return PartOutcome(False, reason="part_gone")

    gate = await evaluate_send_gate(sequence, part)
    if not gate.allowed:
        if gate.retry_at is not None:
            return PartOutcome(False, reason=gate.reason, retry_at=gate.retry_at)
        if gate.reason not in {"part_sent", "part_superseded", "part_failed"}:
            await supersede_sequence(sequence, reason=gate.reason)
        return PartOutcome(False, reason=gate.reason)

    outcome = await send_part(sequence, part)
    await _settle_if_first(sequence, part)
    remaining = [
        other
        for other in sequence.parts
        if other.part_index != part.part_index and other.status == store.PART_PENDING
    ]
    await store.set_sequence_status(
        sequence.id,
        store.STATUS_SENDING if remaining else store.STATUS_COMPLETED,
    )
    return outcome


async def deliver_sequence_now(
    sequence: store.OutboundSequence,
) -> list[str]:
    """Execute a planned sequence without waiting, keeping every send check.

    Used by the owner-only simulator and the offline evaluators. Interruption,
    generation revalidation and receipt persistence behave exactly as they do in
    production; only the wall clock is skipped.
    """
    message_ids: list[str] = []
    current = sequence
    for part in sequence.parts:
        refreshed = await store.get_sequence(sequence.id)
        if refreshed is not None:
            current = refreshed
        live_part = current.part(part.part_index)
        if live_part is None:
            continue
        gate = await evaluate_send_gate(
            current,
            live_part,
            # An immediate run has not waited, so the ordering guard must not
            # read "this bubble is two minutes late".
            now=live_part.due_at,
        )
        if not gate.allowed:
            if gate.reason not in {"part_sent", "part_superseded", "part_failed"}:
                await supersede_sequence(current, reason=gate.reason)
            break
        outcome = await send_part(current, live_part)
        await _settle_if_first(current, live_part)
        if outcome.message_id:
            message_ids.append(str(outcome.message_id))
    final = await store.get_sequence(sequence.id)
    if final is not None and final.status in store.ACTIVE_STATUSES:
        await store.set_sequence_status(
            sequence.id,
            store.STATUS_COMPLETED
            if not final.pending_parts()
            else store.STATUS_SENDING,
        )
    return message_ids
