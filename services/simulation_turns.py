"""Durable, pollable simulated Full Auto turns.

WHY THIS EXISTS
---------------
The Simulator used to run one whole Full Auto turn inside one browser request.
When the V3 writer started pursuing its own model properly — retrying a
rate-limited provider, then trying the same model on another host — a turn
could legitimately outlive the browser's 180-second ceiling. The browser then
reported::

    Timeout: The simulated turn did not finish within 180s and was cancelled

while the backend went on, the fallback answered, and the reply was persisted.
The UI and the database disagreed about whether the turn had happened. From
there: an operator presses Send again, the original turn wakes up, and one fan
receives two contradictory creator replies — after commercial state has already
moved on the first one.

A longer browser timeout would only move the number. The fix is that the
lifetime of model recovery is no longer the lifetime of an HTTP request:

    POST  ->  record the turn, return {turn_id, status: "processing"}
              background task runs the REAL pipeline against that row
    GET   ->  poll the row until it is terminal

FOUR INVARIANTS, AND WHERE EACH ONE ACTUALLY LIVES
--------------------------------------------------
*One submission is one turn.* ``(fan_id, idempotency_key)`` is unique in the
database. A double-clicked Send, a retried POST and a browser that reconnects
and resubmits all resolve to the same row. The dedupe is not a check this
module performs and hopes wins a race; it is a constraint.

*One turn at a time per simulated fan.* A partial unique index over the active
statuses. Serialised deliberately — a second concurrent turn against the same
fan would interleave two pipelines over one commercial state, and nothing about
the Simulator needs that.

*Polling never generates anything.* ``get_turn`` and ``latest_turn`` are reads.
The pipeline runs exactly once, from the task the POST spawned.

*Terminal is terminal.* Every completion is a conditional update that only
matches a still-active row. A task abandoned at the deadline therefore cannot
come back and turn a failed turn into a hidden success, and a completed turn
cannot be overwritten by a straggler.

THE DEADLINE
------------
Durable does not mean unbounded. The turn's ceiling is derived from the writer
ladder the deployment actually configured (``ai/writer_recovery.py``) plus
headroom for the rest of the pipeline, so it comfortably covers the pinned
attempts, their waits, the alternate-provider attempts and the emergency
fallback — and then stops. When it expires the task is CANCELLED, not merely
abandoned, so no outstanding model work can persist a reply afterwards. The
turn is then reconciled once against what actually reached the database, and
that single answer is written as terminal state.

TENANCY AND VISIBILITY
----------------------
Nothing here relaxes the simulator's authorization: the route resolves the
creator/fan pair through the existing six checks before any of this is reached,
and every read below is scoped by both ids. ``public_view`` reduces a turn to a
product-level status for an agency operator; provider names, model names and
raw backend errors are owner-only, exactly as they are for a persisted message.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from core.supabase import get_supabase
from core.tasks import spawn

TABLE = "simulation_turns"

STATUS_ACCEPTED = "accepted"
STATUS_PROCESSING = "processing"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"

ACTIVE_STATUSES: tuple[str, ...] = (STATUS_ACCEPTED, STATUS_PROCESSING)
TERMINAL_STATUSES: tuple[str, ...] = (STATUS_COMPLETED, STATUS_FAILED)

# A turn the backend gave up on. Distinct from the pipeline's own outcomes,
# which describe a turn that ran to a conclusion.
OUTCOME_DEADLINE_EXCEEDED = "deadline_exceeded"
OUTCOME_BACKEND_ERROR = "backend_error"

# Headroom over the writer ladder for everything else one turn does: situation
# analysis, commercial orchestration, session planning, the director, fan
# intelligence extraction and the persistence at the end. Generous, because the
# deadline exists to guarantee termination rather than to be tight — a deadline
# that fires during normal operation is a worse bug than the one it guards.
NON_WRITER_BUDGET_SECONDS = 120.0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def turn_deadline_seconds() -> float:
    """The backend-owned ceiling for ONE simulated turn, in seconds.

    Derived from the writer's own configured recovery ladder rather than from a
    number somebody typed, so tightening or widening the retry schedule cannot
    leave this behind. It is deliberately NOT related to any browser timeout:
    the browser no longer waits.
    """
    from ai.writer_recovery import writer_turn_deadline_seconds

    return float(writer_turn_deadline_seconds()) + NON_WRITER_BUDGET_SECONDS


@dataclass(frozen=True)
class SimulationTurn:
    """One durable simulated turn, as the API reports it."""

    id: str
    creator_id: str
    fan_id: str
    idempotency_key: str
    status: str
    fan_message: str = ""
    fast: bool = True
    fan_message_id: str | None = None
    creator_message_ids: list[str] = field(default_factory=list)
    outcome: str | None = None
    analysis_degraded: bool = False
    error: str | None = None
    error_id: str | None = None
    deadline_seconds: float | None = None
    created_at: str | None = None
    started_at: str | None = None
    finished_at: str | None = None

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "SimulationTurn":
        raw_ids = row.get("creator_message_ids")
        if isinstance(raw_ids, str):
            try:
                raw_ids = json.loads(raw_ids)
            except (TypeError, ValueError):
                raw_ids = []
        return cls(
            id=str(row.get("id") or ""),
            creator_id=str(row.get("creator_id") or ""),
            fan_id=str(row.get("fan_id") or ""),
            idempotency_key=str(row.get("idempotency_key") or ""),
            status=str(row.get("status") or STATUS_ACCEPTED),
            fan_message=str(row.get("fan_message") or ""),
            fast=bool(row.get("fast", True)),
            fan_message_id=(
                str(row["fan_message_id"]) if row.get("fan_message_id") else None
            ),
            creator_message_ids=[
                str(value) for value in (raw_ids or []) if str(value or "").strip()
            ],
            outcome=(str(row["outcome"]) if row.get("outcome") else None),
            analysis_degraded=bool(row.get("analysis_degraded", False)),
            error=(str(row["error"]) if row.get("error") else None),
            error_id=(str(row["error_id"]) if row.get("error_id") else None),
            deadline_seconds=(
                float(row["deadline_seconds"])
                if row.get("deadline_seconds") is not None
                else None
            ),
            created_at=row.get("created_at"),
            started_at=row.get("started_at"),
            finished_at=row.get("finished_at"),
        )

    @property
    def client_status(self) -> str:
        """The three states a client has to distinguish.

        ``accepted`` and ``processing`` are the same thing to a browser — the
        turn is running and the answer is not in yet — and collapsing them here
        means the UI never has to know that a task can exist before it starts.
        The exact row status stays available to an operator as ``stage``.
        """
        return STATUS_PROCESSING if self.status in ACTIVE_STATUSES else self.status

    def public_view(self, *, diagnostics: bool = False) -> dict[str, Any]:
        """What the dashboard receives.

        ``diagnostics`` is the platform-owner tier. An agency operator gets the
        turn's product-level status and its own outcome vocabulary, and never a
        backend exception string — which can name a provider, a model or an
        internal path, none of which an agency is told about anywhere else.
        """
        view: dict[str, Any] = {
            "turn_id": self.id,
            "status": self.client_status,
            "outcome": self.outcome,
            "fan_message_id": self.fan_message_id,
            "creator_message_ids": list(self.creator_message_ids),
            "analysis_degraded": self.analysis_degraded,
            "fast": self.fast,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }
        if diagnostics:
            view["stage"] = self.status
            view["error"] = self.error
            view["error_id"] = self.error_id
            view["deadline_seconds"] = self.deadline_seconds
        elif self.error_id:
            # The correlating id alone is safe and is the one thing an agency
            # can usefully quote when asking for help.
            view["error_id"] = self.error_id
        return view


class SimulationTurnBusy(Exception):
    """Another turn for this fan is still running.

    Carries the active turn so the caller can point the operator's browser at
    the thing it should already be watching, rather than at a bare error.
    """

    def __init__(self, active: SimulationTurn) -> None:
        super().__init__("a simulated turn is already running for this fan")
        self.active = active


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


async def get_turn(
    creator_id: str, fan_id: str, turn_id: str
) -> SimulationTurn | None:
    """One turn, scoped by BOTH ids.

    A pure read. Polling this can never start a generation — the pipeline runs
    exactly once, from the task the POST spawned.
    """

    def _read() -> dict | None:
        response = (
            get_supabase().table(TABLE)
            .select("*")
            .eq("id", str(turn_id))
            .eq("creator_id", str(creator_id))
            .eq("fan_id", str(fan_id))
            .limit(1)
            .execute()
        )
        rows = response.data or []
        return rows[0] if rows else None

    row = await asyncio.to_thread(_read)
    return SimulationTurn.from_row(row) if row else None


async def latest_turn(creator_id: str, fan_id: str) -> SimulationTurn | None:
    """The newest turn for this conversation, terminal or not.

    What a reloaded browser asks for: it knows the fan it is looking at and
    nothing else, and it needs to resume watching whatever is in flight — or
    render the result of the turn that finished while it was away.
    """

    def _read() -> dict | None:
        response = (
            get_supabase().table(TABLE)
            .select("*")
            .eq("creator_id", str(creator_id))
            .eq("fan_id", str(fan_id))
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        rows = response.data or []
        return rows[0] if rows else None

    row = await asyncio.to_thread(_read)
    return SimulationTurn.from_row(row) if row else None


async def _active_turn(fan_id: str) -> SimulationTurn | None:
    def _read() -> dict | None:
        response = (
            get_supabase().table(TABLE)
            .select("*")
            .eq("fan_id", str(fan_id))
            .in_("status", list(ACTIVE_STATUSES))
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        rows = response.data or []
        return rows[0] if rows else None

    row = await asyncio.to_thread(_read)
    return SimulationTurn.from_row(row) if row else None


async def _turn_by_key(fan_id: str, idempotency_key: str) -> SimulationTurn | None:
    def _read() -> dict | None:
        response = (
            get_supabase().table(TABLE)
            .select("*")
            .eq("fan_id", str(fan_id))
            .eq("idempotency_key", str(idempotency_key))
            .limit(1)
            .execute()
        )
        rows = response.data or []
        return rows[0] if rows else None

    row = await asyncio.to_thread(_read)
    return SimulationTurn.from_row(row) if row else None


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------


async def _record_turn(payload: dict[str, Any]) -> SimulationTurn | None:
    """Insert one turn, or return None when the key was already taken.

    ``ignore_duplicates`` makes the conflicting insert a DO NOTHING that
    returns no row, so "did I create this turn?" is answered by the database
    rather than by a read the next request can race.
    """

    def _write() -> list[dict]:
        response = (
            get_supabase().table(TABLE)
            .upsert(
                payload,
                on_conflict="fan_id,idempotency_key",
                ignore_duplicates=True,
            )
            .execute()
        )
        return list(response.data or [])

    rows = await asyncio.to_thread(_write)
    return SimulationTurn.from_row(rows[0]) if rows else None


async def _update_if_active(turn_id: str, changes: dict[str, Any]) -> bool:
    """Apply changes only while the turn is still active.

    This is the whole of the "terminal is terminal" guarantee. A task that was
    abandoned at the deadline, and whose turn has already been written as
    failed, updates nothing here — so a reply it was still holding cannot
    become a completed turn after the operator was told it failed.
    """

    def _write() -> list[dict]:
        response = (
            get_supabase().table(TABLE)
            .update({**changes, "updated_at": _now()})
            .eq("id", str(turn_id))
            .in_("status", list(ACTIVE_STATUSES))
            .execute()
        )
        return list(response.data or [])

    rows = await asyncio.to_thread(_write)
    return bool(rows)


# ---------------------------------------------------------------------------
# Starting a turn
# ---------------------------------------------------------------------------


async def schedule_turn_execution(coro: Any, name: str) -> None:
    """Hand the turn to the background and return immediately.

    The one seam between "the request is done" and "the pipeline is running",
    and the reason it is a named function rather than a bare ``spawn`` call: a
    test substitutes an implementation that awaits the coroutine, so a turn
    becomes deterministic without the test having to race an event loop it does
    not control. Production always returns before the turn finishes — that is
    the entire point of the change.
    """
    spawn(coro, name=name)


async def start_turn(
    *,
    creator_id: str,
    fan_id: str,
    message: str,
    fast: bool = True,
    include_mirrored_catalog: bool = False,
    idempotency_key: str | None = None,
    runner: Any = None,
) -> tuple[SimulationTurn, bool]:
    """Record one simulated turn and start the real pipeline behind it.

    Returns ``(turn, created)``. ``created`` is False when this submission was
    already known — a duplicate POST, a retried request, a browser that
    reconnected and sent the same key — in which case the existing turn is
    returned and NOTHING is re-executed.

    Raises ``SimulationTurnBusy`` when a different turn is still running for
    this fan. Simple serialisation per simulated fan is deliberate: two
    pipelines interleaved over one commercial state is not a thing the
    Simulator needs, and pretending to support it would be a much larger
    correctness problem than the queueing it saves.

    ``runner`` is the coroutine function that actually executes the turn, taken
    as a parameter purely so a test can substitute a deterministic one; it
    defaults to the real Full Auto simulated-inbound path.
    """

    key = str(idempotency_key or "").strip() or uuid.uuid4().hex
    deadline = turn_deadline_seconds()

    existing = await _turn_by_key(fan_id, key)
    if existing is not None:
        print(
            f"[SIM TURN] turn_id={existing.id} status=duplicate_submission "
            f"fan={fan_id} existing_status={existing.status}"
        )
        return existing, False

    active = await _active_turn(fan_id)
    if active is not None:
        raise SimulationTurnBusy(active)

    payload = {
        "creator_id": str(creator_id),
        "fan_id": str(fan_id),
        "idempotency_key": key,
        "status": STATUS_ACCEPTED,
        "fan_message": str(message or ""),
        "fast": bool(fast),
        "include_mirrored_catalog": bool(include_mirrored_catalog),
        "creator_message_ids": [],
        "analysis_degraded": False,
        "deadline_seconds": round(deadline, 2),
        "created_at": _now(),
        "updated_at": _now(),
    }
    turn = await _record_turn(payload)
    if turn is None:
        # The unique key was taken between the read above and this insert, which
        # is exactly the race the constraint exists for. Whoever won owns the
        # generation; this caller joins it.
        joined = await _turn_by_key(fan_id, key)
        if joined is not None:
            print(
                f"[SIM TURN] turn_id={joined.id} status=duplicate_submission "
                f"fan={fan_id} raced=true"
            )
            return joined, False
        # Losing the insert to the ACTIVE-turn index rather than to the key.
        busy = await _active_turn(fan_id)
        if busy is not None:
            raise SimulationTurnBusy(busy)
        raise RuntimeError("could not record the simulated turn")

    print(
        f"[SIM TURN] turn_id={turn.id} status=accepted fan={fan_id} "
        f"creator={creator_id} fast={bool(fast)} deadline={deadline:.0f}s"
    )

    await schedule_turn_execution(
        execute_turn(
            turn_id=turn.id,
            creator_id=str(creator_id),
            fan_id=str(fan_id),
            message=str(message or ""),
            fast=bool(fast),
            include_mirrored_catalog=bool(include_mirrored_catalog),
            deadline_seconds=deadline,
            runner=runner,
        ),
        f"simulation_turn:{turn.id}",
    )
    return turn, True


# ---------------------------------------------------------------------------
# Running a turn
# ---------------------------------------------------------------------------


async def _creator_message_ids(fan_id: str) -> set[str] | None:
    """Ids of this fan's recent creator messages, or None when unreadable.

    Reuses the Simulator's own diff helper rather than a second query, so the
    reconciliation below sees exactly what the turn would have reported.

    None rather than an exception, and None rather than an empty set: this read
    exists only to decide what an abandoned turn actually did, and a transcript
    we could not read is "unknown", not "nothing was written". Treating a
    failed read as an empty snapshot is how a turn that produced no reply would
    later be reported as having produced every message already in the
    conversation.
    """
    from services.suggestions import _recent_creator_message_rows

    try:
        rows = await _recent_creator_message_rows(fan_id)
    except Exception as exc:
        print(f"[SIM TURN] transcript snapshot failed fan={fan_id}: {exc}")
        return None
    return {row["id"] for row in rows}


async def execute_turn(
    *,
    turn_id: str,
    creator_id: str,
    fan_id: str,
    message: str,
    fast: bool,
    include_mirrored_catalog: bool,
    deadline_seconds: float,
    runner: Any = None,
) -> None:
    """Run one simulated turn to a terminal state. Never raises.

    The background half of ``start_turn``. Separate and public so a test can
    drive it directly, and so the only thing the POST does is record and spawn.
    """
    if runner is None:
        from services.suggestions import run_simulated_inbound

        runner = run_simulated_inbound

    started = time.monotonic()
    before = await _creator_message_ids(fan_id)

    await _update_if_active(turn_id, {"status": STATUS_PROCESSING, "started_at": _now()})
    print(f"[SIM TURN] turn_id={turn_id} status=processing fan={fan_id}")

    try:
        # wait_for CANCELS the work on expiry and waits for it to unwind before
        # raising, rather than leaving it running. That is the whole point: an
        # abandoned pipeline that later persists a creator reply is the ghost
        # this design exists to make impossible, and by the time the handler
        # below reads the transcript the task is genuinely finished.
        result = await asyncio.wait_for(
            runner(
                fan_id=fan_id,
                creator_id=creator_id,
                message=message,
                fast=fast,
                include_mirrored_catalog=include_mirrored_catalog,
            ),
            timeout=deadline_seconds,
        )
    except asyncio.TimeoutError:
        await _finalize_after_cancellation(
            turn_id=turn_id,
            fan_id=fan_id,
            before=before,
            elapsed=time.monotonic() - started,
            deadline_seconds=deadline_seconds,
        )
        return
    except asyncio.CancelledError:
        # The process is shutting down, or somebody cancelled us. Record the
        # turn as terminally failed rather than leaving it "processing"
        # forever, then let the cancellation continue.
        #
        # Best effort by construction: this write is itself running inside a
        # cancelled task, so it may not get to finish. It must never replace
        # the cancellation with some other exception, hence the bare guard —
        # a turn left "processing" by a hard shutdown is recoverable from the
        # UI, an unwind that raises the wrong error is not.
        try:
            await _record_failure(
                turn_id,
                outcome=OUTCOME_BACKEND_ERROR,
                error="the simulated turn was cancelled before it finished",
            )
        except BaseException:
            pass
        raise
    except Exception as exc:
        error_id = uuid.uuid4().hex[:12]
        print(
            f"[SIM TURN] turn_id={turn_id} status=failed fan={fan_id} "
            f"error_id={error_id} type={type(exc).__name__} error={exc}"
        )
        import traceback

        traceback.print_exc()
        await _record_failure(
            turn_id,
            outcome=OUTCOME_BACKEND_ERROR,
            error=f"{type(exc).__name__}: {exc}",
            error_id=error_id,
        )
        return

    # A runner that returned nothing is a turn that ran and produced no reply,
    # not a crash. Treated as an empty result rather than being allowed to
    # raise, because raising here would report a completed turn as a backend
    # failure.
    settled: dict[str, Any] = result if isinstance(result, dict) else {}
    creator_messages = list(settled.get("creator_messages") or [])
    changes = {
        "status": STATUS_COMPLETED,
        "fan_message_id": settled.get("fan_message_id"),
        "creator_message_ids": [
            str(row.get("id")) for row in creator_messages if row.get("id")
        ],
        "outcome": settled.get("outcome"),
        "analysis_degraded": bool(settled.get("analysis_degraded")),
        "finished_at": _now(),
    }
    written = await _update_if_active(turn_id, changes)
    # The status in this line is what was RECORDED, not what this task wanted
    # to record. A straggler that lost to an already-terminal turn must not
    # leave a "status=completed" line behind for a turn the operator was told
    # had failed.
    print(
        f"[SIM TURN] turn_id={turn_id} "
        f"status={STATUS_COMPLETED if written else 'superseded'} fan={fan_id} "
        f"outcome={settled.get('outcome')} messages={len(creator_messages)} "
        f"elapsed={time.monotonic() - started:.1f}s"
    )


async def _finalize_after_cancellation(
    *,
    turn_id: str,
    fan_id: str,
    before: set[str] | None,
    elapsed: float,
    deadline_seconds: float,
) -> None:
    """Decide ONCE what an abandoned turn actually did, then write it.

    Cancellation is cooperative, so there is a narrow window in which the
    pipeline had already persisted a creator reply when the deadline fired.
    Reconciling against the transcript here — after the task has finished
    unwinding — is what keeps the terminal state and the database agreeing.
    It happens once, and what it writes is final: a failed turn never becomes
    a hidden success afterwards, and a turn that genuinely produced a reply is
    never reported as a failure the operator would try to repeat.
    """
    after = await _creator_message_ids(fan_id)

    # Both snapshots have to be readable for the diff to mean anything. When
    # either is not, the turn is reported as failed: that is the answer that
    # never claims a reply which may not exist, and the transcript itself is
    # still whatever it is.
    produced: list[str] = []
    if before is not None and after is not None:
        produced = [message_id for message_id in after if message_id not in before]

    if produced:
        await _update_if_active(
            turn_id,
            {
                "status": STATUS_COMPLETED,
                "creator_message_ids": produced,
                "outcome": "replied",
                "finished_at": _now(),
            },
        )
        print(
            f"[SIM TURN] turn_id={turn_id} status=completed fan={fan_id} "
            f"outcome=replied messages={len(produced)} "
            f"note=persisted_before_deadline elapsed={elapsed:.1f}s"
        )
        return

    await _record_failure(
        turn_id,
        outcome=OUTCOME_DEADLINE_EXCEEDED,
        error=(
            f"the simulated turn did not finish within the backend deadline of "
            f"{deadline_seconds:.0f}s and was cancelled"
        ),
    )
    print(
        f"[SIM TURN] turn_id={turn_id} status=failed fan={fan_id} "
        f"outcome={OUTCOME_DEADLINE_EXCEEDED} elapsed={elapsed:.1f}s "
        f"deadline={deadline_seconds:.0f}s"
    )


async def _record_failure(
    turn_id: str,
    *,
    outcome: str,
    error: str,
    error_id: str | None = None,
) -> None:
    await _update_if_active(
        turn_id,
        {
            "status": STATUS_FAILED,
            "outcome": outcome,
            "error": error[:2000],
            "error_id": error_id,
            "finished_at": _now(),
        },
    )


# ---------------------------------------------------------------------------
# What the operator is told while it runs
# ---------------------------------------------------------------------------
#
# Product-level, always. An agency operator is never told that a provider is
# rate-limiting, which provider it is, which model is being pursued, or that a
# fallback exists — the same boundary the persisted message marker respects.
# "The primary writer is temporarily busy" is true, useful and says none of it.

#: After this long the UI stops saying "generating" and starts explaining.
SLOW_TURN_NOTICE_SECONDS = 45


def progress_message(elapsed_seconds: float) -> str:
    """The product-level status for a turn that is still running."""
    if elapsed_seconds >= SLOW_TURN_NOTICE_SECONDS:
        return "Still generating — the primary writer is temporarily busy."
    return "Generating reply…"
