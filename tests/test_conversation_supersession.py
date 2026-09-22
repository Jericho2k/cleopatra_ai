"""Interruption and supersession, decided durably rather than in one process.

The behaviour these cover is the whole reason the sprint exists. Full Auto used
to answer "has a newer message replaced this reply?" from a dictionary in one
Python process, polled every 0.5 s by a sleeping coroutine. That is correct only
while every event for a fan lands in one process, and it charges a live
coroutine for the whole of a deliberate pause.

So the tests below never ask a coroutine anything. They move the durable
conversation generation — exactly what a fan message handled by ANOTHER process
does — and then assert what the send boundary decides.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from db import outbound_queries as store
from services import conversation_generation, outbound_delivery
from services.conversation_generation import bumps_generation
from services.human_delivery import AvailabilityMode, DeliverySchedule
from tests.fake_supabase import FakeSupabase


def run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _reset_module_state():
    store._reset_availability_for_tests()
    conversation_generation._COLUMN_AVAILABLE = True
    conversation_generation._RPC_AVAILABLE = True
    yield
    store._reset_availability_for_tests()
    conversation_generation._COLUMN_AVAILABLE = True
    conversation_generation._RPC_AVAILABLE = True


@pytest.fixture
def db(monkeypatch):
    fake = FakeSupabase(
        {
            "fans": [
                {
                    "id": "fan-1",
                    "creator_id": "creator-1",
                    "conversation_generation": 7,
                    "needs_human_review": False,
                    "auto_mode": True,
                    "platform_fan_id": "test_fan_1",
                    "fansly_group_id": "",
                }
            ],
            "creators": [{"id": "creator-1", "apifansly_account_id": ""}],
            "outbound_sequences": [],
            "outbound_sequence_parts": [],
            "fan_execution_leases": [],
        }
    )
    monkeypatch.setattr(store, "get_supabase", lambda: fake)
    monkeypatch.setattr(conversation_generation, "get_supabase", lambda: fake)
    return fake


def schedule(composition: float = 4.0, gaps: tuple[float, ...] = (3.0, 5.0)):
    return DeliverySchedule(
        availability_delay_seconds=30.0,
        composition_delay_seconds=composition,
        inter_part_delays_seconds=gaps,
        availability_mode=AvailabilityMode.LIVE,
    )


class Recorder:
    """Captures everything a planned sequence would enqueue and send."""

    def __init__(self) -> None:
        self.actions: list[dict] = []
        self.sent: list[tuple[str, int, str]] = []
        self.cancelled: list[str] = []

    async def schedule_action(self, **kwargs):
        self.actions.append(kwargs)

    async def cancel(self, dedupe_key: str) -> None:
        self.cancelled.append(dedupe_key)

    async def save_message(self, fan_id, creator_id, role, content, **kwargs):
        context = kwargs.get("media_context") or {}
        self.sent.append(
            (str(fan_id), int(context.get("part") or 0), str(content))
        )
        return f"msg-{len(self.sent)}"


@pytest.fixture
def recorder(monkeypatch, db):
    recording = Recorder()
    monkeypatch.setattr(outbound_delivery, "schedule_action", recording.schedule_action)
    monkeypatch.setattr(
        outbound_delivery, "cancel_action_by_dedupe_key", recording.cancel
    )
    monkeypatch.setattr(outbound_delivery, "save_message", recording.save_message)

    async def _route(creator_id, fan_id):
        return "", "", True

    monkeypatch.setattr(outbound_delivery, "_delivery_route", _route)

    async def _fan(fan_id):
        rows = db.tables["fans"]
        row = next((r for r in rows if r["id"] == str(fan_id)), None)
        if row is None:
            return None
        return SimpleNamespace(
            id=row["id"],
            needs_human_review=bool(row.get("needs_human_review")),
            auto_mode=row.get("auto_mode"),
            platform_fan_id=row.get("platform_fan_id"),
            fansly_group_id=row.get("fansly_group_id"),
        )

    monkeypatch.setattr("db.queries.get_fan_by_id", _fan)
    return recording


async def plan(recorder, *, parts=("one", "two", "three"), generation=7):
    return await outbound_delivery.schedule_outbound_sequence(
        creator_id="creator-1",
        fan_id="fan-1",
        trigger_identity="message-1",
        turn_id="turn-1",
        parts=list(parts),
        schedule=schedule(),
        conversation_generation=generation,
        metadata=outbound_delivery.sequence_metadata(
            message_metadata={"turn_id": "turn-1"}
        ),
    )


def part_action(sequence_id: str, index: int) -> dict:
    return {
        "id": f"action-{index}",
        "creator_id": "creator-1",
        "fan_id": "fan-1",
        "action_type": outbound_delivery.DELIVER_PART_ACTION,
        "payload": {"sequence_id": sequence_id, "part_index": index},
    }


# --- what moves the generation, and what deliberately does not --------------


def test_a_new_fan_message_moves_the_conversation_on():
    assert bumps_generation("fan", was_ai_suggested=False, inserted=True) is True


def test_a_human_creator_reply_moves_the_conversation_on():
    assert bumps_generation("creator", was_ai_suggested=False, inserted=True) is True


def test_an_automated_bubble_never_invalidates_its_own_siblings():
    """Bubble 1 must not be the reason bubble 2 is refused."""
    assert bumps_generation("creator", was_ai_suggested=True, inserted=True) is False


def test_a_duplicate_platform_delivery_does_not_move_the_conversation():
    """REL-002's ``inserted`` flag is what makes webhook redelivery harmless."""
    assert bumps_generation("fan", was_ai_suggested=False, inserted=False) is False


def test_bumping_is_readable_back(db):
    assert run(conversation_generation.current_generation("fan-1")) == 7
    assert run(conversation_generation.bump_generation("fan-1", reason="test")) == 8
    assert run(conversation_generation.current_generation("fan-1")) == 8


# --- planning ---------------------------------------------------------------


def test_planned_due_times_follow_the_human_delivery_schedule():
    now = datetime(2026, 9, 22, 12, 0, tzinfo=timezone.utc)
    plan_rows = outbound_delivery.plan_due_times(schedule(), 3, now=now)

    assert [row[0] for row in plan_rows] == [4.0, 3.0, 5.0]
    assert [row[1] for row in plan_rows] == [
        now + timedelta(seconds=4),
        now + timedelta(seconds=7),
        now + timedelta(seconds=12),
    ]


def test_the_availability_delay_is_not_charged_twice():
    """It was already spent as the AUTO_REPLY action's own execute_at."""
    now = datetime(2026, 9, 22, 12, 0, tzinfo=timezone.utc)
    first = outbound_delivery.plan_due_times(schedule(), 1, now=now)[0][1]
    assert (first - now).total_seconds() == 4.0


def test_planning_queues_one_durable_action_per_bubble(recorder):
    sequence = run(plan(recorder))

    assert sequence is not None
    assert len(sequence.parts) == 3
    assert [a["action_type"] for a in recorder.actions] == [
        outbound_delivery.DELIVER_PART_ACTION
    ] * 3
    assert [a["payload"]["part_index"] for a in recorder.actions] == [0, 1, 2]
    # Nothing has been sent by planning. That is the point: the worker returns
    # its slot and the bubbles leave from the queue.
    assert recorder.sent == []


def test_the_plan_records_what_production_would_have_waited(recorder):
    sequence = run(plan(recorder))
    timing = sequence.planned_timing

    assert timing["availability_mode"] == "live"
    assert timing["composition_delay_seconds"] == 4.0
    assert timing["inter_part_delays_seconds"] == [3.0, 5.0]


def test_replanning_the_same_trigger_adopts_the_existing_sequence(recorder):
    first = run(plan(recorder))
    second = run(plan(recorder))

    assert second is not None
    assert second.id == first.id


# --- the send boundary ------------------------------------------------------


def test_a_bubble_sends_when_the_generation_still_matches(recorder):
    sequence = run(plan(recorder))
    outcome = run(outbound_delivery.deliver_due_part(part_action(sequence.id, 0)))

    assert outcome.sent is True
    assert recorder.sent == [("fan-1", 0, "one")]


def test_a_fan_reply_before_the_first_bubble_stops_the_whole_reply(recorder, db):
    """Case B: he replied during the composition delay."""
    sequence = run(plan(recorder))
    run(conversation_generation.bump_generation("fan-1", reason="fan replied"))

    outcome = run(outbound_delivery.deliver_due_part(part_action(sequence.id, 0)))

    assert outcome.sent is False
    assert outcome.reason == "stale_generation"
    assert recorder.sent == []
    refreshed = run(store.get_sequence(sequence.id))
    assert refreshed.status == store.STATUS_SUPERSEDED
    assert all(part.status == store.PART_SUPERSEDED for part in refreshed.parts)


def test_a_fan_reply_after_bubble_one_cancels_bubble_two_and_three(recorder):
    """Case C: bubble 1 stays canon; the rest are stale."""
    sequence = run(plan(recorder))
    assert run(outbound_delivery.deliver_due_part(part_action(sequence.id, 0))).sent

    run(conversation_generation.bump_generation("fan-1", reason="fan replied"))

    second = run(outbound_delivery.deliver_due_part(part_action(sequence.id, 1)))
    third = run(outbound_delivery.deliver_due_part(part_action(sequence.id, 2)))

    assert second.sent is False and second.reason == "stale_generation"
    assert third.sent is False
    assert [row[1] for row in recorder.sent] == [0]
    refreshed = run(store.get_sequence(sequence.id))
    assert refreshed.part(0).status == store.PART_SENT
    assert refreshed.part(1).status == store.PART_SUPERSEDED
    assert refreshed.part(2).status == store.PART_SUPERSEDED
    # The queued actions for the abandoned bubbles are retired too, rather than
    # left to be claimed and refused one at a time.
    assert outbound_delivery.part_dedupe_key(sequence.id, 1) in recorder.cancelled


def test_a_restart_between_bubbles_does_not_resend_bubble_one(recorder):
    """Case 6: the process dies after bubble 1. Bubble 2 is still durable."""
    sequence = run(plan(recorder))
    run(outbound_delivery.deliver_due_part(part_action(sequence.id, 0)))

    # A restart loses every coroutine. Re-claiming the SAME action must be a
    # no-op, and bubble 2 must still be deliverable.
    replay = run(outbound_delivery.deliver_due_part(part_action(sequence.id, 0)))
    assert replay.sent is False
    assert replay.reason == "part_sent"

    second = run(outbound_delivery.deliver_due_part(part_action(sequence.id, 1)))
    assert second.sent is True
    assert [row[1] for row in recorder.sent] == [0, 1]


def test_a_bubble_waits_for_its_predecessor_rather_than_overtaking_it(recorder):
    sequence = run(plan(recorder))

    outcome = run(outbound_delivery.deliver_due_part(part_action(sequence.id, 1)))

    assert outcome.sent is False
    assert outcome.reason == "waiting_for_earlier_bubble"
    assert outcome.retry_at is not None
    assert recorder.sent == []


def test_a_review_hold_taken_mid_sequence_stops_the_rest(recorder, db):
    """Case 13: human review becomes active while a future action waits."""
    sequence = run(plan(recorder))
    run(outbound_delivery.deliver_due_part(part_action(sequence.id, 0)))
    db.tables["fans"][0]["needs_human_review"] = True

    outcome = run(outbound_delivery.deliver_due_part(part_action(sequence.id, 1)))

    assert outcome.sent is False
    assert outcome.reason == "human_review_hold"


def test_auto_mode_switched_off_mid_sequence_stops_the_rest(recorder, db):
    """Case 12: auto mode disabled while a future action waits."""
    sequence = run(plan(recorder))
    run(outbound_delivery.deliver_due_part(part_action(sequence.id, 0)))
    db.tables["fans"][0]["auto_mode"] = False

    outcome = run(outbound_delivery.deliver_due_part(part_action(sequence.id, 1)))

    assert outcome.sent is False
    assert outcome.reason == "auto_mode_off"


def test_a_newer_authorized_turn_retires_the_older_plan(recorder):
    sequence = run(plan(recorder))

    superseded = run(
        outbound_delivery.supersede_active_sequences(
            "fan-1", reason="newer_authorized_turn"
        )
    )

    assert superseded == 1
    refreshed = run(store.get_sequence(sequence.id))
    assert refreshed.status == store.STATUS_SUPERSEDED


# --- immediate mode keeps every check --------------------------------------


def test_immediate_delivery_sends_everything_without_waiting(recorder):
    from services.delivery_mode import immediate_delivery_scope

    with immediate_delivery_scope():
        sequence = run(plan(recorder))
        ids = run(outbound_delivery.deliver_sequence_now(sequence))

    assert len(ids) == 3
    assert [row[1] for row in recorder.sent] == [0, 1, 2]
    # No durable part actions: immediate mode is the simulator's, and it does
    # not queue work a test process will never drain.
    assert recorder.actions == []


def test_immediate_delivery_still_obeys_supersession(recorder):
    from services.delivery_mode import immediate_delivery_scope

    with immediate_delivery_scope():
        sequence = run(plan(recorder))
        run(conversation_generation.bump_generation("fan-1", reason="fan replied"))
        ids = run(outbound_delivery.deliver_sequence_now(sequence))

    assert ids == []
    assert recorder.sent == []


# --- the one place the conversation is declared to have moved on -----------


class WriteDouble:
    """A minimal messages/fans store for the save_message chokepoint."""

    def __init__(self) -> None:
        self.messages: list[dict] = []
        self.fans: list[dict] = [
            {"id": "fan-1", "creator_id": "creator-1", "conversation_generation": 3}
        ]

    def table(self, name):
        return _WriteQuery(self, name)


class _WriteQuery:
    def __init__(self, db, table):
        self._db = db
        self._table = table
        self._op = "select"
        self._payload = None
        self._filters: dict = {}

    def select(self, *_a, **_k):
        return self

    def eq(self, column, value):
        self._filters[column] = value
        return self

    def limit(self, _n):
        return self

    def insert(self, payload):
        self._op = "insert"
        self._payload = payload
        return self

    def update(self, payload):
        self._op = "update"
        self._payload = payload
        return self

    def upsert(self, payload, on_conflict=None, ignore_duplicates=False, **_k):
        self._op = "upsert"
        self._payload = payload
        return self

    def _rows(self):
        return self._db.messages if self._table == "messages" else self._db.fans

    def execute(self):
        from types import SimpleNamespace

        rows = self._rows()
        if self._op == "upsert":
            key = (
                self._payload.get("creator_id"),
                self._payload.get("fansly_message_id"),
            )
            if any(
                (row.get("creator_id"), row.get("fansly_message_id")) == key
                for row in rows
            ):
                return SimpleNamespace(data=[])
            row = {**self._payload, "id": f"msg-{len(rows) + 1}"}
            rows.append(row)
            return SimpleNamespace(data=[row])
        if self._op == "insert":
            row = {**self._payload, "id": f"msg-{len(rows) + 1}"}
            rows.append(row)
            return SimpleNamespace(data=[row])
        matches = [
            row
            for row in rows
            if all(str(row.get(k)) == str(v) for k, v in self._filters.items())
        ]
        if self._op == "update":
            for row in matches:
                row.update(self._payload)
            return SimpleNamespace(data=matches)
        return SimpleNamespace(data=matches[:1])


@pytest.fixture
def writes(monkeypatch):
    from db import queries

    double = WriteDouble()
    monkeypatch.setattr(queries, "get_supabase", lambda: double)
    return double


def save(writes_double, role="fan", *, platform_id=None, ai=False, content="hi"):
    from db.queries import save_message_result

    return run(
        save_message_result(
            "fan-1",
            "creator-1",
            role,
            content,
            was_ai_suggested=ai,
            fansly_message_id=platform_id,
        )
    )


def generation_of(writes_double) -> int:
    return int(writes_double.fans[0]["conversation_generation"])


def test_a_fan_message_moves_the_conversation_on_at_the_write(writes):
    save(writes, platform_id="p-1")

    assert generation_of(writes) == 4


def test_a_duplicate_webhook_delivery_changes_nothing(writes):
    """Case 15: redelivery must not bump, reply again, or replan a sequence."""
    first = save(writes, platform_id="p-1")
    second = save(writes, platform_id="p-1")

    assert first.inserted is True and second.inserted is False
    assert len(writes.messages) == 1
    assert generation_of(writes) == 4


def test_a_burst_of_three_messages_leaves_one_current_generation(writes):
    """Case 1: the whole burst is one conversation, answered once."""
    for index in range(3):
        save(writes, platform_id=f"p-{index}")

    assert generation_of(writes) == 6
    # Any sequence planned from an earlier message in the burst is stale
    # against this, which is what collapses three triggers into one reply.


def test_an_automated_bubble_does_not_move_the_conversation_on(writes):
    save(writes, role="creator", ai=True, platform_id="p-out-1")

    assert generation_of(writes) == 3


def test_an_operator_reply_takes_the_conversation_over(writes):
    save(writes, role="creator", ai=False, platform_id="p-human-1")

    assert generation_of(writes) == 4
