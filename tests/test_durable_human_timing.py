"""Human timing restored to Core v1 — durably, and without per-fan polling.

Two regressions this closes, and they pull in opposite directions:

  * Core v1's delivery path had lost the timing mathematics in
    ``services/human_delivery.py`` entirely, so replies arrived instantly.
  * The implementation that HAD that timing paid for it with a sleeping
    coroutine per fan, polled every 0.5 s, whose cancellation lived in one
    process's dictionary.

So the tests below assert both halves: that the schedule is computed and used,
and that no part of the mechanism is a timer, a poll, or a process-local
registry.
"""

from __future__ import annotations

import asyncio
import inspect
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from ai.generation_trace import GenerationTrace
from db import outbound_queries as store
from models.commercial import CreatorPolicy, FanCommercialState, FanStatus
from models.conversation_decision import ConversationDecision
from models.live_orchestration import (
    ApprovedExecution,
    EvidenceSnapshot,
    TurnTrigger,
    ValidationResult,
)
from models.schemas import Fan, Message, Persona
from services import conversation_generation, live_orchestration, outbound_delivery
from services.context_packet import ContextPacket
from services.conversation_core import CORE_CONVERSATIONAL_V1
from services.delivery_mode import immediate_delivery_scope
from services.human_delivery import AvailabilityMode, build_delivery_schedule
from services.reply_provenance import PIPELINE_AUTO, ReplyProvenance
from tests.fake_supabase import FakeSupabase
from workers import scheduled_actions as worker


def run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _reset():
    store._reset_availability_for_tests()
    conversation_generation._COLUMN_AVAILABLE = True
    conversation_generation._RPC_AVAILABLE = True
    yield
    store._reset_availability_for_tests()
    conversation_generation._COLUMN_AVAILABLE = True
    conversation_generation._RPC_AVAILABLE = True


@pytest.fixture
def world(monkeypatch):
    fake = FakeSupabase(
        {
            "fans": [
                {
                    "id": "fan-1",
                    "creator_id": "creator-1",
                    "conversation_generation": 4,
                    "needs_human_review": False,
                    "auto_mode": True,
                    "platform_fan_id": "test_fan_1",
                    "fansly_group_id": "",
                }
            ],
            "creators": [{"id": "creator-1", "apifansly_account_id": ""}],
            "outbound_sequences": [],
            "outbound_sequence_parts": [],
        }
    )
    monkeypatch.setattr(store, "get_supabase", lambda: fake)
    monkeypatch.setattr(conversation_generation, "get_supabase", lambda: fake)

    queued: list = []
    sent: list = []

    async def _schedule(**kwargs):
        queued.append(kwargs)

    async def _save(fan_id, creator_id, role, content, **kwargs):
        sent.append(content)
        return f"m-{len(sent)}"

    async def _route(creator_id, fan_id):
        return "", "", True

    async def _fan(fan_id):
        row = fake.tables["fans"][0]
        return SimpleNamespace(
            id=row["id"],
            needs_human_review=bool(row["needs_human_review"]),
            auto_mode=row["auto_mode"],
            platform_fan_id=row["platform_fan_id"],
            fansly_group_id=row["fansly_group_id"],
        )

    monkeypatch.setattr(outbound_delivery, "schedule_action", _schedule)
    monkeypatch.setattr(outbound_delivery, "save_message", _save)
    monkeypatch.setattr(outbound_delivery, "_delivery_route", _route)
    monkeypatch.setattr("db.queries.get_fan_by_id", _fan)
    return SimpleNamespace(db=fake, queued=queued, sent=sent)


def history(gap_minutes: float) -> list[Message]:
    now = datetime.now(timezone.utc)
    return [
        Message(
            role="creator",
            content="what are you up to",
            sent_at=now - timedelta(minutes=gap_minutes),
        ),
        Message(role="fan", content="thinking about you", sent_at=now),
    ]


def prepared(
    *,
    replies: list[str],
    conversation_history: list[Message],
    generation: int = 4,
    core: str = CORE_CONVERSATIONAL_V1,
) -> live_orchestration.PreparedTurn:
    snapshot = EvidenceSnapshot(
        creator_id="creator-1",
        fan_id="fan-1",
        trigger=TurnTrigger(
            kind="fan_message",
            identity="message-1",
            latest_message="thinking about you",
        ),
        state_revision="rev-1",
    )
    evidence = live_orchestration.LoadedEvidence(
        snapshot=snapshot,
        packet=ContextPacket(),
        history=conversation_history,
        fan=Fan(
            id="fan-1",
            display_name="Fan",
            platform_fan_id="test_fan_1",
            auto_mode=True,
        ),
        persona=Persona(),
        commercial_state=FanCommercialState(status=FanStatus.IDLE),
        policy=CreatorPolicy(),
        next_offer=None,
        active_session=None,
        pending_payment=None,
        sent_ppv=[],
        within_daily_caps=True,
        stack=SimpleNamespace(profile_id="cleo_v3"),
        conversation_generation=generation,
    )
    return live_orchestration.PreparedTurn(
        loaded=evidence,
        decision=ConversationDecision(),
        execution=ApprovedExecution(validation=ValidationResult(True, "none")),
        replies=replies,
        provenance=ReplyProvenance(
            creator_id="creator-1", fan_id="fan-1", mode=PIPELINE_AUTO
        ),
        writer_trace=GenerationTrace(),
        conversation_core=core,
    )


# --- the timing mathematics is back, and it is the old one -----------------


def test_a_live_exchange_is_answered_like_a_live_exchange():
    schedule = build_delivery_schedule(
        "you still there?",
        ["yeah", "just got distracted by you"],
        conversation_history=history(gap_minutes=1),
    )

    assert schedule.availability_mode is AvailabilityMode.LIVE
    assert 2.5 <= schedule.composition_delay_seconds <= 22.0
    assert len(schedule.inter_part_delays_seconds) == 1


def test_a_resumed_conversation_is_not_answered_instantly():
    schedule = build_delivery_schedule(
        "hey stranger",
        ["hi you"],
        conversation_history=history(gap_minutes=60 * 8),
    )

    assert schedule.availability_mode is AvailabilityMode.RETURNING
    assert schedule.availability_delay_seconds > 15


def test_a_longer_bubble_takes_longer_to_type():
    short = build_delivery_schedule(
        "hi", ["mm"], conversation_history=history(1)
    ).composition_delay_seconds
    long = build_delivery_schedule(
        "hi", ["x" * 400], conversation_history=history(1)
    ).composition_delay_seconds

    assert long > short


def test_core_v1_uses_that_schedule(world):
    turn = prepared(
        replies=["first thought|second thought"], conversation_history=history(1)
    )

    result = run(live_orchestration.deliver_reply(turn, expected_revision="rev-1"))

    assert result["outcome"] == live_orchestration.OUTCOME_SCHEDULED
    timing = result["planned_timing"]
    assert timing["availability_mode"] == "live"
    assert timing["composition_delay_seconds"] > 0
    assert len(timing["inter_part_delays_seconds"]) == 1


def test_the_intimate_window_comes_from_intimacy_context_not_a_director(world):
    from models.conversation_decision import IntimacyContext

    turn = prepared(replies=["mm"], conversation_history=history(gap_minutes=10))
    turn.decision = ConversationDecision(
        intimacy_context=IntimacyContext(active=True, content_register="explicit")
    )

    schedule = live_orchestration._delivery_schedule(turn, ["mm"])

    assert schedule.availability_mode is AvailabilityMode.INTIMATE


# --- the waiting is durable, not a coroutine -------------------------------


def test_the_worker_returns_its_slot_instead_of_pretending_to_type(world):
    turn = prepared(replies=["one|two|three"], conversation_history=history(1))

    result = run(live_orchestration.deliver_reply(turn, expected_revision="rev-1"))

    # Nothing was sent by the turn itself, and three durable obligations exist.
    assert world.sent == []
    assert len(world.queued) == 3
    assert {row["action_type"] for row in world.queued} == {
        outbound_delivery.DELIVER_PART_ACTION
    }
    assert result["message_ids"] == []


def test_each_bubble_carries_its_own_due_time(world):
    turn = prepared(replies=["one|two|three"], conversation_history=history(1))
    run(live_orchestration.deliver_reply(turn, expected_revision="rev-1"))

    due = [row["execute_at"] for row in world.queued]
    assert due == sorted(due)
    assert due[0] > datetime.now(timezone.utc)


def test_nothing_in_the_delivery_path_sleeps():
    """A deliberate pause must never be an awaited sleep on a worker slot."""
    source = "\n".join(
        line
        for line in inspect.getsource(outbound_delivery).splitlines()
        if not line.lstrip().startswith("#")
    )
    # Docstrings are prose about the design; only executable lines are asserted.
    import ast

    tree = ast.parse(inspect.getsource(outbound_delivery))
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "sleep":
            raise AssertionError("the delivery path must not sleep")
    assert "time.sleep" not in source


def test_the_durable_path_does_not_consult_the_in_process_registry():
    """Core v1 correctness may not depend on ``_pending_auto_replies``."""
    for module in (outbound_delivery, live_orchestration):
        source = inspect.getsource(module)
        assert "_pending_auto_replies" not in source
        assert "_sleep_while_current" not in source


def test_supersession_is_decided_by_the_database_not_by_a_task(world, monkeypatch):
    """No task is registered anywhere, and the bubble is still refused."""
    from services import suggestions

    monkeypatch.setattr(suggestions, "_pending_auto_replies", {})
    turn = prepared(replies=["one|two"], conversation_history=history(1))
    run(live_orchestration.deliver_reply(turn, expected_revision="rev-1"))
    sequence = run(store.get_sequence_by_trigger("fan-1", "message-1"))

    # A fan message handled by another process: only the number moves.
    world.db.tables["fans"][0]["conversation_generation"] = 5

    outcome = run(
        outbound_delivery.deliver_due_part(
            {
                "id": "a1",
                "creator_id": "creator-1",
                "fan_id": "fan-1",
                "action_type": outbound_delivery.DELIVER_PART_ACTION,
                "payload": {"sequence_id": sequence.id, "part_index": 0},
            }
        )
    )

    assert outcome.sent is False
    assert outcome.reason == "stale_generation"
    assert suggestions._pending_auto_replies == {}


# --- one shared dispatcher, never a timer per fan --------------------------


def test_the_dispatcher_sleeps_until_the_next_thing_is_due(monkeypatch):
    soon = datetime.now(timezone.utc) + timedelta(seconds=1.5)

    async def _next_due():
        return soon

    monkeypatch.setattr(worker, "next_due_at", _next_due)

    assert 1.0 < run(worker._idle_sleep_seconds()) < 2.0


def test_the_dispatcher_never_sleeps_longer_than_the_configured_poll(monkeypatch):
    async def _next_due():
        return datetime.now(timezone.utc) + timedelta(hours=3)

    monkeypatch.setattr(worker, "next_due_at", _next_due)

    assert run(worker._idle_sleep_seconds()) == worker.poll_seconds()


def test_an_empty_queue_falls_back_to_the_idle_poll(monkeypatch):
    async def _next_due():
        return None

    monkeypatch.setattr(worker, "next_due_at", _next_due)

    assert run(worker._idle_sleep_seconds()) == worker.poll_seconds()


def test_an_overdue_action_does_not_turn_the_loop_into_a_spin(monkeypatch):
    async def _next_due():
        return datetime.now(timezone.utc) - timedelta(minutes=5)

    monkeypatch.setattr(worker, "next_due_at", _next_due)

    assert run(worker._idle_sleep_seconds()) == worker.MIN_ADAPTIVE_SLEEP_SECONDS


def test_the_due_probe_is_one_query_regardless_of_how_many_fans(monkeypatch):
    """O(1), not O(number_of_fans). That is the whole point of a shared queue."""
    from db import commercial_queries

    fake = FakeSupabase(
        {
            "scheduled_actions": [
                {
                    "id": f"a{i}",
                    "status": "PENDING",
                    "execute_at": (
                        datetime.now(timezone.utc) + timedelta(seconds=i)
                    ).isoformat(),
                }
                for i in range(500)
            ]
        }
    )
    monkeypatch.setattr(commercial_queries, "get_supabase", lambda: fake)

    run(commercial_queries.next_due_at())

    assert len(fake.queries_for("scheduled_actions")) == 1


# --- the simulator: fast, and structurally identical -----------------------


def test_simulation_executes_the_plan_at_once(world):
    turn = prepared(replies=["one|two"], conversation_history=history(1))

    with immediate_delivery_scope():
        result = run(live_orchestration.deliver_reply(turn, expected_revision="rev-1"))

    assert result["outcome"] == live_orchestration.OUTCOME_REPLIED
    assert world.sent == ["one", "two"]
    # No durable part actions queued: nothing would drain them.
    assert world.queued == []


def test_simulation_still_reports_what_production_would_have_waited(world):
    turn = prepared(replies=["one|two"], conversation_history=history(1))

    with immediate_delivery_scope():
        result = run(live_orchestration.deliver_reply(turn, expected_revision="rev-1"))

    timing = result["planned_timing"]
    assert timing["delivery_mode"] == "immediate"
    assert timing["composition_delay_seconds"] > 0
    assert timing["availability_delay_seconds"] > 0


def test_simulation_skips_sleeping_and_not_correctness(world):
    turn = prepared(replies=["one|two"], conversation_history=history(1))

    with immediate_delivery_scope():
        run(live_orchestration.deliver_reply(turn, expected_revision="rev-1"))
        sequence = run(store.get_sequence_by_trigger("fan-1", "message-1"))

    assert sequence.conversation_generation == 4
    assert sequence.status == store.STATUS_COMPLETED
    assert [part.status for part in sequence.parts] == [
        store.PART_SENT,
        store.PART_SENT,
    ]


# --- other cores are untouched ---------------------------------------------


def test_semantic_v1_keeps_the_previous_inline_delivery(world, monkeypatch):
    from services.conversation_core import CORE_SEMANTIC_V1

    sent: list = []

    async def _inline(turn, *, expected_revision):
        sent.append(expected_revision)
        return ["m-1"]

    monkeypatch.setattr(live_orchestration, "_deliver_plain_parts", _inline)
    turn = prepared(
        replies=["one"], conversation_history=history(1), core=CORE_SEMANTIC_V1
    )

    result = run(live_orchestration.deliver_reply(turn, expected_revision="rev-1"))

    assert result["outcome"] == live_orchestration.OUTCOME_REPLIED
    assert result["message_ids"] == ["m-1"]
    assert sent == ["rev-1"]
