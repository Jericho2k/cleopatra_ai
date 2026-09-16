"""A Simulator turn is a durable record, not the lifetime of an HTTP request.

THE INCIDENT
------------
One simulated fan message produced:

* Kimi @ Inceptron -> 429, three times, over the old 5s/30s/60s schedule;
* the Qwen fallback, which succeeded;
* a creator reply persisted by the backend;
* and, on screen, ``Timeout: The simulated turn did not finish within 180s and
  was cancelled``.

The UI reported failure while the backend was still alive and went on to
persist a reply. An operator acting on that screen presses Send again, the
original turn wakes up, and the fan gets two contradictory creator replies
after commercial state has already moved.

These tests hold the whole corrected path closed, end to end: the writer gets
Kimi from another host without ever calling Qwen, the turn it belongs to
reaches ``completed``, and the operator sees that result even though the
generation lasted longer than any browser request would.

Nothing here waits. The writer's waits go through a fake clock, and the
durable turn's own scheduling seam is replaced (tests/conftest.py) so a turn is
deterministic rather than raced.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from ai import generator, writer_recovery
from ai.generator import CONTRACT_AUTO_MESSAGES, PERSISTENT_PRIMARY_RETRY_POLICY
from ai.openrouter_routing import PROVIDER_MODE_ALTERNATE
from models.model_runtime import ModelResult, ModelTarget, ModelUsage
from models.schemas import Persona
from services import simulation_turns
from services.simulation_turns import (
    OUTCOME_DEADLINE_EXCEEDED,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_PROCESSING,
    SimulationTurnBusy,
    get_turn,
    latest_turn,
    start_turn,
)

CREATOR = "creator-1"
FAN = "fan-test"

KIMI = ModelTarget(
    name="moonshotai/kimi-k2.6",
    provider="openrouter",
    model="moonshotai/kimi-k2.6",
    base_url="https://openrouter.example/v1",
    api_key_env="TEST_API_KEY",
    timeout_seconds=90.0,
    metadata={"openrouter_providers": ["Inceptron"]},
)
QWEN = ModelTarget(
    name="Qwen/Qwen3.7-Plus",
    provider="together",
    model="Qwen/Qwen3.7-Plus",
    base_url="https://together.example/v1",
    api_key_env="TEST_API_KEY",
    timeout_seconds=90.0,
)


class _RateLimited(Exception):
    status_code = 429

    def __init__(self) -> None:
        super().__init__("429 rate limited by Inceptron")


class _Unavailable(Exception):
    status_code = 503

    def __init__(self) -> None:
        super().__init__("503 upstream unavailable")


def _routing_mode(target: ModelTarget) -> str:
    return str((target.metadata or {}).get("openrouter_provider_mode") or "pinned")


@pytest.fixture
def writer(monkeypatch):
    """The real generator, with a fake provider and a fake clock."""
    state: dict = {"calls": [], "sleeps": [], "responses": []}

    async def fake_sleep(seconds):
        state["sleeps"].append(seconds)

    async def fake_complete(target, **_kwargs):
        state["calls"].append((target.model, _routing_mode(target)))
        index = len(state["calls"]) - 1
        behaviour = state["responses"][min(index, len(state["responses"]) - 1)]
        if isinstance(behaviour, Exception):
            raise behaviour
        return ModelResult(
            text=behaviour,
            target=target,
            usage=ModelUsage(input_tokens=10, output_tokens=5),
            latency_ms=1,
            upstream_provider=(
                "Parasail" if _routing_mode(target) == PROVIDER_MODE_ALTERNATE else "Inceptron"
            ),
        )

    async def noop(*_args, **_kwargs):
        return None

    monkeypatch.setattr(generator, "_sleep", fake_sleep)
    monkeypatch.setattr(generator, "complete", fake_complete)
    monkeypatch.setattr(generator, "record_model_result", noop)
    monkeypatch.setattr(generator, "record_model_failure", noop)
    monkeypatch.setattr(generator, "record_writer_recovery_outcome", noop)
    return state


@pytest.fixture
def transcript(monkeypatch):
    """The persisted creator messages, as the turn machinery reads them."""
    rows: list[dict] = []

    async def fake_rows(_fan_id):
        return [dict(row) for row in rows]

    monkeypatch.setattr(
        "services.suggestions._recent_creator_message_rows", fake_rows
    )
    return rows


def _messages(*bubbles: str) -> str:
    return json.dumps({"messages": list(bubbles)})


def _write(writer_state, transcript_rows):
    """A runner that generates through the REAL writer, then persists.

    Not a stub of the pipeline's decisions — the point is that the turn's
    durable state is driven by what the writer actually did, waits and provider
    failover included.
    """

    async def runner(*, fan_id, creator_id, message, fast, include_mirrored_catalog):
        replies = await generator.generate_replies(
            [
                {"role": "system", "content": "system"},
                {"role": "user", "content": message},
            ],
            Persona(),
            telemetry_context={"feature": "auto_reply"},
            target_override=KIMI,
            fallback_target_override=QWEN,
            output_contract=CONTRACT_AUTO_MESSAGES,
            retry_policy=PERSISTENT_PRIMARY_RETRY_POLICY,
            profile_id="cleo_v3",
        )
        if not replies:
            return {
                "status": "ok",
                "simulation": True,
                "fan_message_id": "fan-message-1",
                "creator_messages": [],
                "outcome": "writer_failed",
            }
        produced = []
        for index, reply in enumerate(replies):
            row = {
                "id": f"creator-message-{len(transcript_rows) + index + 1}",
                "role": "creator",
                "content": reply,
                "sent_at": None,
                "media_context": None,
            }
            transcript_rows.append(row)
            produced.append(row)
        return {
            "status": "ok",
            "simulation": True,
            "fan_message_id": "fan-message-1",
            "creator_messages": produced,
            "outcome": "replied",
        }

    return runner


def _models(writer_state) -> list[str]:
    return [model for model, _mode in writer_state["calls"]]


def _modes(writer_state) -> list[str]:
    return [mode for _model, mode in writer_state["calls"]]


# --- THE INCIDENT, end to end ----------------------------------------------


def test_the_incident_now_completes_on_kimi_without_ever_reaching_qwen(
    writer, transcript
):
    """Inceptron 429 x3 -> another Kimi host -> one reply -> completed turn.

    The whole chain in one test, because the two halves failed together: the
    writer went to the wrong model, and the UI was told the turn had failed
    while it was happening.
    """
    writer["responses"] = [
        _RateLimited(),
        _RateLimited(),
        _RateLimited(),
        _messages("hey you", "what are you up to"),
    ]

    turn, created = asyncio.run(
        start_turn(
            creator_id=CREATOR,
            fan_id=FAN,
            message="hii",
            idempotency_key="send-1",
            runner=_write(writer, transcript),
        )
    )

    assert created is True

    # The writer: three cache-affine attempts, then the SAME model elsewhere.
    assert _models(writer) == [KIMI.model] * 4
    assert _modes(writer) == ["pinned"] * 3 + [PROVIDER_MODE_ALTERNATE]
    assert QWEN.model not in _models(writer)
    # 35 seconds of waiting, not 95, and none of it before changing host.
    assert writer["sleeps"] == [5.0, 30.0]

    # Exactly one creator reply, generated once and persisted once.
    assert len(transcript) == 1
    assert transcript[0]["content"] == "hey you | what are you up to"

    # And the turn the operator is watching says so.
    settled = asyncio.run(get_turn(CREATOR, FAN, turn.id))
    assert settled.status == STATUS_COMPLETED
    assert settled.outcome == "replied"
    assert settled.creator_message_ids == ["creator-message-1"]
    assert settled.public_view()["status"] == "completed"


def test_the_operator_sees_a_processing_turn_rather_than_a_timeout(writer, transcript):
    """What the browser is told while a long recovery is in progress.

    The old answer was "Timeout ... and was cancelled", which was false in both
    halves: nothing was cancelled and the turn had not failed. The new answer
    is a status, and a product-level sentence that names no provider.
    """
    assert simulation_turns.progress_message(2) == "Generating reply…"
    assert simulation_turns.progress_message(60) == (
        "Still generating — the primary writer is temporarily busy."
    )
    for elapsed in (0, 10, 44, 45, 120, 600):
        message = simulation_turns.progress_message(elapsed)
        lowered = message.lower()
        for secret in ("inceptron", "openrouter", "kimi", "qwen", "provider", "429"):
            assert secret not in lowered, message


# --- Qwen is the last resort, and it still works ----------------------------


def test_qwen_answers_only_after_every_kimi_host_has_failed(writer, transcript):
    writer["responses"] = [
        _RateLimited(),
        _RateLimited(),
        _RateLimited(),
        _Unavailable(),
        _Unavailable(),
        _messages("qwen wrote this"),
    ]

    turn, _ = asyncio.run(
        start_turn(
            creator_id=CREATOR,
            fan_id=FAN,
            message="hii",
            idempotency_key="send-1",
            runner=_write(writer, transcript),
        )
    )

    assert _models(writer) == [KIMI.model] * 5 + [QWEN.model]
    assert _modes(writer)[3:5] == [PROVIDER_MODE_ALTERNATE] * 2
    assert len(transcript) == 1

    settled = asyncio.run(get_turn(CREATOR, FAN, turn.id))
    assert settled.status == STATUS_COMPLETED
    assert settled.outcome == "replied"
    # No false timeout anywhere in the record.
    assert settled.error is None


# --- the backend deadline, and the ghost it prevents ------------------------


def test_an_expired_turn_is_terminally_failed_and_persists_no_late_reply(
    writer, transcript, monkeypatch
):
    """Kimi gone, Qwen hanging, the deadline fires.

    The pipeline is CANCELLED rather than abandoned, so the reply it was about
    to write is never written at all — cancellation lands on the same await
    points the persistence would have gone through. The turn is failed, and
    the transcript stays empty, which is the whole of "no ghost reply".
    """
    monkeypatch.setattr(simulation_turns, "turn_deadline_seconds", lambda: 0.05)

    async def hangs(*, fan_id, creator_id, message, fast, include_mirrored_catalog):
        await asyncio.sleep(30)
        # Everything past the hang is what the old code would eventually have
        # done while the UI had already reported failure.
        transcript.append(
            {
                "id": "ghost-message",
                "role": "creator",
                "content": "the reply the operator was told would never come",
                "sent_at": None,
                "media_context": None,
            }
        )
        return {
            "status": "ok",
            "simulation": True,
            "fan_message_id": "fan-message-1",
            "creator_messages": [{"id": "ghost-message"}],
            "outcome": "replied",
        }

    turn, _ = asyncio.run(
        start_turn(
            creator_id=CREATOR,
            fan_id=FAN,
            message="hii",
            idempotency_key="send-1",
            runner=hangs,
        )
    )

    settled = asyncio.run(get_turn(CREATOR, FAN, turn.id))
    assert settled.status == STATUS_FAILED
    assert settled.outcome == OUTCOME_DEADLINE_EXCEEDED
    assert settled.creator_message_ids == []
    assert "deadline" in (settled.error or "")
    assert transcript == [], "the abandoned task persisted nothing"


def test_a_terminal_turn_is_never_rewritten_by_a_straggler(writer, transcript):
    """Terminal is terminal, enforced by the write and not by hoping.

    Whichever answer is recorded first is the one the operator keeps. A late
    completion cannot overwrite a failure, and a late failure cannot overwrite
    a completion — both are conditional on the turn still being active.
    """
    writer["responses"] = [_messages("hey you")]

    turn, _ = asyncio.run(
        start_turn(
            creator_id=CREATOR,
            fan_id=FAN,
            message="hii",
            idempotency_key="send-1",
            runner=_write(writer, transcript),
        )
    )
    assert asyncio.run(get_turn(CREATOR, FAN, turn.id)).status == STATUS_COMPLETED

    # A deadline that fires after the turn already completed changes nothing.
    asyncio.run(
        simulation_turns._record_failure(
            turn.id, outcome=OUTCOME_DEADLINE_EXCEEDED, error="expired"
        )
    )
    unchanged = asyncio.run(get_turn(CREATOR, FAN, turn.id))
    assert unchanged.status == STATUS_COMPLETED
    assert unchanged.outcome == "replied"

    applied = asyncio.run(
        simulation_turns._update_if_active(
            turn.id, {"status": STATUS_FAILED, "outcome": "late"}
        )
    )
    assert applied is False


def test_a_failed_turn_can_never_become_a_hidden_success(monkeypatch, transcript):
    """The direction that actually hurt: failure reported, success recorded.

    A turn that expired is failed. If the abandoned work were somehow still
    able to report a reply, the conditional update refuses it — so the screen
    that said "failed" cannot be contradicted afterwards.
    """
    monkeypatch.setattr(simulation_turns, "turn_deadline_seconds", lambda: 0.05)

    async def hangs(**_kwargs):
        await asyncio.sleep(30)

    turn, _ = asyncio.run(
        start_turn(
            creator_id=CREATOR,
            fan_id=FAN,
            message="hii",
            idempotency_key="send-1",
            runner=hangs,
        )
    )
    assert asyncio.run(get_turn(CREATOR, FAN, turn.id)).status == STATUS_FAILED

    late = asyncio.run(
        simulation_turns._update_if_active(
            turn.id,
            {
                "status": STATUS_COMPLETED,
                "outcome": "replied",
                "creator_message_ids": ["ghost-message"],
            },
        )
    )

    assert late is False
    final = asyncio.run(get_turn(CREATOR, FAN, turn.id))
    assert final.status == STATUS_FAILED
    assert final.creator_message_ids == []


def test_a_reply_persisted_just_before_the_deadline_is_reported_as_completed(
    transcript, monkeypatch
):
    """Cancellation is cooperative, so this window is real and is reconciled.

    The turn is decided ONCE, against what actually reached the transcript, so
    the operator is never told a turn failed while a reply from it sits in the
    conversation.
    """
    monkeypatch.setattr(simulation_turns, "turn_deadline_seconds", lambda: 0.05)

    async def writes_then_hangs(
        *, fan_id, creator_id, message, fast, include_mirrored_catalog
    ):
        transcript.append(
            {
                "id": "creator-message-1",
                "role": "creator",
                "content": "made it just in time",
                "sent_at": None,
                "media_context": None,
            }
        )
        await asyncio.sleep(30)
        return {}

    turn, _ = asyncio.run(
        start_turn(
            creator_id=CREATOR,
            fan_id=FAN,
            message="hii",
            idempotency_key="send-1",
            runner=writes_then_hangs,
        )
    )

    settled = asyncio.run(get_turn(CREATOR, FAN, turn.id))
    assert settled.status == STATUS_COMPLETED
    assert settled.outcome == "replied"
    assert settled.creator_message_ids == ["creator-message-1"]


def test_the_turn_deadline_outlives_the_whole_writer_ladder():
    """A ceiling derived from the writer's own configuration, not a UI number.

    It has to comfortably contain the pinned attempts, their waits, the
    alternate-provider attempts and the emergency fallback — otherwise the
    deadline, not the ladder, would decide how hard a turn tries.
    """
    ladder = writer_recovery.writer_turn_deadline_seconds()

    assert simulation_turns.turn_deadline_seconds() > ladder
    assert (
        simulation_turns.turn_deadline_seconds()
        == ladder + simulation_turns.NON_WRITER_BUDGET_SECONDS
    )
    # Nothing here is derived from the browser's old 180-second ceiling.
    assert ladder > 180


# --- one submission is one turn --------------------------------------------


def test_a_duplicate_submission_returns_the_same_turn_and_generates_once(
    writer, transcript
):
    """A double-clicked Send, or a POST retried after a dropped connection."""
    writer["responses"] = [_messages("hey you")]
    runner = _write(writer, transcript)

    first, created_first = asyncio.run(
        start_turn(
            creator_id=CREATOR,
            fan_id=FAN,
            message="hii",
            idempotency_key="send-1",
            runner=runner,
        )
    )
    second, created_second = asyncio.run(
        start_turn(
            creator_id=CREATOR,
            fan_id=FAN,
            message="hii",
            idempotency_key="send-1",
            runner=runner,
        )
    )

    assert created_first is True
    assert created_second is False
    assert second.id == first.id
    assert len(writer["calls"]) == 1, "one submission, one generation"
    assert len(transcript) == 1


def test_a_second_concurrent_turn_for_one_fan_is_refused(monkeypatch):
    """Simple serialisation per simulated fan, rather than interleaving two
    pipelines over one commercial state."""

    async def never_finishes(**_kwargs):
        await asyncio.Event().wait()

    async def leave_it_running(coro, _name):
        # The production seam: schedule and return. Closed at the end of the
        # test so nothing leaks into the next one.
        monkeypatch.setattr(
            simulation_turns, "_pending_test_task", asyncio.ensure_future(coro),
            raising=False,
        )

    async def scenario():
        monkeypatch.setattr(
            simulation_turns, "schedule_turn_execution", leave_it_running
        )
        first, _ = await start_turn(
            creator_id=CREATOR,
            fan_id=FAN,
            message="one",
            idempotency_key="send-1",
            runner=never_finishes,
        )
        await asyncio.sleep(0)
        with pytest.raises(SimulationTurnBusy) as refused:
            await start_turn(
                creator_id=CREATOR,
                fan_id=FAN,
                message="two",
                idempotency_key="send-2",
                runner=never_finishes,
            )
        assert refused.value.active.id == first.id

        task = getattr(simulation_turns, "_pending_test_task", None)
        if task is not None:
            task.cancel()

    asyncio.run(scenario())


def test_a_different_key_starts_a_new_turn_once_the_first_is_terminal(
    writer, transcript
):
    writer["responses"] = [_messages("first"), _messages("second")]
    runner = _write(writer, transcript)

    first, _ = asyncio.run(
        start_turn(
            creator_id=CREATOR, fan_id=FAN, message="one",
            idempotency_key="send-1", runner=runner,
        )
    )
    second, created = asyncio.run(
        start_turn(
            creator_id=CREATOR, fan_id=FAN, message="two",
            idempotency_key="send-2", runner=runner,
        )
    )

    assert created is True
    assert second.id != first.id
    assert len(transcript) == 2


# --- polling and resuming ---------------------------------------------------


def test_polling_a_turn_never_runs_the_pipeline_again(writer, transcript):
    writer["responses"] = [_messages("hey you")]

    turn, _ = asyncio.run(
        start_turn(
            creator_id=CREATOR, fan_id=FAN, message="hii",
            idempotency_key="send-1", runner=_write(writer, transcript),
        )
    )
    calls_after_generation = len(writer["calls"])

    for _ in range(20):
        polled = asyncio.run(get_turn(CREATOR, FAN, turn.id))
        assert polled.status == STATUS_COMPLETED

    assert len(writer["calls"]) == calls_after_generation
    assert len(transcript) == 1


def test_a_reloaded_browser_recovers_the_turn_it_was_watching(writer, transcript):
    """The resume read: the newest turn for this conversation, by fan alone.

    A browser that reloaded knows which fan it is looking at and nothing else.
    """
    writer["responses"] = [_messages("hey you")]

    turn, _ = asyncio.run(
        start_turn(
            creator_id=CREATOR, fan_id=FAN, message="hii",
            idempotency_key="send-1", runner=_write(writer, transcript),
        )
    )

    recovered = asyncio.run(latest_turn(CREATOR, FAN))
    assert recovered is not None
    assert recovered.id == turn.id
    assert recovered.status == STATUS_COMPLETED
    assert recovered.creator_message_ids == ["creator-message-1"]


def test_a_turn_in_flight_reports_processing_to_a_reconnecting_browser():
    async def scenario():
        turn, _ = await start_turn(
            creator_id=CREATOR, fan_id=FAN, message="hii",
            idempotency_key="send-1",
            runner=_never(),
        )
        return turn

    async def _hold(coro, _name):
        coro.close()

    async def run():
        simulation_turns.schedule_turn_execution = _hold
        try:
            turn = await scenario()
            recovered = await latest_turn(CREATOR, FAN)
            assert recovered.id == turn.id
            assert recovered.public_view()["status"] == STATUS_PROCESSING
            assert recovered.terminal is False
        finally:
            pass

    original = simulation_turns.schedule_turn_execution
    try:
        asyncio.run(run())
    finally:
        simulation_turns.schedule_turn_execution = original


def _never():
    async def runner(**_kwargs):
        await asyncio.Event().wait()

    return runner


# --- what an agency is told -------------------------------------------------


def test_an_agency_is_never_shown_the_backend_failure_detail(writer, transcript):
    """Operator diagnostics stay operator diagnostics.

    A backend exception can name a provider, a model or an internal path — the
    three things the AI-stack boundary exists to keep away from a tenant.
    """
    turn = simulation_turns.SimulationTurn(
        id="turn-1",
        creator_id=CREATOR,
        fan_id=FAN,
        idempotency_key="send-1",
        status=STATUS_FAILED,
        outcome=OUTCOME_DEADLINE_EXCEEDED,
        error="openrouter moonshotai/kimi-k2.6 via Inceptron returned 429",
        error_id="abc123",
        deadline_seconds=758.0,
    )

    agency = turn.public_view()
    assert "error" not in agency
    assert agency["error_id"] == "abc123", "a correlating id is safe and useful"
    assert "deadline_seconds" not in agency
    assert "kimi" not in json.dumps(agency).lower()
    assert "inceptron" not in json.dumps(agency).lower()

    owner = turn.public_view(diagnostics=True)
    assert "Inceptron" in owner["error"]
    assert owner["deadline_seconds"] == 758.0
    assert owner["stage"] == STATUS_FAILED
