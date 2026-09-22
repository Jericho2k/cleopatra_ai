"""A promise the conversation made, kept durably and written when it comes due.

Cleopatra could already keep several SPECIFIC promises. What it could not keep
was "wait right there" — so the beat was dropped the moment the reply was sent.

The two properties that make a generic version safe are the ones these tests
are about:

  * nothing fan-facing is frozen in advance, so a due intention is written
    against the conversation as it IS rather than as it was;
  * application code owns the clock, so a model can ask for a delay but can
    never state a time or reuse one it invented.
"""

from __future__ import annotations

import asyncio
import json
import random
from datetime import datetime, timedelta, timezone

import pytest

from models.conversation_decision import ScheduledIntent
from services import scheduled_intent
from services.conversational_decision_contract import parse_semantic_decision
from workers import scheduled_actions as worker


def run(coro):
    return asyncio.run(coro)


NOW = datetime(2026, 9, 22, 12, 0, tzinfo=timezone.utc)


def decision_json(**overrides):
    payload = {
        "disposition": "reply",
        "response_goal": "keep the moment alive",
        "operation_proposal": {"kind": "none"},
    }
    payload.update(overrides)
    return json.dumps(payload)


# --- the contract -----------------------------------------------------------


def test_a_short_continuation_is_read_with_its_default_policy():
    result = parse_semantic_decision(
        decision_json(
            scheduled_intent={
                "kind": "short_continuation",
                "goal": "come back with the next beat if he has not answered",
                "timing": {"relative_minutes": 2},
                "source_ids": ["message-9"],
            }
        )
    )

    intent = result.decision.scheduled_intent
    assert intent.requested is True
    assert intent.kind == "short_continuation"
    assert intent.timing_kind == "relative"
    assert intent.relative_seconds == 120
    assert intent.activity_policy == "cancel_on_activity"
    assert intent.source_ids == ("message-9",)


def test_a_payday_obligation_points_at_evidence_rather_than_a_time():
    result = parse_semantic_decision(
        decision_json(
            scheduled_intent={
                "kind": "payday_followup",
                "goal": "reopen the door he left open",
                "timing": {"reference": "payday"},
            }
        )
    )

    intent = result.decision.scheduled_intent
    assert intent.timing_kind == "reference"
    assert intent.reference == "payday"
    # Not a short continuation, so his talking does not silently delete it.
    assert intent.activity_policy == "revalidate_on_activity"


def test_an_intention_may_never_carry_the_words_to_say():
    result = parse_semantic_decision(
        decision_json(
            scheduled_intent={
                "kind": "short_continuation",
                "goal": "continue",
                "timing": {"relative_minutes": 2},
                "message": "hey, still thinking about you 😏",
            }
        )
    )

    assert result.decision.scheduled_intent.requested is False
    assert "fan-facing wording" in result.degradations["scheduled_intent"]


def test_an_intention_may_never_name_a_price():
    result = parse_semantic_decision(
        decision_json(
            scheduled_intent={
                "kind": "commercial_callback",
                "goal": "come back to the $30 set",
                "timing": {"relative_minutes": 30},
            }
        )
    )

    assert result.decision.scheduled_intent.requested is False
    assert result.degradations["scheduled_intent.goal"] == "named a price; dropped"


def test_an_absurd_delay_is_clamped_rather_than_obeyed():
    result = parse_semantic_decision(
        decision_json(
            scheduled_intent={
                "kind": "check_back",
                "goal": "see how he is",
                "timing": {"relative_minutes": 60 * 24 * 30},
            }
        )
    )

    intent = result.decision.scheduled_intent
    assert intent.relative_seconds == scheduled_intent_max()
    assert "clamped" in result.degradations["scheduled_intent.timing"]


def test_an_invented_clock_time_is_refused():
    result = parse_semantic_decision(
        decision_json(
            scheduled_intent={
                "kind": "check_back",
                "goal": "see how he is",
                "timing": {"at": "2026-09-23T18:00:00Z"},
            }
        )
    )

    assert result.decision.scheduled_intent.requested is False


def test_an_unknown_kind_is_refused():
    result = parse_semantic_decision(
        decision_json(
            scheduled_intent={
                "kind": "escalate_the_funnel",
                "goal": "push harder",
                "timing": {"relative_minutes": 5},
            }
        )
    )

    assert result.decision.scheduled_intent.requested is False


def scheduled_intent_max() -> int:
    from services.conversational_decision_contract import MAX_INTENT_DELAY_SECONDS

    return MAX_INTENT_DELAY_SECONDS


# --- application-owned normalization ---------------------------------------


def test_a_relative_delay_lands_inside_its_own_jitter_band():
    intent = ScheduledIntent(
        kind="short_continuation",
        goal="continue",
        timing_kind="relative",
        relative_seconds=120,
    )
    execute_at = scheduled_intent.resolve_execute_at(
        intent, now=NOW, rng=random.Random(1)
    )

    delta = (execute_at - NOW).total_seconds()
    assert 120 * 0.8 <= delta <= 120 * 1.2


def test_a_payday_reference_resolves_against_evidence_the_app_parsed():
    payday = NOW + timedelta(days=3)
    intent = ScheduledIntent(
        kind="payday_followup",
        goal="reopen it",
        timing_kind="reference",
        reference="payday",
    )

    assert scheduled_intent.resolve_execute_at(
        intent, now=NOW, payday_at=payday
    ) == payday


def test_a_payday_reference_without_evidence_is_refused():
    intent = ScheduledIntent(
        kind="payday_followup",
        goal="reopen it",
        timing_kind="reference",
        reference="payday",
    )
    with pytest.raises(scheduled_intent.IntentRejected):
        scheduled_intent.resolve_execute_at(intent, now=NOW, payday_at=None)


def test_the_persisted_payload_contains_no_copy(monkeypatch):
    recorded: dict = {}

    async def _schedule(**kwargs):
        recorded.update(kwargs)

    monkeypatch.setattr(scheduled_intent, "schedule_action", _schedule)

    async def _sleep_hours(creator_id, execute_at, *, now):
        return execute_at

    monkeypatch.setattr(scheduled_intent, "_apply_sleep_hours", _sleep_hours)

    result = run(
        scheduled_intent.persist_scheduled_intent(
            creator_id="creator-1",
            fan_id="fan-1",
            intent=ScheduledIntent(
                kind="short_continuation",
                goal="pick the tease back up if he has gone quiet",
                timing_kind="relative",
                relative_seconds=90,
                source_ids=("message-3",),
            ),
            conversation_generation=12,
            now=NOW,
            rng=random.Random(3),
        )
    )

    assert result is not None
    assert recorded["action_type"] == "CONVERSATIONAL_INTENT"
    assert recorded["dedupe_key"] == "conv-intent:fan-1:short_continuation"
    payload = recorded["payload"]
    assert set(payload) == set(scheduled_intent.PAYLOAD_KEYS)
    assert payload["created_generation"] == 12
    # Nothing here is a sentence anybody will read.
    for forbidden in ("message", "messages", "reply", "text", "copy", "body"):
        assert forbidden not in payload


def test_one_live_obligation_per_kind_per_fan():
    assert scheduled_intent.dedupe_key("fan-1", "payday_followup") == (
        "conv-intent:fan-1:payday_followup"
    )


def test_the_due_goal_is_an_instruction_to_re_decide_not_a_message():
    goal = scheduled_intent.goal_for_due_intent(
        {"kind": "short_continuation", "goal": "pick the tease back up"}
    )

    assert "pick the tease back up" in goal
    assert "Read the CURRENT conversation first" in goal
    assert "say nothing and hold" in goal


# --- what happens when it comes due ----------------------------------------


def intent_action(policy: str, generation: int = 5) -> dict:
    return {
        "id": "action-1",
        "creator_id": "creator-1",
        "fan_id": "fan-1",
        "action_type": "CONVERSATIONAL_INTENT",
        "payload": {
            "kind": "short_continuation",
            "goal": "pick the tease back up",
            "activity_policy": policy,
            "created_generation": generation,
        },
    }


@pytest.fixture
def fan(monkeypatch):
    state = {"generation": 5}

    async def _get_fan(_fan_id):
        from types import SimpleNamespace

        return SimpleNamespace(
            id="fan-1", needs_human_review=False, auto_mode=True
        )

    async def _generation(_fan_id):
        return state["generation"]

    monkeypatch.setattr("db.queries.get_fan_by_id", _get_fan)
    monkeypatch.setattr(
        "services.conversation_generation.current_generation", _generation
    )
    return state


def test_cancel_on_activity_is_dropped_once_the_fan_speaks(fan):
    fan["generation"] = 6

    check = run(worker._should_still_send(intent_action("cancel_on_activity")))

    assert check.ok is False
    assert "superseded" in check.reason


def test_cancel_on_activity_survives_while_he_stays_quiet(fan):
    check = run(worker._should_still_send(intent_action("cancel_on_activity")))

    assert check.ok is True


def test_revalidate_on_activity_is_not_deleted_by_him_speaking(fan):
    """A payday obligation is not cancelled by chatter; it is re-decided."""
    fan["generation"] = 9

    check = run(worker._should_still_send(intent_action("revalidate_on_activity")))

    assert check.ok is True


def test_a_review_hold_still_stops_a_due_intention(fan, monkeypatch):
    from types import SimpleNamespace

    async def _frozen(_fan_id):
        return SimpleNamespace(id="fan-1", needs_human_review=True, auto_mode=True)

    monkeypatch.setattr("db.queries.get_fan_by_id", _frozen)

    check = run(worker._should_still_send(intent_action("revalidate_on_activity")))

    assert check.ok is False


def test_auto_mode_off_still_stops_a_due_intention(fan, monkeypatch):
    from types import SimpleNamespace

    async def _off(_fan_id):
        return SimpleNamespace(id="fan-1", needs_human_review=False, auto_mode=False)

    monkeypatch.setattr("db.queries.get_fan_by_id", _off)

    check = run(worker._should_still_send(intent_action("revalidate_on_activity")))

    assert check.ok is False


def test_a_due_intention_re_enters_the_semantic_path(monkeypatch):
    """It must go through GLM and Kimi, not read a stored message."""
    seen: dict = {}

    async def _send(*, creator_id, fan_id, goal, action_id, action_payload):
        seen.update(
            {"creator_id": creator_id, "fan_id": fan_id, "goal": goal}
        )
        return True

    monkeypatch.setattr("services.proactive.send_proactive_message", _send)

    result = run(worker._run_conversational_intent(intent_action("cancel_on_activity")))

    assert result.sent_message is True
    assert "pick the tease back up" in seen["goal"]
    assert "Read the CURRENT conversation first" in seen["goal"]


def test_an_intention_that_no_longer_makes_sense_resolves_silently(monkeypatch):
    async def _send(**_kwargs):
        return False

    monkeypatch.setattr("services.proactive.send_proactive_message", _send)

    result = run(worker._run_conversational_intent(intent_action("cancel_on_activity")))

    assert result.sent_message is False
    assert "resolved without sending" in result.reason


def test_the_intent_handler_is_registered():
    assert worker.HANDLERS["CONVERSATIONAL_INTENT"] is worker._run_conversational_intent


# --- the specialised follow-ups are integrated with, not replaced -----------


def payday_action(payday_at: datetime) -> dict:
    return {
        "id": "action-2",
        "creator_id": "creator-1",
        "fan_id": "fan-1",
        "action_type": "PAYDAY_REENGAGEMENT",
        "dedupe_key": "payday:fan-1",
        "payload": {
            "payday_at": payday_at.isoformat(),
            "desired_experience": "the set he picked",
        },
    }


@pytest.fixture
def payday_world(monkeypatch):
    from types import SimpleNamespace

    from models.commercial import CreatorPolicy, FanCommercialState, FanStatus

    payday = datetime.now(timezone.utc) - timedelta(minutes=5)
    state = FanCommercialState(
        status=FanStatus.PAUSED_UNTIL_PAYDAY,
        payday_at=payday,
        next_followup_type="PAYDAY_REENGAGEMENT",
        next_followup_dedupe_key="payday:fan-1",
    )
    holder = {"state": state, "payday": payday}

    async def _fan(_fan_id):
        return SimpleNamespace(id="fan-1", needs_human_review=False, auto_mode=True)

    async def _fan_state(_fan_id):
        return holder["state"]

    async def _policy(_creator_id):
        return CreatorPolicy()

    async def _history(_fan_id, limit=10):
        return []

    async def _sleep_hours(_creator_id):
        return 0, 0

    monkeypatch.setattr("db.queries.get_fan_by_id", _fan)
    monkeypatch.setattr("db.queries.get_conversation_history", _history)
    monkeypatch.setattr("db.queries.get_creator_sleep_hours", _sleep_hours)
    monkeypatch.setattr(worker, "get_fan_state", _fan_state)
    monkeypatch.setattr(worker, "get_creator_policy", _policy)
    return holder


def test_payday_reengagement_still_fires_when_he_is_still_waiting(payday_world):
    check = run(worker._should_still_send(payday_action(payday_world["payday"])))

    assert check.ok is True


def test_payday_reengagement_does_not_chase_a_fan_who_already_bought(payday_world):
    """Case 10: revalidation, not a stored message, is what makes this safe."""
    from models.commercial import FanStatus

    payday_world["state"].status = FanStatus.PAID_SESSION_ACTIVE

    check = run(worker._should_still_send(payday_action(payday_world["payday"])))

    assert check.ok is False
    assert "no longer paused" in check.reason


def test_a_newer_payday_replaces_the_one_this_action_was_built_for(payday_world):
    check = run(
        worker._should_still_send(
            payday_action(datetime.now(timezone.utc) - timedelta(days=7))
        )
    )

    assert check.ok is False
    assert "newer payday" in check.reason
