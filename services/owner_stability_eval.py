"""Measure how often the Conversational Core v1 owner path survives a turn.

WHAT THIS ANSWERS
-----------------
Not "is the conversation good" — ``services/ab_trajectory_eval.py`` and the
blind review own that. This answers the question that has to be settled first:

    Can ordinary messages go through ``conversational_v1`` repeatedly without a
    different orchestration bug every second turn?

One happy-path message proves nothing about that. Response-format failures on a
reasoning model are intermittent by nature: the same prompt returns clean JSON,
then prose, then an empty completion whose budget went into hidden reasoning.
So this runs many turns and reports rates, not anecdotes.

HOW IT AVOIDS MEASURING ITSELF
------------------------------
Every turn goes through the real functions — ``decide_conversational_v1`` for
the owner boundary and ``settle_conversational_v1_turn`` for deterministic
authority — with only the transport replaced. A harness that reimplemented the
failure model would report on the reimplementation, which is the failure mode
of most reliability dashboards.

TWO ARMS
--------
*Synthetic* (the default, and the one CI can run): a seeded generator produces
the response shapes production has actually produced — valid objects, JSON
truncated mid-``state_delta``, prose with no object, empty content after a long
reasoning trace, fenced JSON, out-of-range confidence, unknown enum values,
operations citing references that do not exist. It is deterministic, needs no
network, and its point is the *pipeline's* behaviour under each shape.

*Live*: pass a real ``complete`` and real evidence, and the same accounting
applies to the real endpoint. That arm costs money and is never run by CI.

WHAT IT REPORTS
---------------
owner-failed rate, malformed-output rate, repair-attempt and repair-success
rates, no-send rate, operation-proposal and operation-rejection rates,
state-delta rejection rate, a histogram of failure categories, and p50/p95
latency. Rates are arithmetic over recorded turns; nothing here decides whether
a number is acceptable.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, Sequence

from models.conversational_core import ConversationalWorkingState
from models.live_orchestration import EvidenceFact, EvidenceSnapshot, TurnTrigger
from models.model_runtime import ModelResponseDiagnostics, ModelTarget, ModelUsage
from services.conversational_core import validate_and_apply_delta
from services.live_orchestration import (
    decide_conversational_v1,
    settle_conversational_v1_turn,
)
from models.conversation_decision import ResponseDisposition

#: Bumped when the report shape changes.
REPORT_VERSION = "owner_stability_v1"

OUTCOME_REPLIED = "replied"
OUTCOME_NO_SEND = "no_send"
OUTCOME_HANDOFF = "handoff"
OUTCOME_OWNER_FAILED = "owner_failed"


# ---------------------------------------------------------------------------
# Synthetic evidence
# ---------------------------------------------------------------------------


def synthetic_snapshot(turn_index: int, message: str) -> EvidenceSnapshot:
    """One turn of evidence with no real fan or creator in it."""

    return EvidenceSnapshot(
        creator_id="stability-creator",
        fan_id="stability-fan",
        trigger=TurnTrigger(
            kind="fan_message",
            identity=f"msg-{turn_index}",
            latest_message=message,
        ),
        state_revision=f"revision-{turn_index}",
        creator_facts=(
            EvidenceFact(
                value="favourite season: late autumn",
                source_ref="creator_legend:favourite_season",
                certainty="creator_confirmed",
            ),
        ),
        historical_facts=(
            EvidenceFact(
                value="possible interest: old films",
                source_ref="memory:0",
                certainty="uncertain",
            ),
        ),
    )


def synthetic_loaded(
    snapshot: EvidenceSnapshot,
    *,
    target: ModelTarget,
    max_tokens: int = 8192,
) -> Any:
    """A ``LoadedEvidence``-shaped stand-in with no offer and no pending payment.

    Deliberately barren on the commercial side: every operation the generator
    proposes is therefore refused by the real validator, which is the case this
    harness most needs to exercise — a refused operation must not cost the
    reply, and refusing one must never reach the database.
    """

    spec = SimpleNamespace(
        primary_target=lambda: target,
        fallback_target=lambda: None,
        max_tokens=max_tokens,
        resolved_max_tokens=lambda: max_tokens,
    )
    return SimpleNamespace(
        snapshot=snapshot,
        packet=None,
        history=[],
        fan=SimpleNamespace(
            id="stability-fan",
            needs_human_review=False,
            auto_mode=True,
        ),
        persona=None,
        commercial_state=SimpleNamespace(
            pending_offer=None,
            last_offer_at=None,
            desired_experience=None,
        ),
        policy=SimpleNamespace(
            require_operator_ppv_approval=False,
            pending_offer_expiry_hours=24,
        ),
        next_offer=None,
        active_session=None,
        pending_payment=None,
        sent_ppv=[],
        within_daily_caps=True,
        stack=SimpleNamespace(
            profile_id="owner_stability",
            profile=SimpleNamespace(stage=lambda _name: spec),
        ),
    )


FAN_MESSAGES: tuple[str, ...] = (
    "just subscribed, hi",
    "well 2 much to list it all",
    "what are you up to today",
    "haha that's fair",
    "i liked the last thing you said",
    "tell me something i wouldn't guess",
    "mm",
    "that reminds me of something",
    "i'm not sure about that",
    "ok go on then",
    "you always say that",
    "i was thinking about our last chat",
)


# ---------------------------------------------------------------------------
# Synthetic owner responses
# ---------------------------------------------------------------------------

_VALID_REPLY = "that story about the rain ending mid-sentence has been bothering me all week"


def _valid_object(_reply: str = _VALID_REPLY, **overrides: Any) -> str:
    payload: dict[str, Any] = {
        "disposition": "reply",
        "response_goal": "continue the unfinished rain story",
        "contribution_goal": "take creator initiative",
        "initiative": "creator",
        "pacing": "continue",
        "operation_proposal": {"kind": "none"},
        "hold": "none",
        "confidence": 0.7,
        "state_delta": {"current_focus": "the unfinished story"},
    }
    payload.update(overrides)
    return json.dumps(payload, ensure_ascii=False)


#: Every shape production has produced, named. The generator draws from these.
RESPONSE_SHAPES: tuple[str, ...] = (
    "valid",
    "valid_fenced",
    "valid_extra_prose",
    "invalid_operation_refs",
    "operation_with_price_text",
    "malformed_state_delta",
    "truncated_mid_state_delta",
    "truncated_mid_reply",
    "out_of_range_confidence",
    "unknown_enums",
    "missing_optional_fields",
    "prose_only",
    "empty_content",
    "explicit_silence",
)


def synthetic_response(shape: str, *, turn_index: int) -> str:
    """The raw text one owner call returns, for a named failure shape."""

    reply = f"{_VALID_REPLY} (turn {turn_index})"
    if shape == "valid":
        return _valid_object(reply)
    if shape == "valid_fenced":
        return "```json\n" + _valid_object(reply) + "\n```"
    if shape == "valid_extra_prose":
        return "Here is the object you asked for:\n" + _valid_object(reply)
    if shape == "invalid_operation_refs":
        return _valid_object(
            reply,
            operation_proposal={
                "kind": "present_offer",
                "subject": "the thing they keep circling back to",
                "candidate_handle": "candidate-that-does-not-exist",
            },
        )
    if shape == "operation_with_price_text":
        return _valid_object(
            reply,
            operation_proposal={
                "kind": "present_offer",
                "subject": "the set they asked about for $24",
                "because": "they said $24 was fine",
            },
        )
    if shape == "malformed_state_delta":
        return _valid_object(reply, state_delta="not an object at all")
    if shape == "truncated_mid_state_delta":
        full = _valid_object(reply)
        return full[: full.index('"state_delta"') + 30]
    if shape == "truncated_mid_reply":
        full = _valid_object(reply)
        goal = "continue the unfinished rain story"
        return full[: full.index(goal) + len(goal) - 8]
    if shape == "out_of_range_confidence":
        return _valid_object(
            reply,
            confidence=1.8,
            operation_proposal={"kind": "check_payment_claim"},
        )
    if shape == "unknown_enums":
        return _valid_object(
            reply, initiative="somebody", pacing="upward_only", hold="thinking_about_it"
        )
    if shape == "missing_optional_fields":
        return json.dumps({"disposition": "reply", "response_goal": "continue"})
    if shape == "prose_only":
        return "I think the best move here is to keep the scene going and see what they say."
    if shape == "empty_content":
        return ""
    if shape == "explicit_silence":
        return json.dumps(
            {"disposition": "silence", "hold": "respect_silence"}
        )
    raise ValueError(f"unknown response shape: {shape}")


#: How often each shape appears in a default synthetic run. Weighted so the
#: healthy path dominates, as it does in production, while every failure shape
#: still appears often enough for its rate to mean something.
DEFAULT_SHAPE_WEIGHTS: dict[str, float] = {
    "valid": 46.0,
    "valid_fenced": 4.0,
    "valid_extra_prose": 3.0,
    "invalid_operation_refs": 8.0,
    "operation_with_price_text": 6.0,
    "malformed_state_delta": 6.0,
    "truncated_mid_state_delta": 8.0,
    "truncated_mid_reply": 4.0,
    "out_of_range_confidence": 4.0,
    "unknown_enums": 3.0,
    "missing_optional_fields": 3.0,
    "prose_only": 2.0,
    "empty_content": 2.0,
    "explicit_silence": 1.0,
}


@dataclass
class SyntheticOwner:
    """A seeded stand-in for the provider, including how it answers a repair.

    ``repair_success_rate`` is the fraction of repair calls that come back
    usable. It is a parameter rather than a constant because the interesting
    question is what the pipeline does at BOTH ends: a repair that works must
    produce a sent reply in exactly one extra call, and a repair that fails must
    produce a clean ``owner_failed`` and no legacy fallback.
    """

    seed: int = 7
    shape_weights: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_SHAPE_WEIGHTS)
    )
    repair_success_rate: float = 0.95
    latency_ms_range: tuple[int, int] = (400, 2_600)
    reasoning_tokens_range: tuple[int, int] = (80, 900)

    def __post_init__(self) -> None:
        self._random = random.Random(self.seed)
        self.calls: list[dict[str, Any]] = []

    def _draw_shape(self) -> str:
        shapes = list(self.shape_weights)
        weights = [max(float(self.shape_weights[name]), 0.0) for name in shapes]
        return self._random.choices(shapes, weights=weights, k=1)[0]

    async def __call__(self, target: Any, **kwargs: Any) -> Any:
        system = str(kwargs.get("system") or "")
        is_repair = "previous semantic decision could not be read" in system
        if is_repair:
            usable = self._random.random() < self.repair_success_rate
            shape = "valid" if usable else "prose_only"
        else:
            shape = self._draw_shape()
        turn_index = len(self.calls)
        text = synthetic_response(shape, turn_index=turn_index)
        latency_ms = self._random.randint(*self.latency_ms_range)
        reasoning_tokens = self._random.randint(*self.reasoning_tokens_range)
        truncated = shape.startswith("truncated") or shape == "empty_content"
        self.calls.append(
            {"shape": shape, "repair": is_repair, "latency_ms": latency_ms}
        )
        return SimpleNamespace(
            text=text,
            target=target,
            usage=ModelUsage(input_tokens=2_400, output_tokens=len(text) // 3),
            latency_ms=latency_ms,
            raw_response_id=f"gen-{turn_index}",
            upstream_provider="synthetic",
            reported_cost_usd=0.0,
            diagnostics=ModelResponseDiagnostics(
                provider=str(getattr(target, "provider", "synthetic")),
                model=str(getattr(target, "model", "synthetic-owner")),
                upstream_provider="synthetic",
                response_id=f"gen-{turn_index}",
                latency_ms=latency_ms,
                response_format_requested="json_object",
                reasoning_requested="on,effort=low",
                max_tokens_requested=int(kwargs.get("max_tokens") or 0),
                choice_count=1,
                finish_reason="length" if truncated else "stop",
                message_fields=("reasoning",) if not text else ("content", "reasoning"),
                content_is_null=not text,
                content_chars=len(text),
                reasoning_present=True,
                reasoning_chars=reasoning_tokens * 4,
                prompt_tokens=2_400,
                completion_tokens=reasoning_tokens + len(text) // 3,
                reasoning_tokens=reasoning_tokens,
            ),
        )


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TurnObservation:
    """What one turn did, in terms an operator can add up."""

    turn_index: int
    outcome: str
    latency_ms: int
    owner_calls: int
    repair_attempted: bool
    repaired: bool
    first_response_malformed: bool
    json_status: str
    failure_categories: tuple[str, ...]
    reply_sent: bool
    reply_chars: int
    operation_proposed: str
    operation_executed: str
    operation_rejected: bool
    state_delta_proposed: bool
    state_delta_rejected: bool
    locally_repaired: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _percentile(values: Sequence[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(
        len(ordered) - 1,
        max(0, int(round(fraction * (len(ordered) - 1)))),
    )
    return int(ordered[index])


@dataclass(frozen=True)
class StabilityReport:
    turns: tuple[TurnObservation, ...] = ()
    version: str = REPORT_VERSION
    arm: str = "synthetic"

    def metrics(self) -> dict[str, Any]:
        total = len(self.turns)
        if not total:
            return {"version": self.version, "arm": self.arm, "turns": 0}

        def rate(predicate: Callable[[TurnObservation], bool]) -> float:
            return round(sum(1 for turn in self.turns if predicate(turn)) / total, 4)

        latencies = [turn.latency_ms for turn in self.turns]
        categories: dict[str, int] = {}
        for turn in self.turns:
            for category in turn.failure_categories:
                categories[category] = categories.get(category, 0) + 1
        outcomes: dict[str, int] = {}
        for turn in self.turns:
            outcomes[turn.outcome] = outcomes.get(turn.outcome, 0) + 1

        repairs = [turn for turn in self.turns if turn.repair_attempted]
        return {
            "version": self.version,
            "arm": self.arm,
            "turns": total,
            "owner_failed_rate": rate(lambda t: t.outcome == OUTCOME_OWNER_FAILED),
            "reply_rate": rate(lambda t: t.reply_sent),
            "no_send_rate": rate(lambda t: t.outcome == OUTCOME_NO_SEND),
            "handoff_rate": rate(lambda t: t.outcome == OUTCOME_HANDOFF),
            "malformed_first_response_rate": rate(
                lambda t: t.first_response_malformed
            ),
            "repair_attempt_rate": rate(lambda t: t.repair_attempted),
            "repair_success_rate": (
                round(sum(1 for t in repairs if t.repaired) / len(repairs), 4)
                if repairs
                else 0.0
            ),
            "operation_proposed_rate": rate(lambda t: t.operation_proposed != "none"),
            "operation_rejection_rate": rate(lambda t: t.operation_rejected),
            "state_delta_rejection_rate": rate(lambda t: t.state_delta_rejected),
            "local_repair_rate": rate(lambda t: t.locally_repaired),
            "owner_calls_total": sum(turn.owner_calls for turn in self.turns),
            "owner_calls_per_turn": round(
                sum(turn.owner_calls for turn in self.turns) / total, 3
            ),
            "latency_ms_p50": _percentile(latencies, 0.50),
            "latency_ms_p95": _percentile(latencies, 0.95),
            "latency_ms_max": max(latencies),
            "failure_categories": dict(sorted(categories.items())),
            "outcomes": dict(sorted(outcomes.items())),
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "metrics": self.metrics(),
            "turns": [turn.as_dict() for turn in self.turns],
        }


async def run_turn(
    *,
    turn_index: int,
    loaded: Any,
    working_state: ConversationalWorkingState,
    owner_complete: Callable[..., Any] | None,
    mode: str = "auto",
) -> tuple[TurnObservation, ConversationalWorkingState]:
    """Drive ONE turn through the real owner boundary and real authority."""

    decision, _no_owner_copy, trace, raw_delta = await decide_conversational_v1(
        loaded,
        working_state,
        owner_complete=owner_complete,
    )
    delta_validation = validate_and_apply_delta(
        working_state,
        raw_delta,
        snapshot=loaded.snapshot,
        known_thread_ids=set(),
    )
    replies = (
        [f"{_VALID_REPLY} (turn {turn_index})"]
        if not trace.failure_reason
        and decision.disposition is ResponseDisposition.REPLY
        else []
    )
    settlement = await settle_conversational_v1_turn(
        loaded,
        decision=decision,
        replies=replies,
        mode=mode,
        # Never let an eval reach the delivery planner or the database.
        execute_operations=False,
    )

    owner_failed = bool(trace.failure_reason) and not settlement.replies
    if owner_failed:
        outcome = OUTCOME_OWNER_FAILED
    elif settlement.decision.disposition is ResponseDisposition.HANDOFF:
        outcome = OUTCOME_HANDOFF
    elif settlement.replies:
        outcome = OUTCOME_REPLIED
    else:
        outcome = OUTCOME_NO_SEND

    attempts = list(trace.owner_attempts)
    first = attempts[0] if attempts else {}
    return (
        TurnObservation(
            turn_index=turn_index,
            outcome=outcome,
            latency_ms=int(trace.elapsed_ms),
            owner_calls=len(attempts),
            repair_attempted=bool(trace.repair_attempted),
            repaired=bool(trace.repaired),
            first_response_malformed=not bool(first.get("usable", True)),
            json_status=str(first.get("json_status") or "unknown"),
            failure_categories=tuple(
                str(row.get("failure_category"))
                for row in attempts
                if row.get("failure_category")
            ),
            reply_sent=bool(settlement.replies),
            reply_chars=sum(len(reply) for reply in settlement.replies),
            operation_proposed=settlement.proposed_operation,
            operation_executed=settlement.execution.operation,
            operation_rejected=settlement.operation_rejected,
            state_delta_proposed=bool(delta_validation.proposed),
            state_delta_rejected=bool(delta_validation.rejected_fields)
            or any(
                "state_delta" in (row.get("degraded_fields") or {})
                for row in attempts
            ),
            locally_repaired=settlement.locally_repaired,
        ),
        delta_validation.state_after,
    )


async def run_synthetic_stability(
    *,
    turns: int = 200,
    seed: int = 7,
    repair_success_rate: float = 0.95,
    shape_weights: dict[str, float] | None = None,
) -> StabilityReport:
    """Run many synthetic turns through the real pipeline and report rates."""

    owner = SyntheticOwner(
        seed=seed,
        shape_weights=dict(shape_weights or DEFAULT_SHAPE_WEIGHTS),
        repair_success_rate=repair_success_rate,
    )
    target = ModelTarget(
        name="synthetic:owner",
        provider="synthetic",
        model="synthetic-owner",
        metadata={"reasoning_enabled": True, "reasoning_effort": "low"},
    )
    state = ConversationalWorkingState()
    observations: list[TurnObservation] = []
    for index in range(int(turns)):
        message = FAN_MESSAGES[index % len(FAN_MESSAGES)]
        loaded = synthetic_loaded(
            synthetic_snapshot(index, message),
            target=target,
        )
        observation, state = await run_turn(
            turn_index=index,
            loaded=loaded,
            working_state=state,
            owner_complete=owner,
        )
        observations.append(observation)
    return StabilityReport(turns=tuple(observations), arm="synthetic")
