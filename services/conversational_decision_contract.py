"""Semantic-only contract for the GLM role in Conversational Core v1.

The decision model never owns fan-facing text.  This parser therefore refuses
reply/caption/copy fields instead of trying to salvage them.  A malformed
decision may be retried once by the orchestration layer; it is never handed to
the writer as prose.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

from models.conversation_decision import (
    CANCEL_ON_ACTIVITY,
    INTENT_ACTIVITY_POLICIES,
    INTENT_KINDS,
    REVALIDATE_ON_ACTIVITY,
    ConversationDecision,
    IntimacyContext,
    HoldReason,
    OperationKind,
    ProposedOperation,
    ResponseDisposition,
    ResponseIntent,
    ScheduledIntent,
)

FORBIDDEN_PROSE_FIELDS = frozenset(
    {
        "reply",
        "message",
        "messages",
        "caption",
        "copy",
        "rewrite",
        "candidate_sentences",
        "kimi_prompt_instructions",
        "phrasing",
        "say_something_like",
    }
)

# These fields recreate the linear/scored machinery Core v1 is replacing.
# Intimacy is represented by independent working-state dimensions instead.
FORBIDDEN_FUNNEL_FIELDS = frozenset(
    {
        "engagement_score",
        "fan_engagement",
        "intimacy_stage",
        "sexual_stage",
        "escalation_stage",
        "spending_power",
        "conversion_likelihood",
    }
)

EVIDENCE_CATEGORIES = frozenset(
    {"inventory", "memory", "continuity", "transactions", "creator_voice"}
)
INITIATIVE_VALUES = frozenset({"fan", "creator", "shared"})
PACING_VALUES = frozenset(
    {"build", "hold", "continue", "cool", "redirect", "pause", "resume"}
)
INTIMACY_REGISTERS = frozenset({"none", "flirty", "suggestive", "explicit"})
INTIMACY_SCENE_MODES = frozenset({"none", "conversational", "shared_imagined"})

#: Named times the APPLICATION already holds evidence for. A decision may point
#: at one; it may never state a clock time of its own, because it has no way to
#: know one and every way to guess.
INTENT_TIME_REFERENCES = frozenset({"payday", "pending_offer_expiry"})

#: Deterministic bounds on a model-selected relative delay. Thirty seconds is
#: the shortest pause that reads as "hold on" rather than as a glitch; a day is
#: the longest a conversational continuation can claim before it is really a
#: lifecycle follow-up, which has its own machinery.
MIN_INTENT_DELAY_SECONDS = 30
MAX_INTENT_DELAY_SECONDS = 24 * 60 * 60

#: Kinds whose natural reading is "unless he speaks first".
_CANCEL_BY_DEFAULT = frozenset({"short_continuation", "scene_resume"})


@dataclass(frozen=True)
class SemanticDecisionResult:
    decision: ConversationDecision | None = None
    state_delta: dict[str, Any] = field(default_factory=dict)
    turn_id: str = ""
    conversation_revision: str = ""
    failure: str = ""
    degradations: dict[str, str] = field(default_factory=dict)

    @property
    def usable(self) -> bool:
        return self.decision is not None

    def describe(self) -> str:
        if self.usable:
            return "semantic_decision=usable"
        return "semantic_decision=invalid reason=" + (self.failure or "unknown")


def _object(text: str) -> dict[str, Any] | None:
    raw = str(text or "").strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _clean(value: Any, limit: int = 600) -> str:
    return " ".join(str(value or "").split())[:limit]


def _strings(value: Any, *, limit: int = 12) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(
        dict.fromkeys(_clean(item) for item in value if _clean(item))
    )[:limit]


def _parse_scheduled_intent(
    raw: Any, degradations: dict[str, str]
) -> ScheduledIntent:
    """Read a future conversational obligation, or refuse it.

    Everything here is a narrowing. The kind must be in the vocabulary, the goal
    must exist and must not name a price, the timing must be either a clamped
    relative delay or one of the application's own evidenced references, and the
    activity policy must be one of two values. Anything else is dropped with a
    reason rather than guessed at, because a scheduled intention that is wrong
    reaches the fan hours later with nobody watching.
    """
    if raw in (None, "", {}, []):
        return ScheduledIntent()
    if not isinstance(raw, dict):
        degradations["scheduled_intent"] = "not an object; dropped"
        return ScheduledIntent()

    prose = sorted(FORBIDDEN_PROSE_FIELDS.intersection(raw))
    if prose:
        degradations["scheduled_intent"] = (
            "carried fan-facing wording (" + ", ".join(prose) + "); dropped"
        )
        return ScheduledIntent()

    kind = _clean(raw.get("kind"), 40).lower()
    if kind not in INTENT_KINDS:
        degradations["scheduled_intent.kind"] = "unknown kind; dropped"
        return ScheduledIntent()

    goal = _clean(raw.get("goal"), 400)
    if not goal:
        degradations["scheduled_intent.goal"] = "missing; dropped"
        return ScheduledIntent()
    if "$" in goal or "€" in goal or "£" in goal:
        degradations["scheduled_intent.goal"] = "named a price; dropped"
        return ScheduledIntent()

    timing = raw.get("timing")
    if not isinstance(timing, dict):
        timing = {}
    timing_kind = ""
    relative_seconds = 0
    reference = ""

    raw_reference = _clean(timing.get("reference"), 60).lower()
    if raw_reference:
        if raw_reference not in INTENT_TIME_REFERENCES:
            degradations["scheduled_intent.timing"] = (
                "unknown time reference; dropped"
            )
            return ScheduledIntent()
        timing_kind = "reference"
        reference = raw_reference
    else:
        raw_minutes = timing.get("relative_minutes", timing.get("minutes"))
        raw_seconds = timing.get("relative_seconds", timing.get("seconds"))
        candidate: float | None = None
        for value, multiplier in ((raw_minutes, 60.0), (raw_seconds, 1.0)):
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                candidate = float(value) * multiplier
                break
        if candidate is None:
            degradations["scheduled_intent.timing"] = "no usable delay; dropped"
            return ScheduledIntent()
        clamped = max(
            MIN_INTENT_DELAY_SECONDS, min(MAX_INTENT_DELAY_SECONDS, int(candidate))
        )
        if int(clamped) != int(candidate):
            degradations["scheduled_intent.timing"] = (
                f"delay clamped to {clamped}s from {int(candidate)}s"
            )
        timing_kind = "relative"
        relative_seconds = int(clamped)

    policy = _clean(raw.get("activity_policy"), 40).lower()
    if policy not in INTENT_ACTIVITY_POLICIES:
        if policy:
            degradations["scheduled_intent.activity_policy"] = (
                "unknown value; using the default for this kind"
            )
        policy = (
            CANCEL_ON_ACTIVITY
            if kind in _CANCEL_BY_DEFAULT
            else REVALIDATE_ON_ACTIVITY
        )

    return ScheduledIntent(
        kind=kind,
        goal=goal,
        timing_kind=timing_kind,
        relative_seconds=relative_seconds,
        reference=reference,
        source_ids=_strings(raw.get("source_ids"), limit=8),
        activity_policy=policy,
    )


def parse_semantic_decision(
    text: str,
    *,
    source: str = "conversational_decision_v1",
) -> SemanticDecisionResult:
    payload = _object(text)
    if payload is None:
        return SemanticDecisionResult(failure="response is not one JSON object")

    forbidden = sorted(FORBIDDEN_PROSE_FIELDS.intersection(payload))
    if forbidden:
        return SemanticDecisionResult(
            failure="decision model emitted fan-facing prose fields: "
            + ", ".join(forbidden)
        )
    funnel_fields = sorted(FORBIDDEN_FUNNEL_FIELDS.intersection(payload))
    if funnel_fields:
        return SemanticDecisionResult(
            failure="decision model emitted forbidden funnel fields: "
            + ", ".join(funnel_fields)
        )

    degradations: dict[str, str] = {}
    try:
        disposition = ResponseDisposition(
            _clean(payload.get("disposition") or "reply", 40)
        )
    except ValueError:
        return SemanticDecisionResult(failure="unknown disposition")

    response_goal = _clean(payload.get("response_goal"), 800)
    if disposition is ResponseDisposition.REPLY and not response_goal:
        return SemanticDecisionResult(failure="reply decision has no response_goal")

    must_address: list[str] = []
    supporting: list[str] = []
    raw_must = payload.get("must_address") or []
    if not isinstance(raw_must, list):
        degradations["must_address"] = "not a list; dropped"
        raw_must = []
    for item in raw_must[:12]:
        if isinstance(item, dict):
            need = _clean(item.get("need"), 500)
            if need:
                must_address.append(need)
            supporting.extend(_strings(item.get("source_ids"), limit=8))
        else:
            need = _clean(item, 500)
            if need:
                must_address.append(need)

    evidence_requests: list[str] = []
    raw_requests = payload.get("evidence_requests") or []
    if not isinstance(raw_requests, list):
        degradations["evidence_requests"] = "not a list; dropped"
        raw_requests = []
    for item in raw_requests[:5]:
        category = _clean(
            item.get("category") if isinstance(item, dict) else item,
            40,
        ).lower()
        if category in EVIDENCE_CATEGORIES and category not in evidence_requests:
            evidence_requests.append(category)
        elif category:
            degradations[f"evidence_request:{category}"] = "unknown category; dropped"

    initiative = _clean(payload.get("initiative") or "shared", 20).lower()
    if initiative not in INITIATIVE_VALUES:
        degradations["initiative"] = "unknown value; assumed shared"
        initiative = "shared"
    pacing = _clean(payload.get("pacing") or "continue", 20).lower()
    if pacing not in PACING_VALUES:
        degradations["pacing"] = "unknown value; assumed continue"
        pacing = "continue"

    raw_intimacy = payload.get("intimacy_context") or {}
    if not isinstance(raw_intimacy, dict):
        degradations["intimacy_context"] = "not an object; dropped"
        raw_intimacy = {}
    intimacy_active = bool(raw_intimacy.get("active", False))
    content_register = _clean(
        raw_intimacy.get("content_register") or "none", 30
    ).lower()
    if content_register not in INTIMACY_REGISTERS:
        degradations["intimacy_context.content_register"] = (
            "unknown value; assumed none"
        )
        content_register = "none"
    scene_mode = _clean(raw_intimacy.get("scene_mode") or "none", 30).lower()
    if scene_mode not in INTIMACY_SCENE_MODES:
        degradations["intimacy_context.scene_mode"] = "unknown value; assumed none"
        scene_mode = "none"
    intimacy_direction = _clean(
        raw_intimacy.get("direction") or pacing or "continue", 30
    ).lower()
    if intimacy_direction not in PACING_VALUES:
        degradations["intimacy_context.direction"] = (
            "unknown value; assumed continue"
        )
        intimacy_direction = "continue"
    intimacy_context = IntimacyContext(
        active=intimacy_active,
        content_register=content_register,
        scene_mode=scene_mode,
        direction=intimacy_direction,
        last_beat=_clean(raw_intimacy.get("last_beat"), 500),
        boundaries=_strings(raw_intimacy.get("boundaries"), limit=8),
    )

    raw_operation = payload.get("operation_proposal") or {}
    if raw_operation is None:
        raw_operation = {}
    if not isinstance(raw_operation, dict):
        degradations["operation_proposal"] = "not an object; dropped"
        raw_operation = {}
    try:
        kind = OperationKind(_clean(raw_operation.get("kind") or "none", 80))
    except ValueError:
        degradations["operation_proposal.kind"] = "unknown value; dropped"
        kind = OperationKind.NONE

    try:
        hold = HoldReason(_clean(payload.get("hold") or "none", 80))
    except ValueError:
        degradations["hold"] = "unknown value; assumed none"
        hold = HoldReason.NONE
    if disposition is ResponseDisposition.HANDOFF:
        hold = HoldReason.NEEDS_HUMAN
        if kind is OperationKind.NONE:
            kind = OperationKind.HAND_OFF_TO_HUMAN
    elif disposition is ResponseDisposition.SILENCE and hold is HoldReason.NONE:
        hold = HoldReason.RESPECT_SILENCE

    raw_confidence = payload.get("confidence", 0.0)
    confidence = 0.0
    if isinstance(raw_confidence, (int, float)) and not isinstance(
        raw_confidence, bool
    ):
        candidate = float(raw_confidence)
        if math.isfinite(candidate) and 0.0 <= candidate <= 1.0:
            confidence = candidate
        else:
            degradations["confidence"] = "outside 0..1; treated as 0"
    else:
        degradations["confidence"] = "not numeric; treated as 0"

    state_delta = payload.get("state_delta") or {}
    if not isinstance(state_delta, dict):
        degradations["state_delta"] = "not an object; dropped"
        state_delta = {}
    memory_candidates = payload.get("memory_candidates") or []
    if not isinstance(memory_candidates, list):
        degradations["memory_candidates"] = "not a list; dropped"
        memory_candidates = []

    scheduled_intent = _parse_scheduled_intent(
        payload.get("scheduled_intent"), degradations
    )

    decision = ConversationDecision(
        active_needs=_strings(payload.get("active_needs"), limit=12),
        supporting_messages=tuple(dict.fromkeys(supporting))[:24],
        unresolved_references=_strings(
            payload.get("unresolved_references"), limit=12
        ),
        must_address=tuple(dict.fromkeys(must_address))[:12],
        response_goal=response_goal,
        contribution_goal=_clean(payload.get("contribution_goal"), 600),
        relevant_thread_ids=_strings(payload.get("relevant_thread_ids"), limit=12),
        initiative=initiative,
        pacing=pacing,
        intimacy_context=intimacy_context,
        evidence_requests=tuple(evidence_requests),
        memory_candidates=tuple(
            dict(item) for item in memory_candidates[:12] if isinstance(item, dict)
        ),
        proposed_operation=ProposedOperation(
            kind=kind,
            subject=_clean(raw_operation.get("subject"), 500),
            because=_clean(raw_operation.get("because"), 500),
            candidate_handle=_clean(raw_operation.get("candidate_handle"), 100),
            payment_reference=_clean(raw_operation.get("payment_reference"), 200),
            purchase_id=_clean(raw_operation.get("purchase_id"), 200),
        ),
        scheduled_intent=scheduled_intent,
        response_intent=ResponseIntent.ORDINARY_CONVERSATION,
        disposition=disposition,
        hold=hold,
        hold_detail=_clean(payload.get("hold_detail"), 500),
        source=source,
        confidence=confidence,
    )
    return SemanticDecisionResult(
        decision=decision,
        state_delta=state_delta,
        turn_id=_clean(payload.get("turn_id"), 200),
        conversation_revision=_clean(payload.get("conversation_revision"), 200),
        degradations=degradations,
    )
