"""Semantic-only decision contract for the GLM role in Conversational Core v2.

v2 extends the v1 contract rather than replacing it: the same parser reads the
same decision fields (and refuses the same prose and funnel fields), then this
module reads the two v2 additions from the same object:

* ``next_experience_move`` — what should happen next in the INTERACTION. It is
  read first because it is the first question; content is one possible answer.
* ``session_delta`` — proposed changes to the durable session state, validated
  field by field by :mod:`services.conversational_session`.

A missing or malformed v2 field degrades to "keep talking" instead of failing
the turn: session state is interpretation, and a lost turn of it costs nothing
the fan sees.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from models.conversational_session import ExperienceMove, MoveKind
from services.conversational_decision_contract import (
    FORBIDDEN_PROSE_FIELDS,
    SemanticDecisionResult,
    _object,
    parse_semantic_decision,
)

SOURCE_V2 = "conversational_decision_v2"


@dataclass(frozen=True)
class SessionDecisionResult(SemanticDecisionResult):
    session_delta: dict[str, Any] = field(default_factory=dict)
    next_experience_move: ExperienceMove = field(default_factory=ExperienceMove)


def _contains_prose(value: Any) -> bool:
    if isinstance(value, dict):
        return bool(FORBIDDEN_PROSE_FIELDS.intersection(value)) or any(
            _contains_prose(item) for item in value.values()
        )
    if isinstance(value, list):
        return any(_contains_prose(item) for item in value)
    return False


def parse_session_decision(text: str) -> SessionDecisionResult:
    base = parse_semantic_decision(text, source=SOURCE_V2)
    if not base.usable:
        return SessionDecisionResult(
            failure=base.failure, degradations=dict(base.degradations)
        )
    payload = _object(text) or {}
    degradations = dict(base.degradations)

    move = ExperienceMove()
    raw_move = payload.get("next_experience_move")
    if isinstance(raw_move, str):
        raw_move = {"kind": raw_move}
    if raw_move in (None, "", {}):
        degradations["next_experience_move"] = "missing; assumed converse"
    elif not isinstance(raw_move, dict) or _contains_prose(raw_move):
        degradations["next_experience_move"] = "unreadable or carried wording; assumed converse"
    else:
        try:
            move = ExperienceMove(
                kind=MoveKind(str(raw_move.get("kind") or "converse").strip().lower()),
                intent=" ".join(str(raw_move.get("intent") or "").split())[:300],
            )
        except (ValueError, ValidationError):
            degradations["next_experience_move"] = "unknown kind; assumed converse"
            move = ExperienceMove()

    raw_delta = payload.get("session_delta") or {}
    if not isinstance(raw_delta, dict):
        degradations["session_delta"] = "not an object; dropped"
        raw_delta = {}
    elif _contains_prose(raw_delta):
        degradations["session_delta"] = "carried fan-facing wording; dropped"
        raw_delta = {}

    return SessionDecisionResult(
        decision=base.decision,
        state_delta=base.state_delta,
        turn_id=base.turn_id,
        conversation_revision=base.conversation_revision,
        degradations=degradations,
        session_delta=json.loads(json.dumps(raw_delta, default=str)),
        next_experience_move=move,
    )
