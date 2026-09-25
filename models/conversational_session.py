"""Versioned, session-aware interaction state for ``conversational_v2``.

Core v1 carries a short-horizon reading of the scene. v2 keeps that reading
(``working``, the unchanged v1 model and validator) and adds a first-class,
durable representation of the LONGER interaction the conversation is carrying
out: whether one exists, what premise it maintains, what it is trying to do,
which constraints genuinely matter, a provisional future trajectory, and what
has actually happened.

Four rules shape every type here:

* The trajectory is provisional. A beat in ``tentative_trajectory`` is an
  intention, never a promise, reservation, offer, authorization, send or sale.
* Content lifecycle facts are different facts. ``planned`` lives only in the
  trajectory; ``presented`` / ``accepted`` / ``sent`` / ``purchased`` live only
  in ``used_content``, which the APPLICATION rebuilds from authoritative
  records every turn. The semantic owner has no delta field that can write it.
* Completed history is append-only. Replanning replaces future beats and never
  rewrites ``completed_beats``.
* Constraints are evidence, not inference. A spending constraint exists only
  when the fan stated it (or a transaction ledger established it); wealth or
  budget is never inferred.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from models.conversational_core import (
    ConversationalWorkingState,
    ElementStatus,
    EpistemicType,
    WorldScope,
)

SESSION_STATE_SCHEMA_VERSION = "conversational_session_v2"

MAX_CONSTRAINTS = 16
MAX_INFORMATION_NEEDS = 8
MAX_TRAJECTORY_BEATS = 6
MAX_COMPLETED_BEATS = 40
MAX_USED_CONTENT = 40
MAX_REPLANS = 12
MAX_PREVIOUS_SESSIONS = 8


class SessionStatus(str, Enum):
    """Lifecycle of a longer interaction.

    ``PROPOSED`` covers both directions: the fan suggested one, or the creator
    (the semantic owner) recognised a fitting moment and invited him into one.
    Either way it is an offer awaiting his answer, not an agreement; a declined
    proposal moves to ``ABANDONED``.
    """

    INACTIVE = "inactive"
    PROPOSED = "proposed"
    PLANNING = "planning"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


#: Statuses in which a longer interaction exists and carries a trajectory.
LIVE_STATUSES = frozenset(
    {
        SessionStatus.PROPOSED,
        SessionStatus.PLANNING,
        SessionStatus.ACTIVE,
        SessionStatus.PAUSED,
    }
)
#: Statuses from which the next longer interaction starts fresh.
ENDED_STATUSES = frozenset(
    {SessionStatus.INACTIVE, SessionStatus.COMPLETED, SessionStatus.ABANDONED}
)

#: The only legal status moves. Anything else is refused field-by-field.
STATUS_TRANSITIONS: dict[SessionStatus, frozenset[SessionStatus]] = {
    SessionStatus.INACTIVE: frozenset(
        {SessionStatus.PROPOSED, SessionStatus.PLANNING, SessionStatus.ACTIVE}
    ),
    SessionStatus.PROPOSED: frozenset(
        {
            SessionStatus.PLANNING,
            SessionStatus.ACTIVE,
            SessionStatus.PAUSED,
            SessionStatus.ABANDONED,
        }
    ),
    SessionStatus.PLANNING: frozenset(
        {
            SessionStatus.ACTIVE,
            SessionStatus.PAUSED,
            SessionStatus.ABANDONED,
        }
    ),
    SessionStatus.ACTIVE: frozenset(
        {
            SessionStatus.PAUSED,
            SessionStatus.COMPLETED,
            SessionStatus.ABANDONED,
        }
    ),
    SessionStatus.PAUSED: frozenset(
        {
            SessionStatus.ACTIVE,
            SessionStatus.COMPLETED,
            SessionStatus.ABANDONED,
        }
    ),
    SessionStatus.COMPLETED: frozenset(
        {SessionStatus.PROPOSED, SessionStatus.PLANNING, SessionStatus.ACTIVE}
    ),
    SessionStatus.ABANDONED: frozenset(
        {SessionStatus.PROPOSED, SessionStatus.PLANNING, SessionStatus.ACTIVE}
    ),
}


class Tempo(str, Enum):
    BUILD = "build"
    LINGER = "linger"
    PULL_BACK = "pull_back"
    CONTINUE = "continue"
    TRANSITION = "transition"
    REDIRECT = "redirect"
    PAUSE = "pause"
    CLOSE = "close"


class MoveKind(str, Enum):
    """What should happen next in the INTERACTION. Content is one option."""

    CONVERSE = "converse"
    REACT = "react"
    LINGER = "linger"
    INVITE_PARTICIPATION = "invite_participation"
    PULL_BACK = "pull_back"
    CALLBACK = "callback"
    DEVELOP_PREMISE = "develop_premise"
    ALTER_PREMISE = "alter_premise"
    TRANSITION = "transition"
    DISCOVER = "discover"
    USE_CONTENT = "use_content"
    PAUSE = "pause"
    CLOSE = "close"


class BeatKind(str, Enum):
    CONVERSATION = "conversation"
    CALLBACK = "callback"
    PARTICIPATION = "participation"
    DISCOVERY = "discovery"
    CONTENT = "content"
    TRANSITION = "transition"
    CLOSE = "close"


class ParticipationMode(str, Enum):
    UNKNOWN = "unknown"
    LEADING = "leading"
    CO_CREATING = "co_creating"
    FOLLOWING = "following"
    BRIEF = "brief"
    REDIRECTING = "redirecting"
    COOLING = "cooling"
    PAUSING = "pausing"


class ConstraintKind(str, Enum):
    SPENDING_LIMIT = "spending_limit"
    AVAILABILITY = "availability"
    PREFERENCE = "preference"
    BOUNDARY = "boundary"
    OTHER = "other"


class InformationTopic(str, Enum):
    SPENDING_LIMIT = "spending_limit"
    CONTENT_PREFERENCE = "content_preference"
    AVAILABILITY = "availability"
    DIRECTION = "direction"
    OTHER = "other"


class NeedStatus(str, Enum):
    OPEN = "open"
    RESOLVED = "resolved"
    DROPPED = "dropped"


class ContentLifecycle(str, Enum):
    """Application-observed facts only. ``planned`` is deliberately absent."""

    PRESENTED = "presented"
    ACCEPTED = "accepted"
    SENT = "sent"
    PURCHASED = "purchased"


#: Ordering used only to recognise that a fact ADVANCED between turns.
LIFECYCLE_RANK = {
    ContentLifecycle.PRESENTED: 1,
    ContentLifecycle.ACCEPTED: 2,
    ContentLifecycle.SENT: 3,
    ContentLifecycle.PURCHASED: 4,
}
CONSUMED_LIFECYCLES = frozenset({ContentLifecycle.SENT, ContentLifecycle.PURCHASED})


class BeatSource(str, Enum):
    MODEL = "model"
    APPLICATION = "application"


def _tidy(value: str) -> str:
    return " ".join(str(value or "").split())


class ExperiencePremise(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = Field(default="", max_length=600)
    world_scope: WorldScope = WorldScope.CONVERSATION
    source_refs: list[str] = Field(default_factory=list, max_length=8)
    established_turn_ref: str = Field(default="", max_length=200)

    @field_validator("summary")
    @classmethod
    def _tidy_summary(cls, value: str) -> str:
        return _tidy(value)

    @field_validator("world_scope")
    @classmethod
    def _no_real_world_premise(cls, value: WorldScope) -> WorldScope:
        if value not in {WorldScope.CONVERSATION, WorldScope.IMAGINED_SCENE}:
            raise ValueError("a premise is conversational or imagined, never real-world")
        return value


class FanParticipation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: ParticipationMode = ParticipationMode.UNKNOWN
    gist: str = Field(default="", max_length=300)


class KnownConstraint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    constraint_id: str = Field(min_length=1, max_length=100)
    kind: ConstraintKind
    statement: str = Field(min_length=1, max_length=300)
    amount_cents: int | None = Field(default=None, gt=0)
    source_type: EpistemicType
    source_refs: list[str] = Field(default_factory=list, max_length=8)
    status: ElementStatus = ElementStatus.ACTIVE
    superseded_by: str = Field(default="", max_length=100)
    introduced_turn_ref: str = Field(default="", max_length=200)
    #: The application's content-fact sequence when this constraint began.
    #: Purchases recorded after it are what count against a spending limit.
    after_content_seq: int = Field(default=0, ge=0)

    @field_validator("source_type")
    @classmethod
    def _evidence_only(cls, value: EpistemicType) -> EpistemicType:
        if value not in {
            EpistemicType.EXPLICIT_FAN_STATEMENT,
            EpistemicType.TRANSACTION_FACT,
            EpistemicType.CREATOR_CONFIG,
        }:
            raise ValueError("a known constraint must be evidenced, never inferred")
        return value


class InformationNeed(BaseModel):
    model_config = ConfigDict(extra="forbid")

    need_id: str = Field(min_length=1, max_length=100)
    topic: InformationTopic
    why_material: str = Field(min_length=1, max_length=300)
    status: NeedStatus = NeedStatus.OPEN
    opened_turn_ref: str = Field(default="", max_length=200)
    closed_turn_ref: str = Field(default="", max_length=200)


class PlannedBeat(BaseModel):
    """One provisional future (or current) beat. Never a commitment."""

    model_config = ConfigDict(extra="forbid")

    beat_id: str = Field(min_length=1, max_length=100)
    kind: BeatKind
    intent: str = Field(default="", max_length=300)
    #: Internal approved-set reference, resolved by the application from an
    #: opaque candidate handle. Planning it reserves and promises nothing.
    content_ref: str = Field(default="", max_length=200)
    media_role: str = Field(default="", max_length=300)
    planned_turn_ref: str = Field(default="", max_length=200)
    started_turn_ref: str = Field(default="", max_length=200)


class CompletedBeat(BaseModel):
    model_config = ConfigDict(extra="forbid")

    beat_id: str = Field(min_length=1, max_length=100)
    kind: BeatKind
    intent: str = Field(default="", max_length=300)
    outcome: str = Field(default="", max_length=300)
    content_ref: str = Field(default="", max_length=200)
    completed_turn_ref: str = Field(default="", max_length=200)
    source: BeatSource = BeatSource.MODEL


class ContentUse(BaseModel):
    """A content fact the application observed in authoritative records."""

    model_config = ConfigDict(extra="forbid")

    content_ref: str = Field(min_length=1, max_length=200)
    lifecycle: ContentLifecycle
    source_ref: str = Field(default="", max_length=200)
    observed_turn_ref: str = Field(default="", max_length=200)
    #: Monotonic per fan; orders facts without trusting opaque turn ids.
    seq: int = Field(default=0, ge=0)
    price_cents: int = Field(default=0, ge=0)


class ExperienceMove(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: MoveKind = MoveKind.CONVERSE
    intent: str = Field(default="", max_length=300)


class LastEvent(BaseModel):
    """The newest application-observed content event, if any."""

    model_config = ConfigDict(extra="forbid")

    kind: str = Field(default="none", max_length=40)
    content_ref: str = Field(default="", max_length=200)
    observed_turn_ref: str = Field(default="", max_length=200)


class ReplanRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    turn_ref: str = Field(default="", max_length=200)
    reason: str = Field(min_length=1, max_length=300)
    replaced_beat_ids: list[str] = Field(default_factory=list, max_length=12)


class SessionSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(max_length=100)
    final_status: SessionStatus
    premise: str = Field(default="", max_length=600)
    completed_beats: int = 0
    content_events: int = 0
    ended_turn_ref: str = Field(default="", max_length=200)


class InteractionSession(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(default="", max_length=100)
    status: SessionStatus = SessionStatus.INACTIVE
    experience_premise: ExperiencePremise = Field(default_factory=ExperiencePremise)
    interaction_goal: str = Field(default="", max_length=300)
    fan_participation: FanParticipation = Field(default_factory=FanParticipation)
    known_constraints: list[KnownConstraint] = Field(
        default_factory=list, max_length=MAX_CONSTRAINTS
    )
    information_needs: list[InformationNeed] = Field(
        default_factory=list, max_length=MAX_INFORMATION_NEEDS
    )
    tentative_trajectory: list[PlannedBeat] = Field(
        default_factory=list, max_length=MAX_TRAJECTORY_BEATS
    )
    current_beat: PlannedBeat | None = None
    completed_beats: list[CompletedBeat] = Field(
        default_factory=list, max_length=MAX_COMPLETED_BEATS
    )
    completed_beats_trimmed: int = Field(default=0, ge=0)
    next_experience_move: ExperienceMove = Field(default_factory=ExperienceMove)
    tempo: Tempo = Tempo.CONTINUE
    #: A fan-authored direction for future content ("something outdoors", "a
    #: video"). A hint for which approved inventory is relevant, not a request
    #: the application must satisfy.
    content_direction: str = Field(default="", max_length=200)
    used_content: list[ContentUse] = Field(
        default_factory=list, max_length=MAX_USED_CONTENT
    )
    content_fact_seq: int = Field(default=0, ge=0)
    last_event: LastEvent = Field(default_factory=LastEvent)
    last_replan_reason: str = Field(default="", max_length=300)
    replans: list[ReplanRecord] = Field(default_factory=list, max_length=MAX_REPLANS)
    started_turn_ref: str = Field(default="", max_length=200)
    turns_in_session: int = Field(default=0, ge=0)
    turns_since_content_event: int = Field(default=0, ge=0)

    def active_constraints(self) -> list[KnownConstraint]:
        return [c for c in self.known_constraints if c.status is ElementStatus.ACTIVE]

    def spending_constraint(self) -> KnownConstraint | None:
        return next(
            (
                c
                for c in reversed(self.active_constraints())
                if c.kind is ConstraintKind.SPENDING_LIMIT
            ),
            None,
        )

    def lifecycle_of(self, content_ref: str) -> ContentLifecycle | None:
        found = [u.lifecycle for u in self.used_content if u.content_ref == content_ref]
        if not found:
            return None
        return max(found, key=lambda value: LIFECYCLE_RANK[value])

    def consumed_refs(self) -> set[str]:
        return {
            u.content_ref for u in self.used_content if u.lifecycle in CONSUMED_LIFECYCLES
        }


class ConversationalSessionState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = SESSION_STATE_SCHEMA_VERSION
    revision: int = Field(default=0, ge=0)
    #: The v1 short-horizon scene/flow reading, reused verbatim with its own
    #: validator. v2 keeps it in v2's own row, so v1 and v2 never share state.
    working: ConversationalWorkingState = Field(
        default_factory=ConversationalWorkingState
    )
    session: InteractionSession = Field(default_factory=InteractionSession)
    previous_sessions: list[SessionSummary] = Field(
        default_factory=list, max_length=MAX_PREVIOUS_SESSIONS
    )

    @field_validator("schema_version")
    @classmethod
    def _known_version(cls, value: str) -> str:
        if value != SESSION_STATE_SCHEMA_VERSION:
            raise ValueError(f"unsupported session-state version {value!r}")
        return value

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


# ---------------------------------------------------------------------------
# Owner-proposed deltas. Every field is optional; absence preserves state.
# ``used_content`` and ``last_event`` are deliberately NOT here.
# ---------------------------------------------------------------------------


class ProposedPremise(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1, max_length=600)
    world_scope: WorldScope = WorldScope.CONVERSATION
    source_refs: list[str] = Field(default_factory=list, max_length=8)


class ProposedConstraint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    constraint_id: str = Field(min_length=1, max_length=100)
    kind: ConstraintKind
    statement: str = Field(min_length=1, max_length=300)
    amount_cents: int | None = Field(default=None, gt=0)
    source_type: EpistemicType = EpistemicType.EXPLICIT_FAN_STATEMENT
    source_refs: list[str] = Field(default_factory=list, max_length=8)


class ProposedConstraintCorrection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    replaces_constraint_id: str = Field(min_length=1, max_length=100)
    replacement: ProposedConstraint


class ProposedInformationNeed(BaseModel):
    model_config = ConfigDict(extra="forbid")

    topic: InformationTopic
    why_material: str = Field(min_length=1, max_length=300)


class ProposedBeat(BaseModel):
    model_config = ConfigDict(extra="forbid")

    beat_id: str = Field(default="", max_length=100)
    #: Required for a NEW beat; starting an already-planned beat needs only its id.
    kind: BeatKind | None = None
    intent: str = Field(default="", max_length=300)
    candidate_handle: str = Field(default="", max_length=100)
    media_role: str = Field(default="", max_length=300)


class ProposedTrajectory(BaseModel):
    """Replace the FUTURE beats. Completed beats are out of reach by design."""

    model_config = ConfigDict(extra="forbid")

    reason: str = Field(default="", max_length=300)
    beats: list[ProposedBeat] = Field(default_factory=list, max_length=12)


class ProposedBeatCompletion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    outcome: str = Field(default="", max_length=300)


class SessionDelta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: SessionStatus | None = None
    experience_premise: ProposedPremise | None = None
    interaction_goal: str | None = Field(default=None, max_length=300)
    fan_participation: FanParticipation | None = None
    add_constraints: list[ProposedConstraint] = Field(
        default_factory=list, max_length=6
    )
    constraint_corrections: list[ProposedConstraintCorrection] = Field(
        default_factory=list, max_length=4
    )
    open_information_need: ProposedInformationNeed | None = None
    resolve_information_needs: list[str] = Field(default_factory=list, max_length=8)
    trajectory: ProposedTrajectory | None = None
    complete_current_beat: ProposedBeatCompletion | None = None
    start_beat: ProposedBeat | None = None
    tempo: Tempo | None = None
    content_direction: str | None = Field(default=None, max_length=200)


class SessionDeltaValidation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    proposed: dict[str, Any] = Field(default_factory=dict)
    accepted_fields: list[str] = Field(default_factory=list)
    rejected_fields: dict[str, str] = Field(default_factory=dict)
    #: Deterministic changes the application made from authoritative records
    #: before the owner decided (content lifecycle, consumed beats).
    reconciled: list[str] = Field(default_factory=list)
    state_after: ConversationalSessionState
