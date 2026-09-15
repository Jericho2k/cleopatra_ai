"""Persistent conversational scene state: the Experience Director.

WHY THIS EXISTS, AND WHAT IT IS NOT
-----------------------------------
The commercial layer was deliberately reduced to one unlock at a time
(``services/session_planner.plan_session_for_fan`` builds a one-step session on
purpose). That fixed checkout — a fan is never prepaying for a bundle — but it
also meant the only thing with memory across a purchase was the *commercial*
session, and a one-step session is over the moment it is paid. The conversation
therefore restarted at offer discovery after every unlock.

This module is the missing half. It is a lightweight, persistent record of the
SCENE: what is going on between the two people, which beat that scene is on,
what he just reacted to, and whether the conversation has genuinely earned
another paid moment. It is choreography, not authorization.

The division of labour is absolute:

    Experience Director  ->  WHEN, conversationally, something may happen
    Commercial Policy    ->  WHETHER it may happen at all, and at what price

Nothing here can authorize a send, set or move a price, widen approved content,
or overrule a pause. ``another_unlock_ready`` is a *veto that can only narrow*:
policy asks it before discovering a NEW offer and is free to ignore it for
anything already on the table. The fan is never shown a beat, a scene, a step
count, a future price or a session total — see ``writer_context``, which is the
only projection that reaches the prompt and deliberately carries no money.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ExperienceBeat(str, Enum):
    """Where the current scene is, conversationally."""

    #: Nothing is running yet: reading him, finding the premise.
    SETUP = "SETUP"
    #: A premise exists and is being built on. Tension is rising.
    BUILD = "BUILD"
    #: One unlock is on the table and he has not resolved it.
    OFFER = "OFFER"
    #: He unlocked something and has not reacted to it yet. Nothing new is
    #: offered from here — the next move is his.
    AWAIT_REACTION = "AWAIT_REACTION"
    #: He reacted and the conversation is IN the thing he unlocked.
    PLAY = "PLAY"
    #: The scene has produced a natural next direction. Another unlock may now
    #: become conversationally ready.
    BRIDGE = "BRIDGE"
    #: This scene is finished: paused, declined, handed off, or simply over.
    CLOSE = "CLOSE"


class FanReaction(str, Enum):
    NONE = "NONE"
    POSITIVE = "POSITIVE"
    NEUTRAL = "NEUTRAL"
    NEGATIVE = "NEGATIVE"
    WANTS_MORE = "WANTS_MORE"


#: Beats in which a purchase has happened and the scene still owes him
#: interaction. Discovering a brand-new offer here is the regression this
#: module exists to prevent.
_POST_UNLOCK_BEATS = {ExperienceBeat.AWAIT_REACTION, ExperienceBeat.PLAY}

#: The maximum stored intimacy/tension level. A small integer on purpose: it is
#: a dial the writer reads, not a score anything is optimised against.
MAX_LEVEL = 5


class SceneState(BaseModel):
    """One fan's current scene. Small, durable, and free of money."""

    beat: ExperienceBeat = ExperienceBeat.SETUP
    previous_beat: ExperienceBeat | None = None

    #: A stable identity for the scene, so a later turn can tell "still the
    #: shower thing" from "something new started". Copied from approved set
    #: metadata or from the fan's own stated direction; never invented prose.
    scene_key: str = ""
    #: What the scene is ABOUT, in approved words. Grounded: it comes from
    #: catalog metadata or from what he actually said, never from the writer.
    premise: str = ""

    #: The last thing he actually unlocked, and its approved description.
    last_unlocked_set_id: str | None = None
    last_unlocked_description: str = ""

    last_fan_reaction: FanReaction = FanReaction.NONE
    #: True once a reply has actually engaged with that reaction. Until then
    #: the scene owes him a response to the specific thing he said.
    reaction_processed: bool = False

    intimacy_level: int = Field(default=0, ge=0, le=MAX_LEVEL)
    tension_level: int = Field(default=0, ge=0, le=MAX_LEVEL)

    #: Something raised and not yet resolved — a question he asked, a tease
    #: left hanging, a thing he said he wanted. The writer should land it.
    open_hook: str = ""
    #: Where this is heading, in his words or the scene's. Not a promise.
    desired_direction: str = ""

    #: The gate. True only when the scene has genuinely produced a next moment.
    another_unlock_ready: bool = False

    beats_in_scene: int = 0
    turns_since_unlock: int = 0
    unlocks_in_scene: int = 0

    transition_reason: str = "new_scene"
    scene_version: int = 1
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_context(self) -> dict[str, Any]:
        """The full persisted shape. Every key here is a column."""
        return {
            "beat": self.beat.value,
            "previous_beat": self.previous_beat.value if self.previous_beat else None,
            "scene_key": self.scene_key,
            "premise": self.premise,
            "last_unlocked_set_id": self.last_unlocked_set_id,
            "last_unlocked_description": self.last_unlocked_description,
            "last_fan_reaction": self.last_fan_reaction.value,
            "reaction_processed": self.reaction_processed,
            "intimacy_level": self.intimacy_level,
            "tension_level": self.tension_level,
            "open_hook": self.open_hook,
            "desired_direction": self.desired_direction,
            "another_unlock_ready": self.another_unlock_ready,
            "beats_in_scene": self.beats_in_scene,
            "turns_since_unlock": self.turns_since_unlock,
            "unlocks_in_scene": self.unlocks_in_scene,
            "transition_reason": self.transition_reason,
            "scene_version": self.scene_version,
            "updated_at": self.updated_at.isoformat(),
        }

    def writer_context(self) -> dict[str, Any]:
        """The projection the prompt may see.

        Deliberately narrower than ``to_context``: no set id, no counters, no
        readiness flag, and above all no money. The writer is told what the
        scene IS and what this beat owes him — never how many unlocks have
        happened, how many might, or what any of it costs.
        """
        return {
            "beat": self.beat.value,
            "premise": self.premise,
            "just_unlocked": self.last_unlocked_description,
            "fan_reaction": self.last_fan_reaction.value,
            "reaction_owed": (
                self.beat is ExperienceBeat.AWAIT_REACTION
                or (self.last_fan_reaction is not FanReaction.NONE
                    and not self.reaction_processed)
            ),
            "intimacy_level": self.intimacy_level,
            "tension_level": self.tension_level,
            "open_hook": self.open_hook,
            "desired_direction": self.desired_direction,
        }


def scene_from_row(row: dict[str, Any] | None) -> SceneState:
    if not row:
        return SceneState()
    payload = {
        key: value
        for key, value in dict(row).items()
        if key not in {"fan_id", "creator_id", "created_at"}
    }
    try:
        return SceneState.model_validate(payload)
    except Exception:
        # A stored scene that no longer validates is not worth failing a reply
        # over. Starting a fresh scene loses continuity; refusing to answer
        # loses the conversation.
        return SceneState()


# What "he asked for more, right now" actually looks like. Deliberately narrow:
# this is the ONE thing that lets the scene skip ahead after an unlock, so it
# has to mean he genuinely asked, not that he sounded enthusiastic.
_WANTS_MORE_PATTERNS = (
    r"\bsend (?:me )?(?:more|another|the next|something else)\b",
    r"\bwhat else (?:do you|have you|you)\b",
    r"\bgot (?:any(?:thing)? )?more\b",
    r"\bmore(?: of)? (?:that|this|those|it)\b",
    r"\bcan i (?:see|get|have) more\b",
    r"\bi want more\b",
    r"\bnext one\b",
    r"\bshow me more\b",
)

_POSITIVE_REACTION = (
    r"\b(fuck|god|damn|jesus|wow|holy)\b",
    r"\b(love|loved|loving|amazing|perfect|gorgeous|beautiful|incredible|hot|sexy|insane)\b",
    r"\b(so good|so hot|thank you|thanks)\b",
    r"[😍🥵🔥😈🤤❤️]",
)

_NEGATIVE_REACTION = (
    r"\b(disappoint\w*|expected more|not worth|too short|blurry|meh|waste)\b",
    r"\b(refund|rip off|ripoff|scam)\b",
)


def _search_any(patterns: tuple[str, ...], text: str) -> bool:
    import re

    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


def classify_fan_reaction(message: str, situation: dict[str, Any] | None = None) -> FanReaction:
    """Read what he did with the thing he just unlocked.

    Deterministic on purpose. The analyzer's opinion is consulted only for the
    "wants more" case, where it has a structured field of its own; everything
    else is read from his actual words so a fabricated analysis cannot invent
    a reaction that never happened.
    """
    text = str(message or "").strip()
    situation = situation or {}
    if not text:
        return FanReaction.NONE

    if _search_any(_WANTS_MORE_PATTERNS, text):
        return FanReaction.WANTS_MORE
    if _search_any(_NEGATIVE_REACTION, text):
        return FanReaction.NEGATIVE
    if _search_any(_POSITIVE_REACTION, text):
        return FanReaction.POSITIVE
    return FanReaction.NEUTRAL


def _clamp_level(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, min(MAX_LEVEL, parsed))


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, SceneState):
        return value.to_context()
    if isinstance(value, dict):
        return dict(value)
    return {}


def _beat(value: Any) -> ExperienceBeat:
    try:
        return ExperienceBeat(str(value))
    except ValueError:
        return ExperienceBeat.SETUP


def _reaction(value: Any) -> FanReaction:
    try:
        return FanReaction(str(value))
    except ValueError:
        return FanReaction.NONE


def _text(value: Any, limit: int = 240) -> str:
    return " ".join(str(value or "").split())[:limit]


#: Commercial actions that mean "a locked unlock is going out on this turn".
_DELIVERY_ACTIONS = {"SEND_NEXT_PPV_STEP"}
#: Commercial actions that mean "one unlock is on the table, unresolved".
_OFFER_ACTIONS = {"OFFER_NEXT_UNLOCK", "RESUME_PREVIOUS_OFFER"}
#: Commercial actions that end a scene rather than advance it.
_CLOSING_ACTIONS = {
    "HAND_OFF_TO_HUMAN",
    "PAUSE_UNTIL_PAYDAY",
    "PAUSE_NO_BUDGET",
}


def advance_scene(
    *,
    previous: dict[str, Any] | SceneState | None = None,
    situation: dict[str, Any] | None = None,
    commercial_decision: dict[str, Any] | None = None,
    latest_fan_message: str = "",
    scene_metadata: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> SceneState:
    """Advance the scene by one fan turn.

    Pure. Everything it needs is an argument, and it returns the next
    ``SceneState`` without touching the database — so the whole choreography is
    testable as a table of turns, which is what ``tests/test_experience_director.py``
    does.

    ``scene_metadata`` is approved catalog metadata for the set most recently
    unlocked or on the table (``services/scene_metadata.py``). It is the ONLY
    source of premise/continuation text, because the alternative is a writer
    inventing what is in media it has never seen.
    """
    current = now or datetime.now(timezone.utc)
    prev = _mapping(previous)
    situation = situation or {}
    decision = commercial_decision or {}
    metadata = scene_metadata or {}

    beat = _beat(prev.get("beat"))
    scene_key = _text(prev.get("scene_key"), 120)
    premise = _text(prev.get("premise"))
    last_set_id = prev.get("last_unlocked_set_id") or None
    last_description = _text(prev.get("last_unlocked_description"))
    reaction = _reaction(prev.get("last_fan_reaction"))
    reaction_processed = bool(prev.get("reaction_processed"))
    intimacy = _clamp_level(prev.get("intimacy_level"))
    tension = _clamp_level(prev.get("tension_level"))
    open_hook = _text(prev.get("open_hook"))
    desired = _text(prev.get("desired_direction"), 160)
    beats_in_scene = max(0, int(prev.get("beats_in_scene") or 0))
    turns_since_unlock = max(0, int(prev.get("turns_since_unlock") or 0))
    unlocks_in_scene = max(0, int(prev.get("unlocks_in_scene") or 0))

    action = str(decision.get("action") or "")
    if hasattr(decision.get("action"), "value"):
        action = str(decision["action"].value)
    purchase_signal = str(situation.get("purchase_signal") or "none").lower()
    energy = str(situation.get("conversation_energy") or "").lower()
    wants_explicit = str(situation.get("wants_explicit") or "").lower() in {"true", "1", "yes"}
    wants_media = str(situation.get("wants_media") or "").lower() in {"true", "1", "yes"}
    current_desired = _text(situation.get("desired_experience"), 160)

    reason = "scene_continues"

    # ---- Intensity dials ------------------------------------------------
    # Small, bounded and reversible. They describe the conversation; nothing is
    # optimised against them and no offer is gated on them.
    if wants_explicit:
        intimacy = min(MAX_LEVEL, intimacy + 1)
    if energy == "rising" or wants_media:
        tension = min(MAX_LEVEL, tension + 1)
    elif energy == "dropping":
        tension = max(0, tension - 1)
        intimacy = max(0, intimacy - 1)

    if current_desired:
        desired = current_desired

    # ---- A confirmed unlock resets the scene to AWAIT_REACTION ----------
    # This is the whole point of the redesign: a purchase does not return the
    # conversation to offer discovery, it hands the next move to him.
    unlocked_now = purchase_signal == "bought" or bool(
        decision.get("_unlock_confirmed")
    )
    if unlocked_now:
        set_id = str(decision.get("accepted_offer_set_id") or "") or last_set_id
        return SceneState(
            beat=ExperienceBeat.AWAIT_REACTION,
            previous_beat=beat,
            scene_key=_text(metadata.get("scene_key") or scene_key, 120),
            premise=_text(metadata.get("premise") or premise),
            last_unlocked_set_id=set_id,
            last_unlocked_description=_text(
                metadata.get("reveals") or metadata.get("description") or last_description
            ),
            last_fan_reaction=FanReaction.NONE,
            reaction_processed=False,
            intimacy_level=max(intimacy, _clamp_level(metadata.get("intensity_level"))),
            tension_level=tension,
            open_hook="",
            desired_direction=desired,
            another_unlock_ready=False,
            beats_in_scene=beats_in_scene + 1,
            turns_since_unlock=0,
            unlocks_in_scene=unlocks_in_scene + 1,
            transition_reason="unlock_confirmed_await_reaction",
            updated_at=current,
        )

    # ---- Safety and commercial closure ---------------------------------
    if action in _CLOSING_ACTIONS:
        return SceneState(
            beat=ExperienceBeat.CLOSE,
            previous_beat=beat,
            scene_key=scene_key,
            premise=premise,
            last_unlocked_set_id=last_set_id,
            last_unlocked_description=last_description,
            last_fan_reaction=reaction,
            reaction_processed=True,
            intimacy_level=intimacy,
            tension_level=tension,
            open_hook="",
            desired_direction=desired,
            another_unlock_ready=False,
            beats_in_scene=beats_in_scene + 1,
            turns_since_unlock=turns_since_unlock + 1,
            unlocks_in_scene=unlocks_in_scene,
            transition_reason=f"scene_closed_by_{action.lower() or 'policy'}",
            updated_at=current,
        )

    # ---- He is reacting to what he unlocked ----------------------------
    if beat in _POST_UNLOCK_BEATS:
        turns_since_unlock += 1
        observed = classify_fan_reaction(latest_fan_message, situation)
        if beat is ExperienceBeat.AWAIT_REACTION:
            # His first message after the unlock IS the reaction. Record it and
            # move into playing around it — the reply owes that specific thing
            # an answer, which writer_context carries as reaction_owed.
            reaction = observed if observed is not FanReaction.NONE else FanReaction.NEUTRAL
            reaction_processed = False
            beat = ExperienceBeat.PLAY
            reason = "fan_reacted_to_unlock"
        elif observed is not FanReaction.NONE:
            reaction = observed
            reaction_processed = True

        if reaction is FanReaction.WANTS_MORE or (wants_media and turns_since_unlock >= 1):
            # He asked. That IS the bridge; no message count is required.
            beat = ExperienceBeat.BRIDGE
            reason = "fan_asked_for_more"
            open_hook = _text(metadata.get("continuation") or open_hook)
            return _finalize(
                beat=beat,
                previous_beat=_beat(prev.get("beat")),
                scene_key=scene_key,
                premise=premise,
                last_set_id=last_set_id,
                last_description=last_description,
                reaction=reaction,
                reaction_processed=True,
                intimacy=intimacy,
                tension=tension,
                open_hook=open_hook,
                desired=desired,
                ready=True,
                beats_in_scene=beats_in_scene + 1,
                turns_since_unlock=turns_since_unlock,
                unlocks_in_scene=unlocks_in_scene,
                reason=reason,
                now=current,
            )

        if beat is ExperienceBeat.PLAY and reaction_processed and _bridge_emerged(
            reaction=reaction,
            tension=tension,
            intimacy=intimacy,
            # THIS turn's stated direction, not the scene's sticky one: a
            # direction he named before the purchase is not the conversation
            # producing a bridge after it.
            desired=current_desired,
            metadata=metadata,
        ):
            beat = ExperienceBeat.BRIDGE
            reason = "scene_produced_a_bridge"
            open_hook = _text(metadata.get("continuation") or open_hook)
            return _finalize(
                beat=beat,
                previous_beat=_beat(prev.get("beat")),
                scene_key=scene_key,
                premise=premise,
                last_set_id=last_set_id,
                last_description=last_description,
                reaction=reaction,
                reaction_processed=True,
                intimacy=intimacy,
                tension=tension,
                open_hook=open_hook,
                desired=desired,
                ready=True,
                beats_in_scene=beats_in_scene + 1,
                turns_since_unlock=turns_since_unlock,
                unlocks_in_scene=unlocks_in_scene,
                reason=reason,
                now=current,
            )

        return _finalize(
            beat=beat,
            previous_beat=_beat(prev.get("beat")),
            scene_key=scene_key,
            premise=premise,
            last_set_id=last_set_id,
            last_description=last_description,
            reaction=reaction,
            reaction_processed=reaction_processed,
            intimacy=intimacy,
            tension=tension,
            open_hook=open_hook,
            desired=desired,
            ready=False,
            beats_in_scene=beats_in_scene + 1,
            turns_since_unlock=turns_since_unlock,
            unlocks_in_scene=unlocks_in_scene,
            reason=reason,
            now=current,
        )

    # ---- Ordinary forward motion ---------------------------------------
    if action in _DELIVERY_ACTIONS:
        beat = ExperienceBeat.OFFER
        reason = "unlock_being_delivered"
    elif action in _OFFER_ACTIONS:
        beat = ExperienceBeat.OFFER
        reason = "offer_on_the_table"
    elif beat is ExperienceBeat.OFFER and action:
        # The offer resolved into something that is not a purchase and not an
        # offer: he said no, or the conversation moved on. Back to building.
        beat = ExperienceBeat.BUILD
        reason = "offer_resolved_without_unlock"
    elif beat in {ExperienceBeat.SETUP, ExperienceBeat.CLOSE} and (
        wants_explicit or wants_media or tension >= 2 or intimacy >= 2
    ):
        beat = ExperienceBeat.BUILD
        reason = "momentum_started_a_scene"
    elif beat is ExperienceBeat.BRIDGE and action and action not in _OFFER_ACTIONS:
        # A bridge that produced no offer stays a bridge for one more beat and
        # then relaxes, rather than latching "ready" forever.
        if beats_in_scene - max(0, int(prev.get("beats_in_scene") or 0)) >= 0 and turns_since_unlock >= 2:
            beat = ExperienceBeat.PLAY
            reason = "bridge_did_not_produce_an_offer"

    if not scene_key and (metadata.get("scene_key") or desired):
        scene_key = _text(metadata.get("scene_key") or desired, 120)
    if not premise and metadata.get("premise"):
        premise = _text(metadata.get("premise"))
    if metadata.get("setup") and beat in {ExperienceBeat.BUILD, ExperienceBeat.BRIDGE} and not open_hook:
        open_hook = _text(metadata.get("setup"))

    ready = beat in {ExperienceBeat.BRIDGE, ExperienceBeat.BUILD, ExperienceBeat.OFFER}
    if unlocks_in_scene and beat is ExperienceBeat.BUILD and turns_since_unlock < 2:
        # A scene that has already paid once does not go straight back to
        # building an offer on the very next turn.
        ready = False
        reason = "post_unlock_scene_still_settling"

    return _finalize(
        beat=beat,
        previous_beat=_beat(prev.get("beat")),
        scene_key=scene_key,
        premise=premise,
        last_set_id=last_set_id,
        last_description=last_description,
        reaction=reaction,
        reaction_processed=True,
        intimacy=intimacy,
        tension=tension,
        open_hook=open_hook,
        desired=desired,
        ready=ready,
        beats_in_scene=beats_in_scene + 1,
        turns_since_unlock=turns_since_unlock + (1 if unlocks_in_scene else 0),
        unlocks_in_scene=unlocks_in_scene,
        reason=reason,
        now=current,
    )


def _bridge_emerged(
    *,
    reaction: FanReaction,
    tension: int,
    intimacy: int,
    desired: str,
    metadata: dict[str, Any],
) -> bool:
    """Has the scene itself produced a natural next moment?

    This is the replacement for "wait exactly N messages". A bridge is dialogue
    doing something — he is still hot and pulling the scene forward, or he has
    named a direction, or the approved catalog says this set genuinely continues
    into another one. A flat "cool thanks" is none of those and produces no
    bridge no matter how many turns pass.
    """
    if reaction is FanReaction.NEGATIVE:
        return False
    if reaction is FanReaction.WANTS_MORE:
        return True
    if desired:
        return True
    if metadata.get("continuation") and reaction is FanReaction.POSITIVE:
        return True
    return reaction is FanReaction.POSITIVE and (tension >= 3 or intimacy >= 3)


def _finalize(**kwargs: Any) -> SceneState:
    return SceneState(
        beat=kwargs["beat"],
        previous_beat=kwargs["previous_beat"],
        scene_key=kwargs["scene_key"],
        premise=kwargs["premise"],
        last_unlocked_set_id=kwargs["last_set_id"],
        last_unlocked_description=kwargs["last_description"],
        last_fan_reaction=kwargs["reaction"],
        reaction_processed=kwargs["reaction_processed"],
        intimacy_level=kwargs["intimacy"],
        tension_level=kwargs["tension"],
        open_hook=kwargs["open_hook"],
        desired_direction=kwargs["desired"],
        another_unlock_ready=kwargs["ready"],
        beats_in_scene=kwargs["beats_in_scene"],
        turns_since_unlock=kwargs["turns_since_unlock"],
        unlocks_in_scene=kwargs["unlocks_in_scene"],
        transition_reason=kwargs["reason"],
        updated_at=kwargs["now"],
    )
