"""The Experience Director: one persistent scene per fan.

Reads the stored scene, advances it by one turn, writes it back, and hands the
writer a projection that carries no money. Everything decision-shaped lives in
``models/experience_director.py`` and is pure; this module is I/O and logging.

The gate this exists to provide is ``another_unlock_ready``. The commercial
orchestrator asks it before DISCOVERING a new offer, and only then: an offer
already on the table, an acceptance already made, and a delivery already
authorised are commercial facts that choreography has no business vetoing.
"""

from __future__ import annotations

import os
from typing import Any

from db.experience_director_queries import get_scene, save_scene
from models.experience_director import (
    ExperienceBeat,
    SceneState,
    advance_scene,
    scene_from_row,
)


def experience_director_enabled() -> bool:
    """Default ON.

    Unlike the older flag-gated planners, this is not an experiment layered on
    top of working behaviour — it is the replacement for the post-purchase
    cooldown that the one-unlock commercial model made unreachable. Shipping it
    off by default would mean shipping nothing. It can still be turned off in
    an incident without a deploy.
    """
    return os.getenv("EXPERIENCE_DIRECTOR_ENABLED", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


async def load_scene(fan_id: str) -> SceneState:
    if not experience_director_enabled():
        return SceneState()
    return scene_from_row(await get_scene(fan_id))


async def direct_experience(
    *,
    creator_id: str,
    fan_id: str,
    situation: dict[str, Any] | None = None,
    commercial_decision: dict[str, Any] | None = None,
    latest_fan_message: str = "",
    scene_metadata: dict[str, Any] | None = None,
) -> SceneState:
    """Advance and persist the scene for one fan turn."""
    if not experience_director_enabled():
        return SceneState()

    previous = await get_scene(fan_id)
    scene = advance_scene(
        previous=previous,
        situation=situation,
        commercial_decision=commercial_decision,
        latest_fan_message=latest_fan_message,
        scene_metadata=scene_metadata,
    )
    await _persist(creator_id=creator_id, fan_id=fan_id, scene=scene)
    return scene


async def record_unlock(
    *,
    creator_id: str,
    fan_id: str,
    set_id: str | None,
    description: str = "",
    scene_metadata: dict[str, Any] | None = None,
) -> SceneState:
    """A confirmed purchase moves the scene to AWAIT_REACTION.

    Called from the purchase-confirmation path rather than from the next
    message, because that is where the fact actually becomes true and because
    the one-step commercial session is cleared immediately afterwards — by the
    time he replies there is no session left to infer a purchase from.
    """
    if not experience_director_enabled():
        return SceneState()

    previous = await get_scene(fan_id)
    metadata = dict(scene_metadata or {})
    if description and not metadata.get("reveals"):
        metadata["reveals"] = description
    scene = advance_scene(
        previous=previous,
        commercial_decision={
            "_unlock_confirmed": True,
            "accepted_offer_set_id": set_id,
        },
        scene_metadata=metadata,
    )
    await _persist(creator_id=creator_id, fan_id=fan_id, scene=scene)
    print(
        f"[EXPERIENCE] fan={fan_id} unlock recorded set={set_id} "
        f"beat={scene.beat.value}"
    )
    return scene


async def _persist(*, creator_id: str, fan_id: str, scene: SceneState) -> None:
    try:
        await save_scene(creator_id=creator_id, fan_id=fan_id, scene=scene.to_context())
        print(
            f"[EXPERIENCE] fan={fan_id} beat={scene.beat.value} "
            f"reaction={scene.last_fan_reaction.value} "
            f"intimacy={scene.intimacy_level} tension={scene.tension_level} "
            f"unlock_ready={scene.another_unlock_ready} "
            f"reason={scene.transition_reason}"
        )
    except Exception as exc:
        # A scene that cannot be stored is a scene that restarts. That is a
        # continuity loss, not a reason to send nothing.
        print(f"[EXPERIENCE] persistence failed fan={fan_id}: {exc}")


def scene_allows_new_offer(scene: SceneState | dict[str, Any] | None) -> bool:
    """Whether the conversation has earned a NEW offer.

    False only in the narrow window this redesign is about: he has unlocked
    something in this scene and the scene still owes him interaction. It is
    never a hard message count — "send more" moves the scene to BRIDGE on the
    very next turn, and a flat reaction never does no matter how long he talks.
    """
    if scene is None:
        return True
    state = scene if isinstance(scene, SceneState) else scene_from_row(scene)
    if not state.unlocks_in_scene:
        # Nothing has been bought in this scene; the first offer was never
        # gated on choreography and is not gated now.
        return True
    if state.beat in {ExperienceBeat.AWAIT_REACTION, ExperienceBeat.PLAY}:
        return False
    return state.another_unlock_ready or state.beat is ExperienceBeat.BRIDGE


async def scene_metadata_for(creator_id: str, set_id: str | None) -> dict[str, Any]:
    """Approved scene facts for the set this turn revolves around, or ``{}``.

    The writer may only describe what is grounded here and in the media itself,
    so an empty result is a real answer: it means the scene has no approved
    premise to talk about and ``_bridge_emerged`` must be earned by the
    conversation instead of by catalog metadata.
    """
    if not set_id:
        return {}
    from db.experience_director_queries import get_set_scene_row
    from services.scene_metadata import scene_metadata_from_set

    return scene_metadata_from_set(await get_set_scene_row(creator_id, str(set_id)))
