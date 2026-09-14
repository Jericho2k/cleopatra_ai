"""Media as beats in an experience, not rows in an inventory.

WHY THIS IS GENERATED, NOT ASKED FOR AT WRITE TIME
--------------------------------------------------
A writer asked "what does this set continue into?" in the middle of a
conversation will answer, fluently, about media it has never seen. That is how
a photo set becomes "the video where I finally take it off" in a message the
fan then pays for. So the experience metadata is derived ONCE, during catalog
classification, from the classifier's own structured facts about the exact
media in the set — and the writer is only ever handed the result.

Five small facts turn a row into a beat:

    scene_key       which scene this belongs to, so continuations can be found
    scene_premise   what the scene IS, in approved words
    intensity_level where it sits on the escalation ladder (0-5)
    reveals         what unlocking it actually shows or advances
    setup_line      how it can honestly be led into
    continuation    where it can honestly lead next

Every one of them is a restatement of classified facts. Nothing here invents a
body part, an act, a format, or a promise, and ``continuation`` is a direction
rather than a named next product: the fan is never told a sequence exists.
"""

from __future__ import annotations

from typing import Any, Iterable

from models.content_pricing import is_paid_sellable, normalize_category
from services.vault_metadata import clean_text, normalize_string_list, useful_text

#: Explicitness levels, as the classifier writes them, described in the words a
#: setup or a continuation can be built out of. Deliberately vague at the top:
#: the approved description carries the specifics, and this carries the arc.
_INTENSITY_LANGUAGE: dict[int, tuple[str, str]] = {
    0: ("dressed and playful", "showing a little more"),
    1: ("teasing, still covered", "losing a layer"),
    2: ("stripped back to lingerie", "what is under it"),
    3: ("barely covered", "nothing left on"),
    4: ("undressed", "what happens once she is"),
    5: ("explicit", "the rest of it"),
}


def _level(value: Any) -> int:
    try:
        return max(0, min(5, int(value)))
    except (TypeError, ValueError):
        return 0


def scene_key_for(location: Any, outfit: Any, category: Any) -> str:
    """A stable identity for the scene a set belongs to.

    Location first, because a shoot is a place before it is an outfit: the
    bathroom set and the bathroom clip are the same scene and should bridge
    into each other, while the same lingerie in a different room is not.
    """
    parts = [
        clean_text(useful_text(location)).lower(),
        clean_text(useful_text(outfit)).lower(),
    ]
    parts = [part for part in parts if part]
    if not parts:
        parts = [normalize_category(category)]
    return ":".join(part.replace(" ", "_") for part in parts if part)[:120]


def derive_set_experience(
    *,
    location: Any = "",
    outfit: Any = "",
    category: Any = "",
    explicit_min: int = 0,
    explicit_max: int = 0,
    is_video: bool = False,
    tags: Iterable[str] | None = None,
) -> dict[str, Any]:
    """The experience metadata for one proposed set, from its own media.

    Pure, so catalog generation and the tests agree by construction. Every key
    it returns is a ``vault_sets`` column ``db/experience_director_v1.sql``
    created, so a caller merges the result straight into the insert payload.
    """
    tag_list = normalize_string_list(list(tags or []), limit=16)
    peak = _level(explicit_max or explicit_min)
    entry = _level(explicit_min or explicit_max)

    place = clean_text(useful_text(location))
    wearing = clean_text(useful_text(outfit))
    at_level, leads_to = _INTENSITY_LANGUAGE[peak]

    where = f"in the {place}" if place else ""
    what = f"in {wearing}" if wearing else ""
    premise = clean_text(
        f"{'clip' if is_video else 'photos'} {where} {what}".strip()
    ) or ("a private clip" if is_video else "a private set")

    reveals = clean_text(f"{at_level}{f' {where}' if where else ''}")
    setup_line = clean_text(
        f"what she was doing {where}".strip() if where else "what she was up to"
    )
    continuation = "" if peak >= 5 else clean_text(leads_to)

    # A set that climbs inside itself is its own escalation; say so, because it
    # is the difference between "here is a thing" and "watch this happen".
    if peak > entry:
        reveals = clean_text(f"{reveals}, building from {_INTENSITY_LANGUAGE[entry][0]}")

    return {
        "scene_key": scene_key_for(location, outfit, category),
        "scene_premise": premise[:240],
        "intensity_level": peak,
        "reveals": reveals[:240],
        "setup_line": setup_line[:240],
        "continuation": continuation[:240],
        "paid_sellable": is_paid_sellable(
            {
                "content_category": category,
                "tags": [*tag_list, normalize_category(category)],
            }
        ),
    }


def scene_metadata_from_set(row: dict[str, Any] | None) -> dict[str, Any]:
    """The approved scene facts for one vault set, in ``advance_scene`` shape.

    Falls back to the columns that already existed when a row predates
    ``db/experience_director_v1.sql``, so an unclassified vault still produces a
    coherent scene rather than an empty one. It never fabricates a continuation:
    a row with no stored continuation reports none, and ``_bridge_emerged`` then
    needs the conversation itself to earn the bridge.
    """
    if not row:
        return {}
    location = row.get("location") or row.get("scene_location")
    outfit = row.get("outfit") or row.get("scene_outfit")
    category = row.get("content_category") or ""
    stored_key = clean_text(row.get("scene_key"))
    return {
        "scene_key": stored_key or scene_key_for(location, outfit, category),
        "premise": clean_text(row.get("scene_premise"))
        or clean_text(row.get("title"))
        or clean_text(row.get("description"))[:240],
        "intensity_level": _level(
            row.get("intensity_level")
            if row.get("intensity_level") is not None
            else row.get("explicit_max")
        ),
        "reveals": clean_text(row.get("reveals")) or clean_text(row.get("description"))[:240],
        "setup": clean_text(row.get("setup_line")),
        "continuation": clean_text(row.get("continuation")),
        "paid_sellable": is_paid_sellable(row),
    }


def advances_the_interaction(
    candidate: dict[str, Any],
    scene: dict[str, Any] | None,
) -> float:
    """Does this set actually move the CURRENT interaction forward?

    Content selection already optimises scene continuity and explicitness. This
    is the third question the redesign adds: a set that repeats the beat he just
    unlocked is a worse next offer than one that answers the direction he has
    been pulling towards, even when both are equally explicit and equally "in
    scene". Returns a bonus, never a veto — approved bounds and the sellability
    predicate remain the only things that can exclude a row.
    """
    scene = scene or {}
    score = 0.0
    candidate_meta = scene_metadata_from_set(candidate)

    scene_key = str(scene.get("scene_key") or "")
    if scene_key and candidate_meta["scene_key"] == scene_key:
        score += 2.0

    last_level = _level(scene.get("intimacy_level"))
    if candidate_meta["intensity_level"] > last_level:
        score += 2.5
    elif candidate_meta["intensity_level"] < last_level:
        # Going backwards down the ladder is not a continuation.
        score -= 2.0

    direction = normalize_string_list(
        [scene.get("desired_direction"), scene.get("open_hook")], limit=8
    )
    haystack = " ".join(
        str(value or "").lower()
        for value in (
            candidate.get("title"),
            candidate.get("description"),
            candidate_meta["premise"],
            candidate_meta["reveals"],
            *(candidate.get("tags") or []),
        )
    )
    for phrase in direction:
        for token in phrase.split():
            if len(token) > 3 and token in haystack:
                score += 1.5
                break

    if str(candidate.get("id") or "") == str(scene.get("last_unlocked_set_id") or ""):
        score -= 10.0
    return score
