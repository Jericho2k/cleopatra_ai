"""One video-level semantic record, built from several chronological batches.

The problem with one contact sheet
----------------------------------
Twelve frames flattened into a single 896px square gives each moment about a
200-pixel thumbnail, and a vision model asked to describe that square answers
about the square: "a woman in various states of undress in a bedroom". That
sentence is true of most of the vault. It cannot say what changed, when, or in
what order — which is precisely the commercial content of a long clip.

So a long sample is processed as several SMALL chronological sheets (four
frames each by default), each classified on its own, and the per-batch results
are then combined here into one record describing the whole clip.

The combination is deterministic
--------------------------------
No second model call. Everything below is set algebra and ordering over facts
the per-batch classifier already reported: the first batch's wardrobe is the
beginning, the last batch's is the ending, a value that appears in batch 3 and
not batch 1 is a change at that point in time. A cheap structured pass would
add cost, latency and a second opportunity to invent something; the ordering
information is already present and only needs to be read.

Nothing here invents content
----------------------------
Every field is derived from a batch observation or omitted. There is no
"probably", no genre prior, and no filename. If the batches only ever reported
a bedroom, the record says bedroom and nothing about what might have happened
off-camera. An empty field is a correct answer and is left empty.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable

# Frames per contact sheet. Four is a 2x2 grid at ~440px a cell — large enough
# for wardrobe, props and setting to survive, small enough that the model is
# describing four moments rather than a mosaic.
DEFAULT_FRAMES_PER_SHEET = 4

# The ceiling on sheets, and therefore on VLM calls, for one video. With the
# frame ceiling at 16 this is never binding by default; it exists so that
# raising the frame count by configuration cannot silently multiply the bill.
DEFAULT_MAX_SHEETS = 4

# Nudity, weakest to strongest. The index is what makes "increased nudity"
# a comparison rather than a guess.
NUDITY_ORDER = ("none", "partial", "full")

_EMPTY_TEXT = {"", "unknown", "unclear", "none", "n/a", "na", "null"}


def _clean(value: Any, *, limit: int = 200) -> str:
    text = " ".join(str(value or "").split())[:limit].strip()
    return "" if text.lower() in _EMPTY_TEXT else text


def _clean_list(value: Any, *, limit: int = 12) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        value = [value]
    seen: list[str] = []
    try:
        iterator: Iterable[Any] = value
    except TypeError:
        return []
    for entry in iterator:
        text = _clean(entry, limit=80)
        if text and text.lower() not in {s.lower() for s in seen}:
            seen.append(text)
        if len(seen) >= limit:
            break
    return seen


def _timestamp(seconds: float) -> str:
    """``m:ss``, so a progression reads as a timeline rather than as floats."""
    total = max(int(round(float(seconds or 0))), 0)
    return f"{total // 60}:{total % 60:02d}"


# ---------------------------------------------------------------------------
# Chronological batching
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FrameBatch:
    """One contiguous slice of the sample, in order."""

    index: int
    frames: list[bytes]
    offsets_seconds: list[float]

    @property
    def start_seconds(self) -> float:
        return float(self.offsets_seconds[0]) if self.offsets_seconds else 0.0

    @property
    def end_seconds(self) -> float:
        return float(self.offsets_seconds[-1]) if self.offsets_seconds else 0.0

    @property
    def label(self) -> str:
        if not self.offsets_seconds:
            return ""
        if len(self.offsets_seconds) == 1:
            return _timestamp(self.start_seconds)
        return f"{_timestamp(self.start_seconds)}–{_timestamp(self.end_seconds)}"


def chronological_batches(
    frames: list[bytes],
    offsets_seconds: list[float],
    *,
    frames_per_sheet: int = DEFAULT_FRAMES_PER_SHEET,
    max_sheets: int = DEFAULT_MAX_SHEETS,
) -> list[FrameBatch]:
    """Split a sample into consecutive batches, oldest first.

    When the sample would need more batches than ``max_sheets``, the batches
    grow rather than the sample being truncated: dropping the tail would throw
    away the END of the video, which is the part a progression most needs.
    """
    usable = min(len(frames), len(offsets_seconds))
    if usable <= 0:
        return []
    per_sheet = max(int(frames_per_sheet), 1)
    ceiling = max(int(max_sheets), 1)
    if math.ceil(usable / per_sheet) > ceiling:
        per_sheet = math.ceil(usable / ceiling)

    batches: list[FrameBatch] = []
    for index, start in enumerate(range(0, usable, per_sheet)):
        stop = min(start + per_sheet, usable)
        batches.append(
            FrameBatch(
                index=index,
                frames=list(frames[start:stop]),
                offsets_seconds=[float(o) for o in offsets_seconds[start:stop]],
            )
        )
    return batches


# ---------------------------------------------------------------------------
# One batch's observation
# ---------------------------------------------------------------------------


@dataclass
class BatchObservation:
    """The visual facts one contact sheet produced, with its place in time."""

    index: int
    start_seconds: float
    end_seconds: float
    frames: int = 0
    description: str = ""
    scene_location: str = ""
    scene_outfit: str = ""
    scene_lighting: str = ""
    nudity: str = "none"
    explicitness: int = 0
    category: str = ""
    visible_anatomy: list[str] = field(default_factory=list)
    sexual_activity: list[str] = field(default_factory=list)
    action: str = ""
    props: list[str] = field(default_factory=list)
    wardrobe_items: list[str] = field(default_factory=list)
    continuity_markers: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    confidence: float = 0.0

    @property
    def label(self) -> str:
        if len(set((self.start_seconds, self.end_seconds))) == 1:
            return _timestamp(self.start_seconds)
        return f"{_timestamp(self.start_seconds)}–{_timestamp(self.end_seconds)}"

    @property
    def nudity_rank(self) -> int:
        try:
            return NUDITY_ORDER.index(self.nudity)
        except ValueError:
            return 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "at": self.label,
            "start_seconds": round(self.start_seconds, 2),
            "end_seconds": round(self.end_seconds, 2),
            "frames": self.frames,
            "scene_location": self.scene_location,
            "scene_outfit": self.scene_outfit,
            "nudity": self.nudity,
            "explicitness": self.explicitness,
            "action": self.action,
            "sexual_activity": list(self.sexual_activity),
            "props": list(self.props),
            "description": self.description,
        }


def observation_from_classification(
    data: dict[str, Any],
    *,
    batch: FrameBatch,
    frames_used: int | None = None,
) -> BatchObservation:
    """Read one batch's classification result into a positioned observation.

    Deliberately tolerant: the classifier's optional fields are optional, and a
    batch that only produced local NudeNet evidence still contributes its
    nudity and anatomy to the progression.
    """
    metadata = data.get("classification_metadata")
    source: dict[str, Any] = dict(data)
    if isinstance(metadata, dict):
        # A persisted classification keeps the visual detail one level down.
        source = {**metadata, **{k: v for k, v in data.items() if v not in (None, "")}}

    nudity = _clean(source.get("nudity")).lower()
    if nudity not in NUDITY_ORDER:
        nudity = "none"
    try:
        explicitness = int(source.get("explicitness") or 0)
    except (TypeError, ValueError):
        explicitness = 0
    try:
        confidence = float(source.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0

    return BatchObservation(
        index=batch.index,
        start_seconds=batch.start_seconds,
        end_seconds=batch.end_seconds,
        frames=int(frames_used if frames_used is not None else len(batch.frames)),
        description=_clean(source.get("description"), limit=900),
        scene_location=_clean(source.get("scene_location"), limit=120),
        scene_outfit=_clean(source.get("scene_outfit"), limit=160),
        scene_lighting=_clean(source.get("scene_lighting"), limit=120),
        nudity=nudity,
        explicitness=max(0, min(explicitness, 5)),
        category=_clean(source.get("category") or source.get("content_category")),
        visible_anatomy=_clean_list(source.get("visible_anatomy"), limit=8),
        sexual_activity=_clean_list(source.get("sexual_activity"), limit=8),
        action=_clean(source.get("action"), limit=160),
        props=_clean_list(source.get("props"), limit=10),
        wardrobe_items=_clean_list(source.get("wardrobe_items"), limit=10),
        continuity_markers=_clean_list(source.get("continuity_markers"), limit=8),
        tags=_clean_list(source.get("tags"), limit=16),
        confidence=round(max(0.0, min(confidence, 1.0)), 3),
    )


# ---------------------------------------------------------------------------
# Combining batches into one video-level record
# ---------------------------------------------------------------------------


def _ordered_progression(
    values: list[str],
    *,
    keep_literal: bool = False,
) -> list[str]:
    """Collapse consecutive repeats, keeping first-seen order.

    ``[clothed, clothed, lingerie, lingerie, nude]`` becomes
    ``[clothed, lingerie, nude]`` — the shape of the change, not a transcript
    of every sample. A value that returns later is not re-added, so a record
    reads as a progression rather than oscillating on classifier noise.

    ``keep_literal`` disables the "unknown means absent" filter. Nudity needs
    it: ``none`` is a REAL state and the most important one to keep, because a
    progression that silently drops it starts at "partial" and loses the fact
    that the clip opened clothed.
    """
    progression: list[str] = []
    for value in values:
        text = (
            " ".join(str(value or "").split())[:160].strip()
            if keep_literal
            else _clean(value, limit=160)
        )
        if not text:
            continue
        if progression and progression[-1].lower() == text.lower():
            continue
        if any(entry.lower() == text.lower() for entry in progression):
            continue
        progression.append(text)
    return progression


def _dominant(values: list[str]) -> str:
    """The most frequent non-empty value, earliest wins a tie."""
    counts: dict[str, int] = {}
    order: list[str] = []
    for value in values:
        text = _clean(value, limit=160)
        if not text:
            continue
        key = text.lower()
        if key not in counts:
            counts[key] = 0
            order.append(text)
        counts[key] += 1
    if not order:
        return ""
    return max(order, key=lambda entry: (counts[entry.lower()], -order.index(entry)))


def combine_batch_observations(
    observations: list[BatchObservation],
    *,
    duration_seconds: float = 0.0,
) -> dict[str, Any]:
    """Fold chronological batch observations into one video-level record.

    Returns ``{}`` for no observations, so a caller can tell "nothing was
    sampled" from "sampled and found nothing", which are different facts and
    lead to different classification statuses.
    """
    ordered = sorted(observations, key=lambda o: (o.start_seconds, o.index))
    if not ordered:
        return {}

    first, last = ordered[0], ordered[-1]
    middle = ordered[len(ordered) // 2]

    wardrobe = _ordered_progression([o.scene_outfit for o in ordered])
    nudity_states = _ordered_progression(
        [o.nudity for o in ordered], keep_literal=True
    )
    locations = _ordered_progression([o.scene_location for o in ordered])
    activities: list[str] = []
    for observation in ordered:
        for activity in [*observation.sexual_activity, observation.action]:
            text = _clean(activity, limit=160)
            if text and not any(e.lower() == text.lower() for e in activities):
                activities.append(text)

    # Where something actually changed, with the time it changed at. This is
    # the part a single sheet can never produce.
    scene_changes: list[dict[str, Any]] = []
    for previous, current in zip(ordered, ordered[1:]):
        changed: list[str] = []
        if (
            current.scene_location
            and current.scene_location.lower() != previous.scene_location.lower()
        ):
            changed.append("setting")
        if (
            current.scene_outfit
            and current.scene_outfit.lower() != previous.scene_outfit.lower()
        ):
            changed.append("wardrobe")
        if current.nudity_rank != previous.nudity_rank:
            changed.append(
                "more nudity"
                if current.nudity_rank > previous.nudity_rank
                else "less nudity"
            )
        new_props = [
            prop
            for prop in current.props
            if not any(prop.lower() == known.lower() for known in previous.props)
        ]
        if new_props:
            changed.append("props")
        if not changed:
            continue
        scene_changes.append(
            {
                "at": _timestamp(current.start_seconds),
                "at_seconds": round(current.start_seconds, 2),
                "changed": changed,
                "scene_location": current.scene_location,
                "scene_outfit": current.scene_outfit,
                "nudity": current.nudity,
                "new_props": new_props,
            }
        )

    peak = max(ordered, key=lambda o: (o.explicitness, o.nudity_rank))
    props: list[str] = []
    continuity: list[str] = []
    tags: list[str] = []
    for observation in ordered:
        for target, values in (
            (props, observation.props),
            (continuity, observation.continuity_markers),
            (tags, observation.tags),
        ):
            for value in values:
                if not any(value.lower() == known.lower() for known in target):
                    target.append(value)

    return {
        "duration_seconds": round(max(float(duration_seconds or 0.0), 0.0), 2),
        "duration_label": _timestamp(duration_seconds),
        "sampled_frames": sum(o.frames for o in ordered),
        "batches": len(ordered),
        # The headline: what the clip DOES, over time.
        "progression": wardrobe or nudity_states,
        "wardrobe_progression": wardrobe,
        "nudity_progression": nudity_states,
        "explicitness_progression": [o.explicitness for o in ordered],
        "setting": _dominant([o.scene_location for o in ordered]),
        "setting_progression": locations,
        "beginning": first.description or first.scene_outfit,
        "middle": middle.description or middle.scene_outfit,
        "ending": last.description or last.scene_outfit,
        "beginning_at": first.label,
        "ending_at": last.label,
        "scene_changes": scene_changes,
        "activities": activities,
        "props": props[:12],
        "continuity_markers": continuity[:10],
        "tags": tags[:24],
        "peak_explicitness": peak.explicitness,
        "peak_nudity": peak.nudity,
        "peak_at": peak.label,
        "content_category": peak.category,
        "visible_anatomy": _ordered_progression(
            [a for o in ordered for a in o.visible_anatomy]
        ),
        "timeline": [o.to_dict() for o in ordered],
    }


def describe_video_record(record: dict[str, Any]) -> str:
    """The video-level record as prose, for the stored ``ai_description``.

    Every sentence is conditional on having the fact. A clip whose batches all
    reported the same bedroom and the same outfit produces two short sentences,
    which is the honest description of that clip.
    """
    if not record:
        return ""

    sentences: list[str] = []
    duration = _clean(record.get("duration_label"))
    frames = int(record.get("sampled_frames") or 0)
    batches = int(record.get("batches") or 0)
    if duration and duration != "0:00" and frames:
        sentences.append(
            f"A {duration} video, sampled at {frames} points across its full "
            f"length in {batches} chronological group"
            f"{'s' if batches != 1 else ''}."
        )

    setting = _clean(record.get("setting"))
    setting_progression = record.get("setting_progression") or []
    if len(setting_progression) > 1:
        sentences.append(
            "The setting moves through " + " then ".join(setting_progression) + "."
        )
    elif setting:
        sentences.append(f"The setting is {setting} throughout.")

    progression = record.get("progression") or []
    if len(progression) > 1:
        sentences.append("Progression: " + " → ".join(progression) + ".")
    elif progression:
        sentences.append(f"Wardrobe and nudity stay {progression[0]} throughout.")

    beginning = _clean(record.get("beginning"), limit=320)
    ending = _clean(record.get("ending"), limit=320)
    if beginning:
        sentences.append(f"It opens with {beginning[0].lower()}{beginning[1:]}"
                         if beginning[:1].isupper() else f"It opens with {beginning}")
    if ending and ending.lower() != beginning.lower():
        sentences.append(f"It ends with {ending[0].lower()}{ending[1:]}"
                         if ending[:1].isupper() else f"It ends with {ending}")

    activities = record.get("activities") or []
    if activities:
        sentences.append("Visible activity: " + ", ".join(activities[:6]) + ".")

    props = record.get("props") or []
    if props:
        sentences.append("Visible props: " + ", ".join(props[:6]) + ".")

    changes = record.get("scene_changes") or []
    if changes:
        moments = [
            f"{entry.get('at')} ({', '.join(entry.get('changed') or [])})"
            for entry in changes[:4]
        ]
        sentences.append("Meaningful changes at " + "; ".join(moments) + ".")

    return " ".join(sentence.rstrip(".") + "." for sentence in sentences if sentence)


def commercial_role(record: dict[str, Any]) -> str:
    """Where this clip belongs in a session, from what was actually seen.

    Four values, matching the ``good_for`` contract the vault already stores.
    A clip that escalates is a closer; one that stays clothed is an opener; a
    single-state explicit clip stands alone.
    """
    if not record:
        return "standalone"
    peak = int(record.get("peak_explicitness") or 0)
    states = len(record.get("progression") or [])
    if peak >= 4 and states > 1:
        return "closer"
    if peak >= 4:
        return "standalone"
    if peak <= 1:
        return "opener"
    return "mid_session"
