"""Versioned, provider-neutral metadata helpers for creator vault media.

The vision provider is intentionally kept outside this module.  These helpers
own the deterministic contract consumed by set construction and commercial
package matching so changing providers cannot silently change the database
shape again.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any, Iterable


VAULT_CLASSIFIER_VERSION = 5

_EMPTY_VALUES = {"", "unknown", "unclear", "none", "n/a", "na", "null"}
_MEDIA_CATEGORIES = {
    "teaser_clothed",
    "teaser_bundle",
    "legs_feet",
    "lingerie_photo",
    "lingerie_video",
    "nude_photo",
    "nude_video",
    "striptease_video",
    "closeup_photo",
    "closeup_video",
    "dictate_video",
    "solo_toy_video",
    "solo_toy_photo",
    "explicit_photo",
    "explicit_video",
    "bg_content",
    "task",
    "other",
}


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def useful_text(value: Any) -> str:
    text = clean_text(value)
    return "" if text.lower() in _EMPTY_VALUES else text


def normalize_string_list(value: Any, *, limit: int = 16) -> list[str]:
    raw: Iterable[Any]
    if isinstance(value, str):
        raw = re.split(r"[,|]", value)
    elif isinstance(value, (list, tuple, set)):
        raw = value
    else:
        raw = []

    result: list[str] = []
    seen: set[str] = set()
    for item in raw:
        text = useful_text(item).lower().strip(" .,-_/|")
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
        if len(result) >= limit:
            break
    return result


def classification_confidence(value: Any, *, source: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = 0.5
    parsed = min(max(parsed, 0.0), 1.0)
    # A thumbnail or filename can be useful context, but it must never pretend
    # to be as authoritative as looking at the actual asset.
    caps = {"video_thumbnail": 0.72, "filename_album": 0.25}
    return round(min(parsed, caps.get(source, 1.0)), 3)


def semantic_tags(data: dict[str, Any]) -> list[str]:
    values: list[Any] = []
    for key in (
        "tags",
        "sexual_activity",
        "body_focus",
        "visible_anatomy",
        "props",
        "colors",
    ):
        value = data.get(key)
        if isinstance(value, list):
            values.extend(value)
        elif value:
            values.append(value)
    for key in (
        "mood",
        "action",
        "pose",
        "framing",
        "scene_location",
        "scene_outfit",
        "scene_lighting",
        "nudity",
    ):
        if data.get(key):
            values.append(data[key])
    return normalize_string_list(values, limit=24)


def explicitness_from_evidence(data: dict[str, Any]) -> int:
    """Normalize explicitness from both the score and structured evidence.

    Vision models sometimes choose a conservative numeric score while still
    returning unambiguous nudity/activity fields.  The deterministic contract
    must not silently downgrade those internally inconsistent results.
    """
    try:
        level = int(data.get("explicitness", 0))
    except (TypeError, ValueError):
        level = 0
    level = min(max(level, 0), 5)

    nudity = useful_text(data.get("nudity")).lower()
    anatomy = normalize_string_list(data.get("visible_anatomy"))
    activities = normalize_string_list(data.get("sexual_activity"))
    activities = [
        value
        for value in activities
        if value not in {"none", "no activity", "not visible"}
    ]

    if activities:
        level = max(level, 5)
    elif anatomy or nudity in {"partial", "full"}:
        level = max(level, 4)
    elif nudity == "implied":
        level = max(level, 1)
    return level


def normalize_media_category(
    category: Any,
    *,
    explicitness: int,
    is_video: bool,
) -> str:
    """Repair category/type/evidence contradictions deterministically."""
    result = str(category or "other")
    if result not in _MEDIA_CATEGORIES:
        result = "other"

    if is_video:
        result = {
            "lingerie_photo": "lingerie_video",
            "nude_photo": "nude_video",
            "closeup_photo": "closeup_video",
            "solo_toy_photo": "solo_toy_video",
            "explicit_photo": "explicit_video",
        }.get(result, result)
    else:
        result = {
            "lingerie_video": "lingerie_photo",
            "nude_video": "nude_photo",
            "closeup_video": "closeup_photo",
            "solo_toy_video": "solo_toy_photo",
            "explicit_video": "explicit_photo",
            "striptease_video": "lingerie_photo",
            "dictate_video": "task",
        }.get(result, result)

    if result in {
        "other",
        "teaser_clothed",
        "teaser_bundle",
        "lingerie_photo",
        "lingerie_video",
        "nude_photo",
        "nude_video",
    }:
        if explicitness >= 5:
            return "explicit_video" if is_video else "explicit_photo"
        if explicitness >= 4:
            return "nude_video" if is_video else "nude_photo"
        if explicitness == 3 and result in {
            "teaser_clothed",
            "teaser_bundle",
        }:
            return "lingerie_video" if is_video else "lingerie_photo"
    return result


def media_description(data: dict[str, Any], *, source: str) -> str:
    """Build a detailed factual description without exposing provider syntax."""
    sentences: list[str] = []
    summary = useful_text(data.get("description"))
    if summary:
        sentences.append(summary.rstrip(". ") + ".")

    if data.get("description_complete"):
        sentences.append(
            f"Classification evidence: {source.replace('_', ' ')}; "
            f"explicitness {int(data.get('explicitness') or 0)}/5."
        )
        return " ".join(sentences)[:1600]

    details: list[str] = []
    location = useful_text(data.get("scene_location"))
    outfit = useful_text(data.get("scene_outfit"))
    action = useful_text(data.get("action"))
    pose = useful_text(data.get("pose"))
    framing = useful_text(data.get("framing"))
    lighting = useful_text(data.get("scene_lighting"))
    colours = normalize_string_list(data.get("colors"), limit=6)
    nudity = useful_text(data.get("nudity"))
    anatomy = normalize_string_list(data.get("visible_anatomy"))
    if location:
        details.append(f"location: {location}")
    if outfit:
        details.append(f"outfit: {outfit}")
    if action:
        details.append(f"action: {action}")
    if pose:
        details.append(f"pose: {pose}")
    if framing:
        details.append(f"framing: {framing}")
    if lighting:
        details.append(f"lighting: {lighting}")
    if colours:
        details.append(f"dominant palette: {', '.join(colours)}")
    if nudity:
        details.append(f"nudity: {nudity}")
    if anatomy:
        details.append(f"visible anatomy: {', '.join(anatomy)}")
    if details:
        sentences.append("Visual details — " + "; ".join(details) + ".")

    tags = semantic_tags(data)
    if tags:
        sentences.append("Semantic tags: " + ", ".join(tags) + ".")
    sentences.append(
        f"Classification evidence: {source.replace('_', ' ')}; "
        f"explicitness {int(data.get('explicitness') or 0)}/5."
    )
    return " ".join(sentences)[:1600]


def _mode(values: Iterable[Any]) -> str:
    cleaned = [useful_text(value) for value in values]
    cleaned = [value for value in cleaned if value]
    return Counter(cleaned).most_common(1)[0][0] if cleaned else ""


def build_set_description(items: list[dict[str, Any]]) -> str:
    """Aggregate exact media metadata into a rich, writer-safe set summary."""
    if not items:
        return ""
    count = len(items)
    videos = sum(
        str(item.get("mimetype") or "").startswith("video") for item in items
    )
    location = _mode(item.get("scene_location") for item in items)
    outfit = _mode(item.get("scene_outfit") for item in items)
    lighting = _mode(item.get("scene_lighting") for item in items)
    categories = normalize_string_list(
        [item.get("content_category") or item.get("category") for item in items],
        limit=8,
    )
    tags = normalize_string_list(
        [tag for item in items for tag in (item.get("tags") or [])],
        limit=14,
    )
    metadata = [
        item.get("classification_metadata")
        if isinstance(item.get("classification_metadata"), dict)
        else {}
        for item in items
    ]
    local_fingerprints = [
        ((value.get("shoot_fingerprint") or {}).get("local") or {})
        for value in metadata
    ]
    palettes = normalize_string_list(
        [
            colour
            for fingerprint in local_fingerprints
            for colour in (fingerprint.get("palette_names") or [])
        ],
        limit=6,
    )
    visual_tones = normalize_string_list(
        [fingerprint.get("visual_tone") for fingerprint in local_fingerprints],
        limit=4,
    )
    structured = {
        key: normalize_string_list(
            [
                value
                for item_metadata in metadata
                for value in (
                    item_metadata.get(key)
                    if isinstance(item_metadata.get(key), list)
                    else [item_metadata.get(key)]
                )
            ],
            limit=10,
        )
        for key in (
            "action",
            "pose",
            "framing",
            "props",
            "nudity",
            "visible_anatomy",
            "sexual_activity",
            "body_focus",
        )
    }
    levels = [
        int(item.get("explicitness_level") if item.get("explicitness_level") is not None else item.get("explicitness", 0))
        for item in items
    ]

    media_label = f"{count} media item{'s' if count != 1 else ''}"
    if videos:
        media_label += f", including {videos} video{'s' if videos != 1 else ''}"
    visually_matched = sum(
        bool(
            (value.get("shoot_fingerprint") or {}).get("embedding")
        )
        for value in metadata
    )
    opening = (
        f"A visually matched photoshoot sequence of {media_label}"
        if visually_matched == count
        else f"A coherent sequence of {media_label}"
    )
    if location:
        opening += f" in a {location} setting"
    if outfit:
        opening += f", primarily featuring {outfit}"
    opening += "."

    parts = [opening]
    if min(levels) != max(levels):
        parts.append(
            f"The sequence progresses from explicitness {min(levels)}/5 to {max(levels)}/5."
        )
    else:
        parts.append(f"The sequence is consistently explicitness {levels[0]}/5.")
    if categories:
        parts.append("Content categories: " + ", ".join(categories) + ".")
    if palettes:
        parts.append("Dominant visual palette: " + ", ".join(palettes) + ".")
    setting_details = []
    if lighting:
        setting_details.append(f"lighting: {lighting}")
    if visual_tones:
        setting_details.append(f"visual tone: {', '.join(visual_tones)}")
    if structured["framing"]:
        setting_details.append(
            f"framing: {', '.join(structured['framing'])}"
        )
    if setting_details:
        parts.append("Photoshoot look — " + "; ".join(setting_details) + ".")
    content_details = []
    for label, key in (
        ("actions", "action"),
        ("poses", "pose"),
        ("props", "props"),
        ("nudity", "nudity"),
        ("visible anatomy", "visible_anatomy"),
        ("activities", "sexual_activity"),
        ("body focus", "body_focus"),
    ):
        if structured[key]:
            content_details.append(
                f"{label}: {', '.join(structured[key])}"
            )
    if content_details:
        parts.append("Visible progression — " + "; ".join(content_details) + ".")
    if tags:
        parts.append("Themes and visible details: " + ", ".join(tags) + ".")

    # Preserve useful item-level distinctions without repeating near-identical
    # classifier prose from every image in a shoot.
    summaries: list[str] = []
    seen: set[str] = set()
    for item in items:
        summary = useful_text(item.get("ai_description"))
        if not summary:
            continue
        summary = re.sub(r"\s+", " ", summary).strip()
        key = summary.lower()
        if key in seen:
            continue
        seen.add(key)
        summaries.append(summary[:320])
        if len(summaries) >= 3:
            break
    if summaries:
        parts.append("Representative contents: " + " ".join(summaries))
    return " ".join(parts)[:2000]
