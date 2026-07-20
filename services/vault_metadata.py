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


VAULT_CLASSIFIER_VERSION = 2

_EMPTY_VALUES = {"", "unknown", "unclear", "none", "n/a", "na", "null"}


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
    ):
        if data.get(key):
            values.append(data[key])
    return normalize_string_list(values, limit=24)


def media_description(data: dict[str, Any], *, source: str) -> str:
    """Build a detailed factual description without exposing provider syntax."""
    sentences: list[str] = []
    summary = useful_text(data.get("description"))
    if summary:
        sentences.append(summary.rstrip(". ") + ".")

    details: list[str] = []
    location = useful_text(data.get("scene_location"))
    outfit = useful_text(data.get("scene_outfit"))
    action = useful_text(data.get("action"))
    pose = useful_text(data.get("pose"))
    framing = useful_text(data.get("framing"))
    lighting = useful_text(data.get("scene_lighting"))
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
    categories = normalize_string_list(
        [item.get("content_category") or item.get("category") for item in items],
        limit=8,
    )
    tags = normalize_string_list(
        [tag for item in items for tag in (item.get("tags") or [])],
        limit=14,
    )
    levels = [
        int(item.get("explicitness_level") if item.get("explicitness_level") is not None else item.get("explicitness", 0))
        for item in items
    ]

    media_label = f"{count} media item{'s' if count != 1 else ''}"
    if videos:
        media_label += f", including {videos} video{'s' if videos != 1 else ''}"
    opening = f"A coherent sequence of {media_label}"
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

