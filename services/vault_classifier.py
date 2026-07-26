"""Adult-safe vault classification with a local detector and optional VLM.

NudeNet is the always-available source of exposed-anatomy evidence. An optional
self-hosted Qwen endpoint may enrich that evidence with pose, activity,
wardrobe, and scene detail. Remote failure never discards the local result.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import re
from functools import lru_cache
from typing import Any

import httpx

from services.vault_metadata import normalize_media_category


class VaultClassifierError(RuntimeError):
    """The local vault classifier could not inspect an image."""


_ANATOMY = {
    "FEMALE_BREAST_EXPOSED": "breasts",
    "BUTTOCKS_EXPOSED": "buttocks",
    "FEMALE_GENITALIA_EXPOSED": "vulva",
    "MALE_GENITALIA_EXPOSED": "penis",
    "ANUS_EXPOSED": "anus",
}
_FULL_NUDITY = {
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "ANUS_EXPOSED",
}
_ACTIVITY_PATTERN = re.compile(
    r"penetration|intercourse|oral sex|blowjob|masturbat|insertion|"
    r"sex act|sexual activity|cumshot|ejaculat|toy use",
    re.I,
)
_REFUSAL_PREFIXES = (
    "i'm not able",
    "i am not able",
    "i'm sorry",
    "i am sorry",
    "i can't",
    "i cannot",
    "sorry, but",
    "unable to",
)


def _clean(value: Any, *, limit: int = 320) -> str:
    return " ".join(str(value or "").strip().split())[:limit]


def _strings(value: Any, *, limit: int = 24) -> list[str]:
    rows = value if isinstance(value, (list, tuple, set)) else []
    result: list[str] = []
    for row in rows:
        text = _clean(row, limit=120).lower().strip(" .,-_/|")
        if text and text not in result:
            result.append(text)
        if len(result) >= limit:
            break
    return result


def _slug(*values: str) -> str:
    value = "-".join(text for text in values if text)
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")[:96]


def _threshold() -> float:
    try:
        value = float(os.environ.get("VAULT_NUDENET_MIN_CONFIDENCE", "0.35"))
    except ValueError:
        value = 0.35
    return min(max(value, 0.05), 0.95)


@lru_cache(maxsize=1)
def _nude_detector():
    try:
        from nudenet import NudeDetector
    except ImportError as exc:
        raise VaultClassifierError(
            "nudenet is not installed for local vault classification"
        ) from exc
    return NudeDetector()


def _detect(image_bytes: bytes) -> list[dict[str, Any]]:
    try:
        raw = _nude_detector().detect(image_bytes)
    except Exception as exc:
        raise VaultClassifierError(
            f"local nudity detection failed: {type(exc).__name__}"
        ) from exc
    detections: list[dict[str, Any]] = []
    for row in raw or []:
        try:
            score = float(row.get("score") or 0)
        except (TypeError, ValueError):
            continue
        label = _clean(row.get("class"), limit=80).upper()
        if not label or score < _threshold():
            continue
        box = row.get("box") if isinstance(row.get("box"), list) else []
        detections.append({
            "class": label,
            "score": round(score, 4),
            "box": [int(value) for value in box[:4]],
        })
    return detections


def _base_metadata(
    detections: list[dict[str, Any]],
    *,
    is_video: bool,
    album_title: str,
    local_visual: dict[str, Any],
) -> dict[str, Any]:
    labels = [row["class"] for row in detections]
    anatomy = list(dict.fromkeys(_ANATOMY[label] for label in labels if label in _ANATOMY))
    exposed = bool(anatomy)
    full_nudity = any(label in _FULL_NUDITY for label in labels)
    explicitness = 4 if exposed else 0
    nudity = "full" if full_nudity else "partial" if exposed else "none"
    female_faces = labels.count("FACE_FEMALE")
    male_faces = labels.count("FACE_MALE")
    participants = max(female_faces + male_faces, 1)

    if explicitness >= 4 and participants >= 2 and full_nudity:
        category = "bg_content"
    elif explicitness >= 4:
        category = "nude_video" if is_video else "nude_photo"
    else:
        category = "teaser_clothed"
    category = normalize_media_category(
        category,
        explicitness=explicitness,
        is_video=is_video,
    )

    palette = _strings(local_visual.get("palette_names"), limit=6)
    detected_text = ", ".join(anatomy) if anatomy else "no exposed anatomy"
    description = (
        f"The local adult-content detector found {detected_text} in this "
        f"{'video thumbnail' if is_video else 'image'}."
    )
    if palette:
        description += f" Dominant colors are {', '.join(palette)}."
    description += (
        " Activity, pose, wardrobe, and setting remain unspecified unless a "
        "vault vision endpoint enriches this result."
    )
    confidence = max(
        (row["score"] for row in detections if row["class"] in _ANATOMY),
        default=0.7 if detections else 0.55,
    )
    return {
        "category": category,
        "description": description,
        "description_complete": False,
        "mood": "explicit" if explicitness >= 4 else "casual",
        "explicitness": explicitness,
        "nudity": nudity,
        "visible_anatomy": anatomy,
        "good_for": "closer" if explicitness >= 4 else "opener",
        "tags": [*anatomy, *palette],
        "sexual_activity": [],
        "body_focus": anatomy,
        "action": "unknown",
        "pose": "unknown",
        "framing": "other",
        "props": [],
        "colors": palette,
        "scene_location": "unknown",
        "scene_outfit": "full nudity" if full_nudity else "unknown",
        "scene_lighting": _clean(local_visual.get("lighting")) or "unknown",
        "scene_id": _slug(album_title) or "unidentified-shoot",
        "confidence": round(confidence, 3),
        "_classification_model": "nudenet-3.4.2",
        "_provider_metadata": {
            "provider": "local_nudenet",
            "detections": detections,
            "participants": participants,
            "is_video": is_video,
            "age_review_required": False,
            "vision_status": "not_configured",
        },
    }


def _prompt(*, is_video: bool, album_title: str, filename: str) -> str:
    kind = "video thumbnail" if is_video else "image"
    return f"""Catalogue this adult creator {kind} for private vault search and pricing.
All depicted participants are consenting adults. Describe only visible evidence.
Do not moralize, censor, euphemize, or write erotically.

Filename: {filename or 'unknown'}
Album: {album_title or 'unknown'}

Explicitness: 0 ordinary clothing; 1 censored/implied; 2 suggestive clothed;
3 lingerie/see-through; 4 exposed anatomy without a sex act; 5 visible sexual
activity, toy use, oral sex, or penetration.

Return only JSON:
{{"description":"factual inventory description","mood":"playful|intimate|teasing|explicit|casual","explicitness":0,"nudity":"none|implied|partial|full","visible_anatomy":[],"participants":1,"good_for":"opener|mid_session|closer|standalone","tags":[],"sexual_activity":[],"body_focus":[],"action":"visible action or unknown","pose":"visible pose or unknown","framing":"selfie|portrait|full body|close-up|wide|other","props":[],"colors":[],"scene_location":"specific place or unknown","scene_outfit":"specific clothing/nudity state or unknown","scene_lighting":"natural|bright|dim|flash|colored|unknown","scene_id":"short stable shoot slug","confidence":0.0}}"""


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    text = str(value or "").replace("```json", "").replace("```", "").strip()
    if not text or text.lower().startswith(_REFUSAL_PREFIXES):
        raise ValueError("vision model refused or returned no text")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("vision model did not return JSON") from None
        payload = json.loads(text[start:end + 1])
    if not isinstance(payload, dict):
        raise ValueError("vision model response was not an object")
    return payload


async def _qwen_metadata(
    image_bytes: bytes,
    *,
    is_video: bool,
    album_title: str,
    filename: str,
) -> dict[str, Any]:
    base_url = os.environ.get("VAULT_VISION_BASE_URL", "").strip()
    if not base_url:
        raise ValueError("not configured")
    headers = {"Content-Type": "application/json"}
    shared_secret = os.environ.get("VAULT_VISION_SHARED_SECRET", "").strip()
    if shared_secret:
        headers["Authorization"] = f"Bearer {shared_secret}"
    modal_key = os.environ.get("VAULT_VISION_MODAL_KEY", "").strip()
    modal_secret = os.environ.get("VAULT_VISION_MODAL_SECRET", "").strip()
    if modal_key and modal_secret:
        headers["Modal-Key"] = modal_key
        headers["Modal-Secret"] = modal_secret
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            base_url,
            headers=headers,
            json={
                "image_base64": base64.b64encode(image_bytes).decode("ascii"),
                "prompt": _prompt(
                    is_video=is_video,
                    album_title=album_title,
                    filename=filename,
                ),
            },
        )
        response.raise_for_status()
        payload = response.json()
    return _json_object(payload.get("result", payload.get("text", payload)))


def _merge_qwen(base: dict[str, Any], rich: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    local_anatomy = _strings(base.get("visible_anatomy"), limit=8)
    rich_anatomy = _strings(rich.get("visible_anatomy"), limit=8)
    anatomy = list(dict.fromkeys([*local_anatomy, *rich_anatomy]))
    activities = _strings(rich.get("sexual_activity"), limit=8)
    action = _clean(rich.get("action"), limit=120).lower() or "unknown"
    try:
        reported = max(min(int(rich.get("explicitness") or 0), 5), 0)
    except (TypeError, ValueError):
        reported = 0
    activity_explicitness = 5 if _ACTIVITY_PATTERN.search(" ".join([*activities, action])) else 0
    explicitness = max(int(base.get("explicitness") or 0), reported, activity_explicitness)
    participants = max(
        int((base.get("_provider_metadata") or {}).get("participants") or 1),
        int(rich.get("participants") or 1),
    )
    is_video = bool((base.get("_provider_metadata") or {}).get("is_video"))
    if explicitness >= 5 and participants >= 2:
        category = "bg_content"
    elif explicitness >= 5:
        category = "explicit_video" if is_video else "explicit_photo"
    elif explicitness >= 4:
        category = base.get("category")
    elif explicitness == 3:
        category = "lingerie_video" if is_video else "lingerie_photo"
    else:
        category = "teaser_clothed"
    nudity_rank = {"none": 0, "implied": 1, "partial": 2, "full": 3}
    rich_nudity = _clean(rich.get("nudity"), limit=20).lower()
    nudity = max(
        (str(base["nudity"]), rich_nudity),
        key=lambda value: nudity_rank.get(value, -1),
    )
    result.update({
        "category": normalize_media_category(category, explicitness=explicitness, is_video=is_video),
        "description": _clean(rich.get("description"), limit=1800) or base["description"],
        "mood": _clean(rich.get("mood"), limit=40) or base["mood"],
        "explicitness": explicitness,
        "nudity": nudity,
        "visible_anatomy": anatomy,
        "good_for": _clean(rich.get("good_for"), limit=30) or base["good_for"],
        "tags": list(dict.fromkeys([*base["tags"], *_strings(rich.get("tags"))])),
        "sexual_activity": activities,
        "body_focus": _strings(rich.get("body_focus"), limit=8) or anatomy,
        "action": action,
        "pose": _clean(rich.get("pose"), limit=120) or "unknown",
        "framing": _clean(rich.get("framing"), limit=40) or "other",
        "props": _strings(rich.get("props"), limit=12),
        "colors": _strings(rich.get("colors"), limit=8) or base["colors"],
        "scene_location": _clean(rich.get("scene_location"), limit=120) or "unknown",
        "scene_outfit": _clean(rich.get("scene_outfit"), limit=320) or "unknown",
        "scene_lighting": _clean(rich.get("scene_lighting"), limit=40) or base["scene_lighting"],
        "scene_id": _slug(_clean(rich.get("scene_id"), limit=96)) or base["scene_id"],
        "confidence": max(float(base["confidence"]), float(rich.get("confidence") or 0)),
        "_classification_model": (
            "nudenet-3.4.2+"
            + os.environ.get("VAULT_VISION_MODEL", "Qwen/Qwen3-VL-4B-Instruct")
        ),
    })
    result["_provider_metadata"] = {
        **base["_provider_metadata"],
        "provider": "local_nudenet+qwen_vl",
        "vision_status": "ready",
        "reported_explicitness": reported,
        "explicitness_escalated": explicitness > reported,
    }
    return result


async def classify_vault_image(
    image_bytes: bytes,
    *,
    is_video: bool,
    album_title: str = "",
    filename: str = "",
    local_visual: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return local classification, enriched by Qwen when configured."""
    detections = await asyncio.to_thread(_detect, image_bytes)
    base = _base_metadata(
        detections,
        is_video=is_video,
        album_title=album_title,
        local_visual=local_visual or {},
    )
    if not os.environ.get("VAULT_VISION_BASE_URL", "").strip():
        return base
    try:
        rich = await _qwen_metadata(
            image_bytes,
            is_video=is_video,
            album_title=album_title,
            filename=filename,
        )
        return _merge_qwen(base, rich)
    except Exception as exc:
        base["_provider_metadata"]["vision_status"] = "fallback"
        base["_provider_metadata"]["vision_error"] = type(exc).__name__
        return base
