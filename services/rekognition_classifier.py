"""Deterministic AWS Rekognition metadata for creator vault media.

Rekognition moderation is intentionally used instead of a generative model for
adult-content taxonomy.  It returns explicit labels rather than refusing the
image, while general labels provide conservative scene and object evidence.
"""
from __future__ import annotations

import asyncio
import os
import re
from typing import Any


class RekognitionClassifierError(RuntimeError):
    """Base error for the vault classifier provider."""


class RekognitionConfigurationError(RekognitionClassifierError):
    """AWS credentials, region, or permissions are not usable."""


class RekognitionRequestError(RekognitionClassifierError):
    """AWS accepted configuration but could not analyze the image."""


_EXPLICIT_ACTIVITY = {"explicit sexual activity"}
_SEX_TOYS = {"sex toys"}
_EXPOSED = {
    "explicit nudity",
    "exposed male genitalia",
    "exposed female genitalia",
    "exposed buttocks or anus",
    "exposed female nipple",
}
_PARTIAL_EXPOSURE = {
    "non-explicit nudity of intimate parts and kissing",
    "non-explicit nudity",
    "bare back",
    "partially exposed buttocks",
    "partially exposed female breast",
}
_IMPLIED = {
    "implied nudity",
    "obstructed intimate parts",
    "obstructed female nipple",
    "obstructed male genitalia",
}
_UNDERWEAR = {
    "swimwear or underwear",
    "female swimwear or underwear",
    "male swimwear or underwear",
}
_KISSING = {"kissing on the lips"}

_ANATOMY = {
    "exposed male genitalia": "penis",
    "obstructed male genitalia": "penis",
    "exposed female genitalia": "vulva",
    "exposed buttocks or anus": "buttocks",
    "partially exposed buttocks": "buttocks",
    "exposed female nipple": "breasts",
    "obstructed female nipple": "breasts",
    "partially exposed female breast": "breasts",
    "bare back": "back",
}

_LOCATION_LABELS = (
    ({"bedroom"}, "bedroom"),
    ({"bed", "bedding", "mattress"}, "bedroom"),
    ({"bathroom", "shower", "bathtub", "bath tub", "toilet"}, "bathroom"),
    ({"kitchen"}, "kitchen"),
    ({"living room", "couch", "sofa"}, "living room"),
    ({"swimming pool", "pool"}, "pool"),
    ({"beach", "coast", "seashore"}, "beach"),
    ({"outdoors", "nature"}, "outdoors"),
    ({"indoors", "interior design"}, "indoors"),
)

_POSE_LABELS = (
    ({"lying", "lying down"}, "lying"),
    ({"sitting", "seated"}, "sitting"),
    ({"standing"}, "standing"),
    ({"kneeling"}, "kneeling"),
)

_BODY_LABELS = {
    "face",
    "hair",
    "head",
    "hand",
    "arm",
    "leg",
    "foot",
    "feet",
    "back",
    "torso",
}


def _clean_label(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _dedupe(values: list[str], *, limit: int = 24) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = _clean_label(value)
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        result.append(cleaned)
        if len(result) >= limit:
            break
    return result


def _moderation_evidence(response: dict[str, Any]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for row in response.get("ModerationLabels") or []:
        if not isinstance(row, dict):
            continue
        name = _clean_label(row.get("Name"))
        parent = _clean_label(row.get("ParentName"))
        if not name:
            continue
        evidence.append({
            "name": name,
            "parent": parent,
            "confidence": round(float(row.get("Confidence") or 0), 3),
            "taxonomy_level": int(row.get("TaxonomyLevel") or 0),
        })
    return evidence


def _general_evidence(response: dict[str, Any]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for row in response.get("Labels") or []:
        if not isinstance(row, dict):
            continue
        name = _clean_label(row.get("Name"))
        if not name:
            continue
        evidence.append({
            "name": name,
            "confidence": round(float(row.get("Confidence") or 0), 3),
            "instances": len(row.get("Instances") or []),
        })
    return evidence


def _first_matching(
    names: set[str],
    candidates: tuple[tuple[set[str], str], ...],
) -> str:
    for labels, result in candidates:
        if names.intersection(labels):
            return result
    return ""


def _slug(*values: str) -> str:
    text = "-".join(value for value in values if value)
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:96]


def build_rekognition_metadata(
    moderation_response: dict[str, Any],
    labels_response: dict[str, Any],
    *,
    is_video: bool,
    album_title: str = "",
) -> dict[str, Any]:
    """Map AWS responses into Cleopatra's provider-neutral vault contract."""
    moderation = _moderation_evidence(moderation_response)
    general = _general_evidence(labels_response)
    moderation_names = {
        value
        for row in moderation
        for value in (row["name"], row["parent"])
        if value
    }
    general_names = {row["name"] for row in general}
    has_activity = bool(moderation_names.intersection(_EXPLICIT_ACTIVITY))
    has_toy = bool(moderation_names.intersection(_SEX_TOYS))
    has_exposed = bool(moderation_names.intersection(_EXPOSED))
    has_partial = bool(moderation_names.intersection(_PARTIAL_EXPOSURE))
    has_implied = bool(moderation_names.intersection(_IMPLIED))
    has_underwear = bool(moderation_names.intersection(_UNDERWEAR))
    has_kissing = bool(moderation_names.intersection(_KISSING))

    if has_activity or has_toy:
        explicitness = 5
        nudity = "full" if has_exposed else "partial"
    elif has_exposed:
        explicitness = 4
        nudity = "full"
    elif has_partial:
        explicitness = 3
        nudity = "partial"
    elif has_underwear:
        explicitness = 3
        nudity = "none"
    elif has_implied:
        explicitness = 2
        nudity = "implied"
    elif has_kissing:
        explicitness = 2
        nudity = "none"
    else:
        explicitness = 0
        nudity = "none"

    person_instances = max(
        (
            int(row["instances"])
            for row in general
            if row["name"] in {"person", "people", "human"}
        ),
        default=0,
    )
    is_closeup = bool(general_names.intersection({"close-up", "close up", "macro"}))
    if has_toy:
        category = "solo_toy_video" if is_video else "solo_toy_photo"
    elif has_activity and person_instances >= 2:
        category = "bg_content"
    elif has_activity:
        category = "explicit_video" if is_video else "explicit_photo"
    elif has_exposed and is_closeup:
        category = "closeup_video" if is_video else "closeup_photo"
    elif has_exposed or has_partial:
        category = "nude_video" if is_video else "nude_photo"
    elif has_underwear:
        category = "lingerie_video" if is_video else "lingerie_photo"
    elif general_names.intersection({"foot", "feet", "leg", "armpit"}):
        category = "legs_feet"
    else:
        category = "teaser_clothed"

    visible_anatomy = _dedupe([
        anatomy
        for label, anatomy in _ANATOMY.items()
        if label in moderation_names
    ])
    body_focus = _dedupe(
        visible_anatomy
        + [name for name in general_names if name in _BODY_LABELS]
    )
    sexual_activity = _dedupe(
        (["explicit sexual activity"] if has_activity else [])
        + (["sex toys"] if has_toy else [])
        + (["kissing"] if has_kissing else [])
    )

    location = _first_matching(general_names, _LOCATION_LABELS)
    pose = _first_matching(general_names, _POSE_LABELS)
    if has_exposed or has_activity:
        outfit = "nude"
    elif has_partial or has_implied:
        outfit = "partially nude"
    elif has_underwear:
        outfit = (
            "swimwear"
            if general_names.intersection({"swimwear", "bikini", "swimsuit"})
            else "lingerie or underwear"
        )
    else:
        clothing = next(
            (
                name
                for name in (
                    "dress",
                    "shirt",
                    "t-shirt",
                    "top",
                    "pants",
                    "shorts",
                    "skirt",
                    "clothing",
                )
                if name in general_names
            ),
            "",
        )
        outfit = clothing or "clothed"

    if "selfie" in general_names:
        framing = "selfie"
    elif is_closeup:
        framing = "close-up"
    elif "portrait" in general_names:
        framing = "portrait"
    else:
        framing = "other"

    action = (
        sexual_activity[0]
        if sexual_activity
        else "posing"
        if general_names.intersection({"pose", "photography", "selfie", "portrait"})
        else "none"
    )
    mood = "explicit" if explicitness >= 4 else "teasing" if explicitness >= 2 else "casual"
    good_for = "closer" if explicitness >= 4 else "mid_session" if explicitness >= 2 else "opener"

    moderation_display = [
        row["name"]
        for row in moderation
        if row["taxonomy_level"] >= 2 or not row["parent"]
    ]
    scene_display = [
        row["name"]
        for row in general
        if row["name"] not in {"person", "people", "human"}
    ][:10]
    sentences: list[str] = []
    if moderation_display:
        sentences.append(
            "Adult creator media classified with "
            + ", ".join(_dedupe(moderation_display, limit=6))
            + "."
        )
    else:
        sentences.append("Adult creator media with no explicit moderation labels detected.")
    if scene_display:
        sentences.append(
            "Visible objects and scene concepts include "
            + ", ".join(scene_display)
            + "."
        )
    details = [
        f"location: {location}" if location else "",
        f"outfit: {outfit}" if outfit else "",
        f"pose: {pose}" if pose else "",
        f"framing: {framing}" if framing != "other" else "",
    ]
    details = [value for value in details if value]
    if details:
        sentences.append("Visual details — " + "; ".join(details) + ".")

    moderation_confidence = max(
        (float(row["confidence"]) for row in moderation),
        default=0,
    )
    general_confidence = max(
        (float(row["confidence"]) for row in general),
        default=0,
    )
    confidence = max(moderation_confidence, general_confidence) / 100
    if not moderation and not general:
        confidence = 0.25

    moderation_version = str(
        moderation_response.get("ModerationModelVersion") or "unknown"
    )
    label_version = str(labels_response.get("LabelModelVersion") or "unknown")
    tags = _dedupe(
        moderation_display
        + sexual_activity
        + visible_anatomy
        + scene_display
        + [location, outfit, pose, framing],
        limit=24,
    )
    return {
        "category": category,
        "description": " ".join(sentences),
        "mood": mood,
        "explicitness": explicitness,
        "nudity": nudity,
        "visible_anatomy": visible_anatomy,
        "good_for": good_for,
        "tags": tags,
        "sexual_activity": sexual_activity,
        "body_focus": body_focus,
        "action": action,
        "pose": pose or "unknown",
        "framing": framing,
        "props": _dedupe(
            [
                name
                for name in general_names
                if name in {"bed", "mirror", "phone", "toy", "chair", "couch"}
            ]
        ),
        "colors": [],
        "scene_location": location or "unknown",
        "scene_outfit": outfit or "unknown",
        "scene_lighting": "unknown",
        "scene_id": _slug(album_title, location, outfit) or "unidentified-shoot",
        "confidence": round(min(max(confidence, 0), 1), 3),
        "_classification_model": (
            f"aws-rekognition-moderation-{moderation_version}"
            f"+labels-{label_version}"
        ),
        "_provider_metadata": {
            "provider": "aws_rekognition",
            "moderation_model_version": moderation_version,
            "label_model_version": label_version,
            "moderation_labels": moderation,
            "general_labels": general[:24],
        },
    }


def _aws_region() -> str:
    return str(
        os.environ.get("AWS_REKOGNITION_REGION")
        or os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
        or "us-east-1"
    ).strip()


def _analyze_sync(image_bytes: bytes) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        import boto3
        from botocore.exceptions import (
            BotoCoreError,
            ClientError,
            NoCredentialsError,
            PartialCredentialsError,
        )
    except ImportError as exc:
        raise RekognitionConfigurationError(
            "boto3 is not installed for the Rekognition vault classifier"
        ) from exc

    try:
        client = boto3.client("rekognition", region_name=_aws_region())
        image = {"Bytes": image_bytes}
        moderation = client.detect_moderation_labels(
            Image=image,
            MinConfidence=float(
                os.environ.get("VAULT_REKOGNITION_MIN_CONFIDENCE", "50")
            ),
        )
        labels = client.detect_labels(
            Image=image,
            MaxLabels=int(os.environ.get("VAULT_REKOGNITION_MAX_LABELS", "40")),
            MinConfidence=float(
                os.environ.get("VAULT_REKOGNITION_LABEL_CONFIDENCE", "50")
            ),
        )
        return moderation, labels
    except (NoCredentialsError, PartialCredentialsError) as exc:
        raise RekognitionConfigurationError(
            "AWS credentials are missing or incomplete for vault categorization"
        ) from exc
    except ClientError as exc:
        code = str((exc.response.get("Error") or {}).get("Code") or "")
        message = str((exc.response.get("Error") or {}).get("Message") or code)
        if code in {
            "AccessDeniedException",
            "InvalidSignatureException",
            "UnrecognizedClientException",
            "ExpiredTokenException",
        }:
            raise RekognitionConfigurationError(
                f"AWS Rekognition access is not configured correctly: {message}"
            ) from exc
        if code in {
            "ThrottlingException",
            "ProvisionedThroughputExceededException",
        }:
            raise RekognitionRequestError(
                f"AWS Rekognition request was throttled: {message}"
            ) from exc
        raise RekognitionRequestError(
            f"AWS Rekognition could not analyze the media: {message}"
        ) from exc
    except BotoCoreError as exc:
        raise RekognitionRequestError(
            f"AWS Rekognition request failed: {type(exc).__name__}"
        ) from exc


async def classify_with_rekognition(
    image_bytes: bytes,
    *,
    is_video: bool,
    album_title: str = "",
) -> dict[str, Any]:
    moderation, labels = await asyncio.to_thread(_analyze_sync, image_bytes)
    return build_rekognition_metadata(
        moderation,
        labels,
        is_video=is_video,
        album_title=album_title,
    )
