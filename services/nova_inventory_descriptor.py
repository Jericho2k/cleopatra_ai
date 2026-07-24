"""Rich, neutral visual inventory descriptions via Amazon Nova.

Rekognition remains authoritative for adult-content taxonomy.  This module is
deliberately narrower: it records photoshoot details that object/moderation
labels usually miss, such as the room, wardrobe colours, pose, gaze, camera
angle, background, and distinguishing visual details.

The result is optional and fail-open.  A Bedrock permission, model, parsing, or
provider failure must never discard a successful Rekognition classification.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any


DEFAULT_NOVA_DESCRIPTOR_MODEL = "amazon.nova-lite-v1:0"
_MAX_LIST_ITEMS = 12
_descriptor_disabled_reason: str | None = None


class NovaInventoryDescriptorError(RuntimeError):
    """Amazon Nova could not return usable inventory metadata."""


class NovaInventoryDescriptorConfigurationError(NovaInventoryDescriptorError):
    """Bedrock credentials, permissions, region, or model are unavailable."""


def _env_bool(name: str, default: bool) -> bool:
    value = str(os.environ.get(name, str(default))).strip().lower()
    return value not in {"0", "false", "no", "off"}


def _text(value: Any, *, limit: int = 320) -> str:
    cleaned = " ".join(str(value or "").strip().split())
    if cleaned.lower() in {"", "unknown", "unclear", "none", "n/a", "null"}:
        return ""
    return cleaned[:limit]


def _text_list(value: Any, *, limit: int = _MAX_LIST_ITEMS) -> list[str]:
    raw = value if isinstance(value, list) else []
    result: list[str] = []
    seen: set[str] = set()
    for item in raw:
        cleaned = _text(item, limit=120).lower().strip(" .,-_/|")
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        result.append(cleaned)
        if len(result) >= limit:
            break
    return result


def _json_object(value: str) -> dict[str, Any]:
    text = str(value or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise NovaInventoryDescriptorError(
            "Amazon Nova returned no JSON inventory object"
        )
    try:
        parsed = json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        raise NovaInventoryDescriptorError(
            "Amazon Nova returned invalid JSON inventory metadata"
        ) from exc
    if not isinstance(parsed, dict):
        raise NovaInventoryDescriptorError(
            "Amazon Nova inventory metadata was not an object"
        )
    return parsed


def normalize_inventory_descriptor(value: dict[str, Any]) -> dict[str, Any]:
    """Validate Nova output into a small provider-neutral visual contract."""
    try:
        confidence = float(value.get("confidence") or 0)
    except (TypeError, ValueError):
        confidence = 0
    return {
        "description": _text(value.get("description"), limit=1200),
        "setting_location": _text(value.get("setting_location")),
        "setting_details": _text_list(value.get("setting_details")),
        "background_details": _text_list(value.get("background_details")),
        "wardrobe_items": _text_list(value.get("wardrobe_items")),
        "wardrobe_colors": _text_list(value.get("wardrobe_colors"), limit=8),
        "pose": _text(value.get("pose")),
        "limb_position": _text(value.get("limb_position")),
        "gaze": _text(value.get("gaze")),
        "expression": _text(value.get("expression")),
        "action": _text(value.get("action")),
        "framing": _text(value.get("framing")),
        "camera_angle": _text(value.get("camera_angle")),
        "crop": _text(value.get("crop")),
        "composition": _text(value.get("composition")),
        "props": _text_list(value.get("props")),
        "lighting": _text(value.get("lighting")),
        "visual_style": _text(value.get("visual_style")),
        "distinguishing_details": _text_list(
            value.get("distinguishing_details")
        ),
        "search_tags": _text_list(value.get("search_tags"), limit=16),
        "confidence": round(min(max(confidence, 0), 1), 3),
    }


def _merge_list(*values: Any, limit: int = 24) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        raw = value if isinstance(value, list) else [value]
        for item in raw:
            cleaned = _text(item, limit=120).lower().strip(" .,-_/|")
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            result.append(cleaned)
            if len(result) >= limit:
                return result
    return result


def merge_inventory_descriptor(
    base: dict[str, Any],
    descriptor: dict[str, Any],
) -> dict[str, Any]:
    """Merge visual detail without overriding Rekognition adult evidence."""
    result = dict(base)
    visual = normalize_inventory_descriptor(descriptor)

    base_description = _text(result.get("description"), limit=1200)
    rich_description = visual["description"]
    if rich_description:
        result["description"] = " ".join(
            value for value in (base_description, rich_description) if value
        )[:1800]
    # The deterministic renderer still needs to append controlled fields and
    # local visual evidence after Nova's prose.
    result["description_complete"] = False

    current_location = _text(result.get("scene_location")).lower()
    if visual["setting_location"] and current_location in {
        "",
        "indoors",
        "outdoors",
    }:
        result["scene_location"] = visual["setting_location"]

    wardrobe = " ".join(
        value
        for value in (
            " and ".join(visual["wardrobe_colors"]),
            " and ".join(visual["wardrobe_items"]),
        )
        if value
    )
    adult_clothing_state = _text(result.get("scene_outfit"))
    if wardrobe:
        if adult_clothing_state and adult_clothing_state.lower() not in wardrobe.lower():
            result["scene_outfit"] = (
                f"{wardrobe}; {adult_clothing_state}"
            )[:320]
        else:
            result["scene_outfit"] = wardrobe[:320]

    if visual["pose"]:
        result["pose"] = visual["pose"]
    if visual["framing"]:
        result["framing"] = visual["framing"]
    if visual["lighting"]:
        result["scene_lighting"] = visual["lighting"]

    # Rekognition's adult activity is authoritative.  Nova may provide a more
    # useful ordinary visual action only when no explicit activity is present.
    adult_activity = _text_list(result.get("sexual_activity"))
    if visual["action"] and not adult_activity:
        result["action"] = visual["action"]

    result["props"] = _merge_list(
        result.get("props"),
        visual["props"],
        visual["setting_details"],
        limit=16,
    )
    result["colors"] = _merge_list(
        visual["wardrobe_colors"],
        result.get("colors"),
        limit=8,
    )
    result["tags"] = _merge_list(
        result.get("tags"),
        visual["search_tags"],
        visual["setting_location"],
        visual["setting_details"],
        visual["background_details"],
        visual["wardrobe_items"],
        visual["wardrobe_colors"],
        visual["pose"],
        visual["limb_position"],
        visual["gaze"],
        visual["expression"],
        visual["action"],
        visual["framing"],
        visual["camera_angle"],
        visual["crop"],
        visual["composition"],
        visual["visual_style"],
        visual["distinguishing_details"],
        limit=32,
    )
    return result


_SYSTEM_PROMPT = """You index visual inventory for an adult creator business.
Every depicted person has already been verified as an adult by the account
owner. Do not identify the person or estimate age. This is neutral metadata,
not erotic writing.

Record only directly visible photoshoot facts. Be unusually specific about the
setting, surfaces, background, wardrobe pieces and colours, pose, limb
position, gaze, expression, action, camera angle, crop, composition, props,
lighting, and distinguishing details. Do not omit details merely because the
image contains adult nudity. Do not classify nudity, anatomy, sexual acts,
safety, or price; a separate deterministic system owns those fields. Never
invent something hidden or outside the frame."""


def _prompt(*, is_video: bool, album_title: str) -> str:
    asset = "video thumbnail" if is_video else "photo"
    album = _text(album_title) or "not provided"
    return f"""Analyze this {asset} for search and photoshoot matching.
Album/folder context: {album}

Return ONLY one valid JSON object matching this schema:
{{
  "description": "3-6 factual sentences with the most useful visible details",
  "setting_location": "specific room or environment, or unknown",
  "setting_details": ["surfaces, furniture, architecture and scene details"],
  "background_details": ["specific visible background details"],
  "wardrobe_items": ["specific visible garments or accessories"],
  "wardrobe_colors": ["specific garment colours"],
  "pose": "specific body pose",
  "limb_position": "specific arm and leg positioning",
  "gaze": "gaze direction",
  "expression": "visible facial expression",
  "action": "specific visible action, or posing",
  "framing": "selfie, close-up, medium, three-quarter, full-body or wide",
  "camera_angle": "high, eye-level, low, overhead, mirror or other",
  "crop": "what portion of the subject is in frame",
  "composition": "subject placement and composition",
  "props": ["visible handheld or scene props"],
  "lighting": "natural, bright, dim, flash, warm, cool or coloured",
  "visual_style": "overall non-erotic visual look",
  "distinguishing_details": ["details useful for matching this photoshoot"],
  "search_tags": ["specific factual search terms"],
  "confidence": 0.0
}}
"""


def _describe_sync(
    image_bytes: bytes,
    *,
    is_video: bool,
    album_title: str,
    client: Any | None = None,
) -> tuple[dict[str, Any], str]:
    if client is None:
        try:
            import boto3
            from botocore.exceptions import (
                BotoCoreError,
                ClientError,
                NoCredentialsError,
                PartialCredentialsError,
            )
        except ImportError as exc:
            raise NovaInventoryDescriptorConfigurationError(
                "boto3 is not installed for rich vault descriptions"
            ) from exc
    else:
        boto3 = None

        class BotoCoreError(Exception):
            pass

        class ClientError(Exception):
            response: dict[str, Any] = {}

        class NoCredentialsError(Exception):
            pass

        class PartialCredentialsError(Exception):
            pass

    model_id = str(
        os.environ.get("VAULT_RICH_DESCRIPTION_MODEL")
        or DEFAULT_NOVA_DESCRIPTOR_MODEL
    ).strip()
    region = str(
        os.environ.get("AWS_BEDROCK_REGION") or "us-east-1"
    ).strip()
    runtime = client or boto3.client("bedrock-runtime", region_name=region)
    try:
        response = runtime.converse(
            modelId=model_id,
            system=[{"text": _SYSTEM_PROMPT}],
            messages=[{
                "role": "user",
                "content": [
                    {
                        "image": {
                            "format": "jpeg",
                            "source": {"bytes": image_bytes},
                        },
                    },
                    {"text": _prompt(is_video=is_video, album_title=album_title)},
                ],
            }],
            inferenceConfig={
                "maxTokens": int(
                    os.environ.get(
                        "VAULT_RICH_DESCRIPTION_MAX_TOKENS",
                        "700",
                    )
                ),
                "temperature": 0,
                "topP": 0.9,
            },
        )
        content = (
            ((response.get("output") or {}).get("message") or {}).get("content")
            or []
        )
        text = "\n".join(
            str(block.get("text") or "")
            for block in content
            if isinstance(block, dict) and block.get("text")
        ).strip()
        normalized = normalize_inventory_descriptor(_json_object(text))
        if not normalized["description"] and not any(
            normalized.get(key)
            for key in (
                "setting_location",
                "setting_details",
                "wardrobe_items",
                "pose",
                "action",
                "framing",
            )
        ):
            raise NovaInventoryDescriptorError(
                "Amazon Nova returned empty visual inventory metadata"
            )
        return normalized, model_id
    except (NoCredentialsError, PartialCredentialsError) as exc:
        raise NovaInventoryDescriptorConfigurationError(
            "AWS credentials are missing for rich vault descriptions"
        ) from exc
    except ClientError as exc:
        error = exc.response.get("Error") or {}
        code = str(error.get("Code") or "")
        message = str(error.get("Message") or code)
        if code in {
            "AccessDeniedException",
            "InvalidSignatureException",
            "UnrecognizedClientException",
            "ExpiredTokenException",
            "ValidationException",
            "ResourceNotFoundException",
        }:
            raise NovaInventoryDescriptorConfigurationError(
                f"Amazon Nova rich descriptions are unavailable: {message}"
            ) from exc
        raise NovaInventoryDescriptorError(
            f"Amazon Nova could not describe the image: {message}"
        ) from exc
    except BotoCoreError as exc:
        raise NovaInventoryDescriptorError(
            f"Amazon Nova rich description failed: {type(exc).__name__}"
        ) from exc


async def describe_inventory_visual(
    image_bytes: bytes,
    *,
    is_video: bool,
    album_title: str = "",
) -> dict[str, Any]:
    """Return an optional rich visual descriptor without breaking Rekognition."""
    global _descriptor_disabled_reason
    if not _env_bool("VAULT_RICH_DESCRIPTIONS_ENABLED", True):
        return {"status": "disabled", "reason": "disabled"}
    if _descriptor_disabled_reason:
        return {
            "status": "unavailable",
            "reason": _descriptor_disabled_reason,
        }
    try:
        descriptor, model_id = await asyncio.to_thread(
            _describe_sync,
            image_bytes,
            is_video=is_video,
            album_title=album_title,
        )
    except NovaInventoryDescriptorConfigurationError as exc:
        _descriptor_disabled_reason = str(exc)[:300]
        return {
            "status": "unavailable",
            "reason": _descriptor_disabled_reason,
        }
    except NovaInventoryDescriptorError as exc:
        return {"status": "failed", "reason": str(exc)[:300]}
    return {
        "status": "ready",
        "provider": "amazon_nova",
        "model": model_id,
        "descriptor": descriptor,
    }
