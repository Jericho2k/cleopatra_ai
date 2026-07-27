"""Fast, controlled-vocabulary vault semantics from a self-hosted encoder.

SigLIP2 supplies image embeddings and fixed-label scores.  It does not generate
prose, so the same image and taxonomy produce stable metadata.  NudeNet remains
authoritative for exposed anatomy; Qwen is reserved for genuinely ambiguous or
high-risk cases.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import os
import time
from typing import Any

import httpx


SEMANTIC_MODEL = "google/siglip2-base-patch16-224"
_SEMANTIC_GATE = asyncio.Semaphore(32)
_AXIS_LABELS = {
    "scene_location": {
        "bedroom", "bathroom", "kitchen", "living room", "studio",
        "shower", "bathtub", "outdoors", "vehicle", "hallway",
        "other indoor room",
    },
    "wardrobe_state": {
        "full nudity", "partial nudity", "lingerie", "underwear",
        "casual clothing", "dress", "swimwear", "costume", "sleepwear",
    },
    "pose": {
        "standing", "sitting", "lying down", "kneeling", "crouching",
        "close-up body detail",
    },
    "activity": {
        "posing", "selfie", "mirror selfie", "showering", "dancing",
        "undressing", "sexual activity", "using an adult toy",
    },
    "framing": {
        "close-up", "medium shot", "three-quarter shot", "full-body shot",
        "wide shot",
    },
    "lighting": {
        "warm indoor light", "cool blue light", "pink or purple colored light",
        "natural daylight", "bright studio light", "dim low light",
    },
}
_TAG_LABELS = {
    "background_details": {
        "bed and bedding", "kitchen counter and cabinets", "mirror",
        "shower or bathtub", "sofa", "curtains", "tiled wall", "plain wall",
        "studio backdrop",
    },
    "wardrobe_items": {
        "bra", "panties", "lingerie set", "bodysuit", "stockings",
        "fishnet clothing", "sheer clothing", "lace clothing", "dress",
        "crop top", "shorts", "skirt", "high heels", "jewelry",
    },
}
_HIGH_RISK_ACTIVITIES = {"sexual activity", "using an adult toy"}


def semantic_endpoint_configured() -> bool:
    return bool(os.environ.get("VAULT_SEMANTIC_BASE_URL", "").strip())


def _clean(value: Any, *, limit: int = 160) -> str:
    return " ".join(str(value or "").strip().split())[:limit]


def _bounded_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return round(min(max(parsed, 0.0), 1.0), 4)


def _nonnegative_int(value: Any) -> int:
    try:
        return max(int(value or 0), 0)
    except (TypeError, ValueError):
        return 0


def _headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    modal_key = os.environ.get("VAULT_VISION_MODAL_KEY", "").strip()
    modal_secret = os.environ.get("VAULT_VISION_MODAL_SECRET", "").strip()
    if modal_key and modal_secret:
        headers["Modal-Key"] = modal_key
        headers["Modal-Secret"] = modal_secret
    return headers


def _timeout() -> float:
    try:
        value = float(os.environ.get("VAULT_SEMANTIC_TIMEOUT_SECONDS", "180"))
    except ValueError:
        value = 180
    return min(max(value, 30), 300)


def _axis(value: Any, *, name: str) -> dict[str, Any]:
    row = value if isinstance(value, dict) else {}
    allowed = _AXIS_LABELS[name]
    ranked: list[dict[str, Any]] = []
    for candidate in row.get("ranked") or []:
        if not isinstance(candidate, dict):
            continue
        label = _clean(candidate.get("label"), limit=80).lower()
        if label not in allowed:
            continue
        ranked.append({
            "label": label,
            "score": _bounded_float(candidate.get("score")),
        })
        if len(ranked) >= 3:
            break
    top = ranked[0] if ranked else {"label": "", "score": 0.0}
    return {
        "label": top["label"],
        "score": top["score"],
        "margin": _bounded_float(row.get("margin")),
        "confident": bool(row.get("confident")),
        "ranked": ranked,
    }


def _tags(value: Any, *, name: str) -> list[dict[str, Any]]:
    allowed = _TAG_LABELS[name]
    result: list[dict[str, Any]] = []
    for candidate in value if isinstance(value, list) else []:
        if not isinstance(candidate, dict):
            continue
        label = _clean(candidate.get("label"), limit=80).lower()
        score = _bounded_float(candidate.get("score"))
        if label not in allowed or label in {row["label"] for row in result}:
            continue
        result.append({"label": label, "score": score})
        if len(result) >= 4:
            break
    return result


def _embedding(value: Any) -> dict[str, Any]:
    row = value if isinstance(value, dict) else {}
    if row.get("encoding") != "float16_base64":
        return {}
    try:
        dimensions = int(row.get("dimensions") or 0)
    except (TypeError, ValueError):
        return {}
    data = str(row.get("data") or "")
    if not 16 <= dimensions <= 2048 or not data or len(data) > 8192:
        return {}
    try:
        decoded = base64.b64decode(data, validate=True)
    except Exception:
        return {}
    if len(decoded) != dimensions * 2:
        return {}
    return {
        "model": SEMANTIC_MODEL,
        "encoding": "float16_base64",
        "dimensions": dimensions,
        "data": data,
    }


def normalize_semantic_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("semantic endpoint returned no object")
    axes = payload.get("axes") if isinstance(payload.get("axes"), dict) else {}
    tags = payload.get("tags") if isinstance(payload.get("tags"), dict) else {}
    normalized_axes = {
        name: _axis(axes.get(name), name=name)
        for name in _AXIS_LABELS
    }
    embedding = _embedding(payload.get("embedding"))
    if not embedding:
        raise ValueError("semantic endpoint returned no valid embedding")
    return {
        "model": _clean(payload.get("model"), limit=160) or SEMANTIC_MODEL,
        "revision": _clean(payload.get("revision"), limit=64),
        "request_id": _clean(payload.get("request_id"), limit=64),
        "latency_ms": _nonnegative_int(payload.get("latency_ms")),
        "confidence": _bounded_float(payload.get("confidence")),
        "ambiguous_axes": [
            name
            for name in payload.get("ambiguous_axes") or []
            if name in _AXIS_LABELS
        ],
        "axes": normalized_axes,
        "tags": {
            name: _tags(tags.get(name), name=name)
            for name in _TAG_LABELS
        },
        "embedding": embedding,
    }


async def semantic_metadata(image_bytes: bytes) -> dict[str, Any]:
    base_url = os.environ.get("VAULT_SEMANTIC_BASE_URL", "").strip()
    if not base_url:
        raise ValueError("semantic endpoint is not configured")
    request_id = hashlib.sha256(image_bytes).hexdigest()[:12]
    queued_at = time.monotonic()
    async with _SEMANTIC_GATE:
        queue_ms = round((time.monotonic() - queued_at) * 1000)
        started = time.monotonic()
        try:
            async with httpx.AsyncClient(
                timeout=_timeout(),
                follow_redirects=True,
            ) as client:
                response = await client.post(
                    base_url,
                    headers=_headers(),
                    json={
                        "image_base64": base64.b64encode(image_bytes).decode("ascii"),
                        "request_id": request_id,
                    },
                )
                response.raise_for_status()
                result = normalize_semantic_payload(response.json())
        except Exception as exc:
            print(
                f"[VAULT SEMANTICS] request={request_id} status=failed "
                f"reason={type(exc).__name__} queue_ms={queue_ms} "
                f"round_trip_ms={round((time.monotonic() - started) * 1000)}"
            )
            raise
    result["queue_ms"] = queue_ms
    result["round_trip_ms"] = round((time.monotonic() - started) * 1000)
    print(
        f"[VAULT SEMANTICS] request={request_id} status=ready "
        f"confidence={result['confidence']:.3f} "
        f"ambiguous={','.join(result['ambiguous_axes']) or 'none'} "
        f"queue_ms={queue_ms} round_trip_ms={result['round_trip_ms']} "
        f"inference_ms={result['latency_ms']}"
    )
    return result


def qwen_fallback_reasons(
    semantic: dict[str, Any],
    *,
    exposed_anatomy: list[str],
) -> list[str]:
    """Return only ambiguity that can materially change selling metadata."""
    axes = semantic.get("axes") or {}
    reasons: list[str] = []
    activity = (axes.get("activity") or {}).get("label")
    activity_score = float((axes.get("activity") or {}).get("score") or 0)
    if activity in _HIGH_RISK_ACTIVITIES and activity_score >= 0.25:
        reasons.append("high_risk_activity")
    wardrobe = (axes.get("wardrobe_state") or {}).get("label")
    wardrobe_score = float(
        (axes.get("wardrobe_state") or {}).get("score") or 0
    )
    if (
        wardrobe == "full nudity"
        and wardrobe_score >= 0.12
        and bool((axes.get("wardrobe_state") or {}).get("confident"))
        and not exposed_anatomy
    ):
        reasons.append("nudity_detector_conflict")
    ambiguous = set(semantic.get("ambiguous_axes") or [])
    if {"scene_location", "wardrobe_state", "activity"}.issubset(ambiguous):
        reasons.append("core_semantics_ambiguous")
    return reasons
