"""Visual fingerprints and conservative same-photoshoot clustering.

Rekognition remains the source of factual adult-content taxonomy.  It is not a
photoshoot matcher: location, nudity and an album name are far too coarse for
that job.  This module gives every analyzed frame:

* an Amazon Nova multimodal embedding optimized for clustering;
* small deterministic local image features for palette/lighting/debugging;
* complete-link clustering so one weak bridge cannot merge two shoots.

The embedding is stored inside ``classification_metadata``.  This keeps the
contract restart-safe without adding a second storage system or a SQL
migration.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import math
import os
from collections import Counter
from typing import Any, Iterable

from PIL import Image, ImageStat


SHOOT_FINGERPRINT_VERSION = 1
NOVA_EMBEDDING_MODEL = "amazon.nova-2-multimodal-embeddings-v1:0"
_DEFAULT_DIMENSION = 384
_DEFAULT_MIN_SIMILARITY = 0.86
_GENERIC_ALBUM_PREFIXES = ("album_", "album-")
_EMPTY = {"", "unknown", "unclear", "none", "n/a", "na", "null"}


class ShootEmbeddingError(RuntimeError):
    """Nova could not produce a visual embedding."""


class ShootEmbeddingConfigurationError(ShootEmbeddingError):
    """Bedrock credentials, permissions or model access are unavailable."""


_embedding_disabled_reason: str | None = None


def _env_bool(name: str, default: bool) -> bool:
    value = str(os.environ.get(name, str(default))).strip().lower()
    return value not in {"0", "false", "no", "off"}


def _embedding_dimension() -> int:
    try:
        value = int(
            os.environ.get(
                "VAULT_SHOOT_EMBEDDING_DIMENSION",
                str(_DEFAULT_DIMENSION),
            )
        )
    except (TypeError, ValueError):
        value = _DEFAULT_DIMENSION
    return value if value in {256, 384, 1024, 3072} else _DEFAULT_DIMENSION


def _min_similarity() -> float:
    try:
        value = float(
            os.environ.get(
                "VAULT_SHOOT_MIN_SIMILARITY",
                str(_DEFAULT_MIN_SIMILARITY),
            )
        )
    except (TypeError, ValueError):
        value = _DEFAULT_MIN_SIMILARITY
    return min(max(value, 0.5), 0.99)


def _normalize(values: Iterable[float]) -> list[float]:
    vector = [float(value) for value in values]
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 0:
        return []
    return [round(value / norm, 6) for value in vector]


def cosine_similarity(left: Iterable[float], right: Iterable[float]) -> float:
    a = [float(value) for value in left]
    b = [float(value) for value in right]
    if not a or len(a) != len(b):
        return 0.0
    norm_a = math.sqrt(sum(value * value for value in a))
    norm_b = math.sqrt(sum(value * value for value in b))
    if norm_a <= 0 or norm_b <= 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (norm_a * norm_b)


def _pixels(image: Image.Image) -> list[Any]:
    flattened = getattr(image, "get_flattened_data", None)
    return list(flattened() if callable(flattened) else image.getdata())


def _dhash(image: Image.Image) -> str:
    sample = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = _pixels(sample)
    value = 0
    for row in range(8):
        for column in range(8):
            value <<= 1
            offset = row * 9 + column
            if pixels[offset] > pixels[offset + 1]:
                value |= 1
    return f"{value:016x}"


def _colour_name(red: int, green: int, blue: int) -> str:
    maximum = max(red, green, blue)
    minimum = min(red, green, blue)
    spread = maximum - minimum
    brightness = (maximum + minimum) / 2
    if brightness < 35:
        return "black"
    if brightness > 225 and spread < 28:
        return "white"
    if spread < 22:
        return "gray"
    if red > 1.25 * green and red > 1.25 * blue:
        if blue > green * 1.18:
            return "pink"
        if green > blue * 1.35:
            return "orange"
        return "red"
    if blue > red * 1.2 and blue > green * 1.08:
        return "blue"
    if red > green * 1.12 and blue > green * 1.12:
        return "purple"
    if green > red * 1.15 and green > blue * 1.08:
        return "green"
    if green > red * 1.05 and blue > red * 1.05:
        return "cyan"
    if red > blue * 1.35 and green > blue * 1.25:
        return "beige" if brightness > 140 else "brown"
    return "neutral"


def _palette(image: Image.Image) -> tuple[list[str], list[list[int]]]:
    reduced = image.convert("RGB")
    reduced.thumbnail((192, 192), Image.Resampling.LANCZOS)
    quantized = reduced.quantize(colors=6, method=Image.Quantize.MEDIANCUT)
    raw_palette = quantized.getpalette() or []
    counts = sorted(
        quantized.getcolors(maxcolors=256) or [],
        key=lambda row: row[0],
        reverse=True,
    )
    names: list[str] = []
    colours: list[list[int]] = []
    for _, index in counts[:6]:
        offset = index * 3
        rgb = raw_palette[offset:offset + 3]
        if len(rgb) != 3:
            continue
        name = _colour_name(*rgb)
        if name not in names:
            names.append(name)
        colours.append([int(value) for value in rgb])
    return names[:4], colours[:6]


def _hsv_histogram(image: Image.Image) -> list[float]:
    hsv = image.convert("RGB").resize((128, 128), Image.Resampling.BILINEAR)
    hsv = hsv.convert("HSV")
    hue = [0.0] * 16
    saturation = [0.0] * 4
    value = [0.0] * 4
    pixels = _pixels(hsv)
    for h, s, v in pixels:
        hue[min(int(h) * 16 // 256, 15)] += 1
        saturation[min(int(s) * 4 // 256, 3)] += 1
        value[min(int(v) * 4 // 256, 3)] += 1
    total = float(len(pixels) or 1)
    return [round(item / total, 6) for item in hue + saturation + value]


def build_local_visual_fingerprint(image_bytes: bytes) -> dict[str, Any]:
    image = Image.open(io.BytesIO(image_bytes))
    image.seek(0)
    image = image.convert("RGB")
    width, height = image.size
    hsv = image.resize((128, 128), Image.Resampling.BILINEAR).convert("HSV")
    stats = ImageStat.Stat(hsv)
    saturation = float(stats.mean[1]) / 255
    brightness = float(stats.mean[2]) / 255
    palette_names, palette_rgb = _palette(image)
    if brightness < 0.28:
        lighting = "dim"
    elif brightness > 0.72:
        lighting = "bright"
    else:
        lighting = "balanced"
    if saturation > 0.52 and any(
        colour in {"pink", "purple", "blue", "red"} for colour in palette_names
    ):
        visual_tone = "saturated colored"
    elif any(colour in {"orange", "beige", "brown", "red"} for colour in palette_names):
        visual_tone = "warm"
    elif any(colour in {"blue", "cyan"} for colour in palette_names):
        visual_tone = "cool"
    else:
        visual_tone = "neutral"
    orientation = (
        "portrait" if height > width * 1.12
        else "landscape" if width > height * 1.12
        else "square"
    )
    return {
        "version": SHOOT_FINGERPRINT_VERSION,
        "dhash": _dhash(image),
        "hsv_histogram": _hsv_histogram(image),
        "palette_names": palette_names,
        "palette_rgb": palette_rgb,
        "brightness": round(brightness, 4),
        "saturation": round(saturation, 4),
        "lighting": lighting,
        "visual_tone": visual_tone,
        "orientation": orientation,
        "width": width,
        "height": height,
    }


def _nova_embedding_sync(
    image_bytes: bytes,
    *,
    client: Any | None = None,
) -> tuple[list[float], str]:
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
            raise ShootEmbeddingConfigurationError(
                "boto3 is not installed for visual shoot embeddings"
            ) from exc
    else:
        # Tests and offline calibration can inject a protocol-compatible
        # runtime without importing the AWS SDK.
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
        os.environ.get("VAULT_SHOOT_EMBEDDING_MODEL")
        or NOVA_EMBEDDING_MODEL
    ).strip()
    dimension = _embedding_dimension()
    region = str(
        os.environ.get("AWS_BEDROCK_REGION") or "us-east-1"
    ).strip()
    runtime = client or boto3.client("bedrock-runtime", region_name=region)
    request = {
        "schemaVersion": "nova-multimodal-embed-v1",
        "taskType": "SINGLE_EMBEDDING",
        "singleEmbeddingParams": {
            "embeddingPurpose": "CLUSTERING",
            "embeddingDimension": dimension,
            "image": {
                "detailLevel": "STANDARD_IMAGE",
                "format": "jpeg",
                "source": {
                    "bytes": base64.b64encode(image_bytes).decode("ascii"),
                },
            },
        },
    }
    try:
        response = runtime.invoke_model(
            body=json.dumps(request),
            modelId=model_id,
            accept="application/json",
            contentType="application/json",
        )
        body = response.get("body")
        raw = body.read() if hasattr(body, "read") else body
        payload = json.loads(raw or "{}")
        embeddings = payload.get("embeddings") or []
        vector = embeddings[0].get("embedding") if embeddings else None
        normalized = _normalize(vector or [])
        if len(normalized) != dimension:
            raise ShootEmbeddingError(
                "Nova returned an invalid visual embedding dimension"
            )
        return normalized, model_id
    except (NoCredentialsError, PartialCredentialsError) as exc:
        raise ShootEmbeddingConfigurationError(
            "AWS credentials are missing for visual shoot embeddings"
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
        }:
            raise ShootEmbeddingConfigurationError(
                f"Amazon Bedrock visual embeddings are unavailable: {message}"
            ) from exc
        raise ShootEmbeddingError(
            f"Amazon Bedrock could not fingerprint the image: {message}"
        ) from exc
    except BotoCoreError as exc:
        raise ShootEmbeddingError(
            f"Amazon Bedrock visual embedding failed: {type(exc).__name__}"
        ) from exc


async def build_shoot_fingerprint(image_bytes: bytes) -> dict[str, Any]:
    """Build local visual evidence and, when configured, a Nova embedding.

    A Bedrock failure never discards successful Rekognition classification.
    Configuration failures open a process-local circuit breaker so a whole
    vault run does not repeat the same denied request hundreds of times.
    """
    global _embedding_disabled_reason
    local = build_local_visual_fingerprint(image_bytes)
    result: dict[str, Any] = {
        "version": SHOOT_FINGERPRINT_VERSION,
        "status": "local_only",
        "local": local,
    }
    if not _env_bool("VAULT_SHOOT_EMBEDDINGS_ENABLED", True):
        result["reason"] = "disabled"
        return result
    if _embedding_disabled_reason:
        result["reason"] = _embedding_disabled_reason
        return result
    try:
        vector, model_id = await asyncio.to_thread(
            _nova_embedding_sync,
            image_bytes,
        )
    except ShootEmbeddingConfigurationError as exc:
        _embedding_disabled_reason = str(exc)[:240]
        result["reason"] = _embedding_disabled_reason
        return result
    except ShootEmbeddingError as exc:
        result["reason"] = str(exc)[:240]
        return result
    result.update({
        "status": "ready",
        "provider": "amazon_nova_multimodal_embeddings",
        "model": model_id,
        "purpose": "CLUSTERING",
        "dimension": len(vector),
        "embedding": vector,
    })
    return result


def _metadata(item: dict[str, Any]) -> dict[str, Any]:
    value = item.get("classification_metadata")
    return value if isinstance(value, dict) else {}


def shoot_fingerprint(item: dict[str, Any]) -> dict[str, Any]:
    value = _metadata(item).get("shoot_fingerprint")
    return value if isinstance(value, dict) else {}


def _embedding(item: dict[str, Any]) -> list[float]:
    values = shoot_fingerprint(item).get("embedding")
    if not isinstance(values, list):
        return []
    try:
        return [float(value) for value in values]
    except (TypeError, ValueError):
        return []


def _local(item: dict[str, Any]) -> dict[str, Any]:
    value = shoot_fingerprint(item).get("local")
    return value if isinstance(value, dict) else {}


def _hamming_similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    try:
        distance = (int(left, 16) ^ int(right, 16)).bit_count()
    except (TypeError, ValueError):
        return 0.0
    return 1.0 - (distance / 64)


def _specific_text(value: Any) -> str:
    text = " ".join(str(value or "").lower().split())
    if text in _EMPTY or text in {
        "nude",
        "clothed",
        "partially nude",
        "partially clothed nude",
        "lingerie or underwear",
    }:
        return ""
    return text


def _structured_compatible(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_location = _specific_text(left.get("scene_location"))
    right_location = _specific_text(right.get("scene_location"))
    if left_location and right_location and left_location != right_location:
        return False
    return True


def shoot_similarity(left: dict[str, Any], right: dict[str, Any]) -> float:
    """Return fused visual similarity, or zero without authoritative vectors."""
    if not _structured_compatible(left, right):
        return 0.0
    left_embedding = _embedding(left)
    right_embedding = _embedding(right)
    if not left_embedding or len(left_embedding) != len(right_embedding):
        return 0.0
    nova = cosine_similarity(left_embedding, right_embedding)
    left_local = _local(left)
    right_local = _local(right)
    histogram = cosine_similarity(
        left_local.get("hsv_histogram") or [],
        right_local.get("hsv_histogram") or [],
    )
    duplicate = _hamming_similarity(
        str(left_local.get("dhash") or ""),
        str(right_local.get("dhash") or ""),
    )
    # Nova is authoritative.  Palette helps distinguish lighting/outfit
    # variants; dHash is only a small bonus for crops and near-duplicates.
    return round((nova * 0.86) + (histogram * 0.10) + (duplicate * 0.04), 6)


def _media_id(item: dict[str, Any]) -> str:
    return str(
        item.get("fansly_media_id")
        or item.get("media_id")
        or item.get("id")
        or ""
    )


def _cluster_id(items: list[dict[str, Any]]) -> str:
    digest = hashlib.sha1(
        "|".join(sorted(_media_id(item) for item in items)).encode("utf-8")
    ).hexdigest()[:12]
    return f"shoot-{digest}"


def _named_album(item: dict[str, Any]) -> str:
    value = " ".join(str(item.get("album_title") or "").strip().split())
    if not value:
        return ""
    lowered = value.lower()
    if lowered.startswith(_GENERIC_ALBUM_PREFIXES):
        return ""
    return value


def build_shoot_clusters(
    items: list[dict[str, Any]],
    *,
    min_similarity: float | None = None,
) -> list[dict[str, Any]]:
    """Conservatively cluster media by complete-link visual similarity.

    Complete-link means every item must be similar enough to every other item
    in the merged cluster.  This prevents A≈B and B≈C from merging A with C
    when A and C are actually different shoots.
    """
    threshold = _min_similarity() if min_similarity is None else min_similarity
    embedded = sorted(
        [item for item in items if _embedding(item)],
        key=_media_id,
    )
    pair_scores: dict[tuple[int, int], float] = {}
    for left_index in range(len(embedded)):
        for right_index in range(left_index + 1, len(embedded)):
            pair_scores[(left_index, right_index)] = shoot_similarity(
                embedded[left_index],
                embedded[right_index],
            )

    def pair_score(left_index: int, right_index: int) -> float:
        if left_index == right_index:
            return 1.0
        key = (
            (left_index, right_index)
            if left_index < right_index
            else (right_index, left_index)
        )
        return pair_scores.get(key, 0.0)

    # Build deterministic maximal complete-link groups. Choosing the seed with
    # the most current neighbours avoids an isolated edge claiming the middle
    # of a clear photoshoot. Pairwise scores are calculated exactly once, so a
    # 1,000-item vault does not turn agglomerative clustering into O(n^4).
    remaining = set(range(len(embedded)))
    cluster_indexes: list[list[int]] = []
    while remaining:
        seed = min(
            remaining,
            key=lambda index: (
                -sum(
                    pair_score(index, other) >= threshold
                    for other in remaining
                    if other != index
                ),
                _media_id(embedded[index]),
            ),
        )
        cluster = [seed]
        remaining.remove(seed)
        while True:
            eligible = [
                (
                    min(pair_score(candidate, member) for member in cluster),
                    candidate,
                )
                for candidate in remaining
                if all(
                    pair_score(candidate, member) >= threshold
                    for member in cluster
                )
            ]
            if not eligible:
                break
            eligible.sort(
                key=lambda row: (
                    -row[0],
                    _media_id(embedded[row[1]]),
                )
            )
            _, candidate = eligible[0]
            cluster.append(candidate)
            remaining.remove(candidate)
        cluster_indexes.append(cluster)
    clusters = [
        [embedded[index] for index in indexes]
        for indexes in cluster_indexes
    ]

    results: list[dict[str, Any]] = []
    embedded_ids = {_media_id(item) for item in embedded}
    for cluster in clusters:
        if len(cluster) > 1:
            confidence = min(
                shoot_similarity(cluster[i], cluster[j])
                for i in range(len(cluster))
                for j in range(i + 1, len(cluster))
            )
        else:
            confidence = 1.0
        results.append({
            "shoot_id": _cluster_id(cluster),
            "method": "visual_embedding",
            "confidence": round(confidence, 4),
            "items": sorted(cluster, key=_media_id),
        })

    # Safe compatibility path for pre-fingerprint media: only a creator-named
    # album is accepted. Generic albums and classifier scene slugs are not
    # evidence of a photoshoot and remain intentionally ungrouped.
    legacy: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        if _media_id(item) in embedded_ids:
            continue
        album = _named_album(item)
        if album:
            legacy.setdefault(album.lower(), []).append(item)
        else:
            results.append({
                "shoot_id": _cluster_id([item]),
                "method": "unresolved",
                "confidence": 0.0,
                "items": [item],
            })
    for album, album_items in legacy.items():
        results.append({
            "shoot_id": _cluster_id(album_items),
            "method": "named_album",
            "confidence": 0.6,
            "album": album,
            "items": sorted(album_items, key=_media_id),
        })
    results.sort(
        key=lambda row: (
            -len(row["items"]),
            -float(row["confidence"]),
            row["shoot_id"],
        )
    )
    return results


def cluster_debug_summary(cluster: dict[str, Any]) -> dict[str, Any]:
    items = cluster.get("items") or []
    palettes = Counter(
        colour
        for item in items
        for colour in (_local(item).get("palette_names") or [])
    )
    locations = Counter(
        str(item.get("scene_location") or "")
        for item in items
        if str(item.get("scene_location") or "").lower() not in _EMPTY
    )
    outfits = Counter(
        str(item.get("scene_outfit") or "")
        for item in items
        if str(item.get("scene_outfit") or "").lower() not in _EMPTY
    )
    return {
        "shoot_id": cluster.get("shoot_id"),
        "method": cluster.get("method"),
        "confidence": cluster.get("confidence"),
        "media_ids": [_media_id(item) for item in items],
        "location": locations.most_common(1)[0][0] if locations else None,
        "outfit": outfits.most_common(1)[0][0] if outfits else None,
        "palette": [name for name, _ in palettes.most_common(5)],
    }
