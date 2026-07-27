import asyncio
import base64
import io
import math
import struct

from PIL import Image

from db.queries import propose_sets
from services.shoot_fingerprint import (
    add_semantic_shoot_evidence,
    build_local_visual_fingerprint,
    build_shoot_fingerprint,
    build_shoot_clusters,
    shoot_similarity,
)


def unit(angle_degrees: float) -> list[float]:
    radians = math.radians(angle_degrees)
    return [math.cos(radians), math.sin(radians)]


def item(
    media_id: str,
    vector: list[float] | None,
    *,
    album: str = "Album_123",
    location: str = "bedroom",
    outfit: str = "partially clothed nude",
    colour: str = "pink",
    level: int = 4,
    semantic: dict | None = None,
) -> dict:
    fingerprint = {
        "status": "local_only",
        "provider": "pillow_local",
        "local": ({
            "hsv_histogram": vector,
            "dhash": "0000000000000000",
            "palette_names": [colour, "white"],
            "visual_tone": "warm",
        } if vector else {}),
    }
    if semantic:
        fingerprint.update({
            "status": "local_plus_vision",
            "provider": "pillow_local+qwen_vl",
            "semantic": semantic,
        })
    return {
        "fansly_media_id": media_id,
        "content_category": "nude_photo",
        "explicitness_level": level,
        "scene_id": "album-123-bedroom-partially-clothed-nude",
        "scene_location": location,
        "scene_outfit": outfit,
        "scene_lighting": "bright",
        "album_title": album,
        "mimetype": "image/jpeg",
        "price_min": 15,
        "price_max": 80,
        "tags": ["bedroom", colour, "full nudity"],
        "good_for": "closer",
        "ai_description": (
            f"A {colour}-toned bedroom frame from the same photoshoot."
        ),
        "classification_metadata": {
            "nudity": "full",
            "visible_anatomy": ["breasts"],
            "pose": "lying",
            "framing": "portrait",
            "colors": [colour, "white"],
            "shoot_fingerprint": fingerprint,
        },
    }


def jpeg_bytes(colour=(230, 80, 160)) -> bytes:
    image = Image.new("RGB", (120, 180), colour)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def encoded_embedding(values: list[float]) -> dict:
    return {
        "model": "google/siglip2-base-patch16-224",
        "encoding": "float16_base64",
        "dimensions": len(values),
        "data": base64.b64encode(
            struct.pack(f"<{len(values)}e", *values)
        ).decode(),
    }


def test_local_fingerprint_captures_palette_lighting_and_geometry():
    result = build_local_visual_fingerprint(jpeg_bytes())
    assert result["version"] == 2
    assert result["orientation"] == "portrait"
    assert result["lighting"] in {"balanced", "bright"}
    assert result["palette_names"]
    assert len(result["hsv_histogram"]) == 24
    assert len(result["dhash"]) == 16


def test_shoot_fingerprint_is_fully_local():
    result = asyncio.run(build_shoot_fingerprint(jpeg_bytes()))
    assert result["status"] == "local_only"
    assert result["provider"] == "pillow_local"
    assert "embedding" not in result
    assert result["local"]["hsv_histogram"]


def test_qwen_scene_evidence_is_attached_without_replacing_local_pixels():
    fingerprint = asyncio.run(build_shoot_fingerprint(jpeg_bytes()))
    enriched = add_semantic_shoot_evidence(fingerprint, {
        "rich_visual_descriptor": {
            "status": "ready",
            "descriptor": {
                "setting_location": "bedroom",
                "setting_details": ["white quilted bedding"],
                "background_details": ["dark wood headboard"],
                "wardrobe_items": ["pink mesh lingerie"],
                "wardrobe_colors": ["pink"],
                "wardrobe_materials": ["mesh"],
                "subject_styling": ["short blonde hair"],
                "lighting": "warm camera-left light",
                "visual_style": "warm bedroom selfie",
                "continuity_markers": [
                    "white quilted bedding",
                    "dark wood headboard",
                ],
            },
        },
    })
    assert enriched["status"] == "local_plus_vision"
    assert enriched["local"] == fingerprint["local"]
    assert enriched["semantic"]["setting_location"] == "bedroom"


def test_siglip_embedding_is_attached_without_expanding_railway_runtime():
    fingerprint = asyncio.run(build_shoot_fingerprint(jpeg_bytes()))
    embedding = encoded_embedding([1.0] + ([0.0] * 15))
    enriched = add_semantic_shoot_evidence(fingerprint, {
        "_semantic_fingerprint": {
            "model": "google/siglip2-base-patch16-224",
            "revision": "75de2d55",
            "embedding": embedding,
            "confidence": 0.88,
        },
        "rich_visual_descriptor": {
            "status": "ready",
            "descriptor": {
                "setting_location": "bedroom",
                "background_details": ["bed and bedding"],
                "wardrobe_items": ["lingerie set"],
                "lighting": "warm indoor light",
            },
        },
    })
    assert enriched["provider"] == "pillow_local+siglip2"
    assert enriched["embedding"]["dimensions"] == 16
    assert enriched["semantic_confidence"] == 0.88


def test_strong_scene_continuity_bridges_pose_and_crop_changes():
    semantic = {
        "setting_location": "bedroom",
        "setting_details": ["white quilted bedding", "dark wood headboard"],
        "background_details": ["cream wall", "small bedside table"],
        "wardrobe_items": ["pink mesh lingerie"],
        "wardrobe_colors": ["pink"],
        "wardrobe_materials": ["sheer mesh"],
        "subject_styling": ["short blonde hair", "pink lipstick"],
        "lighting": "warm bedside light from camera left",
        "visual_style": "warm bedroom selfie",
        "continuity_markers": [
            "white quilted bedding",
            "dark wood headboard",
            "pink mesh lingerie",
            "warm camera-left light",
        ],
    }
    rows = [
        item("wide", unit(0), semantic=semantic),
        item("close-crop", unit(45), semantic=semantic),
    ]
    assert shoot_similarity(rows[0], rows[1]) >= 0.94
    clusters = build_shoot_clusters(rows)
    assert len(clusters) == 1
    assert clusters[0]["method"] == "local_visual+vision_metadata"


def test_embedding_and_controlled_tags_bridge_crop_changes_conservatively():
    semantic = {
        "setting_location": "bedroom",
        "setting_details": ["bed and bedding"],
        "background_details": ["bed and bedding", "plain wall"],
        "wardrobe_items": ["lingerie set"],
        "wardrobe_colors": ["pink"],
        "lighting": "warm indoor light",
        "continuity_markers": [
            "bed and bedding",
            "lingerie set",
            "warm indoor light",
        ],
    }
    rows = [
        item("wide", unit(0), semantic=semantic),
        item("crop", unit(43), semantic=semantic),
    ]
    vector = encoded_embedding([1.0] + ([0.0] * 15))
    for row in rows:
        row["classification_metadata"]["shoot_fingerprint"][
            "embedding"
        ] = vector
    assert shoot_similarity(rows[0], rows[1]) >= 0.94
    clusters = build_shoot_clusters(rows)
    assert len(clusters) == 1
    assert clusters[0]["method"] == "local_visual+semantic_embedding"


def test_local_visual_evidence_overrides_same_generic_album_and_scene_slug():
    first = [
        item(f"pink-{index}", unit(index), colour="pink")
        for index in (0, 2, 4)
    ]
    second = [
        item(f"blue-{index}", unit(88 + index), colour="blue")
        for index in (0, 2, 4)
    ]
    clusters = build_shoot_clusters(first + second, min_similarity=0.9)
    grouped = [
        {row["fansly_media_id"] for row in cluster["items"]}
        for cluster in clusters
        if len(cluster["items"]) > 1
    ]
    assert {frozenset(group) for group in grouped} == {
        frozenset({"pink-0", "pink-2", "pink-4"}),
        frozenset({"blue-0", "blue-2", "blue-4"}),
    }


def test_complete_link_does_not_chain_two_different_shoots():
    rows = [
        item("a", unit(0)),
        item("b", unit(20)),
        item("c", unit(40)),
    ]
    assert shoot_similarity(rows[0], rows[1]) > 0.9
    assert shoot_similarity(rows[1], rows[2]) > 0.9
    assert shoot_similarity(rows[0], rows[2]) < 0.9
    clusters = build_shoot_clusters(rows, min_similarity=0.9)
    sizes = sorted(len(cluster["items"]) for cluster in clusters)
    assert sizes == [1, 2]


def test_generic_legacy_album_is_not_accepted_as_shoot_evidence():
    rows = [
        item("a", None),
        item("b", None),
        item("c", None),
    ]
    clusters = build_shoot_clusters(rows)
    assert all(cluster["method"] == "unresolved" for cluster in clusters)
    assert propose_sets(rows) == []


def test_named_creator_album_remains_a_safe_legacy_fallback():
    rows = [
        item(f"named-{index}", None, album="Pink bedroom shoot")
        for index in range(3)
    ]
    proposed = propose_sets(rows)
    assert len(proposed) == 1
    assert proposed[0]["shoot_method"] == "named_album"
    assert proposed[0]["media_ids"] == [
        "named-0",
        "named-1",
        "named-2",
    ]


def test_set_builder_keeps_photoshoot_identity_across_explicitness_progression():
    rows = [
        item(
            f"pink-{index}",
            unit(index),
            colour="pink",
            level=level,
        )
        for index, level in enumerate((2, 3, 4))
    ] + [
        item(
            f"blue-{index}",
            unit(90 + index),
            colour="blue",
            level=level,
        )
        for index, level in enumerate((3, 4, 4))
    ]
    proposed = propose_sets(rows)
    assert len(proposed) == 2
    assert {
        frozenset(vault_set["media_ids"])
        for vault_set in proposed
    } == {
        frozenset({"pink-0", "pink-1", "pink-2"}),
        frozenset({"blue-0", "blue-1", "blue-2"}),
    }
    pink = next(
        vault_set
        for vault_set in proposed
        if "pink-0" in vault_set["media_ids"]
    )
    assert pink["explicit_min"] == 2
    assert pink["explicit_max"] == 4
    assert "visually matched photoshoot" in pink["description"].lower()
    assert "dominant visual palette" in pink["description"].lower()
    assert "visible progression" in pink["description"].lower()
