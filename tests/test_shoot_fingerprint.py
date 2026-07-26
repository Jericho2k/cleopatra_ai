import asyncio
import io
import json
import math
from types import SimpleNamespace

import pytest
from PIL import Image

from db.queries import propose_sets
from services.shoot_fingerprint import (
    ShootEmbeddingConfigurationError,
    _request_embedding,
    build_local_visual_fingerprint,
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
) -> dict:
    fingerprint = {
        "status": "ready" if vector else "local_only",
        "embedding": vector or [],
        "local": {
            "hsv_histogram": [1.0, 0.0],
            "dhash": "0000000000000000",
            "palette_names": [colour, "white"],
            "visual_tone": "warm",
        },
    }
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


def test_local_fingerprint_captures_palette_lighting_and_geometry():
    result = build_local_visual_fingerprint(jpeg_bytes())
    assert result["version"] == 1
    assert result["orientation"] == "portrait"
    assert result["lighting"] in {"balanced", "bright"}
    assert result["palette_names"]
    assert len(result["hsv_histogram"]) == 24
    assert len(result["dhash"]) == 16


def test_embedding_request_sends_the_image_as_a_data_uri(monkeypatch):
    monkeypatch.setenv("VAULT_SHOOT_EMBEDDING_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("VAULT_SHOOT_EMBEDDING_MODEL", "test-embedder")

    class Embeddings:
        request = None

        async def create(self, **kwargs):
            Embeddings.request = kwargs
            return SimpleNamespace(
                data=[SimpleNamespace(embedding=[1.0] + ([0.0] * 383))]
            )

    class Client:
        embeddings = Embeddings()

    vector, model = asyncio.run(_request_embedding(jpeg_bytes(), client=Client()))
    assert Embeddings.request["model"] == "test-embedder"
    assert Embeddings.request["input"][0].startswith("data:image/jpeg;base64,")
    assert len(vector) == 384
    assert model == "test-embedder"


def test_a_missing_endpoint_opens_the_circuit_breaker(monkeypatch):
    # A configuration failure must not be retried once per image for a whole
    # vault run.
    monkeypatch.delenv("VAULT_SHOOT_EMBEDDING_BASE_URL", raising=False)
    monkeypatch.delenv("SELF_HOSTED_BASE_URL", raising=False)
    with pytest.raises(ShootEmbeddingConfigurationError):
        asyncio.run(_request_embedding(jpeg_bytes()))


def test_visual_embedding_overrides_same_generic_album_and_scene_slug():
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
