import base64
import struct

from services.vault_semantics import (
    normalize_semantic_payload,
    qwen_fallback_reasons,
)


def payload():
    data = base64.b64encode(struct.pack("<16e", *([0.25] * 16))).decode()
    labels = {
        "scene_location": "kitchen",
        "wardrobe_state": "full nudity",
        "pose": "sitting",
        "activity": "posing",
        "framing": "medium shot",
        "lighting": "cool blue light",
    }
    return {
        "model": "google/siglip2-base-patch16-224",
        "revision": "75de2d55",
        "latency_ms": "bad",
        "confidence": 0.87,
        "ambiguous_axes": ["pose", "invented"],
        "axes": {
            name: {
                "ranked": [
                    {"label": label, "score": 0.8},
                    {"label": "invented", "score": 0.7},
                ],
                "margin": 0.1,
                "confident": True,
            }
            for name, label in labels.items()
        },
        "tags": {
            "background_details": [
                {"label": "kitchen counter and cabinets", "score": 0.8},
                {"label": "invented", "score": 1},
            ],
            "wardrobe_items": [],
        },
        "embedding": {
            "encoding": "float16_base64",
            "dimensions": 16,
            "data": data,
        },
    }


def test_semantic_payload_is_bounded_to_controlled_taxonomy():
    result = normalize_semantic_payload(payload())
    assert result["latency_ms"] == 0
    assert result["ambiguous_axes"] == ["pose"]
    assert result["axes"]["scene_location"]["label"] == "kitchen"
    assert result["tags"]["background_details"] == [{
        "label": "kitchen counter and cabinets",
        "score": 0.8,
    }]
    assert result["embedding"]["dimensions"] == 16


def test_full_nudity_without_detector_evidence_requests_qwen():
    semantic = normalize_semantic_payload(payload())
    assert qwen_fallback_reasons(
        semantic,
        exposed_anatomy=[],
    ) == ["nudity_detector_conflict"]
    assert qwen_fallback_reasons(
        semantic,
        exposed_anatomy=["breasts"],
    ) == []
