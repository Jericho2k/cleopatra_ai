import io

import pytest
from PIL import Image

import main
from services import vault_classifier


def jpeg_bytes() -> bytes:
    image = Image.new("RGB", (120, 180), (210, 90, 150))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def local_visual() -> dict:
    return {
        "palette_names": ["pink", "white"],
        "lighting": "bright",
        "width": 120,
        "height": 180,
    }


def test_nudenet_evidence_is_authoritative_for_explicitness():
    result = vault_classifier._base_metadata(
        [
            {
                "class": "FEMALE_BREAST_EXPOSED",
                "score": 0.91,
                "box": [10, 20, 30, 40],
            },
            {
                "class": "FEMALE_GENITALIA_EXPOSED",
                "score": 0.87,
                "box": [20, 80, 25, 30],
            },
        ],
        is_video=False,
        album_title="Pink bedroom",
        local_visual=local_visual(),
    )
    assert result["explicitness"] == 4
    assert result["nudity"] == "full"
    assert result["category"] == "nude_photo"
    assert result["visible_anatomy"] == ["breasts", "vulva"]
    assert result["_provider_metadata"]["provider"] == "local_nudenet"


def test_qwen_cannot_lower_nudenet_adult_evidence():
    base = vault_classifier._base_metadata(
        [{
            "class": "FEMALE_BREAST_EXPOSED",
            "score": 0.91,
            "box": [10, 20, 30, 40],
        }],
        is_video=False,
        album_title="Pink bedroom",
        local_visual=local_visual(),
    )
    enriched = vault_classifier._merge_qwen(base, {
        "description": "A creator poses in a bedroom.",
        "explicitness": 1,
        "nudity": "none",
        "visible_anatomy": [],
        "participants": 1,
        "sexual_activity": [],
        "confidence": 0.8,
    })
    assert enriched["explicitness"] == 4
    assert enriched["category"] == "nude_photo"
    assert enriched["visible_anatomy"] == ["breasts"]
    assert enriched["_provider_metadata"]["explicitness_escalated"] is True


@pytest.mark.asyncio
async def test_qwen_failure_falls_back_to_local_result(monkeypatch):
    monkeypatch.setattr(
        vault_classifier,
        "_detect",
        lambda _: [{
            "class": "BUTTOCKS_EXPOSED",
            "score": 0.88,
            "box": [10, 20, 30, 40],
        }],
    )

    async def fail(*args, **kwargs):
        raise RuntimeError("endpoint unavailable")

    monkeypatch.setattr(vault_classifier, "_qwen_metadata", fail)
    monkeypatch.setenv("VAULT_VISION_BASE_URL", "https://example.invalid")

    result = await vault_classifier.classify_vault_image(
        jpeg_bytes(),
        is_video=False,
        local_visual=local_visual(),
    )
    assert result["category"] == "nude_photo"
    assert result["_provider_metadata"]["vision_status"] == "fallback"
    assert result["_provider_metadata"]["vision_error"] == "RuntimeError"


@pytest.mark.asyncio
async def test_main_uses_local_classifier_without_anthropic(monkeypatch):
    async def load_visual(item, *, is_video):
        return jpeg_bytes(), "image", "test"

    async def classify(*args, **kwargs):
        return {
            **vault_classifier._base_metadata(
                [],
                is_video=False,
                album_title="Pink bedroom",
                local_visual=local_visual(),
            ),
            "_classification_model": "nudenet-3.4.2",
        }

    monkeypatch.setattr(main, "_load_vault_visual", load_visual)
    monkeypatch.setattr(main, "classify_vault_image", classify)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)

    result = await main._categorize_single_item({
        "id": "media-1",
        "mimetype": "image/jpeg",
        "filename": "one.jpg",
        "album_title": "Pink bedroom",
    })

    assert result["classification_model"] == "nudenet-3.4.2"
    assert result["classification_metadata"]["classifier_provider"] == "local_nudenet"
    assert result["classification_version"] == 7
