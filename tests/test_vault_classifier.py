import io
from pathlib import Path

import httpx
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


def test_qwen_preserves_rich_scene_and_continuity_details():
    base = vault_classifier._base_metadata(
        [],
        is_video=False,
        album_title="Pink bedroom",
        local_visual=local_visual(),
    )
    enriched = vault_classifier._merge_qwen(base, {
        "description": (
            "The creator poses on white quilted bedding beside a dark wood "
            "headboard. Pink mesh lingerie and warm bedside light define the "
            "shoot's look."
        ),
        "explicitness": 3,
        "nudity": "partial",
        "participants": 1,
        "scene_location": "bedroom",
        "scene_outfit": "partially removed lingerie",
        "setting_details": ["white quilted bedding", "dark wood headboard"],
        "background_details": ["cream wall", "small bedside table"],
        "wardrobe_items": ["mesh lingerie set", "silver necklace"],
        "wardrobe_colors": ["pink", "silver"],
        "wardrobe_materials": ["sheer mesh", "ribbed trim"],
        "subject_styling": ["short blonde hair", "pink lipstick"],
        "scene_lighting": "warm bedside light from camera left",
        "colors": ["pink lingerie", "white bedding", "dark brown headboard"],
        "visual_style": "warm bedroom selfie",
        "continuity_markers": [
            "pink mesh lingerie",
            "white quilted bedding",
            "dark wood headboard",
            "warm camera-left light",
        ],
        "distinguishing_details": ["silver pendant necklace"],
        "tags": ["bedroom", "pink lingerie"],
        "confidence": 0.87,
    })
    descriptor = enriched["rich_visual_descriptor"]["descriptor"]
    assert enriched["description_complete"] is True
    assert enriched["scene_location"] == "bedroom"
    assert "pink" in enriched["scene_outfit"]
    assert descriptor["wardrobe_materials"] == ["sheer mesh", "ribbed trim"]
    assert descriptor["continuity_markers"] == [
        "pink mesh lingerie",
        "white quilted bedding",
        "dark wood headboard",
        "warm camera-left light",
    ]
    assert enriched["_provider_metadata"]["vision_status"] == "ready"


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
    assert result["_provider_metadata"]["vision_error_reason"] == "RuntimeError"


def test_modal_failures_are_reduced_to_safe_actionable_reasons():
    request = httpx.Request("POST", "https://example.modal.run")
    response = httpx.Response(401, request=request)
    assert vault_classifier._vision_failure_reason(
        httpx.HTTPStatusError("denied", request=request, response=response)
    ) == "http_401"
    assert vault_classifier._vision_failure_reason(
        httpx.ReadTimeout("slow", request=request)
    ) == "timeout"


def test_modal_image_includes_qwen_processor_runtime_dependencies():
    source = (
        Path(__file__).resolve().parents[1]
        / "deploy"
        / "modal_qwen_vl.py"
    ).read_text()
    assert '"torchvision>=0.21"' in source


@pytest.mark.asyncio
async def test_qwen_client_follows_modal_result_redirects(monkeypatch):
    observed = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"text": '{"description":"ready","confidence":0.8}'}

    class FakeClient:
        def __init__(self, **kwargs):
            observed["client"] = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, url, **kwargs):
            observed["url"] = url
            observed["request"] = kwargs
            return FakeResponse()

    monkeypatch.setattr(vault_classifier.httpx, "AsyncClient", FakeClient)
    monkeypatch.setenv(
        "VAULT_VISION_BASE_URL",
        "https://example.modal.run",
    )
    monkeypatch.setenv("VAULT_VISION_TIMEOUT_SECONDS", "620")

    result = await vault_classifier._qwen_metadata(
        jpeg_bytes(),
        is_video=False,
        album_title="Bedroom",
        filename="one.jpg",
    )

    assert observed["client"]["follow_redirects"] is True
    assert observed["client"]["timeout"] == 620
    assert observed["url"] == "https://example.modal.run"
    assert result["description"] == "ready"


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
    assert result["classification_version"] == 8
