import io
import json
from types import SimpleNamespace

import pytest
from PIL import Image

import anthropic
import main


def jpeg_bytes() -> bytes:
    image = Image.new("RGB", (120, 180), (210, 90, 150))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def classifier_payload() -> dict:
    return {
        "category": "lingerie_photo",
        "description": "An adult creator poses in a pink bedroom set.",
        "mood": "teasing",
        "explicitness": 3,
        "nudity": "none",
        "visible_anatomy": [],
        "good_for": "mid_session",
        "tags": ["pink lingerie", "bedroom"],
        "sexual_activity": [],
        "body_focus": ["full body"],
        "action": "posing",
        "pose": "standing",
        "framing": "full body",
        "props": [],
        "colors": ["pink"],
        "scene_location": "bedroom",
        "scene_outfit": "pink lingerie",
        "scene_lighting": "bright",
        "scene_id": "pink-bedroom-lingerie",
        "possible_minor": False,
        "age_note": "",
        "confidence": 0.92,
    }


@pytest.mark.asyncio
async def test_vault_classifier_uses_existing_anthropic_configuration(
    monkeypatch,
):
    async def load_visual(item, *, is_video):
        return jpeg_bytes(), "image", "test"

    class Messages:
        async def create(self, **kwargs):
            assert kwargs["model"] == "claude-sonnet-4-6"
            assert kwargs["messages"][0]["content"][0]["type"] == "image"
            return SimpleNamespace(
                content=[
                    SimpleNamespace(
                        type="text",
                        text=json.dumps(classifier_payload()),
                    )
                ],
                stop_reason="end_turn",
            )

    class Client:
        def __init__(self, *, api_key):
            assert api_key == "existing-key"
            self.messages = Messages()

    monkeypatch.setattr(main, "_load_vault_visual", load_visual)
    monkeypatch.setattr(anthropic, "AsyncAnthropic", Client)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "existing-key")
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)

    result = await main._categorize_single_item({
        "id": "media-1",
        "mimetype": "image/jpeg",
        "filename": "one.jpg",
        "album_title": "Pink bedroom",
    })

    assert result["classification_model"] == "claude-sonnet-4-6"
    assert result["classification_metadata"]["classifier_provider"] == "anthropic"
    assert result["classification_metadata"]["shoot_fingerprint"]["provider"] == "pillow_local"
    assert result["classification_version"] == 7


@pytest.mark.asyncio
async def test_vault_classifier_reports_missing_existing_key(monkeypatch):
    async def load_visual(item, *, is_video):
        return jpeg_bytes(), "image", "test"

    monkeypatch.setattr(main, "_load_vault_visual", load_visual)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(
        main.VaultClassifierConfigurationError,
        match="ANTHROPIC_API_KEY",
    ):
        await main._categorize_single_item({
            "id": "media-1",
            "mimetype": "image/jpeg",
        })
