import json

from services.nova_inventory_descriptor import (
    _describe_sync,
    merge_inventory_descriptor,
)
from services.vault_metadata import media_description


def nova_payload() -> dict:
    return {
        "description": (
            "The subject is reclining on white bedding in a bedroom. "
            "She wears a matching pink camisole and underwear, raises one "
            "leg, and looks directly toward the camera while adjusting the "
            "waistband. The frame uses a slightly elevated selfie angle."
        ),
        "setting_location": "bedroom",
        "setting_details": ["white bedding", "stacked pillows"],
        "background_details": ["white pillows", "bedside surface"],
        "wardrobe_items": ["camisole", "underwear", "necklace"],
        "wardrobe_colors": ["pink"],
        "pose": "reclining with one leg raised",
        "limb_position": "one leg raised and one hand at the waistband",
        "gaze": "directly toward camera",
        "expression": "neutral expression",
        "action": "adjusting the underwear waistband",
        "framing": "three-quarter selfie",
        "camera_angle": "slightly elevated",
        "crop": "head through upper thighs",
        "composition": "subject fills most of the landscape frame",
        "props": [],
        "lighting": "soft natural",
        "visual_style": "warm bedroom selfie",
        "distinguishing_details": ["pink matching set", "white bedding"],
        "search_tags": ["pink lingerie", "bedroom selfie", "reclining"],
        "confidence": 0.94,
    }


def test_nova_descriptor_uses_image_and_returns_structured_inventory():
    class Runtime:
        request = None

        def converse(self, **kwargs):
            self.request = kwargs
            return {
                "output": {
                    "message": {
                        "content": [{
                            "text": json.dumps(nova_payload()),
                        }],
                    },
                },
            }

    runtime = Runtime()
    descriptor, model = _describe_sync(
        b"jpeg-data",
        is_video=False,
        album_title="Album_123",
        client=runtime,
    )

    assert model == "amazon.nova-lite-v1:0"
    image = runtime.request["messages"][0]["content"][0]["image"]
    assert image["source"]["bytes"] == b"jpeg-data"
    assert runtime.request["inferenceConfig"]["temperature"] == 0
    assert descriptor["setting_location"] == "bedroom"
    assert descriptor["wardrobe_colors"] == ["pink"]
    assert descriptor["pose"] == "reclining with one leg raised"
    assert descriptor["confidence"] == 0.94


def test_visual_descriptor_enriches_but_cannot_override_adult_evidence():
    base = {
        "description": "Explicit-content photo. Visible anatomy: breasts.",
        "description_complete": False,
        "category": "explicit_photo",
        "explicitness": 5,
        "nudity": "full",
        "visible_anatomy": ["breasts"],
        "sexual_activity": ["explicit sexual activity"],
        "action": "explicit sexual activity",
        "scene_location": "unknown",
        "scene_outfit": "partially clothed nude",
        "scene_lighting": "unknown",
        "pose": "unknown",
        "framing": "other",
        "props": [],
        "colors": [],
        "tags": ["full nudity", "breasts"],
    }
    enriched = merge_inventory_descriptor(base, nova_payload())

    assert enriched["category"] == "explicit_photo"
    assert enriched["explicitness"] == 5
    assert enriched["visible_anatomy"] == ["breasts"]
    assert enriched["sexual_activity"] == ["explicit sexual activity"]
    assert enriched["action"] == "explicit sexual activity"
    assert enriched["scene_location"] == "bedroom"
    assert "pink" in enriched["scene_outfit"]
    assert enriched["pose"] == "reclining with one leg raised"
    assert enriched["framing"] == "three-quarter selfie"
    assert enriched["colors"][0] == "pink"
    assert "bedroom selfie" in enriched["tags"]


def test_final_description_includes_rich_photoshoot_details():
    descriptor = nova_payload()
    data = merge_inventory_descriptor(
        {
            "description": "Full-nudity photo.",
            "explicitness": 4,
            "nudity": "full",
            "visible_anatomy": ["breasts"],
            "scene_location": "unknown",
            "scene_outfit": "partially clothed nude",
            "scene_lighting": "unknown",
            "action": "posing",
            "pose": "unknown",
            "framing": "other",
            "props": [],
            "colors": [],
            "tags": [],
        },
        descriptor,
    )
    data["visual_tone"] = "warm"
    data["orientation"] = "landscape"
    data["image_dimensions"] = "896x662"
    data["rich_visual_descriptor"] = {
        "status": "ready",
        "descriptor": descriptor,
    }

    text = media_description(data, source="image")
    lowered = text.lower()
    assert "reclining on white bedding" in lowered
    assert "location: bedroom" in lowered
    assert "wardrobe: camisole, underwear, necklace" in lowered
    assert "camera angle: slightly elevated" in lowered
    assert "orientation: landscape" in lowered
