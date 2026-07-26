import json
from pathlib import Path

import pytest

from ai.model_providers import (
    ADULT_INELIGIBLE_PROVIDERS,
    AdultPolicyError,
    _anthropic_image_block,
    _attach_images,
    _openai_image_block,
    adult_eligibility,
    assert_adult_eligible,
    get_runtime_target,
)
from models.model_runtime import ModelTarget, VisionImage


IMAGE = VisionImage(data=b"\xff\xd8\xff\xe0binary", media_type="image/jpeg")


def _target(provider: str, adult_policy: str = "unverified") -> ModelTarget:
    return ModelTarget(
        name=f"{provider}:test",
        provider=provider,
        model="test-model",
        adult_policy=adult_policy,
    )


def test_anthropic_is_ineligible_for_adult_work_whatever_the_catalog_says():
    # The catalog already said so; nothing enforced it until now.
    assert "anthropic" in ADULT_INELIGIBLE_PROVIDERS
    assert adult_eligibility(_target("anthropic", "operator_responsibility")) == "ineligible"
    with pytest.raises(AdultPolicyError):
        assert_adult_eligible(_target("anthropic"))


def test_operator_hosted_targets_are_allowed_through():
    assert_adult_eligible(_target("self_hosted", "operator_responsibility"))
    assert adult_eligibility(_target("together", "unverified")) == "unverified"


def test_a_catalog_row_can_still_mark_a_provider_ineligible():
    with pytest.raises(AdultPolicyError):
        assert_adult_eligible(_target("together", "ineligible"))


def test_vision_target_defaults_away_from_anthropic(monkeypatch):
    for name in ("VISION_PROVIDER", "VISION_MODEL", "VISION_BASE_URL", "VISION_API_KEY_ENV"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SELF_HOSTED_BASE_URL", "http://localhost:8000/v1")

    target = get_runtime_target("VISION")
    assert target.provider not in ADULT_INELIGIBLE_PROVIDERS
    assert_adult_eligible(target)


def test_images_lead_the_last_user_turn_for_anthropic():
    messages = _attach_images(
        [{"role": "user", "content": "classify this"}], [IMAGE], _anthropic_image_block
    )
    content = messages[0]["content"]
    assert content[0]["type"] == "image"
    assert content[0]["source"]["media_type"] == "image/jpeg"
    assert content[-1] == {"type": "text", "text": "classify this"}


def test_images_become_data_uris_for_openai_compatible_endpoints():
    messages = _attach_images(
        [{"role": "user", "content": "classify this"}], [IMAGE], _openai_image_block
    )
    block = messages[0]["content"][0]
    assert block["type"] == "image_url"
    assert block["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_multiple_frames_are_attached_in_order():
    frames = [VisionImage(data=bytes([index]) * 8) for index in range(4)]
    messages = _attach_images(
        [{"role": "user", "content": "judge the clip"}], frames, _openai_image_block
    )
    blocks = messages[0]["content"]
    assert sum(1 for block in blocks if block["type"] == "image_url") == 4
    assert blocks[-1]["type"] == "text"


def test_no_images_leaves_messages_untouched():
    original = [{"role": "user", "content": "text only"}]
    assert _attach_images(original, None, _openai_image_block) == original
    assert _attach_images(original, [], _openai_image_block) == original


def test_catalog_ships_an_enabled_vision_target_that_permits_adult_work():
    catalog = json.loads(
        (Path(__file__).resolve().parents[1] / "config" / "model_candidates.json")
        .read_text(encoding="utf-8")
    )
    vision_rows = [
        row
        for row in catalog["models"]
        if row.get("metadata", {}).get("supports_vision") and row.get("enabled")
    ]
    assert vision_rows, "no enabled vision model in the catalog"
    for row in vision_rows:
        assert row["provider"] not in ADULT_INELIGIBLE_PROVIDERS
        assert row["adult_policy"] != "ineligible"
