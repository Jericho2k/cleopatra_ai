"""End-to-end coverage of the vault categorization hot path with IO stubbed."""
import asyncio
import json

import pytest

import main
from models.model_runtime import ModelResult, ModelTarget, ModelUsage
from services.vault_classification import ClassifierVerdict


PHOTO = {
    "id": "item-1",
    "creator_id": "creator-1",
    "url": "https://cdn.example/photo.jpg",
    "mimetype": "image/jpeg",
    "filename": "set-04.jpg",
    "album_title": "Bedroom",
}

VIDEO = {
    "id": "item-2",
    "creator_id": "creator-1",
    "url": "https://cdn.example/clip.mp4",
    "mimetype": "video/mp4",
    "filename": "clip-12.mp4",
    "album_title": "Sets",
}


def _vision_result(payload: dict) -> ModelResult:
    return ModelResult(
        text=json.dumps(payload),
        target=ModelTarget(
            name="self_hosted:test", provider="self_hosted", model="test-vlm"
        ),
        usage=ModelUsage(input_tokens=10, output_tokens=10),
        latency_ms=1,
    )


@pytest.fixture
def stub(monkeypatch):
    """Stub every network boundary and hand back the knobs each test needs."""
    monkeypatch.setenv("MODEL_TELEMETRY_ENABLED", "false")
    monkeypatch.setenv("NSFW_CLASSIFIER_PROVIDER", "sightengine")
    monkeypatch.setenv("VISION_PROVIDER", "self_hosted")
    monkeypatch.setenv("VISION_MODEL", "test-vlm")
    monkeypatch.setenv("SELF_HOSTED_BASE_URL", "http://localhost:8000/v1")

    state: dict = {"vision_calls": [], "classified": 0}

    async def fake_fetch(url, **kwargs):
        return b"\xff\xd8\xff\xe0" + b"x" * 2000

    async def fake_frames(url, **kwargs):
        return state.get("frames", [])

    async def fake_classify(data, **kwargs):
        state["classified"] += 1
        verdicts = state.get("verdicts")
        if isinstance(verdicts, list):
            return verdicts[min(state["classified"] - 1, len(verdicts) - 1)]
        return verdicts

    async def fake_complete(target, **kwargs):
        state["vision_calls"].append(kwargs)
        if state.get("vision_error"):
            raise RuntimeError("vision endpoint down")
        return _vision_result(state["vision"])

    monkeypatch.setattr(main, "_fetch_media_bytes", fake_fetch)
    monkeypatch.setattr(main, "extract_frames", fake_frames)
    monkeypatch.setattr(main, "classify_image", fake_classify)
    monkeypatch.setattr(main, "complete", fake_complete)
    return state


def _confident(explicitness: int, confidence: float = 0.93) -> ClassifierVerdict:
    return ClassifierVerdict(
        explicitness=explicitness,
        top_class="test",
        confidence=confidence,
        scores={"test": confidence},
        available=True,
    )


def test_explicit_photo_the_vision_model_under_reads_is_repriced(stub):
    stub["verdicts"] = _confident(5)
    stub["vision"] = {
        "category": "lingerie_photo",
        "description": "she is on the bed",
        "explicitness": 3,
        "good_for": "opener",
        "tags": ["bed"],
        "scene_location": "bedroom",
        "scene_outfit": "red lingerie",
        "scene_lighting": "dim",
        "scene_id": "bedroom-red-lingerie",
    }

    result = asyncio.run(main._categorize_single_item(PHOTO))

    # Was $10-80 as a lingerie photo; the classifier moves it to its real tier.
    assert result["content_category"] == "closeup_photo"
    assert (result["price_min"], result["price_max"]) == (25, 130)
    assert result["explicitness"] == 5
    assert result["classification_needs_review"] is True
    assert result["classification_disagreement"] == "classifier_above_category"
    assert result["classifier_explicitness"] == 5
    assert result["vision_explicitness"] == 3
    assert result["analyzed_frame_count"] == 1
    # The semantics still come from the vision model.
    assert result["scene_outfit"] == "red lingerie"
    assert "Outfit: red lingerie." in result["ai_description"]


def test_agreement_leaves_the_vision_category_alone(stub):
    stub["verdicts"] = _confident(3)
    stub["vision"] = {
        "category": "lingerie_photo",
        "description": "lingerie set",
        "explicitness": 3,
        "good_for": "opener",
        "tags": [],
    }

    result = asyncio.run(main._categorize_single_item(PHOTO))

    assert result["content_category"] == "lingerie_photo"
    assert result["classification_evidence"] == "high"
    assert result["classification_needs_review"] is False


def test_video_is_classified_from_real_frames_not_its_filename(stub):
    stub["frames"] = [b"frame-a" * 200, b"frame-b" * 200, b"frame-c" * 200]
    # Only the middle frame is explicit; the clip is priced on it.
    stub["verdicts"] = [_confident(0), _confident(5), _confident(0)]
    stub["vision"] = {
        "category": "lingerie_video",
        "description": "she undresses",
        "explicitness": 3,
        "good_for": "closer",
        "tags": [],
    }

    result = asyncio.run(main._categorize_single_item(VIDEO))

    assert stub["classified"] == 3
    assert result["analyzed_frame_count"] == 3
    assert result["content_category"] == "closeup_video"
    assert (result["price_min"], result["price_max"]) == (25, 130)
    assert result["explicitness"] == 5
    # Every frame reached the vision model in one request.
    assert len(stub["vision_calls"]) == 1
    assert len(stub["vision_calls"][0]["images"]) == 3


def test_a_video_with_no_extractable_frames_still_returns_a_row(stub):
    stub["frames"] = []
    stub["verdicts"] = ClassifierVerdict()
    stub["vision"] = {
        "category": "dictate_video",
        "description": "guessed from the filename",
        "explicitness": 2,
        "good_for": "standalone",
        "tags": [],
    }

    result = asyncio.run(main._categorize_single_item(VIDEO))

    assert result["analyzed_frame_count"] == 0
    assert result["content_category"] == "dictate_video"
    assert result["classification_evidence"] == "vision_only"


def test_a_dead_vision_endpoint_still_lands_the_item_in_a_priced_tier(stub):
    stub["verdicts"] = _confident(4)
    stub["vision"] = {}
    stub["vision_error"] = True

    result = asyncio.run(main._categorize_single_item(PHOTO))

    assert result["content_category"] == "nude_photo"
    assert (result["price_min"], result["price_max"]) == (15, 80)
    assert result["classification_evidence"] == "classifier_only"
    assert result["classification_needs_review"] is True


def test_a_disabled_classifier_degrades_to_vision_only(stub, monkeypatch):
    monkeypatch.setenv("NSFW_CLASSIFIER_PROVIDER", "disabled")
    stub["vision"] = {
        "category": "nude_photo",
        "description": "nude",
        "explicitness": 4,
        "good_for": "closer",
        "tags": [],
    }

    result = asyncio.run(main._categorize_single_item(PHOTO))

    assert stub["classified"] == 0
    assert result["content_category"] == "nude_photo"
    assert result["classification_evidence"] == "vision_only"
    assert result["classifier_explicitness"] is None


def test_the_classifier_names_the_outfit_when_the_vision_model_does_not(stub):
    stub["verdicts"] = ClassifierVerdict(
        explicitness=3,
        top_class="very_suggestive",
        confidence=0.9,
        scores={"very_suggestive": 0.9},
        outfit_hint="bikini",
        available=True,
    )
    stub["vision"] = {
        "category": "lingerie_photo",
        "description": "poolside",
        "explicitness": 3,
        "good_for": "opener",
        "tags": [],
    }

    result = asyncio.run(main._categorize_single_item(PHOTO))

    assert result["scene_outfit"] == "bikini"


def test_anthropic_is_refused_before_any_image_is_sent(stub, monkeypatch):
    monkeypatch.setenv("VISION_PROVIDER", "anthropic")
    monkeypatch.setenv("VISION_MODEL", "claude-sonnet-4-6")
    stub["verdicts"] = _confident(5)
    stub["vision"] = {}

    result = asyncio.run(main._categorize_single_item(PHOTO))

    assert stub["vision_calls"] == []
    assert result["classification_evidence"] == "classifier_only"
    assert result["classification_needs_review"] is True


def test_a_hallucinated_category_cannot_invent_a_price(stub):
    stub["verdicts"] = _confident(3)
    stub["vision"] = {
        "category": "super_premium_tier",
        "description": "x",
        "explicitness": 3,
        "good_for": "nonsense",
        "tags": "not-a-list",
    }

    result = asyncio.run(main._categorize_single_item(PHOTO))

    assert result["content_category"] == "other"
    assert (result["price_min"], result["price_max"]) == (0, 0)
    assert result["good_for"] == "standalone"
    assert result["tags"] == []


def test_every_result_shape_is_persistable(stub):
    stub["verdicts"] = _confident(5)
    stub["vision"] = {
        "category": "lingerie_photo",
        "description": "x",
        "explicitness": 3,
        "good_for": "opener",
        "tags": [],
    }

    row = main._vault_classification_row(
        asyncio.run(main._categorize_single_item(PHOTO))
    )

    assert row["classification_evidence"] == "low"
    assert row["classification_needs_review"] is True
    assert row["classifier_explicitness"] == 5
    assert row["analyzed_frame_count"] == 1
    assert isinstance(row["classifier_scores"], dict)


def test_a_total_failure_still_produces_a_storable_row(stub, monkeypatch):
    async def explode(*args, **kwargs):
        raise RuntimeError("cdn gone")

    monkeypatch.setattr(main, "_collect_item_images", explode)

    result = asyncio.run(main._categorize_single_item(PHOTO))
    row = main._vault_classification_row(result)

    assert row["content_category"] == "other"
    assert row["classification_needs_review"] is True
    assert row["classification_disagreement"] == "error"
