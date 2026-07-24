from pathlib import Path

from services.rekognition_classifier import build_rekognition_metadata
from services.vault_metadata import VAULT_CLASSIFIER_VERSION


def moderation(*labels):
    return {
        "ModerationModelVersion": "7.0",
        "ModerationLabels": [
            {
                "Name": name,
                "ParentName": parent,
                "Confidence": confidence,
                "TaxonomyLevel": level,
            }
            for name, parent, confidence, level in labels
        ],
    }


def general(*labels):
    return {
        "LabelModelVersion": "3.0",
        "Labels": [
            {
                "Name": name,
                "Confidence": confidence,
                "Instances": [{} for _ in range(instances)],
            }
            for name, confidence, instances in labels
        ],
    }


def test_explicit_nudity_becomes_searchable_nude_metadata():
    result = build_rekognition_metadata(
        moderation(
            ("Explicit Nudity", "Explicit", 99.2, 2),
            ("Exposed Female Genitalia", "Explicit Nudity", 97.4, 3),
        ),
        general(
            ("Person", 99.8, 1),
            ("Bedroom", 93.0, 0),
            ("Bed", 91.0, 1),
            ("Lying Down", 87.0, 0),
            ("Close Up", 83.0, 0),
        ),
        is_video=False,
        album_title="Bedroom set",
    )

    assert result["category"] == "closeup_photo"
    assert result["explicitness"] == 4
    assert result["nudity"] == "full"
    assert result["visible_anatomy"] == ["vulva"]
    assert result["scene_location"] == "bedroom"
    assert result["scene_outfit"] == "nude"
    assert result["pose"] == "lying"
    assert result["framing"] == "close-up"
    assert "full nudity" in result["tags"]
    assert "vulva" in result["tags"]
    assert "explicit" not in result["tags"]
    assert "bedroom" in result["description"]
    assert "classified with" not in result["description"].lower()


def test_explicit_activity_with_multiple_people_becomes_bg_content():
    result = build_rekognition_metadata(
        moderation(
            ("Explicit Sexual Activity", "Explicit", 98.0, 2),
            ("Explicit Nudity", "Explicit", 96.0, 2),
        ),
        general(("Person", 99.0, 2), ("Indoors", 88.0, 0)),
        is_video=True,
    )

    assert result["category"] == "bg_content"
    assert result["explicitness"] == 5
    assert result["sexual_activity"] == ["explicit sexual activity"]
    assert result["good_for"] == "closer"


def test_toy_content_uses_specific_solo_toy_category():
    result = build_rekognition_metadata(
        moderation(("Sex Toys", "Explicit", 97.0, 2)),
        general(("Person", 99.0, 1), ("Bathroom", 85.0, 0)),
        is_video=True,
    )

    assert result["category"] == "solo_toy_video"
    assert result["explicitness"] == 5
    assert "sex toys" in result["sexual_activity"]


def test_underwear_content_remains_below_nudity():
    result = build_rekognition_metadata(
        moderation(
            (
                "Female Swimwear or Underwear",
                "Swimwear or Underwear",
                96.0,
                2,
            )
        ),
        general(
            ("Lingerie", 93.0, 0),
            ("Selfie", 91.0, 0),
            ("Bedroom", 85.0, 0),
        ),
        is_video=False,
    )

    assert result["category"] == "lingerie_photo"
    assert result["explicitness"] == 3
    assert result["nudity"] == "none"
    assert result["framing"] == "selfie"


def test_implied_and_partial_nudity_are_not_promoted_to_full_nudity():
    partial = build_rekognition_metadata(
        moderation(
            (
                "Partially Exposed Female Breast",
                "Non-Explicit Nudity",
                94.0,
                3,
            )
        ),
        general(("Person", 99.0, 1)),
        is_video=False,
    )
    implied = build_rekognition_metadata(
        moderation(
            ("Obstructed Intimate Parts", "Implied Nudity", 91.0, 3)
        ),
        general(("Person", 99.0, 1)),
        is_video=False,
    )

    assert partial["category"] == "nude_photo"
    assert partial["explicitness"] == 3
    assert partial["nudity"] == "partial"
    assert partial["scene_outfit"] == "partially nude"
    assert implied["category"] == "teaser_clothed"
    assert implied["explicitness"] == 2
    assert implied["nudity"] == "implied"
    assert implied["scene_outfit"] == "partially nude"


def test_live_explicit_sample_is_sanitized_for_package_matching():
    result = build_rekognition_metadata(
        moderation(
            ("Explicit", "", 99.6, 1),
            ("Explicit Nudity", "Explicit", 99.4, 2),
            ("Exposed Female Nipple", "Explicit Nudity", 99.1, 3),
            ("Exposed Female Genitalia", "Explicit Nudity", 98.8, 3),
        ),
        general(
            ("Skin", 99.9, 0),
            ("Tattoo", 99.0, 0),
            ("Blouse", 96.0, 1),
            ("Clothing", 95.0, 1),
            ("Body Part", 94.0, 0),
            ("Finger", 93.0, 0),
            ("Hand", 92.0, 0),
            ("Baby", 91.0, 0),
            ("Face", 90.0, 0),
            ("Head", 89.0, 0),
            ("Bedroom", 88.0, 0),
            ("Bed", 87.0, 1),
            ("Portrait", 86.0, 0),
        ),
        is_video=False,
        album_title="Bedroom",
    )

    assert result["category"] == "nude_photo"
    assert result["scene_outfit"] == "partially clothed nude"
    assert set(result["visible_anatomy"]) == {"breasts", "vulva"}
    assert result["description_complete"] is False
    assert result["description"].count("bedroom") == 1
    assert "classification evidence" not in result["description"].lower()
    assert "baby" not in result["description"].lower()
    assert "baby" not in result["tags"]
    assert "skin" not in result["tags"]
    assert "body part" not in result["tags"]
    assert "finger" not in result["tags"]
    assert "hand" not in result["tags"]
    assert "face" not in result["tags"]
    assert "head" not in result["tags"]
    assert "explicit" not in result["tags"]
    assert "full nudity" in result["tags"]
    assert result["_provider_metadata"]["age_review_required"] is True
    assert result["_provider_metadata"]["age_review_signals"] == [
        {"label": "baby", "confidence": 91.0}
    ]


def test_image_properties_add_clothing_colours_and_lighting():
    labels = general(
        ("Person", 99.0, 1),
        ("Underwear", 91.0, 0),
        ("Bed", 61.0, 1),
    )
    labels["Labels"][1]["Instances"] = [{
        "Confidence": 91.0,
        "BoundingBox": {
            "Width": 0.4,
            "Height": 0.3,
            "Left": 0.2,
            "Top": 0.5,
        },
        "DominantColors": [{
            "SimplifiedColor": "Pink",
            "CSSColor": "HotPink",
            "HexCode": "#FF69B4",
            "PixelPercentage": 72.0,
        }],
    }]
    labels["ImageProperties"] = {
        "Quality": {
            "Brightness": 78.0,
            "Sharpness": 82.0,
            "Contrast": 44.0,
        },
        "DominantColors": [{
            "SimplifiedColor": "White",
            "CSSColor": "White",
            "HexCode": "#FFFFFF",
            "PixelPercentage": 41.0,
        }],
    }
    result = build_rekognition_metadata(
        moderation(
            (
                "Female Swimwear or Underwear",
                "Swimwear or Underwear",
                96.0,
                2,
            )
        ),
        labels,
        is_video=False,
    )

    assert result["scene_location"] == "bedroom"
    assert result["colors"][0] == "hotpink"
    assert result["scene_lighting"] == "bright"
    assert result["_provider_metadata"]["image_properties"]["quality"] == {
        "brightness": 78.0,
        "sharpness": 82.0,
        "contrast": 44.0,
    }


def test_rekognition_is_the_v6_default_and_batch_has_cost_circuit_breaker():
    root = Path(__file__).resolve().parents[1]
    env_example = (root / ".env.example").read_text()
    main_source = (root / "main.py").read_text()

    assert VAULT_CLASSIFIER_VERSION == 6
    assert "VAULT_CLASSIFIER_PROVIDER=rekognition" in env_example
    assert 'or "rekognition"' in main_source
    assert "[CATEGORIZE ABORTED]" in main_source
    assert "provider_failures >= 3" in main_source
