from pathlib import Path

from db.queries import propose_sets
from services.media_packages import sequence_intent_score
from services.vault_metadata import (
    VAULT_CLASSIFIER_VERSION,
    build_set_description,
    classification_confidence,
    explicitness_from_evidence,
    media_description,
    normalize_media_category,
)


def test_video_thumbnail_confidence_cannot_claim_full_video_certainty():
    assert classification_confidence(0.99, source="video_thumbnail") == 0.72
    assert classification_confidence(0.91, source="image") == 0.91


def test_rich_media_description_preserves_searchable_visual_facts():
    description = media_description(
        {
            "description": "The creator poses beneath a running shower.",
            "scene_location": "bathroom shower",
            "scene_outfit": "wet white shirt",
            "action": "showering",
            "pose": "standing",
            "framing": "full body",
            "scene_lighting": "bright",
            "tags": ["shower", "wet look"],
            "explicitness": 3,
        },
        source="image",
    )
    assert "running shower" in description
    assert "wet white shirt" in description
    assert "showering" in description
    assert "shower, wet look" in description


def test_complete_provider_description_is_not_expanded_twice():
    description = media_description(
        {
            "description": (
                "Full-nudity photo in a bedroom setting using portrait framing. "
                "Visible anatomy: breasts, vulva."
            ),
            "description_complete": True,
            "scene_location": "bedroom",
            "scene_outfit": "partially clothed nude",
            "visible_anatomy": ["breasts", "vulva"],
            "tags": ["bedroom", "full nudity"],
            "explicitness": 4,
        },
        source="image",
    )

    assert description.count("bedroom") == 1
    assert description.count("Visible anatomy") == 1
    assert "Semantic tags:" not in description
    assert description.endswith("Classification evidence: image; explicitness 4/5.")


def test_complete_rich_description_is_not_expanded_twice():
    description = media_description(
        {
            "description": (
                "A subject crouches beside a kitchen counter. "
                "Full nudity is visible."
            ),
            "description_complete": True,
            "scene_location": "kitchen",
            "scene_outfit": "full nudity",
            "visible_anatomy": ["breasts", "vulva"],
            "explicitness": 4,
            "rich_visual_descriptor": {
                "status": "ready",
                "descriptor": {
                    "background_details": [
                        "kitchen counter and cabinets",
                    ],
                },
            },
        },
        source="image",
    )

    assert description.count("kitchen counter") == 1
    assert "Visual details" not in description
    assert "Photoshoot details" not in description
    assert "exposed anatomy: breasts, vulva" in description
    assert description.endswith(
        "Classification evidence: image; explicitness 4/5."
    )


def test_complete_provider_description_cannot_omit_adult_content_findings():
    description = media_description(
        {
            "description": (
                "A person with short blonde hair stands in a modern kitchen, "
                "wearing a black and white maid-style apron with lace trim."
            ),
            "description_complete": True,
            "nudity": "partial",
            "visible_anatomy": ["breasts", "buttocks"],
            "sexual_activity": [],
            "explicitness": 4,
        },
        source="image",
    )

    assert "partial nudity is visible" in description
    assert "exposed anatomy: breasts, buttocks" in description
    assert description.endswith(
        "Classification evidence: image; explicitness 4/5."
    )


def test_complete_provider_description_does_not_repeat_content_findings():
    description = media_description(
        {
            "description": (
                "Full nudity is visible, including breasts and vulva."
            ),
            "description_complete": True,
            "nudity": "full",
            "visible_anatomy": ["breasts", "vulva"],
            "explicitness": 4,
        },
        source="image",
    )

    assert "Classifier findings" not in description
    assert description.count("Full nudity") == 1
    assert description.count("breasts") == 1
    assert description.count("vulva") == 1


def test_explicit_visual_evidence_cannot_be_silently_downgraded():
    nude = {
        "category": "other",
        "explicitness": 0,
        "nudity": "full",
        "visible_anatomy": ["vulva"],
        "sexual_activity": [],
    }
    level = explicitness_from_evidence(nude)
    assert level == 4
    assert normalize_media_category(
        nude["category"],
        explicitness=level,
        is_video=False,
    ) == "nude_photo"

    explicit = {
        "category": "nude_photo",
        "explicitness": 4,
        "nudity": "full",
        "visible_anatomy": ["vulva"],
        "sexual_activity": ["toy use"],
    }
    level = explicitness_from_evidence(explicit)
    assert level == 5
    assert normalize_media_category(
        explicit["category"],
        explicitness=level,
        is_video=False,
    ) == "explicit_photo"


def test_explicit_anatomy_is_preserved_in_searchable_description():
    description = media_description(
        {
            "description": "A close-up nude inventory frame",
            "nudity": "full",
            "visible_anatomy": ["vulva"],
            "framing": "close-up",
            "explicitness": 4,
        },
        source="image",
    )
    assert "nudity: full" in description
    assert "visible anatomy: vulva" in description
    assert "vulva" in description


def test_set_description_is_built_from_exact_media_and_matches_intent():
    items = [
        {
            "content_category": "lingerie_photo",
            "explicitness_level": level,
            "scene_location": "bathroom shower",
            "scene_outfit": "red lingerie",
            "tags": ["shower", "wet", "red lingerie"],
            "ai_description": f"A shower sequence frame at level {level}.",
            "mimetype": "image/jpeg",
        }
        for level in (2, 3, 4)
    ]
    description = build_set_description(items)
    assert "bathroom shower" in description
    assert "red lingerie" in description
    assert "progresses from explicitness 2/5 to 4/5" in description
    assert sequence_intent_score(
        [{"title": "Private sequence", "description": description}],
        "show me the shower set",
    ) > 0


def test_set_description_keeps_qwen_environment_and_continuity_evidence():
    descriptor = {
        "setting_details": ["white quilted bedding"],
        "background_details": ["dark wood headboard", "cream wall"],
        "wardrobe_items": ["pink mesh lingerie"],
        "wardrobe_colors": ["pink"],
        "wardrobe_materials": ["sheer mesh"],
        "subject_styling": ["short blonde hair"],
        "color_details": ["pink lingerie", "white bedding"],
        "continuity_markers": [
            "white quilted bedding",
            "dark wood headboard",
            "warm camera-left light",
        ],
    }
    items = [
        {
            "content_category": "lingerie_photo",
            "explicitness_level": level,
            "scene_location": "bedroom",
            "scene_outfit": "pink mesh lingerie",
            "scene_lighting": "warm bedside light",
            "tags": ["bedroom", "pink lingerie"],
            "ai_description": f"Bedroom sequence frame {level}.",
            "mimetype": "image/jpeg",
            "classification_metadata": {
                "rich_visual_descriptor": {
                    "status": "ready",
                    "descriptor": descriptor,
                },
            },
        }
        for level in (2, 3, 4)
    ]
    description = build_set_description(items)
    assert "Environment continuity" in description
    assert "white quilted bedding" in description
    assert "Styling continuity" in description
    assert "pink mesh lingerie" in description
    assert "Same-shoot markers" in description


def test_automatic_set_builder_keeps_video_sets_and_writes_description():
    items = [
        {
            "fansly_media_id": f"video-{index}",
            "content_category": "lingerie_video",
            "explicitness_level": 3,
            "scene_id": "bathroom-blue-lingerie",
            "scene_location": "bathroom",
            "scene_outfit": "blue lingerie",
            "album_title": "Bathroom shoot",
            "mimetype": "video/mp4",
            "price_min": 15,
            "price_max": 90,
            "tags": ["shower", "blue lingerie"],
            "good_for": "mid_session",
            "ai_description": "A short blue-lingerie bathroom video.",
        }
        for index in range(3)
    ]
    proposed = propose_sets(items)
    assert len(proposed) == 1
    assert proposed[0]["media_ids"] == ["video-0", "video-1", "video-2"]
    assert "including 3 videos" in proposed[0]["description"]
    assert proposed[0]["metadata_version"] == VAULT_CLASSIFIER_VERSION


def test_migration_and_analyzer_expose_the_v2_contract():
    root = Path(__file__).resolve().parents[1]
    migration = (root / "db" / "vault_metadata_v2.sql").read_text()
    analyzer = (root / "ai" / "situation_analyzer.py").read_text()
    assert "classification_version" in migration
    assert "classification_confidence" in migration
    assert "add column if not exists description" in migration
    assert '"shower set"' in analyzer
    assert "Never collapse a concrete" in analyzer
