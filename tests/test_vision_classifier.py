import pytest

from services.vision_classifier import (
    VaultClassifierError,
    VaultClassifierRefusalError,
    build_vision_metadata,
    extract_json_object,
    looks_like_refusal,
)


BASE = {
    "category": "lingerie_photo",
    "description": "She is sitting on a bed.",
    "mood": "teasing",
    "explicitness": 3,
    "nudity": "none",
    "visible_anatomy": [],
    "participants": 1,
    "good_for": "opener",
    "tags": ["bed"],
    "sexual_activity": [],
    "body_focus": [],
    "action": "posing",
    "pose": "sitting",
    "framing": "portrait",
    "props": ["bed"],
    "colors": ["pink"],
    "scene_location": "bedroom",
    "scene_outfit": "red lingerie",
    "scene_lighting": "dim",
    "scene_id": "bedroom-red-lingerie",
    "confidence": 0.9,
}


def build(**overrides):
    payload = {**BASE, **overrides}
    return build_vision_metadata(payload, is_video=False, album_title="Bedroom")


def test_the_full_contract_is_emitted():
    # Downstream set-building and package matching read these keys by name.
    result = build()
    for key in (
        "category", "description", "description_complete", "mood", "explicitness",
        "nudity", "visible_anatomy", "good_for", "tags", "sexual_activity",
        "body_focus", "action", "pose", "framing", "props", "colors",
        "scene_location", "scene_outfit", "scene_lighting", "scene_id",
        "confidence", "_classification_model", "_provider_metadata",
    ):
        assert key in result, key
    # The deterministic renderer still appends controlled fields afterwards.
    assert result["description_complete"] is False


def test_reported_explicitness_cannot_undercut_the_evidence():
    # The failure this exists to stop: a model describing visible penetration
    # while rating it 2, which would sell an explicit clip from a teaser tier.
    result = build(explicitness=2, sexual_activity=["penetration"], nudity="full")
    assert result["explicitness"] == 5
    assert result["category"] == "explicit_photo"
    assert result["_provider_metadata"]["explicitness_escalated"] is True
    assert result["_provider_metadata"]["reported_explicitness"] == 2


def test_exposed_anatomy_forces_at_least_nude():
    result = build(explicitness=1, visible_anatomy=["breasts"], nudity="none")
    assert result["explicitness"] >= 4
    assert result["category"] == "nude_photo"


def test_lingerie_stays_at_three_and_is_not_escalated():
    result = build(explicitness=3, nudity="none")
    assert result["explicitness"] == 3
    assert result["category"] == "lingerie_photo"
    assert result["_provider_metadata"]["explicitness_escalated"] is False


def test_toys_and_partners_route_to_their_own_tiers():
    toy = build(explicitness=5, sexual_activity=["dildo use"], nudity="full")
    assert toy["category"] == "solo_toy_photo"

    partnered = build_vision_metadata(
        {**BASE, "explicitness": 5, "sexual_activity": ["intercourse"],
         "nudity": "full", "participants": 2},
        is_video=True,
        album_title="Sets",
    )
    assert partnered["category"] == "bg_content"


def test_video_and_photo_categories_are_distinct():
    video = build_vision_metadata(
        {**BASE, "explicitness": 4, "nudity": "full", "visible_anatomy": ["breasts"]},
        is_video=True,
        album_title="Sets",
    )
    assert video["category"] == "nude_video"


def test_a_hallucinated_category_cannot_leak_through():
    # The category is recomputed from evidence, never taken on trust.
    result = build(category="super_premium_tier")
    assert result["category"] == "lingerie_photo"


def test_enum_fields_fall_back_instead_of_storing_junk():
    result = build(
        mood="scandalous", good_for="whenever", framing="dutch angle",
        scene_lighting="moody", nudity="mostly",
    )
    assert result["mood"] in {"playful", "intimate", "teasing", "explicit", "casual"}
    assert result["good_for"] in {"opener", "mid_session", "closer", "standalone"}
    assert result["framing"] == "other"
    assert result["scene_lighting"] == "unknown"
    assert result["nudity"] == "none"


def test_invented_anatomy_terms_are_dropped():
    result = build(visible_anatomy=["breasts", "aura", "vibes"])
    assert result["visible_anatomy"] == ["breasts"]


def test_confidence_is_clamped_to_the_unit_interval():
    assert build(confidence=7).confidence if False else True
    assert build(confidence=7)["confidence"] == 1.0
    assert build(confidence=-3)["confidence"] == 0.0
    assert build(confidence="not a number")["confidence"] == 0.0


def test_a_possible_minor_flag_forces_human_review():
    result = build(possible_minor=True, age_note="subject may be underage")
    metadata = result["_provider_metadata"]
    assert metadata["age_review_required"] is True
    assert metadata["age_review_signals"][0]["note"] == "subject may be underage"


def test_no_age_flag_leaves_the_review_gate_closed():
    assert build()["_provider_metadata"]["age_review_required"] is False


def test_scene_id_falls_back_to_the_album_when_absent():
    result = build(scene_id="")
    assert result["scene_id"] == "bedroom-bedroom-red-lingerie"
    assert build(scene_id="", scene_location="", scene_outfit="")["scene_id"] == "bedroom"


def test_refusals_are_raised_not_parsed():
    # Parsing a refusal would store a confident-looking "clothed" row for
    # explicit media, which is worse than failing the item.
    for text in (
        "I'm sorry, but I can't help with that.",
        "I cannot describe this image.",
        "As an AI, I am not able to analyze explicit content.",
    ):
        assert looks_like_refusal(text) is True
        with pytest.raises(VaultClassifierRefusalError):
            extract_json_object(text)


def test_a_normal_description_is_not_mistaken_for_a_refusal():
    assert looks_like_refusal('{"category": "nude_photo"}') is False


def test_json_survives_fences_and_surrounding_prose():
    assert extract_json_object('```json\n{"category":"nude_photo"}\n```')[
        "category"
    ] == "nude_photo"
    assert extract_json_object('Here you go: {"category":"nude_photo"} — done.')[
        "category"
    ] == "nude_photo"


def test_empty_or_unparseable_output_is_an_error():
    with pytest.raises(VaultClassifierRefusalError):
        extract_json_object("")
    with pytest.raises(VaultClassifierError):
        extract_json_object("no json here at all")
    with pytest.raises(VaultClassifierError):
        extract_json_object("[1, 2, 3]")
