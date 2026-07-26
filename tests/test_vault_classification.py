from services.vault_classification import (
    CATEGORY_EXPLICITNESS_BANDS,
    CLASSIFIER_VERSION,
    DISAGREEMENT_SCORE_CEILING,
    NUDITY_CLASS_EXPLICITNESS,
    VAULT_CATEGORIES,
    ClassifierVerdict,
    clamp_explicitness,
    escalation_category,
    explicitness_scale_text,
    merge_frame_verdicts,
    normalize_category,
    outfit_hint_from_scores,
    price_bounds,
    reconcile,
    verdict_from_nudity,
)


def _confident(explicitness: int, **overrides) -> ClassifierVerdict:
    defaults = {
        "explicitness": explicitness,
        "top_class": "test",
        "confidence": 0.95,
        "available": True,
    }
    defaults.update(overrides)
    return ClassifierVerdict(**defaults)


def test_every_category_has_a_declared_band():
    assert set(CATEGORY_EXPLICITNESS_BANDS) == set(VAULT_CATEGORIES)


def test_nudity_payload_maps_to_the_highest_scoring_class():
    verdict = verdict_from_nudity(
        {
            "sexual_activity": 0.91,
            "erotica": 0.05,
            "none": 0.01,
            "suggestive_classes": {"lingerie": 0.8},
        }
    )
    assert verdict.available is True
    assert verdict.explicitness == 5
    assert verdict.top_class == "sexual_activity"
    assert verdict.confidence == 0.91
    assert verdict.outfit_hint == "lingerie"


def test_unknown_classes_never_invent_an_explicitness():
    assert verdict_from_nudity({"brand_new_class": 0.99}).available is False
    assert verdict_from_nudity(None).available is False
    assert verdict_from_nudity({}).available is False


def test_ties_resolve_toward_the_more_explicit_class():
    # Under-reading costs revenue; over-reading only costs a review.
    verdict = verdict_from_nudity({"erotica": 0.5, "suggestive": 0.5})
    assert verdict.top_class == "erotica"
    assert verdict.explicitness == NUDITY_CLASS_EXPLICITNESS["erotica"]


def test_outfit_hint_ignores_weak_scores():
    assert outfit_hint_from_scores({"bikini": 0.9}) == "bikini"
    assert outfit_hint_from_scores({"bikini": 0.1}) == ""
    assert outfit_hint_from_scores(None) == ""


def test_video_is_scored_on_its_peak_frame_not_its_average():
    merged = merge_frame_verdicts([
        _confident(0, confidence=0.99, scores={"none": 0.99}),
        _confident(5, confidence=0.80, scores={"sexual_activity": 0.80}),
        _confident(0, confidence=0.99, scores={"none": 0.95}),
    ])
    assert merged.explicitness == 5
    assert merged.frames_analyzed == 3
    # Merged scores cover the whole clip, not just the peak frame.
    assert merged.scores == {"none": 0.99, "sexual_activity": 0.80}


def test_merging_nothing_is_unavailable_not_zero():
    assert merge_frame_verdicts([]).available is False
    assert merge_frame_verdicts([ClassifierVerdict()]).available is False


def test_agreement_keeps_the_vision_category():
    decision = reconcile(
        vision_category="lingerie_photo",
        vision_explicitness=3,
        verdict=_confident(3),
        is_video=False,
    )
    assert decision.category == "lingerie_photo"
    assert decision.explicitness == 3
    assert decision.evidence == "high"
    assert decision.needs_review is False


def test_under_read_explicit_media_is_escalated_out_of_the_cheap_tier():
    # The failure this exists to stop: a sex-act still sold as a $10-80
    # lingerie photo because the vision model would not look at it.
    decision = reconcile(
        vision_category="lingerie_photo",
        vision_explicitness=3,
        verdict=_confident(5),
        is_video=False,
    )
    assert decision.category == "closeup_photo"
    assert decision.explicitness == 5
    assert decision.needs_review is True
    assert decision.disagreement == "classifier_above_category"
    assert price_bounds(decision.category) == (25, 130)


def test_escalation_respects_photo_versus_video():
    assert escalation_category(4, is_video=False) == "nude_photo"
    assert escalation_category(4, is_video=True) == "striptease_video"
    assert escalation_category(0, is_video=False) == "other"


def test_a_low_classifier_reading_never_demotes_paid_content_to_free():
    decision = reconcile(
        vision_category="nude_photo",
        vision_explicitness=4,
        verdict=_confident(1),
        is_video=False,
    )
    assert decision.category == "nude_photo"
    assert decision.disagreement == "classifier_below_category"
    assert decision.needs_review is True
    assert price_bounds(decision.category) != (0, 0)


def test_an_unconfident_classifier_takes_the_higher_reading_and_asks_for_review():
    decision = reconcile(
        vision_category="lingerie_photo",
        vision_explicitness=3,
        verdict=ClassifierVerdict(
            explicitness=1, confidence=0.20, top_class="mildly_suggestive", available=True
        ),
        is_video=False,
    )
    assert decision.explicitness == 3
    assert decision.evidence == "low"
    assert decision.disagreement == "classifier_unconfident"


def test_intent_defined_categories_are_never_escalated():
    decision = reconcile(
        vision_category="dictate_video",
        vision_explicitness=0,
        verdict=_confident(5),
        is_video=True,
    )
    assert decision.category == "dictate_video"
    assert decision.explicitness == 5
    assert decision.needs_review is False


def test_classifier_only_still_lands_in_a_priced_tier():
    decision = reconcile(
        vision_category=None,
        vision_explicitness=0,
        verdict=_confident(4),
        is_video=True,
        vision_available=False,
    )
    assert decision.category == "striptease_video"
    assert decision.evidence == "classifier_only"
    assert decision.needs_review is True
    assert price_bounds(decision.category) == (15, 100)


def test_vision_only_is_usable_but_labelled():
    decision = reconcile(
        vision_category="lingerie_photo",
        vision_explicitness=3,
        verdict=ClassifierVerdict(),
        is_video=False,
    )
    assert decision.evidence == "vision_only"
    assert decision.classifier_explicitness is None
    assert decision.needs_review is False


def test_no_evidence_at_all_is_held_for_review():
    decision = reconcile(
        vision_category=None,
        vision_explicitness=0,
        verdict=ClassifierVerdict(),
        is_video=False,
        vision_available=False,
    )
    assert decision.category == "other"
    assert decision.evidence == "unavailable"
    assert decision.needs_review is True
    assert decision.disagreement == "no_evidence"


def test_category_and_explicitness_inputs_are_sanitized():
    assert normalize_category("NUDE_PHOTO") == "nude_photo"
    assert normalize_category("hallucinated_category") == "other"
    assert normalize_category(None) == "other"
    assert clamp_explicitness(99) == 5
    assert clamp_explicitness(-99) == -1
    assert clamp_explicitness("not a number") == 0
    assert clamp_explicitness("4") == 4


def test_prompt_scale_is_generated_from_the_mapping():
    # The classifier mapping and the vision prompt must not drift apart.
    text = explicitness_scale_text()
    assert "5 = explicit/lewd" in text
    assert "0 = casual/SFW" in text
    assert text.count("\n") == 6


def test_confidence_score_stays_numeric_for_the_dashboard_panel():
    # The dashboard renders classification_confidence as a percentage, so this
    # column must stay a 0..1 number even though the label is text.
    agreed = reconcile(
        vision_category="lingerie_photo",
        vision_explicitness=3,
        verdict=_confident(3, confidence=0.88),
        is_video=False,
    )
    assert agreed.confidence_score == 0.88
    assert agreed.source == "hybrid"

    # A disagreement is capped however sure the classifier sounded.
    disputed = reconcile(
        vision_category="lingerie_photo",
        vision_explicitness=3,
        verdict=_confident(5, confidence=0.99),
        is_video=False,
    )
    assert disputed.confidence_score == DISAGREEMENT_SCORE_CEILING
    assert disputed.source == "hybrid"

    failed = reconcile(
        vision_category=None,
        vision_explicitness=0,
        verdict=ClassifierVerdict(),
        is_video=False,
        vision_available=False,
    )
    assert failed.confidence_score == 0.0
    assert failed.source == "failed"


def test_classifier_version_is_ahead_of_the_dashboards_legacy_default():
    # The dashboard falls back to v2 for rows with no version, so anything
    # written by the old Claude-only path must sort below the current version.
    assert CLASSIFIER_VERSION > 2
