"""Deterministic fusion of adult-classifier evidence and vision-model semantics.

Two sources describe one asset and neither is trusted alone.

The adult classifier owns *how explicit* the asset is. It is purpose-built for
this catalogue, it does not refuse, and its scores are stable. A general
vision model owns *what is in* the asset — outfit, location, lighting, mood —
which no moderation endpoint reports.

The two are fused rather than averaged because they fail in opposite
directions. A general vision model under-reads explicit frames: refusal and
prudishness both collapse toward "lingerie", and since ``price_min`` and
``price_max`` are derived from the category, an under-read is a silent
underpricing bug. So the classifier may *escalate* a category but never
*demote* one — a demotion would move paid content toward the free teaser
tiers on classifier noise alone. Every disagreement is recorded instead of
being quietly resolved, so the operator can see which items the two sources
read differently.

This module is pure. Network access lives in ``services.adult_classifier``
and ``services.video_frames``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


EXPLICITNESS_MIN = -1
EXPLICITNESS_MAX = 5

# Bumped when a classifier change makes existing metadata worth redoing. The
# dashboard counts rows below this as stale and offers a paid re-analysis.
# v3 = adult classifier owns explicitness, vision model owns semantics, and
# videos are read from sampled keyframes instead of their filename.
CLASSIFIER_VERSION = 3

# Numeric stand-ins for the cases where the classifier reports no score of its
# own. The dashboard renders this column as an evidence percentage.
EVIDENCE_SCORES = {
    "vision_only": 0.50,
    "classifier_only": 0.60,
    "unavailable": 0.0,
}
# A disagreement is capped here however sure the classifier sounded: the two
# sources contradicting each other is itself the reason not to trust the row.
DISAGREEMENT_SCORE_CEILING = 0.50

# Cleopatra's explicitness scale, kept verbatim from the operator-facing
# prompt so the classifier mapping and the vision prompt cannot drift apart.
EXPLICITNESS_LABELS = {
    -1: "junk/noise (screenshot, text, meme, unrelated)",
    0: "casual/SFW (normal selfie, street clothes, nothing suggestive)",
    1: "teaser (blurred, pixelated, censored, or fully covered)",
    2: "suggestive but not provocative (hint of skin, flirty, still clothed)",
    3: "sexy: lingerie / bikini / see-through, nothing explicit shown",
    4: "nude: breasts, butt, or genitals exposed",
    5: "explicit/lewd: spread, penetration, sex-act still",
}

VAULT_CATEGORIES: dict[str, dict[str, Any]] = {
    "teaser_clothed":   {"min": 0,   "max": 0,   "label": "Clothed teaser (free)"},
    "teaser_bundle":    {"min": 0,   "max": 0,   "label": "Teaser bundle no nudity (free)"},
    "legs_feet":        {"min": 15,  "max": 70,  "label": "Legs / feet / armpits"},
    "lingerie_photo":   {"min": 10,  "max": 80,  "label": "Lingerie photo"},
    "lingerie_video":   {"min": 15,  "max": 90,  "label": "Lingerie video"},
    "nude_photo":       {"min": 15,  "max": 80,  "label": "Nude photo"},
    "striptease_video": {"min": 15,  "max": 100, "label": "Striptease video"},
    "closeup_photo":    {"min": 25,  "max": 130, "label": "Closeup photo"},
    "closeup_video":    {"min": 25,  "max": 130, "label": "Closeup video"},
    "dictate_video":    {"min": 15,  "max": 50,  "label": "Dictate / dirty talk video"},
    "solo_toy_video":   {"min": 30,  "max": 150, "label": "Solo / toy / orgasm video"},
    "solo_toy_photo":   {"min": 20,  "max": 80,  "label": "Solo / toy photo"},
    "bg_content":       {"min": 50,  "max": 300, "label": "BG (boy-girl) content"},
    "task":             {"min": 10,  "max": 50,  "label": "Task / custom request"},
    "other":            {"min": 0,   "max": 0,   "label": "Other / unclear"},
}

CATEGORY_LIST = "\n".join(
    f"- {key}: {value['label']} (price range ${value['min']}-${value['max']})"
    for key, value in VAULT_CATEGORIES.items()
)

# Sightengine nudity-2.1 raw classes mapped onto the scale above. Kept as one
# table so a provider swap is a table edit, not a logic change.
NUDITY_CLASS_EXPLICITNESS: dict[str, int] = {
    "sexual_activity": 5,
    "sexual_display": 5,
    "erotica": 4,
    "very_suggestive": 3,
    "suggestive": 2,
    "mildly_suggestive": 1,
    "none": 0,
}

# Sightengine reports the garment that made a frame suggestive. It is a
# useful fallback for ``scene_outfit`` when the vision model omits one.
SUGGESTIVE_CLASS_OUTFITS: dict[str, str] = {
    "bikini": "bikini",
    "cleavage": "revealing top",
    "lingerie": "lingerie",
    "male_underwear": "underwear",
    "miniskirt": "miniskirt",
    "schoolgirl_uniform": "schoolgirl uniform",
    "swimwear_one_piece": "one-piece swimsuit",
    "visible_underwear": "visible underwear",
}

# The explicitness a category is expected to carry. ``None`` marks categories
# that are defined by intent rather than by what is on screen, so no amount of
# classifier disagreement makes them wrong.
CATEGORY_EXPLICITNESS_BANDS: dict[str, tuple[int, int] | None] = {
    "teaser_clothed":   (-1, 2),
    "teaser_bundle":    (-1, 2),
    "legs_feet":        (0, 3),
    "lingerie_photo":   (1, 3),
    "lingerie_video":   (1, 3),
    "nude_photo":       (3, 4),
    "striptease_video": (2, 4),
    "closeup_photo":    (3, 5),
    "closeup_video":    (3, 5),
    "dictate_video":    None,
    "solo_toy_video":   (3, 5),
    "solo_toy_photo":   (3, 5),
    "bg_content":       (4, 5),
    "task":             None,
    "other":            None,
}

# Where an escalation lands. Only reached when the classifier reads an asset
# as more explicit than the vision model's category can hold.
ESCALATION_CATEGORIES: dict[int, dict[str, str]] = {
    3: {"photo": "lingerie_photo", "video": "lingerie_video"},
    4: {"photo": "nude_photo", "video": "striptease_video"},
    5: {"photo": "closeup_photo", "video": "closeup_video"},
}

# Below this the classifier is treated as evidence rather than as the answer.
CLASSIFIER_CONFIDENCE_FLOOR = 0.55

# A garment score has to clear this before it is used as an outfit fallback.
OUTFIT_HINT_FLOOR = 0.30


def explicitness_scale_text() -> str:
    """The scale as prompt text, generated so it cannot drift from the mapping."""
    return "\n".join(
        f"{level:>2} = {label}"
        for level, label in sorted(EXPLICITNESS_LABELS.items())
    )


def clamp_explicitness(value: Any) -> int:
    """Coerce any model or API output onto the -1..5 scale."""
    try:
        number = int(value)
    except (TypeError, ValueError):
        return 0
    return max(min(number, EXPLICITNESS_MAX), EXPLICITNESS_MIN)


@dataclass(frozen=True)
class ClassifierVerdict:
    """What the adult classifier saw. ``available`` is false when it did not run."""

    explicitness: int = 0
    top_class: str = ""
    confidence: float = 0.0
    scores: dict[str, float] = field(default_factory=dict)
    outfit_hint: str = ""
    frames_analyzed: int = 0
    available: bool = False

    @property
    def is_confident(self) -> bool:
        return self.available and self.confidence >= CLASSIFIER_CONFIDENCE_FLOOR


@dataclass(frozen=True)
class ClassificationDecision:
    """The reconciled answer plus the evidence that produced it.

    ``evidence`` is the operator-facing label; ``confidence_score`` is the
    0..1 number the dashboard already renders as an evidence percentage.
    """

    category: str
    explicitness: int
    evidence: str
    needs_review: bool
    disagreement: str
    classifier_explicitness: int | None
    vision_explicitness: int
    outfit_hint: str
    confidence_score: float = 0.0

    @property
    def source(self) -> str:
        """Which sources actually contributed, for ``classification_source``."""
        if self.evidence in {"high", "low"}:
            return "hybrid"
        if self.evidence == "unavailable":
            return "failed"
        return self.evidence


def _evidence_score(evidence: str, verdict: "ClassifierVerdict") -> float:
    if evidence in EVIDENCE_SCORES:
        return EVIDENCE_SCORES[evidence]
    if evidence == "low":
        return round(min(verdict.confidence, DISAGREEMENT_SCORE_CEILING), 4)
    return round(verdict.confidence, 4)


def verdict_from_nudity(
    nudity: Mapping[str, Any] | None,
    *,
    frames_analyzed: int = 1,
) -> ClassifierVerdict:
    """Turn one nudity payload into a verdict on Cleopatra's scale.

    Unknown keys are ignored rather than guessed at, so a provider adding a
    class cannot silently change an existing item's explicitness.
    """
    if not isinstance(nudity, Mapping):
        return ClassifierVerdict()

    scores: dict[str, float] = {}
    for name in NUDITY_CLASS_EXPLICITNESS:
        raw = nudity.get(name)
        if isinstance(raw, (int, float)):
            scores[name] = float(raw)

    if not scores:
        return ClassifierVerdict()

    # Ties resolve toward the more explicit class: under-reading costs revenue,
    # over-reading only costs an operator review.
    top_class = max(
        scores,
        key=lambda name: (scores[name], NUDITY_CLASS_EXPLICITNESS[name]),
    )

    return ClassifierVerdict(
        explicitness=NUDITY_CLASS_EXPLICITNESS[top_class],
        top_class=top_class,
        confidence=round(scores[top_class], 4),
        scores={name: round(value, 4) for name, value in scores.items()},
        outfit_hint=outfit_hint_from_scores(nudity.get("suggestive_classes")),
        frames_analyzed=max(int(frames_analyzed), 0),
        available=True,
    )


def outfit_hint_from_scores(suggestive_classes: Any) -> str:
    """Name the garment the classifier is most sure about, if any."""
    if not isinstance(suggestive_classes, Mapping):
        return ""

    best_name = ""
    best_score = OUTFIT_HINT_FLOOR
    for name, label in SUGGESTIVE_CLASS_OUTFITS.items():
        raw = suggestive_classes.get(name)
        if isinstance(raw, (int, float)) and float(raw) > best_score:
            best_score = float(raw)
            best_name = label
    return best_name


def merge_frame_verdicts(
    verdicts: Iterable[ClassifierVerdict],
) -> ClassifierVerdict:
    """Collapse per-frame verdicts into one verdict for a video.

    A video is priced on its peak, not its average: two explicit seconds in a
    four-minute clip are what the fan is paying for. Scores are merged
    per-class by maximum so the stored evidence covers the whole clip.
    """
    available = [verdict for verdict in verdicts if verdict.available]
    if not available:
        return ClassifierVerdict()

    peak = max(available, key=lambda verdict: (verdict.explicitness, verdict.confidence))

    merged_scores: dict[str, float] = {}
    for verdict in available:
        for name, value in verdict.scores.items():
            merged_scores[name] = max(merged_scores.get(name, 0.0), value)

    outfit_hint = next(
        (verdict.outfit_hint for verdict in available if verdict.outfit_hint),
        "",
    )

    return ClassifierVerdict(
        explicitness=peak.explicitness,
        top_class=peak.top_class,
        confidence=peak.confidence,
        scores=merged_scores,
        outfit_hint=peak.outfit_hint or outfit_hint,
        frames_analyzed=len(available),
        available=True,
    )


def normalize_category(category: Any) -> str:
    key = str(category or "").strip().lower()
    return key if key in VAULT_CATEGORIES else "other"


def escalation_category(explicitness: int, *, is_video: bool) -> str:
    """The category an asset falls into when only its explicitness is trusted."""
    options = ESCALATION_CATEGORIES.get(explicitness)
    if not options:
        return "other"
    return options["video" if is_video else "photo"]


def reconcile(
    *,
    vision_category: Any,
    vision_explicitness: Any,
    verdict: ClassifierVerdict,
    is_video: bool,
    vision_available: bool = True,
) -> ClassificationDecision:
    """Fuse both sources into the category and explicitness that get stored."""
    category = normalize_category(vision_category)
    vision_level = clamp_explicitness(vision_explicitness)

    if not vision_available and verdict.available:
        # Classifier-only. Deriving the category from explicitness keeps the
        # item in a priced tier; falling through to "other" would publish it
        # at $0 purely because the vision model was unreachable.
        classifier_only_level = clamp_explicitness(verdict.explicitness)
        return ClassificationDecision(
            category=escalation_category(classifier_only_level, is_video=is_video),
            explicitness=classifier_only_level,
            evidence="classifier_only",
            confidence_score=EVIDENCE_SCORES["classifier_only"],
            needs_review=True,
            disagreement="vision_unavailable",
            classifier_explicitness=classifier_only_level,
            vision_explicitness=vision_level,
            outfit_hint=verdict.outfit_hint,
        )

    if not verdict.available:
        # Vision-only. Usable, but the operator should know the explicitness
        # was never checked by anything purpose-built. With neither source
        # there is nothing to trust at all, so the item is held for review.
        return ClassificationDecision(
            category=category,
            explicitness=vision_level,
            evidence="vision_only" if vision_available else "unavailable",
            confidence_score=EVIDENCE_SCORES[
                "vision_only" if vision_available else "unavailable"
            ],
            needs_review=not vision_available,
            disagreement="" if vision_available else "no_evidence",
            classifier_explicitness=None,
            vision_explicitness=vision_level,
            outfit_hint="",
        )

    classifier_level = clamp_explicitness(verdict.explicitness)

    if not verdict.is_confident:
        # The classifier ran but hedged. Take whichever source read the asset
        # as more explicit and send it for review rather than picking a winner.
        level = max(classifier_level, vision_level)
        return ClassificationDecision(
            category=category,
            explicitness=level,
            evidence="low",
            confidence_score=_evidence_score("low", verdict),
            needs_review=True,
            disagreement="classifier_unconfident",
            classifier_explicitness=classifier_level,
            vision_explicitness=vision_level,
            outfit_hint=verdict.outfit_hint,
        )

    band = CATEGORY_EXPLICITNESS_BANDS.get(category)

    if band is None:
        # Intent-defined category. The classifier still owns explicitness.
        return ClassificationDecision(
            category=category,
            explicitness=classifier_level,
            evidence="high",
            confidence_score=_evidence_score("high", verdict),
            needs_review=False,
            disagreement="",
            classifier_explicitness=classifier_level,
            vision_explicitness=vision_level,
            outfit_hint=verdict.outfit_hint,
        )

    band_min, band_max = band

    if classifier_level > band_max:
        # The vision model under-read the asset. Escalate so the item is not
        # sold out of a cheaper tier than what it actually shows.
        return ClassificationDecision(
            category=escalation_category(classifier_level, is_video=is_video),
            explicitness=classifier_level,
            evidence="low",
            confidence_score=_evidence_score("low", verdict),
            needs_review=True,
            disagreement="classifier_above_category",
            classifier_explicitness=classifier_level,
            vision_explicitness=vision_level,
            outfit_hint=verdict.outfit_hint,
        )

    if classifier_level < band_min:
        # The vision model read it as more explicit than the classifier did.
        # Keep the category — demoting here would push paid content toward the
        # free teaser tiers on classifier noise — but flag it.
        return ClassificationDecision(
            category=category,
            explicitness=classifier_level,
            evidence="low",
            confidence_score=_evidence_score("low", verdict),
            needs_review=True,
            disagreement="classifier_below_category",
            classifier_explicitness=classifier_level,
            vision_explicitness=vision_level,
            outfit_hint=verdict.outfit_hint,
        )

    return ClassificationDecision(
        category=category,
        explicitness=classifier_level,
        evidence="high",
        confidence_score=_evidence_score("high", verdict),
        needs_review=False,
        disagreement="",
        classifier_explicitness=classifier_level,
        vision_explicitness=vision_level,
        outfit_hint=verdict.outfit_hint,
    )


def price_bounds(category: str) -> tuple[int, int]:
    info = VAULT_CATEGORIES.get(normalize_category(category), VAULT_CATEGORIES["other"])
    return int(info["min"]), int(info["max"])
