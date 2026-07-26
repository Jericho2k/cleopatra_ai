"""Vault media classification via a vision model that permits adult content.

This replaces the AWS pair that came before it: Rekognition supplied the adult
taxonomy, Amazon Nova bolted rich scene prose on top, and the two had to be
merged. One vision pass now produces the whole contract, so the taxonomy and
the description describe the same reading of the same image instead of being
reconciled after the fact.

The provider must be one whose terms permit adult imagery. A general-purpose
assistant model will refuse or euphemize this catalogue, and a silent refusal
is worse than an error here: ``price_min``/``price_max`` derive from the
category, so an under-read is an underpricing bug. Refusals are therefore
detected and raised rather than parsed into a bland ``teaser_clothed``.

``build_vision_metadata`` is pure and holds the whole contract, so provider
swaps and prompt edits cannot silently change the database shape.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

from ai.model_providers import assert_adult_eligible, complete, get_runtime_target
from models.model_runtime import VisionImage
from services.vault_metadata import normalize_media_category


class VaultClassifierError(RuntimeError):
    """Base error for the vault vision classifier."""


class VaultClassifierConfigurationError(VaultClassifierError):
    """The vision endpoint, credentials, or model are not usable."""


class VaultClassifierRefusalError(VaultClassifierError):
    """The provider declined to describe the media."""


DEFAULT_VISION_MODEL = "Qwen/Qwen2.5-VL-72B-Instruct"
CLASSIFIER_PROVIDER_NAME = "vision"

_NUDITY_VALUES = {"none", "implied", "partial", "full"}
_MOOD_VALUES = {"playful", "intimate", "teasing", "explicit", "casual"}
_GOOD_FOR_VALUES = {"opener", "mid_session", "closer", "standalone"}
_FRAMING_VALUES = {"selfie", "portrait", "full body", "close-up", "wide", "other"}
_LIGHTING_VALUES = {"natural", "bright", "dim", "flash", "colored", "unknown"}

_ANATOMY_VALUES = {"breasts", "buttocks", "vulva", "penis", "anus", "back"}

# A refusal usually arrives as prose where JSON was demanded. Parsing it would
# store a confident-looking "clothed" row for explicit media.
_REFUSAL_PREFIXES = (
    "i'm not able",
    "i am not able",
    "i'm sorry",
    "i am sorry",
    "i can't",
    "i cannot",
    "i won't",
    "i will not",
    "sorry, but",
    "unable to",
    "as an ai",
)

# Explicitness implied by what the model reports as visible. The model's own
# number is cross-checked against this so a low rating cannot survive an
# admission of visible penetration.
_ACTIVITY_PATTERN = re.compile(
    r"penetration|intercourse|oral sex|blowjob|masturbat|insertion|"
    r"sex act|sexual activity|cumshot|ejaculat",
    re.I,
)
_TOY_PATTERN = re.compile(r"\b(dildo|vibrator|sex toy|toy use|butt plug)\b", re.I)


def _clean(value: Any, *, limit: int = 320) -> str:
    return " ".join(str(value or "").strip().split())[:limit]


def _choice(value: Any, allowed: set[str], fallback: str) -> str:
    text = _clean(value).lower()
    return text if text in allowed else fallback


def _string_list(value: Any, *, limit: int = 24) -> list[str]:
    raw = value if isinstance(value, (list, tuple, set)) else []
    result: list[str] = []
    seen: set[str] = set()
    for item in raw:
        text = _clean(item, limit=120).lower().strip(" .,-_/|")
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
        if len(result) >= limit:
            break
    return result


def _slug(*values: str) -> str:
    text = "-".join(value for value in values if value)
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:96]


def _clamp_explicitness(value: Any) -> int:
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return 0
    return max(min(number, 5), 0)


def looks_like_refusal(text: str) -> bool:
    stripped = _clean(text, limit=200).lower()
    return bool(stripped) and stripped.startswith(_REFUSAL_PREFIXES)


def extract_json_object(text: str) -> dict[str, Any]:
    """Pull the JSON object out of a model response, fences and all."""
    cleaned = str(text or "").replace("```json", "").replace("```", "").strip()
    if not cleaned:
        raise VaultClassifierRefusalError("classifier returned no text")
    if looks_like_refusal(cleaned):
        raise VaultClassifierRefusalError(
            "The vision provider refused the media. Point VISION_PROVIDER at a "
            "provider whose terms permit adult imagery."
        )

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        # Some models wrap the object in a sentence despite instructions.
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end <= start:
            raise VaultClassifierError(
                f"classifier did not return JSON: {cleaned[:200]}"
            ) from None
        payload = json.loads(cleaned[start : end + 1])

    if not isinstance(payload, dict):
        raise VaultClassifierError("classifier did not return a JSON object")
    return payload


def _implied_explicitness(
    *,
    nudity: str,
    sexual_activity: list[str],
    visible_anatomy: list[str],
    action: str,
) -> int:
    """The floor the reported evidence supports, independent of the model's number."""
    joined = " ".join([*sexual_activity, action])
    if _ACTIVITY_PATTERN.search(joined) or _TOY_PATTERN.search(joined):
        return 5
    if nudity == "full" or visible_anatomy:
        return 4
    if nudity == "partial":
        return 3
    if nudity == "implied":
        return 2
    return 0


def _category_for(
    *,
    explicitness: int,
    nudity: str,
    sexual_activity: list[str],
    framing: str,
    participants: int,
    is_video: bool,
) -> str:
    """Derive the category from evidence rather than trusting the model's label.

    The model picks a category too, but that field is the one most prone to
    drifting toward a softer tier. Pricing hangs off it, so it is recomputed
    from the same evidence the model reported.
    """
    joined = " ".join(sexual_activity)
    has_toy = bool(_TOY_PATTERN.search(joined))
    has_activity = bool(_ACTIVITY_PATTERN.search(joined)) or explicitness >= 5

    if has_toy:
        return "solo_toy_video" if is_video else "solo_toy_photo"
    if has_activity and participants >= 2:
        return "bg_content"
    if has_activity:
        return "explicit_video" if is_video else "explicit_photo"
    if explicitness >= 4 and framing == "close-up":
        return "closeup_video" if is_video else "closeup_photo"
    if explicitness >= 4:
        return "nude_video" if is_video else "nude_photo"
    if explicitness == 3:
        return "lingerie_video" if is_video else "lingerie_photo"
    if nudity == "implied" or explicitness == 2:
        return "teaser_bundle" if is_video else "teaser_clothed"
    return "teaser_clothed"


def build_vision_metadata(
    payload: dict[str, Any],
    *,
    is_video: bool,
    album_title: str = "",
    model: str = "",
) -> dict[str, Any]:
    """Normalize one vision response into the provider-neutral vault contract."""
    nudity = _choice(payload.get("nudity"), _NUDITY_VALUES, "none")
    visible_anatomy = [
        value
        for value in _string_list(payload.get("visible_anatomy"), limit=8)
        if value in _ANATOMY_VALUES
    ]
    sexual_activity = _string_list(payload.get("sexual_activity"), limit=8)
    action = _clean(payload.get("action"), limit=120).lower() or "none"
    framing = _choice(payload.get("framing"), _FRAMING_VALUES, "other")
    try:
        participants = max(int(payload.get("participants") or 1), 1)
    except (TypeError, ValueError):
        participants = 1

    reported = _clamp_explicitness(payload.get("explicitness"))
    implied = _implied_explicitness(
        nudity=nudity,
        sexual_activity=sexual_activity,
        visible_anatomy=visible_anatomy,
        action=action,
    )
    # Take the higher of the two. Under-reading loses revenue on the most
    # valuable inventory; over-reading only sends an item for a second look.
    explicitness = max(reported, implied)

    # _category_for derives the tier from evidence; normalize_media_category
    # is the repo's existing repair layer for category/type contradictions.
    category = normalize_media_category(
        _category_for(
            explicitness=explicitness,
            nudity=nudity,
            sexual_activity=sexual_activity,
            framing=framing,
            participants=participants,
            is_video=is_video,
        ),
        explicitness=explicitness,
        is_video=is_video,
    )

    location = _clean(payload.get("scene_location"), limit=120).lower()
    outfit = _clean(payload.get("scene_outfit"), limit=320).lower()
    pose = _clean(payload.get("pose"), limit=120).lower()

    try:
        confidence = float(payload.get("confidence") or 0)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = round(min(max(confidence, 0.0), 1.0), 3)

    age_flagged = bool(payload.get("possible_minor"))
    age_note = _clean(payload.get("age_note"), limit=240)

    tags = _string_list(
        [
            *( [f"{nudity} nudity"] if nudity in {"full", "partial", "implied"} else [] ),
            *sexual_activity,
            *visible_anatomy,
            *_string_list(payload.get("tags"), limit=16),
            location,
            outfit,
            pose,
            framing,
        ],
        limit=24,
    )

    return {
        "category": category,
        "description": _clean(payload.get("description"), limit=1800),
        # The deterministic renderer still appends controlled fields and local
        # visual evidence, exactly as it did for the previous provider.
        "description_complete": False,
        "mood": _choice(
            payload.get("mood"),
            _MOOD_VALUES,
            "explicit" if explicitness >= 4 else "teasing" if explicitness >= 2 else "casual",
        ),
        "explicitness": explicitness,
        "nudity": nudity,
        "visible_anatomy": visible_anatomy,
        "good_for": _choice(
            payload.get("good_for"),
            _GOOD_FOR_VALUES,
            "closer" if explicitness >= 4 else "mid_session" if explicitness >= 2 else "opener",
        ),
        "tags": tags,
        "sexual_activity": sexual_activity,
        "body_focus": _string_list(payload.get("body_focus"), limit=8) or visible_anatomy,
        "action": action,
        "pose": pose or "unknown",
        "framing": framing,
        "props": _string_list(payload.get("props"), limit=12),
        "colors": _string_list(payload.get("colors"), limit=8),
        "scene_location": location or "unknown",
        "scene_outfit": outfit or "unknown",
        "scene_lighting": _choice(
            payload.get("scene_lighting"), _LIGHTING_VALUES, "unknown"
        ),
        "scene_id": (
            _slug(_clean(payload.get("scene_id"), limit=96))
            or _slug(album_title, location, outfit)
            or "unidentified-shoot"
        ),
        "confidence": confidence,
        "_classification_model": model or "vision",
        "_provider_metadata": {
            "provider": CLASSIFIER_PROVIDER_NAME,
            "model": model,
            "reported_explicitness": reported,
            "implied_explicitness": implied,
            "explicitness_escalated": explicitness > reported,
            "participants": participants,
            "setting": _clean(payload.get("setting"), limit=240),
            "background": _clean(payload.get("background"), limit=240),
            "camera_angle": _clean(payload.get("camera_angle"), limit=120),
            "distinguishing_details": _string_list(
                payload.get("distinguishing_details"), limit=12
            ),
            "wardrobe_items": _string_list(payload.get("wardrobe_items"), limit=12),
            "age_review_signals": (
                [{"label": "possible_minor", "note": age_note, "confidence": 100.0}]
                if age_flagged
                else []
            ),
            # Any doubt at all routes the item to a human. A generative model
            # is weaker at this than a dedicated detector, so the flag is
            # treated as a hard stop rather than a score to threshold.
            "age_review_required": age_flagged,
        },
    }


def _prompt(*, is_video: bool, album_title: str, filename: str) -> str:
    asset_kind = "video thumbnail" if is_video else "image"
    partial_note = (
        "A video thumbnail is partial evidence, so describe the visible frame "
        "and lower confidence when the complete video cannot be inferred.\n"
        if is_video
        else ""
    )
    return f"""Classify this adult creator {asset_kind} for private vault search and package matching.
The creator and all depicted participants must be adults. Describe only what is visibly supported; never invent an act, location, outfit, or prop.
{partial_note}
Filename: {filename or 'unknown'}
Album/folder: {album_title or 'unknown'}

EXPLICITNESS is strictly what is visible:
- 0 = ordinary SFW clothing/selfie
- 1 = censored, blurred, implied, or fully covered teaser
- 2 = suggestive/flirty but clothed
- 3 = lingerie, bikini, or see-through with no explicit nudity
- 4 = exposed breasts, butt, or genitals without an explicit sex act
- 5 = explicit sexual activity, toy use, spread pose, oral sex, or penetration
Lingerie alone is never above 3. Do not infer explicitness from the album name.

This is neutral inventory metadata, not erotic writing. Do not omit or euphemize
visible nudity, exposed anatomy, or sexual activity. Record them factually using
the controlled fields below, while never guessing details that are not visible.
Set "possible_minor" to true if anyone depicted might not be an adult; that item
is withheld for human review rather than sold.

Return ONLY one JSON object with exactly these keys:
{{"category":"category_key","description":"3-5 factual, non-erotic sentences covering the visible subject, clothing/nudity state, pose/action, setting and distinguishing details","mood":"playful|intimate|teasing|explicit|casual","explicitness":0,"nudity":"none|implied|partial|full","visible_anatomy":["only visibly exposed: breasts|buttocks|vulva|penis|anus"],"participants":1,"good_for":"opener|mid_session|closer|standalone","tags":["specific searchable themes"],"sexual_activity":["only visibly supported activities"],"body_focus":["visible focal areas"],"action":"specific visible action or none","pose":"specific pose","framing":"selfie|portrait|full body|close-up|wide|other","props":["visible props"],"colors":["dominant outfit/scene colors"],"wardrobe_items":["visible garments"],"setting":"specific room or place","background":"what is behind the subject","camera_angle":"eye level|high angle|low angle|overhead|other","distinguishing_details":["details that identify this specific shoot"],"scene_location":"specific location or unknown","scene_outfit":"specific outfit/nudity state or unknown","scene_lighting":"natural|bright|dim|flash|colored|unknown","scene_id":"short stable shoot slug derived from album/location/outfit","possible_minor":false,"age_note":"","confidence":0.0}}
"""


def vision_target():
    """Resolve and validate the configured vault vision target."""
    target = get_runtime_target("VISION")
    if not target.model:
        raise VaultClassifierConfigurationError(
            "VISION_MODEL is not set for the vault classifier."
        )
    try:
        assert_adult_eligible(target)
    except Exception as exc:
        raise VaultClassifierConfigurationError(str(exc)) from exc
    return target


async def classify_with_vision(
    image_bytes: bytes,
    *,
    is_video: bool,
    album_title: str = "",
    filename: str = "",
) -> dict[str, Any]:
    """Classify one image and return the provider-neutral vault contract."""
    if not image_bytes:
        raise VaultClassifierError("no image bytes to classify")

    target = vision_target()
    try:
        result = await complete(
            target,
            system=(
                "You catalogue an adult creator's own media vault so it can be "
                "priced and merchandised. Return the requested JSON object and "
                "nothing else."
            ),
            messages=[
                {
                    "role": "user",
                    "content": _prompt(
                        is_video=is_video,
                        album_title=album_title,
                        filename=filename,
                    ),
                }
            ],
            max_tokens=int(os.environ.get("VAULT_VISION_MAX_TOKENS", "900")),
            temperature=0.0,
            images=[VisionImage(data=image_bytes, media_type="image/jpeg")],
        )
    except VaultClassifierError:
        raise
    except Exception as exc:
        raise VaultClassifierError(
            f"vision classifier request failed: {type(exc).__name__}: {exc}"
        ) from exc

    payload = extract_json_object(result.text)
    return build_vision_metadata(
        payload,
        is_video=is_video,
        album_title=album_title,
        model=f"{target.provider}:{target.model}",
    )
