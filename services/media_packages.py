"""Pure helpers for coherent, budget-aware content packages."""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Iterable

from models.commercial import CreatorPolicy, PackageOption
from models.content_pricing import DEFAULT_PRICE_STEP_CENTS
from models.price_learning import PriceLearningPolicy, probe_price_cents
from models.vault_pricing import (
    allocate_step_prices,
    cents_from_row,
    sequence_bounds,
)

VIDEO_REQUEST_RE = re.compile(r"\b(video|videos|vid|vids|clip|clips|movie)\b", re.IGNORECASE)

# How much further into an approved content range a premium package probes than
# an opener does. Expressed in basis points on top of the policy cold-start
# position, so an agency retunes both from one dial.
PREMIUM_PROBE_BONUS_BPS = 2_500


def normalize_text(value: Any) -> str:
    text = str(value or "").lower().strip()
    text = re.sub(r"\b(level|lvl|part|set|scene|bundle)\s*\d+\b", " ", text)
    text = re.sub(r"\b\d+\b", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())

_EXPERIENCE_STOPWORDS = {
    "a", "an", "and", "at", "be", "can", "content", "do", "for",
    "from", "have", "her", "him", "i", "in", "it", "me", "more",
    "my", "of", "on", "one", "pics", "pictures", "please", "private",
    "send", "session", "set", "show", "something", "the", "this",
    "to", "video", "videos", "want", "wanted", "wants", "wanna",
    "with", "you", "your",
}


def experience_tokens(value: Any) -> set[str]:
    """Return the meaningful semantic tokens in a requested experience."""
    return {
        token
        for token in normalize_text(value).split()
        if len(token) > 1 and token not in _EXPERIENCE_STOPWORDS
    }


def row_experience_tokens(row: dict[str, Any]) -> set[str]:
    values = [
        row.get("title"),
        row.get("description"),
        row.get("location"),
        row.get("outfit"),
        *(row.get("tags") or []),
    ]
    return {
        token
        for value in values
        for token in normalize_text(value).split()
        if len(token) > 1
    }


def sequence_intent_score(
    sequence: list[dict[str, Any]],
    desired_experience: str | None,
) -> float:
    """Score a coherent sequence against the fan's current concrete request.

    A positive match is deliberately much stronger than price proximity. Price
    targets are soft; a requested shower set must not silently become a cheaper
    unrelated set merely because the latter is closer to the default target.
    """
    desired = experience_tokens(desired_experience)
    if not desired:
        return 0.0
    available = {token for row in sequence for token in row_experience_tokens(row)}
    overlap = desired & available
    if not overlap:
        return 0.0
    coverage = len(overlap) / len(desired)
    return (coverage * 8.0) + (len(overlap) * 2.0)


def _clean_experience_part(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\b(level|lvl|part|set|scene|bundle)\s*\d+\b", "", text, flags=re.I)
    return " ".join(text.split()).strip(" -·|—,")


# Format words that are only true of a clip. A photo set's own metadata can
# legitimately contain them — the tag list of a shoot that also produced a video,
# an album title like "shower video day" — and that description is handed to the
# writer as the approved experience. Left in, it is an invitation to promise
# video that this package does not contain.
_VIDEO_FORMAT_RE = re.compile(
    r"\b(videos?|vids?|clips?|movies?|footage|recording|filmed?|filming)\b",
    re.IGNORECASE,
)


def strip_video_format_words(value: str) -> str:
    """Remove clip-only format words from a photo-set description fragment."""
    cleaned = _VIDEO_FORMAT_RE.sub(" ", str(value or ""))
    return " ".join(cleaned.split()).strip(" -·|—,")


def describe_sequence(sequence: list[dict[str, Any]]) -> str | None:
    """Produce writer-safe semantic context for the exact approved package.

    Format words are filtered against the package's ACTUAL contents: a package
    with no video in it never describes itself using the word video, whatever
    the source rows happen to be tagged with.
    """
    has_video = any(is_video_row(row) for row in sequence)
    parts: list[str] = []
    seen: set[str] = set()
    for row in sequence:
        raw_title = str(row.get("title") or "")
        title_stem = re.split(r"\s*[·|—]\s*", raw_title, maxsplit=1)[0]
        values = [
            title_stem,
            row.get("description"),
            row.get("location"),
            row.get("outfit"),
            *(row.get("tags") or []),
        ]
        for value in values:
            cleaned = _clean_experience_part(value)
            if not has_video:
                cleaned = strip_video_format_words(cleaned)
            key = cleaned.lower()
            if not cleaned or key in seen:
                continue
            seen.add(key)
            parts.append(cleaned)
    return ", ".join(parts)[:240] or None


def continuity_key(row: dict[str, Any]) -> str:
    location = normalize_text(row.get("location"))
    outfit = normalize_text(row.get("outfit"))
    raw_title = str(row.get("title") or "")
    # Vault-set titles generated from one source shoot use the shoot/album name
    # before separators such as "·". Keep that stem so two different bedroom +
    # black-lingerie shoots are not merged merely because metadata is generic.
    title_stem = re.split(r"\s*[·|—]\s*", raw_title, maxsplit=1)[0]
    title = normalize_text(title_stem)
    if title or location or outfit:
        return f"{title}|{location}|{outfit}"
    tags = ",".join(sorted(normalize_text(tag) for tag in (row.get("tags") or []) if tag))
    return tags or str(row.get("id") or "unknown")


def explicitness(row: dict[str, Any]) -> float:
    values = [row.get("explicit_min"), row.get("explicit_max")]
    parsed: list[float] = []
    for value in values:
        try:
            parsed.append(float(value))
        except (TypeError, ValueError):
            pass
    return sum(parsed) / len(parsed) if parsed else 0.0


def price_cents(row: dict[str, Any]) -> int:
    return cents_from_row(row)


def usable_sets(rows: Iterable[dict[str, Any]], sent_set_ids: set[str] | None = None) -> list[dict[str, Any]]:
    sent = sent_set_ids or set()
    result = []
    for row in rows:
        set_id = str(row.get("id") or "")
        media_ids = row.get("media_ids") or []
        if not set_id or set_id in sent or not media_ids:
            continue
        copy = dict(row)
        copy["id"] = set_id
        copy["media_ids"] = [str(value) for value in media_ids if value]
        if copy["media_ids"]:
            result.append(copy)
    return result


def group_coherent_sets(rows: Iterable[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[continuity_key(row)].append(dict(row))
    ordered = []
    for group in groups.values():
        group.sort(key=lambda row: (explicitness(row), price_cents(row), str(row.get("id"))))
        ordered.append(group)
    ordered.sort(key=lambda group: (-len(group), continuity_key(group[0]) if group else ""))
    return ordered


def choose_sequence(
    rows: list[dict[str, Any]],
    *,
    target_cents: int,
    min_steps: int,
    max_steps: int,
    preferred_tags: list[str] | None = None,
    excluded_set_ids: set[str] | None = None,
    desired_experience: str | None = None,
    hard_ceiling_cents: int | None = None,
) -> list[dict[str, Any]]:
    excluded = excluded_set_ids or set()
    preferred = {normalize_text(tag) for tag in (preferred_tags or []) if normalize_text(tag)}
    candidates = [row for row in rows if str(row.get("id")) not in excluded]
    if not candidates:
        return []

    ceiling = int(hard_ceiling_cents) if hard_ceiling_cents else None
    effective_target = min(target_cents, ceiling) if ceiling else target_cents
    best: tuple[float, list[dict[str, Any]]] | None = None

    for group in group_coherent_sets(candidates):
        upper = min(max(1, max_steps), len(group))
        lower = min(max(1, min_steps), upper)
        for count in range(lower, upper + 1):
            # Keep escalation chronological by explicitness; for large groups, a
            # contiguous window avoids stitching unrelated sub-shoots together.
            for start in range(0, len(group) - count + 1):
                sequence = group[start : start + count]
                _, sequence_floor, sequence_ceiling = sequence_bounds(sequence)
                if sequence_ceiling <= 0:
                    # No approved paid value on this content at all.
                    continue
                if ceiling is not None and sequence_floor > ceiling:
                    # Its cheapest approved price already breaks his stated
                    # limit. Discounting below approved value is not an option.
                    continue

                raw_total = sum(price_cents(row) for row in sequence)
                distance = abs(raw_total - effective_target) / max(effective_target, 1)
                continuity_bonus = 0.35 * (count - 1)
                tag_overlap = 0
                if preferred:
                    sequence_tags = {
                        normalize_text(tag)
                        for row in sequence
                        for tag in (row.get("tags") or [])
                        if normalize_text(tag)
                    }
                    tag_overlap = len(preferred & sequence_tags)

                # A concrete current request is an offer anchor, not a mild
                # preference. The large multiplier makes semantic fulfilment
                # outrank closeness to a soft package target.
                intent_score = sequence_intent_score(sequence, desired_experience)
                score = intent_score * 100.0 + tag_overlap * 2.0 + continuity_bonus - distance
                if best is None or score > best[0]:
                    best = (score, sequence)

    # A real explicit current ceiling may make every approved sequence
    # unavailable. Returning no package is safer than silently breaking it.
    return best[1] if best else []


def is_video_row(row: dict[str, Any]) -> bool:
    tags = {str(tag).strip().lower() for tag in (row.get("tags") or [])}
    if "individual_video" in tags:
        return True
    return str(row.get("asset_type") or "").lower() == "video"


def split_media_types(
    rows: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Separate lower-friction photo content from individual video assets."""
    photos: list[dict[str, Any]] = []
    videos: list[dict[str, Any]] = []
    for row in rows:
        (videos if is_video_row(row) else photos).append(row)
    return photos, videos


def wants_video(desired_experience: str | None) -> bool:
    """True when the fan explicitly asked for video, not merely 'content'."""
    return bool(VIDEO_REQUEST_RE.search(str(desired_experience or "")))


def order_steps_for_progression(
    sequence: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Order one package so it escalates: softer photos first, video last.

    Explicitness still leads. Media type is the tiebreak, because a photo set is
    the lower-friction way into a scene and a clip is the natural payoff.
    """
    return sorted(
        sequence,
        key=lambda row: (
            explicitness(row),
            1 if is_video_row(row) else 0,
            price_cents(row),
            str(row.get("id")),
        ),
    )


def choose_video_finale(
    sequence: list[dict[str, Any]],
    videos: list[dict[str, Any]],
    *,
    desired_experience: str | None = None,
    preferred_tags: list[str] | None = None,
    excluded_set_ids: set[str] | None = None,
) -> dict[str, Any] | None:
    """Pick the video that best continues an already-chosen photo progression.

    Scene continuity outranks raw explicitness: one coherent experience that
    ends on a clip beats a stronger but unrelated clip bolted onto the end.
    """
    excluded = excluded_set_ids or set()
    pool = [row for row in videos if str(row.get("id")) not in excluded]
    if not sequence or not pool:
        return None

    scene_tokens = {token for row in sequence for token in row_experience_tokens(row)}
    preferred = {normalize_text(tag) for tag in (preferred_tags or []) if normalize_text(tag)}
    desired = experience_tokens(desired_experience)
    peak = max(explicitness(row) for row in sequence)

    best: tuple[tuple[float, float, int, str], dict[str, Any]] | None = None
    for row in pool:
        tokens = row_experience_tokens(row)
        continuity = len(scene_tokens & tokens)
        intent = len(desired & tokens) if desired else 0
        preference = len(preferred & {normalize_text(tag) for tag in (row.get("tags") or [])})
        score = (
            continuity * 4.0
            + intent * 3.0
            + preference * 2.0
            + (1.5 if explicitness(row) >= peak else 0.0)
        )
        key = (score, explicitness(row), price_cents(row), str(row.get("id")))
        if best is None or key > best[0]:
            best = (key, row)
    return best[1] if best else None


def allocate_step_pricing(
    total_cents: int,
    rows: list[dict[str, Any]],
    *,
    step_cents: int = DEFAULT_PRICE_STEP_CENTS,
) -> list[int] | None:
    """Split one sold session total into human-looking per-step PPV prices.

    Returns ``None`` when no valid distribution exists. The previous weighted
    division always produced *a* number — which is how a $25 session became a
    $10.63 PPV followed by a $14.37 one.
    """
    return allocate_step_prices(total_cents, rows, step_cents=step_cents)


def package_from_sequence(
    sequence: list[dict[str, Any]],
    *,
    label: str,
    package_key: str,
    price_learning: dict[str, Any] | None = None,
    pricing_policy: PriceLearningPolicy | None = None,
    hard_ceiling_cents: int | None = None,
    probe_bonus_bps: int = 0,
) -> PackageOption | None:
    """Turn an approved sequence into one offer at one approved, clean price.

    The price is decided in this order and no other: the content's approved
    range, then where inside that range this fan should currently be probed,
    then a human-looking price on the agency's grid. A package target is a
    shape hint for choosing content, never a price.
    """
    if not sequence:
        return None
    ordered = order_steps_for_progression(sequence)
    _, floor_cents, ceiling_cents = sequence_bounds(ordered)
    if ceiling_cents <= 0:
        return None

    policy = pricing_policy or PriceLearningPolicy()
    if probe_bonus_bps:
        policy = policy.model_copy(
            update={
                "cold_start_probe_bps": min(
                    10_000, policy.cold_start_probe_bps + max(0, probe_bonus_bps)
                )
            }
        )
    probe = probe_price_cents(
        floor_cents,
        ceiling_cents,
        price_learning=price_learning,
        policy=policy,
        hard_ceiling_cents=hard_ceiling_cents,
    )
    if probe is None or probe.price_cents <= 0:
        return None

    # An offer is only presentable if it can actually be delivered as planned
    # steps. Discovering that after he has said yes is how a $25 offer turns
    # into two fractional PPVs.
    if allocate_step_pricing(
        probe.price_cents,
        ordered,
        step_cents=policy.customer_price_step_cents,
    ) is None:
        return None

    set_ids = [str(row["id"]) for row in ordered]
    legal_description = describe_sequence(ordered)
    return PackageOption(
        package_id=f"package:{package_key}:{'-'.join(set_ids)}",
        label=label,
        price_cents=probe.price_cents,
        set_id=set_ids[0],
        set_ids=set_ids,
        experience=legal_description,
        legal_description=legal_description,
        step_count=len(ordered),
        media_count=sum(len(row.get("media_ids") or []) for row in ordered),
        asset_types=[("video" if is_video_row(row) else "photo_set") for row in ordered],
        content_floor_cents=floor_cents,
        content_ceiling_cents=ceiling_cents,
        price_reason_codes=list(probe.reason_codes),
    )


def build_offer_packages(
    rows: list[dict[str, Any]],
    policy: CreatorPolicy,
    *,
    preferred_tags: list[str] | None = None,
    price_learning: dict[str, Any] | None = None,
    desired_experience: str | None = None,
    hard_ceiling_cents: int | None = None,
    pricing_policy: PriceLearningPolicy | None = None,
) -> list[PackageOption]:
    """Build up to two approved offers, photo-first unless told otherwise.

    Commercial progression, not a media-type rule: a photo tease is the
    lower-friction way into a paid session, so a generic offer opens on photos
    and escalates into video. An explicit request for video, or a vault with
    nothing else in it, overrides that immediately.
    """
    if not rows:
        return []

    pricing_policy = pricing_policy or PriceLearningPolicy()
    photo_rows, video_rows = split_media_types(rows)
    fan_asked_for_video = wants_video(desired_experience)

    if fan_asked_for_video or (video_rows and not photo_rows):
        video_packages = _build_video_packages(
            video_rows,
            policy,
            preferred_tags=preferred_tags,
            price_learning=price_learning,
            desired_experience=desired_experience,
            hard_ceiling_cents=hard_ceiling_cents,
            pricing_policy=pricing_policy,
        )
        if video_packages:
            return video_packages

    sequence_rows = photo_rows or rows
    if not sequence_rows:
        return []

    quick_sequence = choose_sequence(
        sequence_rows,
        target_cents=policy.quick_package_target_cents,
        min_steps=policy.session_min_steps,
        max_steps=min(policy.session_max_steps, 3),
        preferred_tags=preferred_tags,
        desired_experience=desired_experience,
        hard_ceiling_cents=hard_ceiling_cents,
    )
    quick = package_from_sequence(
        quick_sequence,
        label="quick private session",
        package_key="quick",
        price_learning=price_learning,
        pricing_policy=pricing_policy,
        hard_ceiling_cents=hard_ceiling_cents,
    )
    packages = [quick] if quick else []

    if policy.offer_two_packages:
        # Reuse of the first step is allowed only if there is not enough coherent
        # media. Prefer a larger progression for the premium package.
        full_sequence = choose_sequence(
            sequence_rows,
            target_cents=policy.full_package_target_cents,
            min_steps=max(policy.session_min_steps, len(quick_sequence) + 1),
            max_steps=policy.session_max_steps,
            preferred_tags=preferred_tags,
            desired_experience=desired_experience,
            hard_ceiling_cents=hard_ceiling_cents,
        )
        full_sequence = _with_video_finale(
            full_sequence,
            video_rows,
            policy,
            preferred_tags=preferred_tags,
            desired_experience=desired_experience,
        )
        full = package_from_sequence(
            full_sequence,
            label="full private session",
            package_key="full",
            price_learning=price_learning,
            pricing_policy=pricing_policy,
            hard_ceiling_cents=hard_ceiling_cents,
            probe_bonus_bps=PREMIUM_PROBE_BONUS_BPS,
        )
        if full and (not quick or full.set_ids != quick.set_ids):
            packages.append(full)

    return sorted(packages, key=lambda package: package.price_cents)


def _with_video_finale(
    sequence: list[dict[str, Any]],
    video_rows: list[dict[str, Any]],
    policy: CreatorPolicy,
    *,
    preferred_tags: list[str] | None,
    desired_experience: str | None,
) -> list[dict[str, Any]]:
    """Let a premium session end on a coherent clip when there is room for one."""
    if not sequence or not video_rows:
        return sequence
    if len(sequence) >= policy.session_max_steps:
        return sequence
    finale = choose_video_finale(
        sequence,
        video_rows,
        desired_experience=desired_experience,
        preferred_tags=preferred_tags,
        excluded_set_ids={str(row.get("id")) for row in sequence},
    )
    return [*sequence, finale] if finale else sequence


def _build_video_packages(
    video_rows: list[dict[str, Any]],
    policy: CreatorPolicy,
    *,
    preferred_tags: list[str] | None,
    price_learning: dict[str, Any] | None,
    desired_experience: str | None,
    hard_ceiling_cents: int | None,
    pricing_policy: PriceLearningPolicy,
) -> list[PackageOption]:
    packages: list[PackageOption] = []
    excluded: set[str] = set()
    for key, label, target, bonus in (
        ("video-quick", "private video", policy.quick_package_target_cents, 0),
        (
            "video-premium",
            "premium private video",
            policy.full_package_target_cents,
            PREMIUM_PROBE_BONUS_BPS,
        ),
    ):
        sequence = choose_sequence(
            video_rows,
            target_cents=target,
            min_steps=1,
            max_steps=1,
            preferred_tags=preferred_tags,
            excluded_set_ids=excluded,
            desired_experience=desired_experience,
            hard_ceiling_cents=hard_ceiling_cents,
        )
        package = package_from_sequence(
            sequence,
            label=label,
            package_key=key,
            price_learning=price_learning,
            pricing_policy=pricing_policy,
            hard_ceiling_cents=hard_ceiling_cents,
            probe_bonus_bps=bonus,
        )
        if package:
            packages.append(package)
            excluded.update(package.set_ids)
        if not policy.offer_two_packages:
            break
    return sorted(packages, key=lambda package: package.price_cents)
