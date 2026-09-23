"""Pure helpers for coherent, budget-aware content packages."""
from __future__ import annotations

import re
from typing import Any, Iterable

from models.commercial import CreatorPolicy, Offer
from models.content_pricing import (
    DEFAULT_PRICE_STEP_CENTS,
    paid_sellable_block_reason,
)
from models.price_learning import PriceLearningPolicy, probe_price_cents
from services.scene_metadata import advances_the_interaction
from models.vault_pricing import (
    allocate_step_prices,
    cents_from_row,
    sequence_bounds,
)

VIDEO_REQUEST_RE = re.compile(r"\b(video|videos|vid|vids|clip|clips|movie)\b", re.IGNORECASE)

# How much further into an approved content range each confirmed purchase moves
# this fan's probe position, in basis points on top of the policy cold-start
# position. This is the whole of "the next price adapts": a fan who has already
# bought twice is probed higher inside the NEXT set's own approved range, and
# never above it. Nothing here tells him a range exists.
PURCHASE_PROBE_BONUS_BPS = 1_250
MAX_PURCHASE_PROBE_BONUS_BPS = 3_750


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
    """Approved sets this fan may actually be offered or delivered, right now.

    The one chokepoint every commercial read passes through: offer construction
    (``db.commercial_queries.get_next_offer_with_inventory``) and delivery
    planning (``services.session_planner.plan_session_for_fan``) both start
    here. That is why the paid-sellable boundary is enforced HERE rather than
    as another ad-hoc category filter at each call site — teaser inventory used
    to be filtered out of the media queries and not out of this one, so it
    reached automatic offers through vault_sets.
    """
    sent = sent_set_ids or set()
    result = []
    for row in rows:
        set_id = str(row.get("id") or "")
        media_ids = row.get("media_ids") or []
        if not set_id or set_id in sent or not media_ids:
            continue
        blocked = paid_sellable_block_reason(row)
        if blocked:
            print(f"[SELLABILITY] set={set_id} excluded reason={blocked}")
            continue
        copy = dict(row)
        copy["id"] = set_id
        copy["media_ids"] = [str(value) for value in media_ids if value]
        if copy["media_ids"]:
            result.append(copy)
    return result



def sets_with_sellable_media_evidence(
    sets: Iterable[dict[str, Any]],
    media_rows: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Reject legacy sets whose actual child media is known to be free-only.

    Set-level metadata remains the primary authority and explicit
    paid_sellable=false still wins in usable_sets. This second check closes the
    legacy gap where an old set has weak/missing tags but every media item
    inside it is classified as teaser/free-only.

    We fail closed only when child evidence is present AND every matched child
    is clearly free-only. A mixed set with at least one priced child remains
    eligible, and a hand-curated set with no child classification evidence keeps
    its existing behavior rather than being silently disabled.
    """
    media_by_id = {
        str(row.get("media_id") or ""): row
        for row in media_rows
        if str(row.get("media_id") or "")
    }
    result: list[dict[str, Any]] = []
    for row in sets:
        ids = [str(value) for value in (row.get("media_ids") or []) if value]
        children = [media_by_id[mid] for mid in ids if mid in media_by_id]
        if children and all(paid_sellable_block_reason(child) for child in children):
            print(
                f"[SELLABILITY] set={row.get('id')} excluded "
                "reason=all_child_media_free_only"
            )
            continue
        result.append(row)
    return result


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


def purchase_probe_bonus_bps(confirmed_purchase_count: int) -> int:
    """How much higher inside an approved range a proven buyer is probed."""
    try:
        count = max(0, int(confirmed_purchase_count or 0))
    except (TypeError, ValueError):
        count = 0
    return min(MAX_PURCHASE_PROBE_BONUS_BPS, count * PURCHASE_PROBE_BONUS_BPS)


def offer_from_row(
    row: dict[str, Any] | None,
    *,
    label: str,
    price_learning: dict[str, Any] | None = None,
    pricing_policy: PriceLearningPolicy | None = None,
    hard_ceiling_cents: int | None = None,
    probe_bonus_bps: int = 0,
) -> Offer | None:
    """Turn ONE approved vault set into ONE next unlock at one approved price.

    The price is decided in this order and no other: the set's approved range,
    then where inside that range this fan should currently be probed, then a
    human-looking price on the agency's grid. Nothing about a session total, a
    number of future parts, or an intended eventual spend exists here, because
    none of those are things the fan is ever told.
    """
    if not row:
        return None
    _, floor_cents, ceiling_cents = sequence_bounds([row])
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

    # One unlock is one PPV, so the price has to be a valid single step price on
    # the agency grid. Discovering otherwise after he has said yes is how a $25
    # offer used to turn into two fractional PPVs.
    if allocate_step_pricing(
        probe.price_cents, [row], step_cents=policy.customer_price_step_cents
    ) is None:
        return None

    set_id = str(row["id"])
    legal_description = describe_sequence([row])
    return Offer(
        offer_id=f"offer:{set_id}",
        label=label,
        price_cents=probe.price_cents,
        set_id=set_id,
        experience=legal_description,
        legal_description=legal_description,
        media_count=len(row.get("media_ids") or []),
        asset_type="video" if is_video_row(row) else "photo_set",
        content_floor_cents=floor_cents,
        content_ceiling_cents=ceiling_cents,
        price_reason_codes=list(probe.reason_codes),
    )


def _escalation_rank(row: dict[str, Any], reference: dict[str, Any] | None) -> tuple:
    """Order candidates so the next unlock is a genuine step up, not a repeat.

    Continuity with what he has already unlocked outranks raw intensity: one
    scene that keeps going beats a stronger but unrelated set. Within that, the
    softest thing that is still MORE than the last piece comes first, so the
    ladder climbs one rung at a time instead of jumping to the strongest asset.
    """
    if not reference:
        return (0, 0.0, explicitness(row), price_cents(row), str(row.get("id")))
    same_scene = continuity_key(row) == continuity_key(reference)
    level = explicitness(row)
    previous_level = explicitness(reference)
    steps_up = level > previous_level
    return (
        0 if same_scene else 1,
        0 if steps_up else 1,
        level if steps_up else -level,
        price_cents(row),
        str(row.get("id")),
    )


def plan_progression(
    rows: list[dict[str, Any]],
    *,
    desired_experience: str | None = None,
    preferred_tags: list[str] | None = None,
    last_unlocked: dict[str, Any] | None = None,
    scene: dict[str, Any] | None = None,
    max_steps: int = 4,
) -> list[dict[str, Any]]:
    """The internal ladder: which approved sets this could walk through, in order.

    This is choreography, never a contract. Nothing here is priced as a whole,
    nothing here is presented to the fan, and no step after the first is
    promised: each rung becomes an offer of its own only when the conversation
    actually reaches it, at a price decided then. It exists so the NEXT unlock
    is a coherent continuation rather than a random set, and so the writer can
    be told what the scene is without being told where it ends.
    """
    usable = [row for row in rows if row.get("media_ids")]
    if not usable:
        return []

    photo_rows, video_rows = split_media_types(usable)
    asked_for_video = wants_video(desired_experience)
    # He asked for a clip in as many words, and one exists: that IS the next
    # thing. Otherwise a photo set is the lower-friction way into a scene, and
    # the clip is the payoff the ladder climbs toward.
    if asked_for_video and video_rows:
        pool = video_rows
    elif photo_rows:
        pool = photo_rows
    else:
        pool = usable

    preferred = {normalize_text(tag) for tag in (preferred_tags or []) if normalize_text(tag)}

    def _intent(row: dict[str, Any]) -> float:
        score = sequence_intent_score([row], desired_experience)
        if preferred:
            tags = {
                normalize_text(tag)
                for tag in (row.get("tags") or [])
                if normalize_text(tag)
            }
            score += 2.0 * len(preferred & tags)
        # The third question, alongside scene continuity and explicitness:
        # does this actually advance the interaction he is having right now?
        # A set that repeats the beat he just unlocked scores worse than one
        # that answers the direction he has been pulling towards, even when
        # both are equally in-scene and equally explicit.
        if scene:
            score += advances_the_interaction(row, scene)
        return score

    opener = min(
        pool,
        key=lambda row: (-_intent(row), _escalation_rank(row, last_unlocked)),
    )
    ladder = [opener]
    used = {str(opener.get("id"))}

    # Continue inside the opener's own scene while it escalates, then allow a
    # coherent clip to be the payoff if one exists.
    remaining = [row for row in usable if str(row.get("id")) not in used]
    while len(ladder) < max(1, max_steps) and remaining:
        nxt = min(remaining, key=lambda row: _escalation_rank(row, ladder[-1]))
        if continuity_key(nxt) != continuity_key(ladder[-1]) and not is_video_row(nxt):
            break
        ladder.append(nxt)
        used.add(str(nxt.get("id")))
        remaining = [row for row in remaining if str(row.get("id")) not in used]
        if is_video_row(nxt):
            break

    if video_rows and len(ladder) < max(1, max_steps) and not any(
        is_video_row(row) for row in ladder
    ):
        finale = choose_video_finale(
            ladder,
            video_rows,
            desired_experience=desired_experience,
            preferred_tags=preferred_tags,
            excluded_set_ids=used,
        )
        if finale:
            ladder.append(finale)
    return ladder


def build_next_offer(
    rows: list[dict[str, Any]],
    policy: CreatorPolicy,
    *,
    preferred_tags: list[str] | None = None,
    price_learning: dict[str, Any] | None = None,
    desired_experience: str | None = None,
    hard_ceiling_cents: int | None = None,
    pricing_policy: PriceLearningPolicy | None = None,
    last_unlocked: dict[str, Any] | None = None,
    confirmed_purchase_count: int = 0,
    scene: dict[str, Any] | None = None,
) -> Offer | None:
    """The ONE next unlock to put in front of this fan, or None.

    ``policy.next_offer_target_cents`` sizes the CONTENT considered, exactly as
    the pair of budgets it replaces did; it is not a price and is never quoted.
    The returned offer is a single approved set at a single approved price, and
    it is the only commercial thing the writer is ever handed.
    """
    if not rows:
        return None

    pricing_policy = pricing_policy or PriceLearningPolicy()
    ladder = plan_progression(
        rows,
        desired_experience=desired_experience,
        preferred_tags=preferred_tags,
        last_unlocked=last_unlocked,
        scene=scene,
    )
    if not ladder:
        return None

    target = max(1, int(policy.next_offer_target_cents or 0) or 1)
    bonus = purchase_probe_bonus_bps(confirmed_purchase_count)

    # Walk the ladder from its next rung outward: the first rung that can
    # actually be priced inside its own approved bounds, under any explicit
    # current ceiling, is the offer. A rung that cannot be is skipped rather
    # than discounted.
    for row in ladder:
        offer = offer_from_row(
            row,
            label=_offer_label(row),
            price_learning=price_learning,
            pricing_policy=pricing_policy,
            hard_ceiling_cents=hard_ceiling_cents,
            probe_bonus_bps=bonus,
        )
        if offer:
            return offer

    # Nothing on the planned ladder is priceable. Fall back to the single
    # approved set closest to the content target that is.
    for row in sorted(rows, key=lambda item: abs(price_cents(item) - target)):
        offer = offer_from_row(
            row,
            label=_offer_label(row),
            price_learning=price_learning,
            pricing_policy=pricing_policy,
            hard_ceiling_cents=hard_ceiling_cents,
            probe_bonus_bps=bonus,
        )
        if offer:
            return offer
    return None


def _offer_label(row: dict[str, Any]) -> str:
    """A plain description of the thing itself. Never a tier name."""
    if is_video_row(row):
        return "private video"
    count = len(row.get("media_ids") or [])
    return "private photo set" if count != 1 else "private photo"
