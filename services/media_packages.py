"""Pure helpers for coherent, budget-aware content packages."""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Iterable

from models.commercial import CreatorPolicy, PackageOption
from models.vault_pricing import (
    approved_target_from_learning,
    cents_from_row,
    resolve_sequence_price,
)


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


def describe_sequence(sequence: list[dict[str, Any]]) -> str | None:
    """Produce writer-safe semantic context for the exact approved package."""
    parts: list[str] = []
    seen: set[str] = set()
    for row in sequence:
        raw_title = str(row.get("title") or "")
        title_stem = re.split(r"\s*[·|—]\s*", raw_title, maxsplit=1)[0]
        values = [title_stem, row.get("location"), row.get("outfit"), *(row.get("tags") or [])]
        for value in values:
            cleaned = _clean_experience_part(value)
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
                resolved_price = resolve_sequence_price(sequence, effective_target)
                if ceiling is not None and resolved_price > ceiling:
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


def allocate_budget(total_cents: int, rows: list[dict[str, Any]]) -> list[int]:
    if not rows:
        return []
    total = max(len(rows), int(total_cents))
    # Later/more explicit steps get more weight while preserving exact total.
    base_weights = []
    for index, row in enumerate(rows):
        raw = price_cents(row)
        escalation = 1.0 + (index * 0.35)
        base_weights.append(max(1.0, raw / 100.0) * escalation)
    weight_total = sum(base_weights)
    allocations = [max(1, int(total * weight / weight_total)) for weight in base_weights]
    delta = total - sum(allocations)
    allocations[-1] += delta
    # Guard against a negative final adjustment with very tiny budgets.
    for index in range(len(allocations) - 1, -1, -1):
        if allocations[index] <= 0:
            needed = 1 - allocations[index]
            allocations[index] = 1
            for donor in range(len(allocations)):
                if donor != index and allocations[donor] > needed:
                    allocations[donor] -= needed
                    break
    return allocations


def package_from_sequence(
    sequence: list[dict[str, Any]],
    *,
    label: str,
    target_cents: int,
    package_key: str,
) -> PackageOption | None:
    if not sequence:
        return None
    set_ids = [str(row["id"]) for row in sequence]
    legal_description = describe_sequence(sequence)
    return PackageOption(
        package_id=f"package:{package_key}:{'-'.join(set_ids)}",
        label=label,
        price_cents=resolve_sequence_price(sequence, int(target_cents)),
        set_id=set_ids[0],
        set_ids=set_ids,
        experience=legal_description,
        legal_description=legal_description,
    )


def build_offer_packages(
    rows: list[dict[str, Any]],
    policy: CreatorPolicy,
    *,
    preferred_tags: list[str] | None = None,
    price_learning: dict[str, Any] | None = None,
    desired_experience: str | None = None,
    hard_ceiling_cents: int | None = None,
) -> list[PackageOption]:
    if not rows:
        return []

    quick_target_cents = approved_target_from_learning(
        price_learning, fallback_cents=policy.quick_package_target_cents
    )
    if hard_ceiling_cents:
        quick_target_cents = min(quick_target_cents, int(hard_ceiling_cents))
    quick_sequence = choose_sequence(
        rows,
        target_cents=quick_target_cents,
        min_steps=policy.session_min_steps,
        max_steps=min(policy.session_max_steps, 3),
        preferred_tags=preferred_tags,
        desired_experience=desired_experience,
        hard_ceiling_cents=hard_ceiling_cents,
    )
    quick = package_from_sequence(
        quick_sequence,
        label="quick private session",
        target_cents=quick_target_cents,
        package_key="quick",
    )
    packages = [quick] if quick else []

    if policy.offer_two_packages:
        # Reuse of the first step is allowed only if there is not enough coherent
        # media. Prefer a larger progression for the premium package.
        full_target_cents = approved_target_from_learning(
            price_learning,
            fallback_cents=policy.full_package_target_cents,
            use_ceiling=True,
        )
        if hard_ceiling_cents:
            full_target_cents = min(full_target_cents, int(hard_ceiling_cents))
        full_sequence = choose_sequence(
            rows,
            target_cents=full_target_cents,
            min_steps=max(policy.session_min_steps, len(quick_sequence) + 1),
            max_steps=policy.session_max_steps,
            preferred_tags=preferred_tags,
            desired_experience=desired_experience,
            hard_ceiling_cents=hard_ceiling_cents,
        )
        full = package_from_sequence(
            full_sequence,
            label="full private session",
            target_cents=full_target_cents,
            package_key="full",
        )
        if full and (not quick or full.set_ids != quick.set_ids):
            packages.append(full)

    return sorted(packages, key=lambda package: package.price_cents)
