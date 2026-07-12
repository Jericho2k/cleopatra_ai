"""Pure helpers for coherent, budget-aware content packages."""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Iterable

from models.commercial import CreatorPolicy, PackageOption


def normalize_text(value: Any) -> str:
    text = str(value or "").lower().strip()
    text = re.sub(r"\b(level|lvl|part|set|scene|bundle)\s*\d+\b", " ", text)
    text = re.sub(r"\b\d+\b", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


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
    try:
        return max(0, int(round(float(row.get("suggested_price") or 0) * 100)))
    except (TypeError, ValueError):
        return 0


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
) -> list[dict[str, Any]]:
    excluded = excluded_set_ids or set()
    preferred = {normalize_text(tag) for tag in (preferred_tags or []) if normalize_text(tag)}
    candidates = [row for row in rows if str(row.get("id")) not in excluded]
    if not candidates:
        return []

    best: tuple[float, list[dict[str, Any]]] | None = None
    for group in group_coherent_sets(candidates):
        upper = min(max(1, max_steps), len(group))
        lower = min(max(1, min_steps), upper)
        for count in range(lower, upper + 1):
            # Keep escalation chronological by explicitness; for large groups, a
            # contiguous window avoids stitching unrelated sub-shoots together.
            for start in range(0, len(group) - count + 1):
                sequence = group[start : start + count]
                raw_total = sum(price_cents(row) for row in sequence)
                distance = abs(raw_total - target_cents) / max(target_cents, 1)
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
                score = tag_overlap * 2.0 + continuity_bonus - distance
                if best is None or score > best[0]:
                    best = (score, sequence)

    # If no group can satisfy the requested minimum, prefer the strongest single
    # coherent set instead of inventing continuity.
    if best is None:
        row = min(candidates, key=lambda item: abs(price_cents(item) - target_cents))
        return [row]
    return best[1]


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
    tags = [str(tag) for row in sequence for tag in (row.get("tags") or [])]
    experience = ", ".join(dict.fromkeys(tags))[:200] or None
    return PackageOption(
        package_id=f"package:{package_key}:{'-'.join(set_ids)}",
        label=label,
        price_cents=int(target_cents),
        set_id=set_ids[0],
        set_ids=set_ids,
        experience=experience,
    )


def build_offer_packages(
    rows: list[dict[str, Any]],
    policy: CreatorPolicy,
    *,
    preferred_tags: list[str] | None = None,
) -> list[PackageOption]:
    if not rows:
        return []
    quick_sequence = choose_sequence(
        rows,
        target_cents=policy.quick_package_target_cents,
        min_steps=policy.session_min_steps,
        max_steps=min(policy.session_max_steps, 3),
        preferred_tags=preferred_tags,
    )
    quick = package_from_sequence(
        quick_sequence,
        label="quick private session",
        target_cents=policy.quick_package_target_cents,
        package_key="quick",
    )
    packages = [quick] if quick else []

    if policy.offer_two_packages:
        # Reuse of the first step is allowed only if there is not enough coherent
        # media. Prefer a larger progression for the premium package.
        full_sequence = choose_sequence(
            rows,
            target_cents=policy.full_package_target_cents,
            min_steps=max(policy.session_min_steps, len(quick_sequence) + 1),
            max_steps=policy.session_max_steps,
            preferred_tags=preferred_tags,
        )
        full = package_from_sequence(
            full_sequence,
            label="full private session",
            target_cents=policy.full_package_target_cents,
            package_key="full",
        )
        if full and (not quick or full.set_ids != quick.set_ids):
            packages.append(full)

    return sorted(packages, key=lambda package: package.price_cents)
