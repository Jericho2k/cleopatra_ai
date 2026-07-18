"""Deterministic vault-operation policy.

The expensive AI work remains in ``main.py`` for now, while this module owns
the small contract that must stay stable across the API, scheduler and tests.
"""
from __future__ import annotations

from typing import Iterable


MANUAL_RECATEGORIZATION_DAILY_LIMIT = 5


def normalize_media_ids(media_ids: Iterable[str] | None) -> list[str]:
    """Return unique, non-empty media identifiers without changing order."""
    seen: set[str] = set()
    normalized: list[str] = []
    for raw in media_ids or []:
        media_id = str(raw or "").strip()
        if not media_id or media_id in seen:
            continue
        seen.add(media_id)
        normalized.append(media_id)
    return normalized


def manual_recategorization_usage(
    used: int,
    daily_limit: int = MANUAL_RECATEGORIZATION_DAILY_LIMIT,
) -> dict[str, int | bool]:
    safe_limit = max(int(daily_limit), 0)
    safe_used = max(int(used), 0)
    remaining = max(safe_limit - safe_used, 0)
    return {
        "used": safe_used,
        "remaining": remaining,
        "daily_limit": safe_limit,
        "allowed": remaining > 0,
    }


def categorize_new_batch_enabled(auto_enabled: bool, media_ids: Iterable[str] | None) -> bool:
    """New-media categorization is opt-in and never falls back to the full vault."""
    return bool(auto_enabled and normalize_media_ids(media_ids))

