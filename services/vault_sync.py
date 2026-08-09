"""Pure policy helpers for low-cost incremental vault synchronization."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from typing import Any

DEFAULT_VAULT_SYNC_INTERVAL_HOURS = 24


def parse_utc(value: object) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    elif value:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def vault_sync_cooldown(
    last_sync_at: object,
    *,
    now: datetime | None = None,
    interval_hours: float = DEFAULT_VAULT_SYNC_INTERVAL_HOURS,
) -> dict[str, Any]:
    """Return compatibility cooldown fields for a 24-hour sync interval."""
    current = parse_utc(now) or datetime.now(timezone.utc)
    last = parse_utc(last_sync_at)
    if last is None:
        return {
            "allowed": True,
            "last_at": str(last_sync_at) if last_sync_at else None,
            "hours_remaining": 0,
            "days_remaining": 0,
            "next_allowed_at": None,
        }

    next_allowed = last + timedelta(hours=max(1.0, float(interval_hours)))
    if current >= next_allowed:
        return {
            "allowed": True,
            "last_at": last.isoformat(),
            "hours_remaining": 0,
            "days_remaining": 0,
            "next_allowed_at": None,
        }

    remaining_seconds = (next_allowed - current).total_seconds()
    return {
        "allowed": False,
        "last_at": last.isoformat(),
        "hours_remaining": round(remaining_seconds / 3600, 1),
        # Kept for older dashboard builds that still render this field.
        "days_remaining": round(remaining_seconds / 86400, 1),
        "next_allowed_at": next_allowed.isoformat(),
    }


def ordered_vault_albums(albums: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Scan custom albums before Fansly's overlapping catch-all album.

    A media row currently stores one album association. Processing custom albums
    first preserves the useful album name, while the final ``All`` pass catches
    media that is not organized anywhere else.
    """

    def key(album: dict[str, Any]) -> tuple[int, int, str]:
        title = str(album.get("title") or "")
        is_all = int(album.get("type") or 0) == 38000 or title.strip().lower() == "all"
        return (
            1 if is_all else 0,
            -int(album.get("itemCount") or 0),
            title.lower(),
        )

    return sorted((dict(album) for album in albums), key=key)


def should_stop_album_scan(
    *,
    items: list[dict[str, Any]],
    next_cursor: object,
    is_first_page: bool,
    last_item_id: object,
    all_items_known: bool,
    consecutive_known_batches: int,
) -> bool:
    """Stop safely once the API proves the newest page is already local."""
    if not next_cursor:
        return True
    first_entry_id = str((items[0] if items else {}).get("id") or "")
    latest_entry_id = str(last_item_id or "")
    confirmed_latest_page = bool(
        is_first_page
        and latest_entry_id
        and first_entry_id == latest_entry_id
    )
    return bool(
        (all_items_known and confirmed_latest_page)
        or consecutive_known_batches >= 3
    )
