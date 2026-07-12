"""Resolve stated payday expressions into timezone-aware datetimes."""
import re
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

_WEEKDAYS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}


def _zone(name: str | None):
    try:
        return ZoneInfo(name or "UTC")
    except ZoneInfoNotFoundError:
        return timezone.utc


def resolve_payday(
    raw: str,
    now: datetime | None = None,
    send_hour: int = 18,
    timezone_name: str = "UTC",
) -> tuple[datetime | None, float]:
    """Return (payday_at, confidence), using the creator/fan local timezone."""
    if not raw:
        return None, 0.0

    tz = _zone(timezone_name)
    if now is None:
        now = datetime.now(tz)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=tz)
    else:
        now = now.astimezone(tz)

    text = raw.strip().lower()

    for name, weekday in _WEEKDAYS.items():
        if name in text:
            days_ahead = (weekday - now.weekday()) % 7
            if days_ahead == 0:
                days_ahead = 7
            if "next" in text and days_ahead < 7:
                days_ahead += 7
            target = now + timedelta(days=days_ahead)
            return target.replace(hour=send_hour, minute=0, second=0, microsecond=0), 0.9

    if "tomorrow" in text:
        target = now + timedelta(days=1)
        return target.replace(hour=send_hour, minute=0, second=0, microsecond=0), 0.9

    if "next week" in text or re.search(r"\bin a week\b", text):
        target = now + timedelta(days=7)
        return target.replace(hour=send_hour, minute=0, second=0, microsecond=0), 0.7

    ordinal = re.search(r"\b(?:the\s+)?(\d{1,2})(?:st|nd|rd|th)\b", text)
    if ordinal:
        day = int(ordinal.group(1))
        if 1 <= day <= 31:
            year, month = now.year, now.month
            if day <= now.day:
                month += 1
                if month > 12:
                    month, year = 1, year + 1
            try:
                return datetime(year, month, day, send_hour, tzinfo=tz), 0.8
            except ValueError:
                return None, 0.0

    relative = re.search(r"\bin (\d{1,2}) days?\b", text)
    if relative:
        target = now + timedelta(days=int(relative.group(1)))
        return target.replace(hour=send_hour, minute=0, second=0, microsecond=0), 0.8

    return None, 0.0
