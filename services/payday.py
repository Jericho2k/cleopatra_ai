"""Resolve a fan's stated payday ("Friday", "the 1st", "next week") into a concrete
datetime, so a promise becomes a scheduled action.

Deliberately conservative: if we can't resolve an expression confidently we return
None and the fan goes to PAUSED_NO_BUDGET rather than getting a follow-up on a date
we guessed. A wrong-day follow-up is worse than none.
"""
import re
from datetime import datetime, timedelta, timezone

_WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}


def resolve_payday(
    raw: str,
    now: datetime | None = None,
    send_hour: int = 18,
) -> tuple[datetime | None, float]:
    """Return (payday_at, confidence). Confidence 0.0 means unresolved."""
    if not raw:
        return None, 0.0
    now = now or datetime.now(timezone.utc)
    text = raw.strip().lower()

    # "friday", "on friday", "this friday", "next friday"
    for name, idx in _WEEKDAYS.items():
        if name in text:
            days_ahead = (idx - now.weekday()) % 7
            if days_ahead == 0:
                days_ahead = 7  # "friday" said on a Friday means next Friday
            if "next" in text:
                days_ahead += 7 if days_ahead <= 7 else 0
            target = now + timedelta(days=days_ahead)
            return target.replace(hour=send_hour, minute=0, second=0, microsecond=0), 0.9

    # "tomorrow"
    if "tomorrow" in text:
        target = now + timedelta(days=1)
        return target.replace(hour=send_hour, minute=0, second=0, microsecond=0), 0.9

    # "in a week", "next week"
    if "next week" in text or re.search(r"in a week", text):
        target = now + timedelta(days=7)
        return target.replace(hour=send_hour, minute=0, second=0, microsecond=0), 0.7

    # "the 1st", "on the 15th"
    m = re.search(r"\b(\d{1,2})(st|nd|rd|th)\b", text)
    if m:
        day = int(m.group(1))
        if 1 <= day <= 31:
            year, month = now.year, now.month
            if day <= now.day:  # already passed this month -> next month
                month += 1
                if month > 12:
                    month, year = 1, year + 1
            try:
                target = datetime(year, month, day, send_hour, 0, 0, tzinfo=timezone.utc)
                return target, 0.8
            except ValueError:
                return None, 0.0

    # "in N days"
    m = re.search(r"in (\d{1,2}) days?", text)
    if m:
        target = now + timedelta(days=int(m.group(1)))
        return target.replace(hour=send_hour, minute=0, second=0, microsecond=0), 0.8

    return None, 0.0