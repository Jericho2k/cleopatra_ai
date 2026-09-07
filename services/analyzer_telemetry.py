"""Countable analyzer outcomes (REL-001).

The question this exists to answer, without reading Railway logs:

    "How many suggestions/replies this hour had a failed analyzer?"

Counters are in-process and bucketed by wall-clock hour. That matches the
existing runtime-health surfaces (``/model-runtime-health`` is likewise in
memory) and is enough for incident triage: a restart loses history, but a
provider incident is visible while it is happening, which is the point.
"""
from __future__ import annotations

import threading
from collections import defaultdict
from datetime import datetime, timedelta, timezone

# Two days of hourly buckets is far more than the "this hour" question needs and
# still bounds memory to a few dozen small entries.
_RETAINED_HOURS = 48

_lock = threading.Lock()
_counts: dict[tuple[str, str, str], int] = defaultdict(int)


def _hour_key(moment: datetime | None = None) -> str:
    now = moment or datetime.now(timezone.utc)
    return now.replace(minute=0, second=0, microsecond=0).isoformat()


def _prune(now: datetime) -> None:
    cutoff = _hour_key(now - timedelta(hours=_RETAINED_HOURS))
    for key in [key for key in _counts if key[0] < cutoff]:
        _counts.pop(key, None)


def record_analysis_outcome(
    *,
    degraded: bool,
    reason: str = "",
    feature: str = "unknown",
) -> None:
    """Record one analyzer result. Never raises — telemetry must not break a reply."""
    try:
        now = datetime.now(timezone.utc)
        key = (_hour_key(now), feature, reason if degraded else "")
        with _lock:
            _counts[key] += 1
            _prune(now)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[ANALYZER TELEMETRY] not recorded: {exc}")


def analyzer_health(hours: int = 1) -> dict:
    """Degraded vs total analyses over the last ``hours`` whole hours."""
    now = datetime.now(timezone.utc)
    wanted = {
        _hour_key(now - timedelta(hours=offset)) for offset in range(max(hours, 1))
    }
    total = 0
    degraded = 0
    by_reason: dict[str, int] = defaultdict(int)
    by_feature: dict[str, int] = defaultdict(int)
    with _lock:
        for (hour, feature, reason), count in _counts.items():
            if hour not in wanted:
                continue
            total += count
            if reason:
                degraded += count
                by_reason[reason] += count
                by_feature[feature] += count
    return {
        "window_hours": max(hours, 1),
        "analyses": total,
        "degraded": degraded,
        "degraded_by_reason": dict(by_reason),
        "degraded_by_feature": dict(by_feature),
    }


def reset_for_tests() -> None:
    with _lock:
        _counts.clear()
