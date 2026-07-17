"""Length-aware, jittered delivery timing for full-auto messages.

The generator can finish instantly, but the send path should still look like a
person reading and typing. These helpers are pure so timing remains testable.
"""
from __future__ import annotations

from dataclasses import dataclass
import random
import re
from typing import Sequence


_PPV_MARKER_RE = re.compile(r"\[PPV:[^\]]+\]")


@dataclass(frozen=True)
class DeliverySchedule:
    initial_delay_seconds: float
    inter_part_delays_seconds: tuple[float, ...]


def visible_text(value: str) -> str:
    return _PPV_MARKER_RE.sub("", str(value or "")).strip()


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def build_delivery_schedule(
    incoming_text: str,
    outgoing_parts: Sequence[str],
    *,
    rng: random.Random | None = None,
) -> DeliverySchedule:
    """Return realistic delays that grow with reading and typing length.

    The pre-generation debounce already absorbs rapid multi-message bursts, so
    these delays are intentionally bounded. Long messages take longer, but never
    create absurd multi-minute waits.
    """
    generator = rng or random.Random()
    parts = [visible_text(part) for part in outgoing_parts if visible_text(part)]
    first = parts[0] if parts else ""

    reading_seconds = min(4.5, len(str(incoming_text or "")) / generator.uniform(18.0, 30.0))
    typing_seconds = len(first) / generator.uniform(6.0, 10.0)
    reaction_pause = generator.uniform(1.2, 3.8)
    initial = _clamp(reading_seconds + typing_seconds + reaction_pause, 2.5, 18.0)

    between: list[float] = []
    for part in parts[1:]:
        bubble_pause = generator.uniform(0.8, 2.4)
        part_typing = len(part) / generator.uniform(6.5, 10.5)
        between.append(_clamp(bubble_pause + part_typing, 1.5, 14.0))

    return DeliverySchedule(
        initial_delay_seconds=round(initial, 2),
        inter_part_delays_seconds=tuple(round(value, 2) for value in between),
    )
