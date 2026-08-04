"""Context-aware delivery timing for full-auto messages.

Human response time is mostly availability, not reading and typing speed. A live
exchange should feel immediate, while a casual or resumed conversation should
sometimes wait minutes. Availability and composition are kept separate so the
platform typing indicator is never shown throughout an "away" delay.
"""
from __future__ import annotations

import random
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

_PPV_MARKER_RE = re.compile(r"\[PPV:[^\]]+\]")
_INTIMATE_PHASES = {"TENSION", "PAID_SESSION"}


class AvailabilityMode(str, Enum):
    LIVE = "live"
    INTIMATE = "intimate"
    WARM = "warm"
    CASUAL = "casual"
    RETURNING = "returning"
    NEW = "new"


@dataclass(frozen=True)
class DeliverySchedule:
    availability_delay_seconds: float
    composition_delay_seconds: float
    inter_part_delays_seconds: tuple[float, ...]
    availability_mode: AvailabilityMode

    @property
    def initial_delay_seconds(self) -> float:
        """Backward-compatible total delay before the first outgoing bubble."""
        return round(
            self.availability_delay_seconds + self.composition_delay_seconds, 2
        )


def visible_text(value: str) -> str:
    return _PPV_MARKER_RE.sub("", str(value or "")).strip()


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _message_value(message: Any, name: str) -> Any:
    if isinstance(message, dict):
        return message.get(name)
    return getattr(message, name, None)


def _as_utc(value: Any) -> datetime | None:
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def infer_availability_mode(
    conversation_history: Sequence[Any],
    *,
    conversation_phase: str | None = None,
    active_session: dict[str, Any] | None = None,
) -> AvailabilityMode:
    """Infer whether the creator is actively present or returning later.

    The gap that matters is from the creator's last message to the newest fan
    message. This avoids treating a burst of fan messages as an established live
    exchange. Missing timestamps deliberately fall back to a normal new-chat delay.
    """
    messages = list(conversation_history or [])
    latest_fan_index: int | None = None
    for index in range(len(messages) - 1, -1, -1):
        if str(_message_value(messages[index], "role") or "").lower() == "fan":
            latest_fan_index = index
            break
    if latest_fan_index is None:
        return AvailabilityMode.NEW

    latest_fan_at = _as_utc(_message_value(messages[latest_fan_index], "sent_at"))
    last_creator_at: datetime | None = None
    creator_count = 0
    for index in range(latest_fan_index - 1, -1, -1):
        if str(_message_value(messages[index], "role") or "").lower() != "creator":
            continue
        creator_count += 1
        if last_creator_at is None:
            last_creator_at = _as_utc(_message_value(messages[index], "sent_at"))

    if creator_count == 0:
        return AvailabilityMode.NEW
    if latest_fan_at is None or last_creator_at is None:
        return AvailabilityMode.CASUAL

    gap_seconds = max(0.0, (latest_fan_at - last_creator_at).total_seconds())
    phase = str(conversation_phase or "").upper()
    session_active = str((active_session or {}).get("status") or "").lower() == "active"

    # An immediate response to the creator is an actual live exchange regardless
    # of commercial phase. A sexting/paid session gets a wider active window.
    if gap_seconds <= 4 * 60:
        return AvailabilityMode.LIVE
    if gap_seconds <= 15 * 60 and (session_active or phase in _INTIMATE_PHASES):
        return AvailabilityMode.INTIMATE
    if gap_seconds <= 20 * 60:
        return AvailabilityMode.WARM
    if gap_seconds <= 4 * 60 * 60:
        return AvailabilityMode.CASUAL
    return AvailabilityMode.RETURNING


def _sample_availability_delay(
    mode: AvailabilityMode, generator: random.Random
) -> float:
    """Sample an intentionally uneven availability pause.

    Ordinary chats use mixtures rather than one uniform range. Most replies remain
    commercially responsive, but a meaningful minority take several minutes and a
    resumed conversation can occasionally take about half an hour.
    """
    roll = generator.random()
    if mode == AvailabilityMode.LIVE:
        return generator.uniform(0.5, 8.0)
    if mode == AvailabilityMode.INTIMATE:
        return generator.uniform(3.0, 25.0)
    if mode == AvailabilityMode.WARM:
        if roll < 0.20:
            return generator.uniform(8.0, 25.0)
        if roll < 0.75:
            return generator.uniform(25.0, 90.0)
        if roll < 0.95:
            return generator.uniform(90.0, 240.0)
        return generator.uniform(240.0, 420.0)
    if mode == AvailabilityMode.CASUAL:
        if roll < 0.12:
            return generator.uniform(15.0, 60.0)
        if roll < 0.60:
            return generator.uniform(60.0, 240.0)
        if roll < 0.90:
            return generator.uniform(240.0, 600.0)
        return generator.uniform(600.0, 1200.0)
    if mode == AvailabilityMode.RETURNING:
        if roll < 0.08:
            return generator.uniform(20.0, 75.0)
        if roll < 0.40:
            return generator.uniform(75.0, 300.0)
        if roll < 0.80:
            return generator.uniform(300.0, 900.0)
        if roll < 0.95:
            return generator.uniform(900.0, 1800.0)
        return generator.uniform(1800.0, 2400.0)

    # A first interaction should not look permanently staffed, but it should be
    # answered sooner than a cold conversation resumed after many hours or days.
    if roll < 0.15:
        return generator.uniform(15.0, 45.0)
    if roll < 0.75:
        return generator.uniform(45.0, 180.0)
    if roll < 0.95:
        return generator.uniform(180.0, 420.0)
    return generator.uniform(420.0, 720.0)


def build_availability_delay(
    conversation_history: Sequence[Any],
    *,
    conversation_phase: str | None = None,
    active_session: dict[str, Any] | None = None,
    rng: random.Random | None = None,
) -> tuple[AvailabilityMode, float]:
    """Choose a durable pre-composition pause from conversation cadence."""
    generator = rng or random.Random()
    mode = infer_availability_mode(
        conversation_history,
        conversation_phase=conversation_phase,
        active_session=active_session,
    )
    return mode, round(_sample_availability_delay(mode, generator), 2)


def build_delivery_schedule(
    incoming_text: str,
    outgoing_parts: Sequence[str],
    *,
    conversation_history: Sequence[Any] = (),
    conversation_phase: str | None = None,
    active_session: dict[str, Any] | None = None,
    rng: random.Random | None = None,
) -> DeliverySchedule:
    """Return availability, reading/typing, and split-bubble delays."""
    generator = rng or random.Random()
    parts = [visible_text(part) for part in outgoing_parts if visible_text(part)]
    first = parts[0] if parts else ""
    mode, availability = build_availability_delay(
        conversation_history,
        conversation_phase=conversation_phase,
        active_session=active_session,
        rng=generator,
    )

    reading_seconds = min(
        7.0, len(str(incoming_text or "")) / generator.uniform(16.0, 28.0)
    )
    typing_seconds = len(first) / generator.uniform(6.0, 10.0)
    reaction_pause = generator.uniform(1.2, 3.8)
    composition = _clamp(
        reading_seconds + typing_seconds + reaction_pause, 2.5, 22.0
    )

    between: list[float] = []
    for part in parts[1:]:
        bubble_pause = generator.uniform(0.8, 2.4)
        part_typing = len(part) / generator.uniform(6.5, 10.5)
        between.append(_clamp(bubble_pause + part_typing, 1.5, 14.0))

    return DeliverySchedule(
        availability_delay_seconds=round(availability, 2),
        composition_delay_seconds=round(composition, 2),
        inter_part_delays_seconds=tuple(round(value, 2) for value in between),
        availability_mode=mode,
    )
