"""Deterministic message-shape policy for Full Auto output.

The writer prompt has always said "most replies are one bubble", and Full Auto
still sent exactly two bubbles roughly nine turns in ten. Prompt text cannot fix
that, because nothing downstream ever looked at the shape: Auto takes option 1
verbatim and splits it on " | ". Whatever two-part rhythm the model settles into
becomes the creator's entire texting personality.

So the shape is decided here, deterministically, and the writer is told the
target before it writes. Two properties matter:

* No randomness. The same conversation state always produces the same shape, so
  the behaviour is testable and reproducible in the simulator.
* Never split a coherent sentence. This policy only ever *merges* bubbles the
  writer chose to separate, or leaves them alone. A three-bubble burst happens
  when the writer genuinely wrote three and the policy has room for it.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

SEPARATOR = " | "

_PPV_TAG_RE = re.compile(r"\[PPV:[^\]]+\]", re.IGNORECASE)

# One deterministic cycle over twelve creator turns: eight singles, three
# doubles, one triple — 67% / 25% / 8%. The agency wants singles dominant,
# doubles common, and triples occasional rather than absent.
SHAPE_CYCLE: tuple[int, ...] = (1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 3, 1)

# Below this, a reply is one thought and splitting it reads as padding.
SHORT_REPLY_WORDS = 8
# A long reply earns a second bubble even when the cycle asked for one.
LONG_REPLY_WORDS = 34
# How many identical consecutive multi-bubble shapes before the policy
# deliberately breaks the pattern. 2/2/2/2/2 is the exact failure being fixed.
# Runs of single bubbles are left alone: singles are supposed to dominate, and
# breaking them up is how the bias toward doubles came back.
REPETITION_LIMIT = 3


@dataclass(frozen=True)
class MessageShape:
    """The bubble count this turn should use, and why."""

    target_bubbles: int
    reason: str

    def to_context(self) -> dict[str, Any]:
        return {"target_bubbles": self.target_bubbles, "reason": self.reason}


def bubble_count(text: str) -> int:
    return max(1, len([part for part in str(text or "").split("|") if part.strip()]))


def recent_bubble_counts(messages: Iterable[Any], *, limit: int = 6) -> list[int]:
    """Bubble counts of the most recent creator turns, oldest first.

    Each stored creator message is one bubble, so consecutive creator messages
    are grouped into the turns they were sent as.
    """
    roles_and_text = [
        (str(_field(message, "role") or ""), str(_field(message, "content") or ""))
        for message in messages
    ]
    turns: list[int] = []
    run = 0
    for role, _ in roles_and_text:
        if role == "creator":
            run += 1
            continue
        if run:
            turns.append(run)
        run = 0
    if run:
        turns.append(run)
    return turns[-limit:]


def choose_message_shape(
    *,
    fan_id: str | None = None,
    recent_counts: Sequence[int] | None = None,
    turn_index: int | None = None,
    turn_key: str | None = None,
    max_messages: int | None = None,
    is_ppv_delivery: bool = False,
    writer_word_count: int | None = None,
) -> MessageShape:
    """Pick this turn's target bubble count. Pure and deterministic.

    Callers with a genuine monotonic turn counter pass ``turn_index`` and walk
    the cycle in order. Production passes ``turn_key`` instead — the message
    being answered — because the conversation history it can see is capped at
    the most recent messages, so a counter derived from it stops advancing in a
    long chat and would freeze one shape forever. Hashing the turn key samples
    the same cycle uniformly, which holds the same distribution without needing
    a counter that does not exist.
    """
    history = [int(value) for value in (recent_counts or []) if int(value) > 0]
    if turn_key is not None:
        position = _stable_index(f"{fan_id or ''}|{turn_key}")
    else:
        index = len(history) if turn_index is None else int(turn_index)
        position = (index + _stable_index(str(fan_id or ""))) % len(SHAPE_CYCLE)
    target = SHAPE_CYCLE[position % len(SHAPE_CYCLE)]
    reason = "shape_cycle"

    # Break a run of identical multi-bubble shapes even when the cycle repeats.
    recent = history[-REPETITION_LIMIT:]
    if len(recent) == REPETITION_LIMIT and len(set(recent)) == 1 and recent[0] > 1:
        if target == recent[0]:
            target = 1
            reason = "break_repeated_shape"

    if writer_word_count is not None:
        if writer_word_count <= SHORT_REPLY_WORDS and target > 1:
            target = 1
            reason = "short_reply_is_one_thought"
        elif writer_word_count >= LONG_REPLY_WORDS and target == 1:
            target = 2
            reason = "long_reply_earns_a_second_bubble"

    if is_ppv_delivery:
        # A locked send is a short line plus the attachment. More parts around
        # it reads as a pitch.
        target = min(target, 2)
        reason = "ppv_delivery_turn"

    if max_messages is not None:
        capped = max(1, int(max_messages))
        if capped < target:
            target = capped
            reason = "commercial_max_messages"

    return MessageShape(target_bubbles=max(1, target), reason=reason)


def apply_message_shape(reply: str, target_bubbles: int) -> str:
    """Merge a reply down to at most ``target_bubbles`` bubbles.

    Merging only. A writer that produced one flowing sentence keeps it; a turn
    that genuinely needed three rapid-fire bubbles is never padded up to three
    from two. A bubble carrying a [PPV:...] tag is never merged into another,
    because the delivery layer reads the text before the tag as the caption.
    """
    parts = [part.strip() for part in str(reply or "").split("|") if part.strip()]
    if not parts:
        return str(reply or "").strip()
    target = max(1, int(target_bubbles))
    if len(parts) <= target:
        return SEPARATOR.join(parts)

    protected = [bool(_PPV_TAG_RE.search(part)) for part in parts]
    merged = list(parts)
    flags = list(protected)
    while len(merged) > target:
        index = _merge_index(flags)
        if index is None:
            break
        merged[index] = _join(merged[index], merged[index + 1])
        flags[index] = flags[index] or flags[index + 1]
        del merged[index + 1]
        del flags[index + 1]
    return SEPARATOR.join(merged)


def _merge_index(flags: list[bool]) -> int | None:
    """First position whose merge would not absorb a PPV bubble."""
    for index in range(len(flags) - 1):
        if not flags[index + 1]:
            return index
    return None


def _join(left: str, right: str) -> str:
    left = left.rstrip()
    if not left:
        return right
    if left[-1] in ".!?…,:;":
        return f"{left} {right}"
    return f"{left}, {right}"


def _stable_index(value: str) -> int:
    """Deterministic cycle position, so two chats are not in lockstep."""
    if not value:
        return 0
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % len(SHAPE_CYCLE)


def _field(message: Any, name: str) -> Any:
    if isinstance(message, dict):
        return message.get(name)
    return getattr(message, name, None)
