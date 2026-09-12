"""Platform semantics for paid content delivery.

Fansly PPV media is attached directly to a chat message. There is no link to
send, click, or open, and Cleopatra never moves a fan off-platform. A writer
that offers "the link" is describing a product that does not exist, and it
reads as a scam to the fan.

The writer prompt says so, but a prompt is a request. This module is the
deterministic backstop for turns that actually sell or deliver content, and it
is deliberately narrow: ordinary conversational uses of the word "link" —
a linked Instagram post he mentions, "link up", a chain link — are left alone.
"""

from __future__ import annotations

import re
from typing import Iterable

# Actions the commercial layer can decide that involve offering or delivering
# Cleopatra-controlled content. Only these turns are sanitized.
DELIVERY_ACTIONS = frozenset(
    {
        "PRESENT_SESSION_OPTIONS",
        "END_TEASER_AND_OFFER",
        "CREATE_PAID_SESSION",
        "SEND_NEXT_PPV_STEP",
        "RESUME_PREVIOUS_OFFER",
        "PAYDAY_REENGAGEMENT",
    }
)

_PPV_TAG_RE = re.compile(r"\[PPV:[^\]]+\]", re.IGNORECASE)

# Each rule is (pattern, replacement). Replacements keep the creator's register:
# short, lowercase, and about sending the thing itself.
_DELIVERY_LINK_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bwant(?:\s+the|\s+a|\s+my)?\s+link\b", re.IGNORECASE), "want it"),
    (re.compile(r"\bwan+a\s+(?:the\s+)?link\b", re.IGNORECASE), "want it"),
    (
        re.compile(r"\b(?:i(?:'| a)?ll|i will|lemme|let me)\s+(?:send|drop|shoot|dm)\s+(?:you\s+)?(?:the|a|my)\s+link\b", re.IGNORECASE),
        "i'll send it",
    ),
    (
        re.compile(r"\b(?:here(?:'s| is)|there(?:'s| is))\s+(?:the|a|my|ur|your)\s+link\b", re.IGNORECASE),
        "here it is",
    ),
    (
        re.compile(r"\b(?:click|open|tap|follow|check)\s+(?:on\s+)?(?:the|this|that|my)\s+link\b", re.IGNORECASE),
        "open it",
    ),
    (
        re.compile(r"\b(?:sending|dropping|posting)\s+(?:you\s+)?(?:the|a|my)\s+link\b", re.IGNORECASE),
        "sending it",
    ),
    (
        re.compile(r"\b(?:send|drop|shoot|dm)\s+(?:you\s+)?(?:the|a|my)\s+link\b", re.IGNORECASE),
        "send it",
    ),
    (
        re.compile(r"\b(?:the|a|my)\s+link\s+(?:to|for)\s+(?:it|the|that|this|those|these)\b", re.IGNORECASE),
        "it",
    ),
    (re.compile(r"\b(?:the|a|my)\s+link\s+is\s+(?:below|here|coming|above)\b", re.IGNORECASE), "it's right here"),
    (re.compile(r"\blink(?:ed)?\s+(?:it|them|those)\s+(?:below|here|above)\b", re.IGNORECASE), "sent it here"),
    (re.compile(r"\bunlock\s+(?:the|this|that)\s+link\b", re.IGNORECASE), "unlock it"),
)


def is_delivery_turn(
    *,
    decision_action: str | None = None,
    reply: str | None = None,
    active_session: dict | None = None,
) -> bool:
    """True when this turn offers or delivers Cleopatra-controlled content."""
    if str(decision_action or "").upper() in DELIVERY_ACTIONS:
        return True
    if reply and _PPV_TAG_RE.search(reply):
        return True
    return bool(active_session and active_session.get("status") == "active")


def contains_delivery_link_language(text: str) -> bool:
    """True when the text promises a link for content we attach in chat."""
    return any(pattern.search(str(text or "")) for pattern, _ in _DELIVERY_LINK_RULES)


def repair_delivery_link_language(text: str) -> str:
    """Rewrite delivery-link phrasing into how content is actually sent.

    Deterministic and idempotent. The repaired text is still checked by the
    caller; anything this cannot express naturally is dropped rather than sent.
    """
    repaired = str(text or "")
    for pattern, replacement in _DELIVERY_LINK_RULES:
        repaired = pattern.sub(replacement, repaired)
    repaired = re.sub(r"[ \t]{2,}", " ", repaired)
    repaired = re.sub(r"\s+([,.!?])", r"\1", repaired)
    return repaired.strip()


def sanitize_delivery_language(
    text: str,
    *,
    decision_action: str | None = None,
    active_session: dict | None = None,
) -> tuple[str, bool]:
    """Return ``(text, repaired)`` for one outgoing reply.

    Non-commercial turns are returned untouched, so "send me the link to your
    spotify" stays exactly as written.
    """
    original = str(text or "")
    if not is_delivery_turn(
        decision_action=decision_action,
        reply=original,
        active_session=active_session,
    ):
        return original, False
    if not contains_delivery_link_language(original):
        return original, False
    return repair_delivery_link_language(original), True


def sanitize_candidates(
    candidates: Iterable[str],
    *,
    decision_action: str | None = None,
    active_session: dict | None = None,
) -> list[str]:
    return [
        sanitize_delivery_language(
            candidate,
            decision_action=decision_action,
            active_session=active_session,
        )[0]
        for candidate in candidates
    ]
