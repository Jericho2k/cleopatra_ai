"""Deterministic, provenance-carrying readings of what the FAN actually said.

Three production failures this sprint is fixing all have the same shape: the
conversation layer had no deterministic statement of what the evidence does and
does not support, so the models filled the gap.

1. **Invented publication.** Approved vault inventory was treated as proof that
   something had been posted, and Cleopatra told a fan to "check my feed" for a
   set that exists only as private inventory. Inventory is permission to OFFER;
   it is not evidence that anything was published, that the creator is wearing
   anything, or that a page will show something if the fan refreshes.

2. **Money allergy.** After #68 taught the system that intimacy does not require
   monetization, it started declining explicit, fan-created buying
   opportunities. A fan saying "just tell me the price" is a fact about the
   conversation, not an inference about his wallet, and it belongs in evidence.

3. **Purchase claims read as receipts.** "I bought it" is something a fan said.
   It becomes true when the payment ledger says so and not before.

Everything here is computed from text the fan actually wrote, and every reading
carries the message ids it came from. Nothing here infers wealth, spending
power, a budget, or an escalation stage — those would be the funnel this
architecture removed.
"""

from __future__ import annotations

import re
from typing import Any, Sequence


def _speaker(row: dict[str, Any]) -> str:
    """Raw message rows name the speaker ``speaker``; history rows use ``role``."""
    return str(row.get("speaker") or row.get("role") or "").lower()


def _text(row: dict[str, Any]) -> str:
    return str(row.get("text") or row.get("content") or "")


def _reference(row: dict[str, Any]) -> str:
    return str(row.get("message_id") or row.get("id") or "")


# --- 1. Publication / feed grounding ---------------------------------------

#: Source-ref prefixes that would constitute AUTHORITATIVE evidence that the
#: creator published something. There is no feed integration today, so this set
#: never matches; it exists so that adding one is a data change rather than a
#: prompt rewrite, and so that the rule is stated rather than assumed.
PUBLICATION_SOURCE_PREFIXES = ("feed_post:", "platform_post:", "published_post:")

#: A creator asserting that something was published, or telling the fan to go
#: and look at a page. Deliberately narrow: it matches the CLAIM, never the noun.
#: "that bikini post you mentioned" is fan-supplied context and must survive;
#: "the bedroom set I just posted" and "check my feed" must not.
_PUBLICATION_CLAIM = re.compile(
    # The transitive forms need an object. "I just posted a new set" is a
    # publication claim; "I posted up on the couch all evening" is a Tuesday.
    r"\b(?:i|i['’]ve|i have)\s+(?:just\s+|literally\s+|already\s+)?"
    r"(?:posted|uploaded|put\s+up|dropped)\s+"
    r"(?:a|an|the|some|my|that|this|new|another|it|them|these|those)\b"
    r"|\b(?:just|literally)\s+(?:posted|uploaded)\s+"
    r"(?:a|an|the|some|my|that|this|new|another|it|them|these|those)\b"
    r"|\bjust\s+put\s+it\s+up\b"
    # The relative-clause form, where the object comes first: "the bedroom set
    # I just posted". Anchored to the end of the clause so "I posted up on the
    # couch" still reads as furniture.
    r"|\b(?:i|i['’]ve|i have)\s+(?:just\s+|literally\s+|already\s+)?"
    r"(?:posted|uploaded)"
    r"(?:\s+(?:earlier|today|yesterday|tonight|last\s+night|this\s+morning))?"
    r"\s*(?=$|[.,!?;)\"'])"
    r"|\b(?:check|go\s+(?:check|look|see)|look\s+at|peep|see)\s+"
    r"(?:out\s+)?(?:my|the)\s+(?:feed|page|profile|wall|timeline|posts?|stories|story)\b"
    r"|\b(?:my|the)\s+(?:latest|newest|new|last)\s+post\b"
    r"|\bon\s+my\s+(?:feed|page|profile|wall|timeline)\b"
    r"|\brefresh\s+(?:my|the|your)\s+(?:feed|page|profile)\b"
    r"|\bit['’]?s\s+(?:up\s+)?on\s+my\s+(?:feed|page|profile)\b"
    r"|\bnew\s+post\s+is\s+up\b",
    re.IGNORECASE,
)

#: What the fan may legitimately have brought up themselves. When the fan
#: mentions a post, the creator may talk about it as something HE referred to.
_FAN_PUBLICATION_REFERENCE = re.compile(
    r"\b(?:post(?:ed|s)?|feed|page|profile|story|stories|timeline|upload(?:ed|s)?)\b",
    re.IGNORECASE,
)


def publication_evidence(
    creator_facts: Sequence[Any],
) -> dict[str, Any]:
    """State, in evidence, exactly what is known about the creator's feed."""
    sourced = [
        fact
        for fact in creator_facts
        if str(getattr(fact, "source_ref", "") or "").startswith(
            PUBLICATION_SOURCE_PREFIXES
        )
    ]
    return {
        "authoritative_posts_available": bool(sourced),
        "known_posts": [
            {
                "value": str(getattr(fact, "value", "")),
                "source_ref": str(getattr(fact, "source_ref", "")),
            }
            for fact in sourced[:10]
        ],
        "rule": (
            "Approved vault inventory is permission to offer privately. It is "
            "NOT evidence that anything was published, that a feed post exists, "
            "that the creator is wearing or doing anything now, or that the fan "
            "will find something by refreshing a page."
        ),
        "integration": "none" if not sourced else "sourced",
    }


def fan_publication_references(
    latest_burst: Sequence[dict[str, Any]],
    recent_messages: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Feed/post references the FAN supplied, which may be discussed as his."""
    seen: list[dict[str, Any]] = []
    for row in list(recent_messages)[-30:] + list(latest_burst):
        if _speaker(row) != "fan":
            continue
        text = _text(row)
        if not text or not _FAN_PUBLICATION_REFERENCE.search(text):
            continue
        entry = {
            "source_ref": _reference(row),
            "fan_words": text[:300],
        }
        if entry not in seen:
            seen.append(entry)
    return tuple(seen[-5:])


def unsupported_publication_claim(
    text: str,
    *,
    evidence: dict[str, Any] | None,
    fan_references: Sequence[dict[str, Any]] = (),
) -> bool:
    """Whether this creator copy claims a publication nothing supports.

    Returns False when authoritative post evidence exists. It also returns False
    for wording that only *echoes* a reference the fan made, because discussing
    "that bikini post" he brought up is legitimate and refusing it would be the
    opposite mistake.
    """
    if (evidence or {}).get("authoritative_posts_available"):
        return False
    match = _PUBLICATION_CLAIM.search(str(text or ""))
    if match is None:
        return False
    claim = match.group(0).lower()
    # An echo of the fan's own words is his context, not our invention.
    for reference in fan_references:
        fan_words = str(reference.get("fan_words") or "").lower()
        if claim and claim in fan_words:
            return False
    return True


# --- 2. Fan-created commercial opportunity ---------------------------------

#: Explicit, fan-authored buying signals. Every one of these is something the
#: fan SAID; none of them is an inference about how much money he has.
_PURCHASE_SIGNALS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "asked_price",
        re.compile(
            r"\bhow\s+much\b|\bwhat['’]?s\s+(?:the\s+)?(?:price|cost)\b"
            r"|\b(?:just\s+)?(?:tell|say|name)\s+me\s+(?:the\s+)?price\b"
            r"|\b(?:just\s+)?say\s+the\s+price\b|\bwhat\s+does\s+it\s+cost\b"
            r"|\bhow\s+much\s+(?:is|for|would)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "offered_to_pay",
        re.compile(
            r"\bi['’]?ll\s+pay\b|\bi\s+can\s+pay\b|\bi\s+want\s+to\s+pay\b"
            r"|\bpay\s+(?:extra|more|whatever|double)\b|\btake\s+my\s+money\b"
            r"|\bi\s+can\s+do\s+it\b|\bmoney\s+(?:is\s+)?(?:no|not\s+an)\s+(?:object|issue|problem)\b"
            r"|\bname\s+your\s+price\b|\bi['’]?ll\s+(?:buy|take)\s+it\b",
            re.IGNORECASE,
        ),
    ),
    (
        "asked_to_buy",
        re.compile(
            r"\bwhat\s+can\s+i\s+(?:buy|get)\b|\bwhat\s+(?:do\s+)?you\s+(?:sell|have\s+for\s+sale)\b"
            r"|\bsend\s+me\s+(?:more|another|the\s+(?:rest|next))\b"
            r"|\bi\s+want\s+to\s+(?:buy|unlock|order)\b|\bcan\s+i\s+(?:buy|unlock|order)\b"
            r"|\bunlock\s+it\s+for\s+me\b|\bshow\s+me\s+what\s+you\s+(?:got|have)\b",
            re.IGNORECASE,
        ),
    ),
)


def purchase_intent(
    latest_burst: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Explicit fan-created buying opportunity, with the messages that show it.

    Deliberately restricted to the CURRENT burst. A price question three days
    ago is not a live opportunity, and treating it as one is how a system starts
    selling at people.
    """
    kinds: list[str] = []
    sources: list[str] = []
    for row in latest_burst:
        text = _text(row)
        if not text:
            continue
        for name, pattern in _PURCHASE_SIGNALS:
            if pattern.search(text):
                if name not in kinds:
                    kinds.append(name)
                reference = _reference(row)
                if reference and reference not in sources:
                    sources.append(reference)
    return {
        "fan_stated_buying_signal": bool(kinds),
        "signal_kinds": kinds,
        "source_ids": sources[:8],
        "rule": (
            "This records what the fan said, never what he can afford. Intimacy "
            "alone never creates a commercial opportunity, and an explicit "
            "fan-created one is not cancelled by the conversation being intimate."
        ),
    }


# --- 3. Purchase claims are claims -----------------------------------------

_PURCHASE_CLAIM = re.compile(
    r"\bi\s+(?:just\s+)?(?:bought|paid|purchased|unlocked|got)\s+"
    r"(?:it|this|that|them|the\s+\w+)\b"
    r"|\bi['’]?ve\s+(?:already\s+)?(?:bought|paid|purchased|unlocked)\b"
    r"|\bpayment\s+(?:is\s+)?sent\b|\bjust\s+(?:paid|bought|unlocked)\b"
    r"|\bi\s+sent\s+(?:the\s+)?(?:money|payment|tip)\b",
    re.IGNORECASE,
)

#: Creator copy that treats an unconfirmed claim as a settled transaction.
#: Only consulted when the fan has claimed a purchase that no receipt supports,
#: which is what keeps it from touching ordinary warmth.
_PURCHASE_ACKNOWLEDGEMENT = re.compile(
    r"\b(?:told|knew)\s+you\s+(?:it|that)\s*(?:['’]d|\s+would)\s+be\s+worth\s+it\b"
    r"|\bworth\s+(?:every\s+penny|it)\s*(?:,|!|\.|$)"
    r"|\bthanks?\s+(?:you\s+)?for\s+(?:buying|unlocking|grabbing|the\s+(?:purchase|unlock))\b"
    r"|\bnow\s+that\s+you(?:['’]ve|\s+have)\s+(?:got|bought|unlocked|paid)\b"
    r"|\benjoy\s+(?:it|them|your\s+(?:unlock|purchase))\b"
    r"|\bglad\s+you\s+(?:bought|unlocked|grabbed)\b"
    r"|\byou['’]?(?:re|ve)\s+(?:got|unlocked)\s+(?:it|access)\b",
    re.IGNORECASE,
)


def purchase_claim(
    latest_burst: Sequence[dict[str, Any]],
    *,
    confirmed_purchases: Sequence[dict[str, Any]] = (),
    pending_payment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Whether the fan said he paid, and whether anything authoritative agrees."""
    sources = [
        _reference(row)
        for row in latest_burst
        if _PURCHASE_CLAIM.search(_text(row))
    ]
    claimed = bool(sources)
    return {
        "fan_claimed_purchase": claimed,
        "source_ids": [ref for ref in sources if ref][:8],
        "authoritative_confirmation": bool(confirmed_purchases),
        "pending_payment_exists": bool(pending_payment),
        "rule": (
            "A fan saying he paid is a claim. Until the payment ledger confirms "
            "it, do not acknowledge the purchase, grant access, advance a paid "
            "session, or deliver anything that requires payment."
        ),
    }


def unverified_purchase_acknowledgement(text: str, claim: dict[str, Any]) -> bool:
    """Creator copy treating an unconfirmed purchase claim as settled."""
    if not claim.get("fan_claimed_purchase"):
        return False
    if claim.get("authoritative_confirmation"):
        return False
    return bool(_PURCHASE_ACKNOWLEDGEMENT.search(str(text or "")))


# --- 4. Voice rhythm (soft) ------------------------------------------------

_EMOJI = re.compile(
    "[\U0001F300-\U0001FAFF←-⇿☀-➿️⬀-⯿]"
)


def recent_creator_emoji(
    recent_messages: Sequence[dict[str, Any]], *, limit: int = 6
) -> dict[str, Any]:
    """What the last few creator bubbles leaned on, as soft rhythm context.

    Soft on purpose. Repetition is sometimes exactly right, creator voice wins,
    and nothing here blocks a send — it only lets the writer notice that it has
    ended four bubbles in a row the same way.
    """
    counts: dict[str, int] = {}
    consecutive: list[str] = []
    creator_rows = [
        row
        for row in recent_messages
        if _speaker(row) == "creator"
    ][-limit:]
    for row in creator_rows:
        found = _EMOJI.findall(_text(row))
        unique = list(dict.fromkeys(found))
        consecutive.append("".join(unique))
        for mark in unique:
            counts[mark] = counts.get(mark, 0) + 1
    overused = sorted(
        (mark for mark, count in counts.items() if count >= 2),
        key=lambda mark: -counts[mark],
    )
    return {
        "recent_creator_emoji": list(counts),
        "repeated_in_recent_turns": overused[:5],
        "note": (
            "Soft guidance only. Vary the ending rhythm rather than closing "
            "every bubble the same way; repetition is fine when it is natural, "
            "and this creator's own voice always wins."
        ),
    }
