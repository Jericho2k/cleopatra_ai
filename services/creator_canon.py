"""Ordinary creator self-facts, captured from what she actually sent.

WHY THIS EXISTS
---------------
The creator legend (``creators.legend``, ``db/queries.py``) already is the
canonical store of what the creator has established about herself, and it
already has a writeback path: ``services/suggestions._update_fan_memory`` asks a
model for ``model_facts`` every tenth fan message and merges them in
first-established-wins. That path is not replaced here and is not changed.

What it does not do is the thing ``writer_v3`` needs. It runs on the ASSISTED
path only — Full Auto never calls it — and its extraction is aimed at identity
("name", "origin", "age", "job", "background"), with a freeform ``other`` bucket
for anything else. So under Full Auto, a creator who was asked her favourite
colour and answered "probably dark green" established nothing at all, and three
weeks later answered something different.

``writer_v3`` deliberately lets the writer improvise an ordinary personal detail
rather than say "I don't have one" because a field was never filled in. That is
only an improvement if the improvised detail becomes canon, so this module is
the small piece that makes it one:

* it runs **after the message was actually sent and persisted**, never on a
  candidate. An Assisted suggestion nobody picked, or an Auto turn that aborted
  before delivery, cannot mutate the creator's canon;
* it writes into the **existing** legend, through the existing
  ``update_creator_legend`` merge. There is no second memory system, no new
  table, and no migration;
* it can only ever add to the freeform ``other`` list. The protected identity
  keys are not passed to the merge at all, so nothing here can fill in — let
  alone overwrite — a name, age, origin, job or background;
* it is first-wins per topic, like the rest of the legend. Once "favorite
  color: dark green" is canon, a later "favorite color: blue" is dropped rather
  than appended next to it.

COST
----
A model call per sent reply would be absurd for a fact that appears in maybe one
turn in ten, so a deterministic pre-filter runs first: either the fan asked
something personal, or the reply contains a self-preference marker. Everything
else returns without spending anything.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable, Sequence

from ai.model_providers import complete
from ai.stack_profiles import STAGE_FAN_INTELLIGENCE, get_profile
from db.queries import get_creator_legend, update_creator_legend
from models.model_runtime import ModelTelemetryContext
from services.model_telemetry import record_model_result


# The legend keys the writer may never establish by improvising. They describe
# who she *is* rather than what she likes, and they are configured by the
# operator or learned from a deliberate statement through the slower extraction
# path. This module never passes them to the merge; the tuple is here so the
# topic filter below can also refuse to smuggle one in through ``other``.
PROTECTED_LEGEND_KEYS: tuple[str, ...] = (
    "name",
    "origin",
    "age",
    "job",
    "background",
)

# Words that make a "topic" an identity or reality claim rather than a
# preference. A fact whose topic matches any of these is dropped before it can
# reach the legend, whatever the model decided to call it.
_PROTECTED_TOPIC_MARKERS: tuple[str, ...] = (
    "name",
    "age",
    "old",
    "birth",
    "born",
    "origin",
    "nationality",
    "country",
    "city",
    "town",
    "where she",
    "where i",
    "live",
    "living",
    "location",
    "address",
    "job",
    "work",
    "career",
    "occupation",
    "profession",
    "study",
    "school",
    "university",
    "college",
    "meet",
    "meeting",
    "phone",
    "number",
    "email",
    "snapchat",
    "instagram",
    "telegram",
    "whatsapp",
    "account",
    "platform",
    "subscription",
    "price",
    "prices",
    "pricing",
    "content",
    "vault",
    "boyfriend",
    "husband",
    "married",
    "kids",
    "children",
    "family",
    "real",
)

# What the fan's message has to look like for a personal answer to be likely.
#
# Two shapes, both of which have to be a question. Either he names a preference
# frame outright ("what's your favourite colour?", "do you like coffee?"), or he
# uses a possessive about one of the ordinary topics ("what's your go-to
# drink?"). "did you see the game?" is neither, and correctly costs nothing.
_FAN_PREFERENCE_WORDS: tuple[str, ...] = (
    "favorite",
    "favourite",
    "fav ",
    "fav?",
    "like",
    "love",
    "prefer",
    "into",
    "hobby",
    "hobbies",
)

_FAN_TOPIC_WORDS: tuple[str, ...] = (
    "music",
    "song",
    "band",
    "artist",
    "food",
    "eat",
    "drink",
    "coffee",
    "tea",
    "wine",
    "beer",
    "color",
    "colour",
    "movie",
    "film",
    "show",
    "series",
    "book",
    "read",
    "game",
    "gaming",
    "sport",
    "pet",
    "dog",
    "cat",
    "travel",
    "season",
    "type",
    "go to",
    "go-to",
)

_FAN_POSSESSIVE = re.compile(r"\b(your|yours|ur)\b")

# What a self-disclosure looks like in the reply itself, for the case where the
# fan did not ask a recognisable question.
_SELF_PREFERENCE_MARKERS: tuple[str, ...] = (
    "my favorite",
    "my favourite",
    "my fav",
    "i love",
    "i like",
    "i prefer",
    "i hate",
    "i adore",
    "i enjoy",
    "im into",
    "i'm into",
    "im obsessed",
    "i'm obsessed",
    "obsessed with",
    "i can't stand",
    "i cant stand",
    "my go to",
    "my go-to",
    "i usually",
    "i always",
    "i never",
)

_SYSTEM_PROMPT = """You extract ordinary personal facts that a CREATOR has just stated about herself in a chat message she sent to a fan.

Return ONLY valid JSON, no markdown:
{"facts": [{"topic": "favorite color", "value": "dark green"}]}

Rules:
- Only facts the CREATOR stated about HERSELF in her message. Never anything about the fan, and never anything the fan said.
- Only ordinary, harmless preferences and opinions: a favourite colour, a food, a drink, music, a film or show, a book, a hobby, a pet, a small like or dislike.
- NEVER extract her name, age, where she is from, where she lives, her job, her studies, her background, her family or relationship status, whether she can meet anyone, prices, or anything about the platform or her account. Those are not yours to record.
- "topic" is a short lowercase noun phrase naming what the fact is about ("favorite color", "coffee", "music taste"). "value" is what she said, in a few words.
- If she stated no such fact, return {"facts": []}. An empty list is the correct answer far more often than not.
"""

# Rendered form of one soft fact, and how it is read back apart again.
_ENTRY_SEPARATOR = ": "

_MAX_FACTS_PER_TURN = 4
_MAX_TOPIC_CHARS = 40
_MAX_VALUE_CHARS = 80


def _normalized(value: object) -> str:
    return " ".join(str(value or "").split()).strip()


def mentions_self_fact(*, reply: str, fan_message: str = "") -> bool:
    """Cheap gate: could this exchange plausibly have established a self-fact?

    False here costs nothing but a missed capture on this turn. True costs one
    extraction call, so the test is deliberately narrow: the fan asked something
    personal, or the reply says what she likes.
    """
    reply_text = _normalized(reply).lower()
    if not reply_text:
        return False

    if any(marker in reply_text for marker in _SELF_PREFERENCE_MARKERS):
        return True

    fan_text = _normalized(fan_message).lower()
    if not fan_text or "?" not in fan_text:
        return False
    if any(word in fan_text for word in _FAN_PREFERENCE_WORDS):
        return True
    return bool(_FAN_POSSESSIVE.search(fan_text)) and any(
        word in fan_text for word in _FAN_TOPIC_WORDS
    )


def is_protected_topic(topic: str) -> bool:
    """Whether this topic names identity or platform reality rather than taste."""
    text = _normalized(topic).lower()
    if not text:
        return True
    if text in PROTECTED_LEGEND_KEYS:
        return True
    words = set(re.findall(r"[a-z']+", text))
    return any(
        marker in words or (" " in marker and marker in text)
        for marker in _PROTECTED_TOPIC_MARKERS
    )


def existing_topics(legend: dict[str, Any] | None) -> set[str]:
    """Topics the legend already has an answer for, protected keys included."""
    topics: set[str] = set()
    for key in PROTECTED_LEGEND_KEYS:
        if _normalized((legend or {}).get(key)):
            topics.add(key)
    other = (legend or {}).get("other") or []
    if isinstance(other, str):
        other = [other]
    for item in other:
        text = _normalized(item)
        if not text:
            continue
        head = text.split(_ENTRY_SEPARATOR, 1)[0] if _ENTRY_SEPARATOR in text else text
        topics.add(head.strip().lower())
    return topics


def new_legend_entries(
    facts: Iterable[dict[str, Any]],
    legend: dict[str, Any] | None,
) -> list[str]:
    """Render the facts worth keeping as ``topic: value`` legend entries.

    Drops anything protected, anything empty, and anything about a topic the
    legend already answers — the same first-established-wins rule the stable
    keys have always used, applied to the freeform ones.
    """
    known = existing_topics(legend)
    entries: list[str] = []
    for fact in facts or []:
        if not isinstance(fact, dict):
            continue
        topic = _normalized(fact.get("topic")).lower()[:_MAX_TOPIC_CHARS]
        value = _normalized(fact.get("value"))[:_MAX_VALUE_CHARS]
        if not topic or not value:
            continue
        if is_protected_topic(topic):
            continue
        if topic in known:
            continue
        known.add(topic)
        entries.append(f"{topic}{_ENTRY_SEPARATOR}{value}")
        if len(entries) >= _MAX_FACTS_PER_TURN:
            break
    return entries


def _parse_facts(text: str) -> list[dict[str, Any]]:
    cleaned = "\n".join(
        line for line in str(text or "").splitlines()
        if not line.lstrip().startswith("```")
    ).strip()
    if not cleaned:
        return []
    payload = json.loads(cleaned)
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    facts = payload.get("facts")
    return [item for item in (facts or []) if isinstance(item, dict)]


def _transcript(history: Sequence[Any], *, limit: int = 6) -> str:
    lines: list[str] = []
    for message in list(history or [])[-limit:]:
        role = getattr(message, "role", None)
        if role is None and isinstance(message, dict):
            role = message.get("role")
        content = getattr(message, "content", None)
        if content is None and isinstance(message, dict):
            content = message.get("content")
        text = _normalized(content)
        if text:
            lines.append(f"{'Fan' if role == 'fan' else 'Creator'}: {text}")
    return "\n".join(lines)


async def persist_sent_creator_facts(
    *,
    creator_id: str,
    sent_reply: str,
    fan_message: str = "",
    conversation_history: Sequence[Any] | None = None,
    fan_id: str | None = None,
    profile_id: str | None = None,
) -> list[str]:
    """Record ordinary self-facts from a reply that was actually sent.

    Returns the entries added to the legend, which is the empty list on every
    path that decides there is nothing to record — including every failure. This
    runs after delivery, so it must never raise into the caller: a legend that
    missed a favourite colour is a worse conversation later, while an exception
    here would be a broken send now.
    """
    reply = _normalized(sent_reply)
    if not creator_id or not reply:
        return []
    if not mentions_self_fact(reply=reply, fan_message=fan_message):
        return []

    try:
        profile = get_profile(profile_id)
        spec = profile.stages.get(STAGE_FAN_INTELLIGENCE)
        if spec is None:  # pragma: no cover - every profile defines the stage
            return []

        context = _transcript(conversation_history or [])
        user_prompt = (
            "RECENT CONTEXT (reference only, never a source of facts):\n"
            + (context or "[none]")
            + "\n\nTHE FAN'S LAST MESSAGE:\n"
            + (_normalized(fan_message) or "[none]")
            + "\n\nTHE MESSAGE THE CREATOR JUST SENT (the only source of facts):\n"
            + reply
        )

        result = await complete(
            spec.primary_target(),
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=spec.resolved_max_tokens(),
            temperature=spec.temperature if spec.temperature is not None else 0.0,
        )
        telemetry = ModelTelemetryContext(
            feature="creator_canon_extraction",
            creator_id=creator_id,
            fan_id=fan_id,
            metadata={"ai_stack_profile": profile.profile_id},
        )
        try:
            facts = _parse_facts(result.text)
        except Exception as exc:
            await record_model_result(
                result,
                telemetry,
                success=False,
                parse_valid=False,
                error=f"invalid creator canon JSON: {exc}",
            )
            print(f"[CREATOR CANON] invalid extraction creator={creator_id}: {exc}")
            return []

        await record_model_result(result, telemetry, success=True, parse_valid=True)

        legend = await get_creator_legend(creator_id)
        entries = new_legend_entries(facts, legend)
        if not entries:
            return []

        await update_creator_legend(creator_id, {"other": entries})
        print(
            f"[CREATOR CANON] creator={creator_id} established={entries}"
        )
        return entries
    except Exception as exc:
        print(f"[CREATOR CANON ERROR] creator={creator_id} error={exc}")
        return []
