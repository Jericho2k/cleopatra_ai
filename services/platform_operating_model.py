"""One product-world contract shared by the conversational owner and writer.

Cleopatra's transaction boundaries were already deterministic, but the facts
that explain *what product those boundaries describe* lived in several prompt
paragraphs and validators.  This module keeps the concise operating model,
application-owned platform context, action meanings, and narrow language
backstops together so GLM, Kimi, and deterministic validation cannot drift into
different worlds.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import Any

PLATFORM_OPERATING_MODEL = """CLEOPATRA ENVIRONMENT
- You operate one creator's private conversation with one fan on a paid creator platform. The fan is already inside this private chat; free text and paid-media offers happen here, and there is no second DM room to find.
- The creator may have application-approved private media inventory. Paid media can be offered and sent as a locked message in this same conversation.
- Application code — never either language model — owns inventory, recipient, price, payment, purchase, permissions, attachment, delivery, persistence, and operation results.
- Narration is not an operation: saying "I'll show you", "here goes", "just sent it", or "look at this" cannot make media appear. Only approved prepared delivery facts mean media is attached.
- You cannot see the fan. You do not know what the creator is physically doing now unless sourced evidence says so. Vault media is neither present activity nor proof of public-feed content.
- Shared imagined roleplay is conversation, not real-world evidence. Keep imagined actions inside an explicit hypothetical/shared-imagined frame; never silently turn them into current facts."""


CLEOPATRA_MISSION = """CLEOPATRA MISSION
Operate the creator's private inbox as an excellent long-term chatter: sustain an engaging, believable, context-aware relationship and, when genuine fan interest supports it, naturally convert that interest into authorized paid media. Conversation quality, retention, and monetization are compatible. Do not hard-sell, infer wealth, or treat sexual explicitness as a sale; also do not ignore a direct media or buying request merely to avoid selling. GLM owns the semantic business judgment. Kimi owns all fan-facing words. Application code owns authority and execution."""


_ACTION_EFFECTS = {
    "none": (
        "Continue the text conversation only. No offer, attachment, payment "
        "check, access repair, handoff, or other external product action occurs."
    ),
    "present_offer": (
        "Present the currently authorized private paid content naturally in "
        "this chat. This does not purchase, attach, unlock, or deliver media; "
        "it only makes the offer available."
    ),
    "send_locked_paid_message": (
        "Actually attach and send the exact authorized paid media behind the "
        "platform paywall in this chat. Use only when deterministic state says "
        "the exact offer can be sent; this is real same-chat media delivery."
    ),
    "check_payment_claim": (
        "Ask application authority to verify the fan's claim that they paid or "
        "unlocked. The fan's words are not purchase truth."
    ),
    "repair_content_access": (
        "Support an existing confirmed purchase the fan cannot access. This is "
        "never a new sale, and requesting repair does not prove its result."
    ),
    "hand_off_to_human": (
        "Request human attention for a genuine unresolved authority, support, "
        "or safety situation; ordinary wording mistakes are repaired locally."
    ),
}


SCHEDULED_INTENT_EFFECT = (
    "Create a future semantic obligation or callback, never prewritten copy. "
    "The application reloads current state and re-runs judgment when it is due."
)


def platform_context() -> dict[str, Any]:
    """Application-owned environment facts safe for either model to receive."""
    return {
        "surface": "private_creator_fan_chat",
        "fan_is_already_in_private_chat": True,
        "free_text_surface": "this_chat",
        "paid_media_delivery_surface": "this_chat",
        "paid_media_delivery_mechanism": "locked_message_attachment",
        "can_see_fan": False,
        "creator_current_activity_known": False,
        "narration_causes_product_action": False,
        "roleplay_is_real_world_evidence": False,
    }


def operation_affordances(legal_operations: Iterable[str]) -> dict[str, dict[str, Any]]:
    """Describe product effects as well as the state-legal choices.

    All live operation meanings are shown so an illegal action is understood as
    a real capability that is unavailable *now*, rather than an unknown enum.
    Exact inventory handles remain in the evidence snapshot.
    """
    legal = {str(value) for value in legal_operations}
    return {
        name: {"legal": name in legal, "effect": effect}
        for name, effect in _ACTION_EFFECTS.items()
    }


def prepared_execution_reality(prepared: dict[str, Any]) -> dict[str, Any]:
    """What Kimi may truthfully imply happened after backend authority ruled."""
    operation = str(prepared.get("operation") or "none")
    delivery = prepared.get("delivery") or None
    approved_delivery = bool(
        operation == "send_locked_paid_message"
        and delivery
        and not prepared.get("approval_required")
    )
    if approved_delivery:
        effect = (
            "The authorized paid media is attached to this same outgoing locked "
            "message. Kimi may describe this delivery without inventing details."
        )
    elif operation == "present_offer":
        effect = (
            "An authorized offer is being presented in this message. No media "
            "has been delivered, purchased, or unlocked."
        )
    elif operation == "check_payment_claim":
        effect = "A payment claim is being checked. Payment and access are not yet confirmed."
    elif operation == "repair_content_access":
        effect = (
            "Support for an existing purchase is being requested. Do not invent "
            "the repair result or a new sale."
        )
    elif operation == "hand_off_to_human":
        effect = "Human attention is requested. Do not invent what the human will do."
    elif operation == "send_locked_paid_message":
        effect = (
            "No approved delivery is prepared for this outgoing message. Do not "
            "imply that media was attached, sent, unlocked, or made visible."
        )
    else:
        effect = (
            "Text conversation only. Nothing external was offered, attached, "
            "sent, purchased, unlocked, repaired, or handed off."
        )
    return {
        "approved_operation": operation,
        "external_product_action_happened": bool(
            operation != "none"
            and (operation != "send_locked_paid_message" or approved_delivery)
        ),
        "new_media_attached_to_this_message": approved_delivery,
        "effect": effect,
        "rule": "Write from approved execution, never from a rejected proposal.",
    }


_OTHER_CHAT_PATTERNS = (
    re.compile(
        r"\b(?:come|go|head|move|hop|switch)(?:\s+over)?\s+(?:find\s+me\s+)?"
        r"(?:to|into|in|on)\s+(?:my|the)?\s*(?:dms?|direct\s+messages?|private\s+messages?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bfind\s+me\s+(?:in|on)\s+(?:my|the)?\s*"
        r"(?:dms?|direct\s+messages?|private\s+messages?|messages?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:message|text|dm)\s+me\s+(?:in|on|through|via)\s+"
        r"(?:my|the)?\s*(?:dms?|direct\s+messages?|private\s+messages?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:message|text|dm)\s+me\s+(?:somewhere|someplace)\s+else\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:dm|message|text)\s+me\s+(?:for|to\s+(?:get|see|receive|unlock))\s+"
        r"(?:(?:the|a|your|that|this|those|these|my)\s+)?"
        r"(?:links?|photos?|pics?|videos?|content|media|set|access)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:send|sent|sending)\s+(?:you\s+)?(?:the|a|that|this)\s+link\b",
        re.IGNORECASE,
    ),
)


def redirects_to_other_chat(text: str) -> bool:
    """Whether creator copy sends an already-present fan to another chat."""
    value = str(text or "")
    return any(pattern.search(value) for pattern in _OTHER_CHAT_PATTERNS)


_IMAGINED_SCOPE = re.compile(
    r"\b(?:imagine(?:\s+(?:me|us))?|picture\s+(?:me|us)|pretend|"
    r"if\s+(?:you|u)\s+(?:were|was)\s+here|if\s+we\s+were\s+together|"
    r"i(?:['’]d|\s+would|\s+could|\s+might)|"
    r"you(?:['’]d|\s+would)|we(?:['’]d|\s+would)|"
    r"in\s+(?:our|this|that)\s+(?:scene|fantasy))\b",
    re.IGNORECASE,
)


_PRESENT_CREATOR_ACTIONS = (
    re.compile(
        r"\b(?:i(?:['’]?m|\s+am)\s+)?(?:already|right\s+now|now|just)\s+"
        r"(?:unclipping|unhooking|undoing|unzipping|unbuttoning|taking|pulling|"
        r"sliding|peeling|dropping|removing|undressing|stripping)\b[^.!?\n|]*",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bi(?:['’]?m|\s+am)\s+(?:unclipping|unhooking|undoing|unzipping|"
        r"unbuttoning|taking|pulling|sliding|peeling|dropping|removing|"
        r"undressing|stripping)\b[^.!?\n|]*",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:taking|pulling|sliding|peeling|dropping)\s+"
        r"(?:it|this|that|them|my\s+\w+)\s+(?:off|down|up)(?:\s+now)?\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:i\s+)?(?:just\s+)?got\s+(?:fully\s+)?undressed\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:my\s+)?(?:straps?|top|bra|dress|shirt|skirt|shorts|underwear|"
        r"panties|thong)\s+(?:are\s+|is\s+)?(?:off|down|open|undone)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:the\s+rest|it|this|that|my\s+\w+)\s+"
        r"(?:(?:is|['’]s)\s+)?coming\s+off(?:\s+(?:slow|slowly|now))?\b",
        re.IGNORECASE,
    ),
)


def _normalized(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.lower()).split())


def unsupported_present_creator_action(
    text: str,
    *,
    creator_facts: Sequence[Any] = (),
) -> bool:
    """Detect a narrow unsupported assertion of current creator activity.

    Explicit hypothetical/imagined scope is always conversationally legal. A
    factual form is legal only when a sourced creator fact supports the matched
    action. ``scene_mode`` is intentionally not an exemption: a shared scene
    does not turn an unscoped present-tense assertion into a real-world fact.
    """
    value = str(text or "")
    match = next(
        (
            found
            for pattern in _PRESENT_CREATOR_ACTIONS
            if (found := pattern.search(value))
        ),
        None,
    )
    if match is None:
        return False
    clause_start = max(
        value.rfind(mark, 0, match.start()) for mark in (".", "!", "?", "\n", "|")
    )
    clause_end_candidates = [
        position
        for mark in (".", "!", "?", "\n", "|")
        if (position := value.find(mark, match.end())) >= 0
    ]
    clause_end = min(clause_end_candidates) if clause_end_candidates else len(value)
    clause = value[clause_start + 1 : clause_end]
    if _IMAGINED_SCOPE.search(clause):
        return False
    claim = _normalized(match.group(0))
    for fact in creator_facts:
        source = str(getattr(fact, "source_ref", "") or "")
        fact_value = _normalized(str(getattr(fact, "value", "") or ""))
        if source and fact_value and (claim in fact_value or fact_value in claim):
            return False
    return True


def repair_present_action_as_imagined(text: str) -> str:
    """Rewrite the narrow intimate-action forms into explicit imagination."""
    value = str(text or "")
    replacements = (
        (
            re.compile(
                r"\b(?:i(?:['’]?m|\s+am)\s+)?already\s+unclipping\b", re.IGNORECASE
            ),
            "imagine me slowly unclipping",
        ),
        (
            re.compile(
                r"\b(?:i(?:['’]?m|\s+am)\s+)?taking\s+it\s+off(?:\s+now)?\b",
                re.IGNORECASE,
            ),
            "imagine me taking it off",
        ),
        (
            re.compile(
                r"\b(?:i(?:['’]?m|\s+am)\s+)?pulling\s+it\s+down(?:\s+now)?\b",
                re.IGNORECASE,
            ),
            "imagine me pulling it down slowly",
        ),
        (
            re.compile(r"\b(?:my\s+)?straps?\s+(?:are\s+|is\s+)?off\b", re.IGNORECASE),
            "imagine the straps coming off",
        ),
        (
            re.compile(
                r"\bthe\s+rest\s+(?:(?:is|['’]s)\s+)?coming\s+off(?:\s+slow)?\b",
                re.IGNORECASE,
            ),
            "imagine the rest coming off slowly",
        ),
        (
            re.compile(
                r"\b(?:i\s+)?just\s+got\s+(?:fully\s+)?undressed\b", re.IGNORECASE
            ),
            "imagine me undressed",
        ),
    )
    for pattern, replacement in replacements:
        value, count = pattern.subn(replacement, value, count=1)
        if count:
            return value
    return value


_MEDIA_COMPLETION_CLAIM = re.compile(
    r"\b(?:(?:i\s+)?(?:just\s+)?(?:sent|delivered|attached|dropped|shared|uploaded)"
    r"\s+(?:you\s+)?(?:it|this|that|them|something|(?:the|your|those|these)\s+"
    r"(?:photos?|pics?|videos?|content|media))|"
    r"(?:check|open|look\s+in)\s+(?:it|that|your\s+(?:inbox|messages))|"
    r"(?:it['’]?s|its|they['’]?re|it\s+is|they\s+are|should\s+be)\s+"
    r"(?:there|in\s+your\s+inbox|waiting\s+for\s+you)|"
    r"here(?:\s+it|\s+this|['’]?s\s+it)\s+(?:is|comes)|"
    r"look\s+at\s+(?:this|these))\b",
    re.IGNORECASE,
)


def claims_media_delivery(text: str) -> bool:
    """Whether copy implies new creator-controlled media is now visible."""
    return bool(_MEDIA_COMPLETION_CLAIM.search(str(text or "")))
