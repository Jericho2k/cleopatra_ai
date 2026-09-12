"""What media Cleopatra may actually promise on this turn.

The writer never learns what is in the vault by reading a prompt about content
strategy. It learns it here, from the approved rows that were actually loaded,
and it is held to it deterministically afterwards.

Why this exists
---------------
Full Auto steered a conversation toward video for a creator whose approved vault
contained photo sets and nothing else. Nothing in the pipeline had told the
writer that. ``session_planner`` and ``media_packages`` are already correct — a
video finale is appended only when real ``video_rows`` exist — but correctness in
the *planner* does not reach the *writer*: the model saw the fan asking for a
clip, a generic "escalate the tension" objective, and an approved experience
description whose text can legitimately contain the word "video" (a photo set
shot on the same day as a clip, a tag, an album title). From there, "wait till
you see the video" is a perfectly reasonable next token and a promise the
creator cannot keep.

So this module does two things, in the same shape as ``services.ppv_language``:

1. It states the inventory. ``build_media_inventory`` turns the authoritative
   rows/packages/session the turn already loaded into a small, writer-safe
   capability record, which the prompt renders verbatim. The model is never
   asked to infer what exists.

2. It enforces the statement. ``sanitize_media_promises`` is the deterministic
   backstop for turns that offer or deliver content: a promise of a media type
   that is not in the authorised inventory is repaired into the type that IS
   available, or removed. Merely *talking* about video ("do you watch a lot of
   videos?") is untouched — only a first-person offer of creator inventory is.

Nothing here decides commerce. It cannot add inventory, change a price, or
select a package; it can only narrow what may be said to what was approved.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

# The two asset types the commercial layer actually produces. ``media_packages``
# emits exactly these strings on PackageOption.asset_types and session steps.
ASSET_PHOTO_SET = "photo_set"
ASSET_VIDEO = "video"

# Turns that offer or deliver Cleopatra-controlled content. Same list as
# ``services.ppv_language.DELIVERY_ACTIONS`` plus the teaser/offer transitions,
# because a promise made while teasing is still a promise.
COMMERCIAL_ACTIONS = frozenset(
    {
        "PRESENT_SESSION_OPTIONS",
        "END_TEASER_AND_OFFER",
        "CREATE_PAID_SESSION",
        "SEND_NEXT_PPV_STEP",
        "RESUME_PREVIOUS_OFFER",
        "PAYDAY_REENGAGEMENT",
        "START_FREE_TEASER",
        "CONTINUE_FREE_TEXT",
    }
)

_HUMAN_ASSET_NAMES = {
    ASSET_VIDEO: "video",
    ASSET_PHOTO_SET: "photo set",
}

_VIDEO_NOUN = r"(?:video|videos|vid|vids|clip|clips|movie|movies|footage|recording)"
_PHOTO_NOUN = r"(?:photo|photos|pic|pics|picture|pictures|set|sets|album|shoot)"


def human_asset_name(asset_type: str, *, plural: bool = False) -> str:
    """A customer-facing noun for one internal asset type."""
    name = _HUMAN_ASSET_NAMES.get(str(asset_type or ""), str(asset_type or "content"))
    if not plural:
        return name
    return f"{name}s" if not name.endswith("s") else name


def is_video_asset_type(value: Any) -> bool:
    return str(value or "").strip().lower() == ASSET_VIDEO


def _normalize_types(values: Iterable[Any]) -> tuple[str, ...]:
    """Deduplicate asset types, photo-first, so rendering is deterministic."""
    seen: list[str] = []
    for value in values:
        text = str(value or "").strip().lower()
        if not text:
            continue
        text = ASSET_VIDEO if text in {"video", "videos", "clip"} else text
        if text not in seen:
            seen.append(text)
    order = {ASSET_PHOTO_SET: 0, ASSET_VIDEO: 1}
    return tuple(sorted(seen, key=lambda item: (order.get(item, 2), item)))


def asset_types_from_rows(rows: Iterable[dict[str, Any]]) -> tuple[str, ...]:
    """Asset types present in a list of approved vault-set rows."""
    from services.media_packages import is_video_row

    return _normalize_types(
        ASSET_VIDEO if is_video_row(row) else ASSET_PHOTO_SET for row in rows or []
    )


def asset_types_from_packages(packages: Iterable[Any]) -> tuple[str, ...]:
    """Asset types across offer options (PackageOption or its dict form)."""
    values: list[str] = []
    for package in packages or []:
        if isinstance(package, dict):
            values.extend(package.get("asset_types") or [])
        else:
            values.extend(getattr(package, "asset_types", None) or [])
    return _normalize_types(values)


def asset_types_from_session(session: dict[str, Any] | None) -> tuple[str, ...]:
    """Asset types in the steps of an active/paused session plan."""
    plan = (session or {}).get("plan") or []
    return _normalize_types(step.get("asset_type") for step in plan)


def next_step_asset_type(session: dict[str, Any] | None) -> str | None:
    """The asset type of the next unsent step of an active session, if any."""
    if not session:
        return None
    plan = session.get("plan") or []
    try:
        index = int(session.get("current_index", 0) or 0)
    except (TypeError, ValueError):
        index = 0
    for step in plan[max(0, index):]:
        if not step.get("sent"):
            value = str(step.get("asset_type") or "").strip().lower()
            return value or None
    return None


@dataclass(frozen=True)
class MediaInventory:
    """The authoritative statement of what may be promised on this turn.

    ``authorized_asset_types`` is the narrow list: what the currently authorised
    package, selected offer or active session actually contains. It is what the
    writer may promise. ``available_package_asset_types`` is what could still be
    offered from the current options, and ``vault_asset_types`` what exists in
    approved, unsent inventory at all — both are context for an honest pivot,
    never a licence to promise.
    """

    authorized_asset_types: tuple[str, ...] = ()
    available_package_asset_types: tuple[str, ...] = ()
    vault_asset_types: tuple[str, ...] = ()
    next_step_asset_type: str | None = None
    video_requested: bool = False
    known: bool = True
    reason_codes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def promisable_asset_types(self) -> tuple[str, ...]:
        """Every type this turn is allowed to name as creator inventory.

        The authorised package first; the wider current offer set counts too,
        because presenting two options is itself an authorised promise of both.
        """
        return _normalize_types(
            [*self.authorized_asset_types, *self.available_package_asset_types]
        )

    @property
    def may_promise_video(self) -> bool:
        return ASSET_VIDEO in self.promisable_asset_types

    @property
    def video_exists_in_vault(self) -> bool:
        return ASSET_VIDEO in self.vault_asset_types

    @property
    def video_requested_but_unavailable(self) -> bool:
        return bool(self.video_requested) and not self.may_promise_video

    def to_context(self) -> dict[str, Any]:
        """The writer-safe capability record. No cents, no ids, no row data."""
        return {
            "authorized_asset_types": list(self.authorized_asset_types),
            "available_package_asset_types": list(self.available_package_asset_types),
            "vault_asset_types": list(self.vault_asset_types),
            "next_step_asset_type": self.next_step_asset_type,
            "promisable_asset_types": list(self.promisable_asset_types),
            "may_promise_video": self.may_promise_video,
            "video_requested": bool(self.video_requested),
            "video_requested_but_unavailable": self.video_requested_but_unavailable,
            "known": bool(self.known),
            "reason_codes": list(self.reason_codes),
        }


# An inventory we could not establish. Deliberately NOT "everything is allowed":
# an unknown inventory promises nothing, which is the same posture the rest of
# the commercial layer takes when it cannot prove a fact.
UNKNOWN_INVENTORY = MediaInventory(known=False, reason_codes=("inventory_unknown",))


def build_media_inventory(
    *,
    decision: Any = None,
    package_options: Sequence[Any] | None = None,
    active_session: dict[str, Any] | None = None,
    approved_rows: Sequence[dict[str, Any]] | None = None,
    desired_experience: str | None = None,
    fan_message: str | None = None,
) -> MediaInventory:
    """Derive the turn's authoritative media capabilities.

    Every argument is something the caller already loaded; nothing here reads the
    database, so the inventory can never disagree with the rows the planner used.
    """
    from services.media_packages import wants_video

    reason_codes: list[str] = []

    decision_options: list[Any] = []
    decision_action = ""
    if decision is not None:
        if isinstance(decision, dict):
            decision_action = str(decision.get("action") or "")
            decision_options = list(decision.get("package_options") or [])
        else:
            action = getattr(decision, "action", None)
            decision_action = str(getattr(action, "value", action) or "")
            decision_options = list(getattr(decision, "package_options", None) or [])

    session_types = asset_types_from_session(active_session)
    decision_types = asset_types_from_packages(decision_options)
    option_types = asset_types_from_packages(package_options or [])
    vault_types = asset_types_from_rows(approved_rows or [])

    # An active plan is the narrowest and most authoritative statement there is:
    # it names the exact steps that will be delivered.
    if session_types:
        authorized = session_types
        reason_codes.append("authorized_from_active_session")
    elif decision_types:
        authorized = decision_types
        reason_codes.append("authorized_from_commercial_decision")
    elif option_types:
        authorized = option_types
        reason_codes.append("authorized_from_offer_options")
    else:
        authorized = ()
        reason_codes.append("no_authorized_media")

    available = option_types or decision_types
    if not available and session_types:
        available = session_types

    requested = bool(
        wants_video(desired_experience) or wants_video(fan_message)
    )
    if requested:
        reason_codes.append("fan_requested_video")

    inventory = MediaInventory(
        authorized_asset_types=authorized,
        available_package_asset_types=available,
        vault_asset_types=vault_types,
        next_step_asset_type=next_step_asset_type(active_session),
        video_requested=requested,
        known=True,
        reason_codes=tuple(reason_codes),
    )
    if inventory.video_requested_but_unavailable:
        object.__setattr__(
            inventory,
            "reason_codes",
            (*inventory.reason_codes, "video_requested_but_unavailable"),
        )
    if decision_action:
        object.__setattr__(
            inventory,
            "reason_codes",
            (*inventory.reason_codes, f"action:{decision_action}"),
        )
    return inventory


# ---------------------------------------------------------------------------
# The deterministic backstop
# ---------------------------------------------------------------------------
#
# Each rule is (pattern, replacement-template). ``{noun}`` is filled with the
# singular noun for an authorised type, ``{nouns}`` with its plural. When no
# media type is authorised at all the whole match is deleted and the sentence is
# tidied, because there is nothing honest to substitute.
#
# The rules only fire on a FIRST-PERSON offer or promise of creator inventory,
# or a second-person tease about one. "did you see that video on twitter" and
# "do you prefer videos or pics" are ordinary conversation and are left alone.

_PROMISE_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    # "I have a video", "I've got some clips", "I do have videos"
    (
        re.compile(
            r"\bi(?:'ve|\s+have|\s+ve|\s+got|'ll\s+have)?\s*(?:do\s+)?"
            r"(?:have|got|made|shot|filmed|recorded)\s+"
            r"(?:a|an|this|some|a\s+few|another|one|new|my)?\s*"
            r"(?:new\s+|hot\s+|little\s+|short\s+|naughty\s+)*" + _VIDEO_NOUN + r"\b",
            re.IGNORECASE,
        ),
        "i have a {noun}",
    ),
    # "wait till you see the video", "just wait until you see my clip"
    (
        re.compile(
            r"\bwait\s+(?:un)?til+\s+(?:you|u)\s+see\s+"
            r"(?:the|my|this|that)?\s*"
            r"(?:new\s+|last\s+|next\s+)*" + _VIDEO_NOUN + r"\b",
            re.IGNORECASE,
        ),
        "wait till you see the {noun}",
    ),
    # "I'll send you a clip", "lemme send the video", "i can send u a vid"
    (
        re.compile(
            r"\b(?:i(?:'|\s+a)?ll|i\s+will|i\s+can|i\s+could|lemme|let\s+me|"
            r"gonna|i'?m\s+gonna|i\s+wanna|i\s+want\s+to)\s+"
            r"(?:send|drop|shoot|film|record|make|show)\s+"
            r"(?:you|u|ya)?\s*(?:a|an|the|my|this|some|another)?\s*"
            r"(?:new\s+|hot\s+|little\s+|short\s+|naughty\s+)*" + _VIDEO_NOUN + r"\b",
            re.IGNORECASE,
        ),
        "i'll send you a {noun}",
    ),
    # "sending you the video", "dropping a clip"
    (
        re.compile(
            r"\b(?:sending|dropping|filming|recording|making)\s+"
            r"(?:you|u|ya)?\s*(?:a|an|the|my|this|some)?\s*"
            r"(?:new\s+|hot\s+|little\s+|short\s+)*" + _VIDEO_NOUN + r"\b",
            re.IGNORECASE,
        ),
        "sending you a {noun}",
    ),
    # "the video gets better", "my clip is worth it", "this vid is insane"
    (
        re.compile(
            r"\b(?:the|my|this|that)\s+"
            r"(?:new\s+|next\s+|last\s+|second\s+|other\s+)*" + _VIDEO_NOUN
            + r"\s+(?:is|was|gets|goes|hits|will)\b",
            re.IGNORECASE,
        ),
        "the {noun} is",
    ),
    # "want the video?", "want a clip?", "do you want my vid"
    (
        re.compile(
            r"\b(?:do\s+)?(?:you\s+|u\s+)?wan+(?:t|na)\s+"
            r"(?:to\s+see\s+)?(?:the|a|an|my|this|some)\s*"
            r"(?:new\s+|hot\s+|little\s+)*" + _VIDEO_NOUN + r"\b",
            re.IGNORECASE,
        ),
        "want the {noun}",
    ),
    # "there's a video waiting", "i got something on video for you"
    (
        re.compile(
            r"\bthere(?:'s|\s+is|\s+are)\s+"
            r"(?:a|an|some|this|my)?\s*"
            r"(?:new\s+|hot\s+|little\s+)*" + _VIDEO_NOUN + r"\b",
            re.IGNORECASE,
        ),
        "there's a {noun}",
    ),
    # "on video", "caught it on camera" — only as an offer of what she has.
    (
        re.compile(
            r"\b(?:it'?s|that'?s|this\s+is|i\s+got\s+it)\s+on\s+"
            r"(?:video|camera|film)\b",
            re.IGNORECASE,
        ),
        "i have a {noun}",
    ),
)

# A sentence that still names creator video after repair is dropped whole.
_RESIDUAL_PROMISE_RE = re.compile(
    r"\b(?:my|the|a|an|this|that|another|some)\s+"
    r"(?:new\s+|hot\s+|little\s+|short\s+|next\s+|last\s+)*" + _VIDEO_NOUN + r"\b",
    re.IGNORECASE,
)

_PPV_TAG_RE = re.compile(r"\[PPV:[^\]]+\]", re.IGNORECASE)


def is_commercial_turn(
    *,
    decision_action: str | None = None,
    reply: str | None = None,
    active_session: dict[str, Any] | None = None,
) -> bool:
    """True when this turn offers or delivers Cleopatra-controlled content."""
    if str(decision_action or "").upper() in COMMERCIAL_ACTIONS:
        return True
    if reply and _PPV_TAG_RE.search(reply):
        return True
    return bool(active_session and active_session.get("status") in {"active", "paused"})


def _substitute_noun(inventory: MediaInventory) -> str | None:
    """The honest replacement noun, or None when nothing may be promised."""
    for asset_type in inventory.promisable_asset_types:
        if asset_type != ASSET_VIDEO:
            return human_asset_name(asset_type)
    return None


def _tidy(text: str) -> str:
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\s+([,.!?])", r"\1", text)
    text = re.sub(r"([,;])\s*([.!?])", r"\2", text)
    text = re.sub(r"\|\s*\|", "|", text)
    text = re.sub(r"(?:^|\s)\|\s*$", "", text)
    text = re.sub(r"^\s*\|\s*", "", text)
    return text.strip(" \t,;")


def _drop_clauses_naming_video(text: str) -> str:
    """Remove only the fragments that still promise video, keeping the reply."""
    bubbles_out: list[str] = []
    for bubble in str(text or "").split("|"):
        kept: list[str] = []
        for sentence in re.split(r"(?<=[.!?])\s+", bubble):
            if _RESIDUAL_PROMISE_RE.search(sentence):
                continue
            kept.append(sentence)
        joined = _tidy(" ".join(part for part in kept if part.strip()))
        if joined:
            bubbles_out.append(joined)
    return " | ".join(bubbles_out)


def promises_unavailable_media(text: str, inventory: MediaInventory) -> bool:
    """True when the copy offers creator video that is not authorised."""
    if inventory.may_promise_video:
        return False
    body = str(text or "")
    if any(pattern.search(body) for pattern, _ in _PROMISE_RULES):
        return True
    return bool(_RESIDUAL_PROMISE_RE.search(body))


def repair_media_promises(text: str, inventory: MediaInventory) -> str:
    """Rewrite promises of unavailable media into what is actually authorised.

    Deterministic and idempotent. When an authorised photo type exists the video
    noun becomes that type; when nothing is authorised the promise is removed
    rather than softened, because there is no honest version of it.
    """
    body = str(text or "")
    if inventory.may_promise_video:
        return body

    noun = _substitute_noun(inventory)
    if noun is None:
        return _drop_clauses_naming_video(body)

    nouns = human_asset_name(
        next(
            (item for item in inventory.promisable_asset_types if item != ASSET_VIDEO),
            ASSET_PHOTO_SET,
        ),
        plural=True,
    )
    repaired = body
    for pattern, template in _PROMISE_RULES:
        repaired = pattern.sub(
            template.format(noun=noun, nouns=nouns).replace("\\", ""), repaired
        )
    repaired = _tidy(repaired)
    if _RESIDUAL_PROMISE_RE.search(repaired):
        repaired = _drop_clauses_naming_video(repaired)
    return repaired


def sanitize_media_promises(
    text: str,
    inventory: MediaInventory | None,
    *,
    decision_action: str | None = None,
    active_session: dict[str, Any] | None = None,
) -> tuple[str, bool]:
    """Return ``(text, repaired)`` for one outgoing reply.

    Non-commercial turns are returned untouched, so ordinary talk about videos
    the fan brings up is never rewritten.
    """
    original = str(text or "")
    if inventory is None or not inventory.known:
        return original, False
    if inventory.may_promise_video:
        return original, False
    if not is_commercial_turn(
        decision_action=decision_action,
        reply=original,
        active_session=active_session,
    ):
        return original, False
    if not promises_unavailable_media(original, inventory):
        return original, False
    return repair_media_promises(original, inventory), True


def choose_inventory_safe_reply(
    candidates: Sequence[str],
    inventory: MediaInventory | None,
    *,
    decision_action: str | None = None,
    active_session: dict[str, Any] | None = None,
) -> tuple[str | None, bool]:
    """Pick the first candidate that keeps the inventory invariant.

    Mirrors how the PPV-link guard is used, one level up: prefer a candidate the
    writer produced cleanly, fall back to a repaired one, and only report failure
    when every candidate collapses to nothing after repair. Returns
    ``(reply_or_None, repaired)``.
    """
    repaired_fallback: str | None = None
    for candidate in candidates or []:
        text, repaired = sanitize_media_promises(
            candidate,
            inventory,
            decision_action=decision_action,
            active_session=active_session,
        )
        if not repaired:
            return text, False
        if text.strip() and not promises_unavailable_media(text, inventory or UNKNOWN_INVENTORY):
            if repaired_fallback is None:
                repaired_fallback = text
    if repaired_fallback is not None:
        return repaired_fallback, True
    return None, True


def render_inventory_block(inventory: MediaInventory | None) -> str:
    """The writer-facing statement of what exists. Never infer, always state."""
    if inventory is None or not inventory.known:
        return ""

    promisable = inventory.promisable_asset_types
    lines: list[str] = []
    if promisable:
        lines.append(
            "content you may offer, promise or describe as yours this turn: "
            + ", ".join(human_asset_name(item, plural=True) for item in promisable)
        )
    else:
        lines.append(
            "you have NOTHING authorised to offer this turn: do not promise, "
            "tease or describe any content of your own as available."
        )
    if inventory.next_step_asset_type:
        lines.append(
            "the next planned piece is a "
            + human_asset_name(inventory.next_step_asset_type)
            + "."
        )
    if not inventory.may_promise_video:
        lines.append(
            "you have NO video available. Never say or imply you have a video, "
            "a clip, footage, or a recording, never say one is coming, and never "
            "hint that anything gets better on video. This holds even if he asks "
            "for one directly."
        )
        if inventory.video_requested:
            lines.append(
                "he just asked for video and there is none. Do not stall, do not "
                "apologise at length, and do not explain inventory, systems or "
                "approvals. Respond warmly in your own voice and move him onto "
                "what you do have, describing it honestly."
            )
    else:
        lines.append(
            "video is authorised this turn, so it may be offered as planned."
        )
    lines.append(
        "This list is the complete truth about what exists. Do not infer more "
        "from tags, titles, descriptions, or from anything he says."
    )
    return "CONTENT INVENTORY (authoritative — overrides any other content hint):\n- " + "\n- ".join(lines)
