"""Whether SEXUAL TEXT is appropriate — which is not a commercial question.

THE BUG THIS MODULE IS THE FIX FOR
----------------------------------
``CommercialDecision.may_be_explicit`` defaults to ``False`` and the writer
prompt turned that into the literal sentence "Keep this response non-explicit."
Because a ``CONTINUE_NORMAL_CHAT`` decision carries that default, the practical
rule became:

    the commercial engine is not selling anything this turn
      -> the creator may not talk to him sexually

which is nonsense. A fan who arrives from TikTok already explicit, on a turn
where there is correctly nothing to sell, got a chaperone. The commercial layer
was answering a question it was never asked.

Two genuinely different questions, answered by two different modules:

    services/commercial_policy.py   may media/PPV be offered, priced, sent?
    this module                     may the TEXT be sexual, and how sexual?

Commercial policy stays absolute over price, media and delivery — nothing here
can offer, price, attach or promise anything, and it never widens what the
commercial decision authorised. It only decides the REGISTER of the words.

WHAT STILL CONSTRAINS IT
------------------------
Four things, in order of authority:

1. Safety. Crisis, human review and a frozen account are absolute: NONE.
2. Truthfulness about a locked PPV. While an unlock is awaiting payment the
   reply must not imply he opened it — but that is a constraint on CLAIMS, and
   saying so is the commercial decision's job. It does not gag the register.
3. The agency's configured sexting mode. PAID_ONLY still means explicit text is
   something he pays for; HYBRID_TEASER and FREE_TEXT_ALLOWED still spend a
   bounded allowance. This is the control the agency bought and it is preserved
   exactly — including, crucially, that explicit free text now CONSUMES that
   allowance (``consumes_free_allowance``). Without that, decoupling would have
   quietly created unlimited free sexting on ``CONTINUE_NORMAL_CHAT`` turns.
4. Him. The creator does not escalate into a conversation that is not there.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from models.commercial import CreatorPolicy, FanStatus, SextingMode


class TextIntimacy(str, Enum):
    #: No flirtation at all. Safety only.
    NONE = "NONE"
    #: Warm and flirty. Suggestive, not graphic.
    FLIRTY = "FLIRTY"
    #: Explicitly sexual language is in bounds.
    EXPLICIT = "EXPLICIT"


_ORDER = {TextIntimacy.NONE: 0, TextIntimacy.FLIRTY: 1, TextIntimacy.EXPLICIT: 2}


def _min(left: TextIntimacy, right: TextIntimacy) -> TextIntimacy:
    return left if _ORDER[left] <= _ORDER[right] else right


class TextIntimacyDecision(BaseModel):
    level: TextIntimacy = TextIntimacy.FLIRTY
    #: True when explicit sexual text is in bounds this turn.
    may_be_explicit: bool = False
    #: True when this turn spends one of the creator's configured free
    #: allowance messages. The orchestrator increments the counter; without
    #: this the agency's cap would silently stop applying.
    consumes_free_allowance: bool = False
    reason: str = ""
    reason_codes: list[str] = Field(default_factory=list)

    def to_context(self) -> dict[str, Any]:
        return {
            "level": self.level.value,
            "may_be_explicit": self.may_be_explicit,
            "consumes_free_allowance": self.consumes_free_allowance,
            "reason": self.reason,
            "reason_codes": list(self.reason_codes),
        }


#: Commercial actions during which a paid context is genuinely active, so
#: explicit text is part of the thing he bought rather than a free service.
_PAID_CONTEXT_ACTIONS = frozenset(
    {
        "SEND_NEXT_PPV_STEP",
        "OFFER_NEXT_UNLOCK",
        "RESUME_PREVIOUS_OFFER",
        "CONTINUE_FREE_TEXT",
        "START_FREE_TEASER",
    }
)

#: Commercial actions that are a hard stop on everything, register included.
_SAFETY_ACTIONS = frozenset({"HAND_OFF_TO_HUMAN"})

#: Beats in which he has paid for the thing being talked about. Reacting to it
#: in its own register is what he bought; making it chaste is the regression.
_PAID_SCENE_BEATS = frozenset({"AWAIT_REACTION", "PLAY", "BRIDGE"})


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def decide_text_intimacy(
    *,
    policy: CreatorPolicy,
    situation: dict[str, Any] | None = None,
    commercial_decision: dict[str, Any] | None = None,
    scene: dict[str, Any] | None = None,
    fan_status: FanStatus | str | None = None,
    teaser_messages_used: int = 0,
    frozen_for_review: bool = False,
    free_mode_on_cooldown: bool = False,
    now: datetime | None = None,
) -> TextIntimacyDecision:
    """How sexual this reply may be. Pure; nothing here writes or authorizes."""
    _ = now or datetime.now(timezone.utc)
    situation = situation or {}
    decision = commercial_decision or {}
    scene = scene or {}
    codes: list[str] = []

    action = str(decision.get("action") or "")
    if hasattr(decision.get("action"), "value"):
        action = str(decision["action"].value)

    # ---- 1. Safety is absolute -----------------------------------------
    crisis = str(situation.get("crisis_signal") or "none").lower()
    if crisis not in {"none", ""}:
        return TextIntimacyDecision(
            level=TextIntimacy.NONE,
            reason=f"crisis signal ({crisis}); nothing sexual",
            reason_codes=["crisis"],
        )
    if frozen_for_review or action in _SAFETY_ACTIONS:
        return TextIntimacyDecision(
            level=TextIntimacy.NONE,
            reason="handed to a human; nothing sexual",
            reason_codes=["human_review"],
        )

    # ---- 2. Is he actually there? --------------------------------------
    wants_explicit = _truthy(situation.get("wants_explicit"))
    intimacy_level = int(scene.get("intimacy_level") or 0)
    tension_level = int(scene.get("tension_level") or 0)
    beat = str(scene.get("beat") or "")

    fan_ceiling = TextIntimacy.FLIRTY
    if wants_explicit or intimacy_level >= 3:
        fan_ceiling = TextIntimacy.EXPLICIT
        codes.append("fan_intent_explicit" if wants_explicit else "scene_intimacy_high")
    elif intimacy_level >= 1 or tension_level >= 2:
        codes.append("conversation_has_heat")
    else:
        fan_ceiling = TextIntimacy.FLIRTY
        codes.append("no_sexual_intent_yet")

    # ---- 3. The agency's configured mode -------------------------------
    paid_context = action in _PAID_CONTEXT_ACTIONS or beat in _PAID_SCENE_BEATS
    mode_ceiling, consumes, mode_code = _mode_ceiling(
        policy=policy,
        paid_context=paid_context,
        teaser_messages_used=teaser_messages_used,
        free_mode_on_cooldown=free_mode_on_cooldown,
    )
    codes.append(mode_code)

    # ---- 4. A paused fan is not given the paid experience for free -----
    status = fan_status.value if isinstance(fan_status, FanStatus) else str(fan_status or "")
    if status in {FanStatus.PAUSED_NO_BUDGET.value, FanStatus.PAUSED_UNTIL_PAYDAY.value}:
        if policy.sexting_mode is not SextingMode.FREE_TEXT_ALLOWED or free_mode_on_cooldown:
            mode_ceiling = _min(mode_ceiling, TextIntimacy.FLIRTY)
            consumes = False
            codes.append("paused_no_free_paid_experience")

    level = _min(fan_ceiling, mode_ceiling)
    explicit = level is TextIntimacy.EXPLICIT
    return TextIntimacyDecision(
        level=level,
        may_be_explicit=explicit,
        consumes_free_allowance=bool(consumes and explicit and not paid_context),
        reason=_describe(level, codes),
        reason_codes=codes,
    )


def _mode_ceiling(
    *,
    policy: CreatorPolicy,
    paid_context: bool,
    teaser_messages_used: int,
    free_mode_on_cooldown: bool,
) -> tuple[TextIntimacy, bool, str]:
    """What the creator's configured sexting mode allows, and what it costs."""
    mode = policy.sexting_mode
    if paid_context:
        # He is buying, has bought, or is inside a configured free window that
        # the commercial layer has already opened and is already counting.
        return TextIntimacy.EXPLICIT, False, "paid_or_configured_context"

    if mode is SextingMode.PAID_ONLY:
        # The agency sells explicit. Flirting is free; graphic is the product.
        return TextIntimacy.FLIRTY, False, "paid_only_mode_caps_free_text"

    allowance = (
        policy.free_text_max_messages
        if mode is SextingMode.FREE_TEXT_ALLOWED
        else policy.teaser_max_messages
    )
    if free_mode_on_cooldown:
        return TextIntimacy.FLIRTY, False, "free_allowance_cooling_down"
    if teaser_messages_used >= max(0, allowance):
        return TextIntimacy.FLIRTY, False, "free_allowance_exhausted"
    return TextIntimacy.EXPLICIT, True, "free_allowance_available"


def _describe(level: TextIntimacy, codes: list[str]) -> str:
    return f"{level.value.lower()} ({', '.join(codes)})"
