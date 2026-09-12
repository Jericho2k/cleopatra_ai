"""Deterministic production writer routing for Cleopatra.

This module answers one question: given conversation state that has already been
computed, which of three writer ROUTES does this turn take — ordinary,
commercially complex, or safety-sensitive? That decision is shared application
logic and is identical under every AI Stack Profile, because it describes the
conversation rather than the brain answering it. The router never asks a model
which business action should happen.

Which MODEL each route points at is a property of the turn's AI Stack Profile
(ai/stack_profiles.py), not of this module:

``cleo_legacy_v1``
    Kimi K2.6 via OpenRouter writes ordinary conversation; Qwen3.7-Plus on
    Together writes every commercial and safety-sensitive turn. The frozen
    behaviour of main before the V2 pass.

``cleo_v2``
    Kimi K2.6 writes ordinary conversation AND commercial expression, with
    Qwen3.7-Plus on Together as its deterministic fallback, so a sale no longer
    arrives in a different voice than the conversation around it. The
    safety-sensitive route stays on Together: a crisis turn is not commercial
    expression, and keeping it on a second provider means one OpenRouter
    incident cannot take every writer down at once.

Provider diversity is deliberate in both profiles. The constants below remain
the canonical default targets and are read by the availability check and the
smoke script.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ai.stack_profiles import (
    STAGE_WRITER_COMMERCIAL,
    STAGE_WRITER_DEFAULT,
    STAGE_WRITER_SAFETY,
    get_profile,
)
from ai.model_providers import get_runtime_target
from models.model_runtime import ModelTarget


# Ordinary conversational writer. Kimi K2.6 is reached through OpenRouter and
# pinned to a single upstream provider (see ai/openrouter_routing.py).
DEFAULT_WRITER_PROVIDER = "openrouter"
DEFAULT_WRITER_MODEL = "moonshotai/kimi-k2.6"

# Commercially complex and safety-sensitive turns. Deliberately not routed
# through OpenRouter, so an OpenRouter incident cannot take both writers down.
#
# Was deepseek-ai/DeepSeek-V4-Pro. Together rejects that handle for this account
# with a live 400 — "Unable to access non-serverless model ... Please create and
# start a dedicated endpoint" — so it was not a fallback at all: every ordinary
# turn whose Kimi attempts failed ended with no reply. Qwen3.7-Plus is serverless
# on Together, reachable with the same TOGETHER_API_KEY, and carries
# reasoning_enabled=false in the catalog so it returns the writer's JSON array
# rather than chain-of-thought.
COMPLEX_WRITER_PROVIDER = "together"
COMPLEX_WRITER_MODEL = "Qwen/Qwen3.7-Plus"


class WriterRoute(str, Enum):
    DEFAULT = "default"
    COMMERCIAL_COMPLEX = "commercial_complex"
    SAFETY_SENSITIVE = "safety_sensitive"


@dataclass(frozen=True)
class WriterRouteDecision:
    route: WriterRoute
    reason: str
    primary_target: ModelTarget
    fallback_target: ModelTarget | None = None
    # Which AI Stack Profile produced these targets, and which writer voice goes
    # with them. Carried on the decision so telemetry, the Railway log line and
    # the persisted message metadata all name the same brain.
    ai_stack_profile: str = ""
    prompt_version: str = ""

    def telemetry_metadata(self) -> dict[str, Any]:
        return {
            "writer_route": self.route.value,
            "writer_route_reason": self.reason,
            "ai_stack_profile": self.ai_stack_profile,
            "writer_prompt_version": self.prompt_version,
            "writer_primary_provider": self.primary_target.provider,
            "writer_primary_model": self.primary_target.model,
            "writer_fallback_provider": (
                self.fallback_target.provider if self.fallback_target else None
            ),
            "writer_fallback_model": (
                self.fallback_target.model if self.fallback_target else None
            ),
        }


def _enabled(name: str, default: bool = True) -> bool:
    fallback = "true" if default else "false"
    return os.getenv(name, fallback).strip().lower() in {"1", "true", "yes", "on"}


def _normalized(value: Any) -> str:
    if value is None:
        return ""
    raw = getattr(value, "value", value)
    return str(raw).strip().upper()


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        dumped = value.model_dump(mode="json")
        return dumped if isinstance(dumped, dict) else {}
    return {}


def _same_target(left: ModelTarget, right: ModelTarget | None) -> bool:
    return bool(
        right
        and left.provider == right.provider
        and left.model == right.model
        and left.base_url == right.base_url
    )


def select_writer_route(
    ctx: Any,
    *,
    profile_id: str | None = None,
) -> WriterRouteDecision:
    """Choose a writer deterministically from already-computed conversation state.

    Which *models* the three routes point at is a property of the turn's AI
    Stack Profile (ai/stack_profiles.py). Which *route* a turn takes is not: the
    conditions below are shared application logic and are identical under every
    profile, because they describe the conversation, not the brain answering it.
    """

    profile = get_profile(profile_id or getattr(ctx, "ai_stack_profile", None))
    default_spec = profile.stage(STAGE_WRITER_DEFAULT)
    commercial_spec = profile.stage(STAGE_WRITER_COMMERCIAL)
    safety_spec = profile.stage(STAGE_WRITER_SAFETY)

    def decide(
        route: WriterRoute,
        reason: str,
        primary: ModelTarget,
        fallback: ModelTarget | None = None,
    ) -> WriterRouteDecision:
        return WriterRouteDecision(
            route=route,
            reason=reason,
            primary_target=primary,
            fallback_target=None if _same_target(primary, fallback) else fallback,
            ai_stack_profile=profile.profile_id,
            prompt_version=default_spec.prompt_version,
        )

    if not _enabled("WRITER_ROUTING_ENABLED", True):
        target = get_runtime_target("CHAT")
        return decide(WriterRoute.DEFAULT, "routing_disabled", target)

    default_target = default_spec.primary_target()
    default_fallback = default_spec.fallback_target()
    complex_target = commercial_spec.primary_target()
    complex_fallback = commercial_spec.fallback_target()
    safety_target = safety_spec.primary_target()
    safety_fallback = safety_spec.fallback_target()

    situation = _mapping(getattr(ctx, "situation", None))
    commercial_decision = _mapping(getattr(ctx, "commercial_decision", None))
    buyer_lifecycle = _mapping(getattr(ctx, "buyer_lifecycle", None))
    lifecycle_stage = _normalized(buyer_lifecycle.get("stage"))
    fan = getattr(ctx, "fan_profile", None)

    crisis_signal = _normalized(situation.get("crisis_signal") or "none")
    action = _normalized(commercial_decision.get("action"))
    stage = _normalized(getattr(ctx, "conversation_stage", None))
    purchase_signal = _normalized(situation.get("purchase_signal"))

    needs_human_review = bool(getattr(fan, "needs_human_review", False))
    if (
        crisis_signal not in {"", "NONE"}
        or needs_human_review
        or action == "HAND_OFF_TO_HUMAN"
    ):
        return decide(
            WriterRoute.SAFETY_SENSITIVE,
            (
                f"crisis:{crisis_signal.lower()}"
                if crisis_signal not in {"", "NONE"}
                else "human_review"
            ),
            safety_target,
            safety_fallback,
        )

    complex_actions = {
        "ASK_ONE_QUALIFYING_QUESTION",
        "END_TEASER_AND_OFFER",
        "PRESENT_SESSION_OPTIONS",
        "CREATE_PAID_SESSION",
        "SEND_NEXT_PPV_STEP",
        "PAUSE_NO_BUDGET",
        "PAUSE_UNTIL_PAYDAY",
        "RESUME_PREVIOUS_OFFER",
        "PAYDAY_REENGAGEMENT",
    }
    if action in complex_actions:
        return decide(
            WriterRoute.COMMERCIAL_COMPLEX,
            f"commercial_action:{action.lower()}",
            complex_target,
            complex_fallback,
        )

    if bool(getattr(ctx, "active_session", None)):
        return decide(
            WriterRoute.COMMERCIAL_COMPLEX,
            "active_paid_session",
            complex_target,
            complex_fallback,
        )

    if purchase_signal in {"DECLINED", "MONEY_AVAILABLE", "READY_TO_BUY", "BOUGHT"}:
        return decide(
            WriterRoute.COMMERCIAL_COMPLEX,
            f"purchase_signal:{purchase_signal.lower()}",
            complex_target,
            complex_fallback,
        )

    if stage in {"OBJECTION", "UPSELL_ACTIVE", "HIGH_VALUE"}:
        return decide(
            WriterRoute.COMMERCIAL_COMPLEX,
            f"conversation_stage:{stage.lower()}",
            complex_target,
            complex_fallback,
        )

    if lifecycle_stage == "VIP":
        return decide(
            WriterRoute.COMMERCIAL_COMPLEX,
            "buyer_lifecycle:vip",
            complex_target,
            complex_fallback,
        )

    spend_tier = str(getattr(fan, "spend_tier", "") or "").strip().lower()
    try:
        total_spent = float(getattr(fan, "total_spent", 0) or 0)
    except (TypeError, ValueError):
        total_spent = 0.0
    try:
        high_value_threshold = float(
            os.getenv("WRITER_HIGH_VALUE_THRESHOLD_USD", "100") or 100
        )
    except ValueError:
        high_value_threshold = 100.0

    if spend_tier in {"whale", "vip", "high_value"} or total_spent >= high_value_threshold:
        return decide(
            WriterRoute.COMMERCIAL_COMPLEX,
            "high_value_fan",
            complex_target,
            complex_fallback,
        )

    return decide(
        WriterRoute.DEFAULT,
        "ordinary_conversation",
        default_target,
        default_fallback,
    )
