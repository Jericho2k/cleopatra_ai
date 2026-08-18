"""Deterministic production writer routing for Cleopatra.

Kimi handles ordinary conversation. DeepSeek handles commercially complex,
high-value, session-active, and safety-sensitive turns. The router never asks a
model to decide which business action should happen; it only chooses the writer
that expresses the already-known context.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ai.model_migrations import resolve_supported_model
from ai.model_providers import find_catalog_target, get_runtime_target
from models.model_runtime import ModelTarget


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

    def telemetry_metadata(self) -> dict[str, Any]:
        return {
            "writer_route": self.route.value,
            "writer_route_reason": self.reason,
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


def _resolve_target(
    *,
    provider_env: str,
    model_env: str,
    default_provider: str,
    default_model: str,
) -> ModelTarget:
    provider = os.getenv(provider_env, default_provider).strip().lower()
    model = resolve_supported_model(
        provider,
        os.getenv(model_env, default_model),
    )

    catalog_target = find_catalog_target(provider, model)
    if catalog_target is not None:
        return catalog_target

    base_url: str | None = None
    api_key_env: str | None = None
    if provider == "together":
        base_url = "https://api.together.xyz/v1"
        api_key_env = "TOGETHER_API_KEY"
    elif provider == "anthropic":
        api_key_env = "ANTHROPIC_API_KEY"
    elif provider in {"self_hosted", "openai_compatible"}:
        base_url = os.getenv("SELF_HOSTED_BASE_URL")
        api_key_env = "SELF_HOSTED_API_KEY"

    return ModelTarget(
        name=f"{provider}:{model}",
        provider=provider,
        model=model,
        base_url=base_url,
        api_key_env=api_key_env,
    )


def _same_target(left: ModelTarget, right: ModelTarget | None) -> bool:
    return bool(
        right
        and left.provider == right.provider
        and left.model == right.model
        and left.base_url == right.base_url
    )


def select_writer_route(ctx: Any) -> WriterRouteDecision:
    """Choose a writer deterministically from already-computed conversation state."""

    if not _enabled("WRITER_ROUTING_ENABLED", True):
        target = get_runtime_target("CHAT")
        return WriterRouteDecision(
            route=WriterRoute.DEFAULT,
            reason="routing_disabled",
            primary_target=target,
        )

    default_target = _resolve_target(
        provider_env="WRITER_DEFAULT_PROVIDER",
        model_env="WRITER_DEFAULT_MODEL",
        default_provider="together",
        default_model="moonshotai/Kimi-K3",
    )
    complex_target = _resolve_target(
        provider_env="WRITER_COMPLEX_PROVIDER",
        model_env="WRITER_COMPLEX_MODEL",
        default_provider="together",
        default_model="deepseek-ai/DeepSeek-V4-Pro",
    )

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
        return WriterRouteDecision(
            route=WriterRoute.SAFETY_SENSITIVE,
            reason=(
                f"crisis:{crisis_signal.lower()}"
                if crisis_signal not in {"", "NONE"}
                else "human_review"
            ),
            primary_target=complex_target,
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
        return WriterRouteDecision(
            route=WriterRoute.COMMERCIAL_COMPLEX,
            reason=f"commercial_action:{action.lower()}",
            primary_target=complex_target,
        )

    if bool(getattr(ctx, "active_session", None)):
        return WriterRouteDecision(
            route=WriterRoute.COMMERCIAL_COMPLEX,
            reason="active_paid_session",
            primary_target=complex_target,
        )

    if purchase_signal in {"DECLINED", "MONEY_AVAILABLE", "READY_TO_BUY", "BOUGHT"}:
        return WriterRouteDecision(
            route=WriterRoute.COMMERCIAL_COMPLEX,
            reason=f"purchase_signal:{purchase_signal.lower()}",
            primary_target=complex_target,
        )

    if stage in {"OBJECTION", "UPSELL_ACTIVE", "HIGH_VALUE"}:
        return WriterRouteDecision(
            route=WriterRoute.COMMERCIAL_COMPLEX,
            reason=f"conversation_stage:{stage.lower()}",
            primary_target=complex_target,
        )

    if lifecycle_stage == "VIP":
        return WriterRouteDecision(
            route=WriterRoute.COMMERCIAL_COMPLEX,
            reason="buyer_lifecycle:vip",
            primary_target=complex_target,
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
        return WriterRouteDecision(
            route=WriterRoute.COMMERCIAL_COMPLEX,
            reason="high_value_fan",
            primary_target=complex_target,
        )

    fallback = None if _same_target(default_target, complex_target) else complex_target
    return WriterRouteDecision(
        route=WriterRoute.DEFAULT,
        reason="ordinary_conversation",
        primary_target=default_target,
        fallback_target=fallback,
    )
