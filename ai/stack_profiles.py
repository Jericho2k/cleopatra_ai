"""AI Stack Profiles — the whole conversational AI brain, named and versioned.

WHAT A PROFILE IS
-----------------
An *AI Stack Profile* is not a model. It is the complete configuration of every
model-powered stage in the conversational pipeline: which provider and model
each stage targets, what it falls back to, which prompt version it uses, whether
reasoning is on, and the generation parameters that materially change what comes
back.

It exists so the owner can run two different AI brains side by side against the
same fixed runtime and compare them. ``cleo_legacy_v1`` is a frozen snapshot of
the AI configuration that shipped on main before this change; ``cleo_v2`` is the
new one.

WHAT A PROFILE IS NOT
---------------------
A profile does **not** fork application behaviour. Simulator isolation, the
inventory authority, the commercial engine, session recovery, pricing, money
handling and every safety rule are shared application logic and are identical
under both profiles. "Legacy" means *old prompts and old model routing*, never
"restore old bugs".

Nothing here reads a provider or model string supplied by a client. The
frontend selects a profile by its stable identifier; this registry is the only
place a provider/model pair is written down.

ENVIRONMENT OVERRIDES
---------------------
``cleo_legacy_v1`` keeps the ``WRITER_DEFAULT_*`` / ``WRITER_COMPLEX_*`` /
``ANALYZER_*`` / ``EXTRACTOR_*`` escape hatches that current main honours,
because preserving today's behaviour means preserving them too: a deployment
that has one of those set is running that model today, and the frozen profile
must reproduce that.

``cleo_v2`` is pinned. Its routing is the point of the profile, so it is not
silently re-pointed by a variable somebody set months ago for the old stack.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Any

from ai.model_migrations import resolve_supported_model
from ai.model_providers import find_catalog_target, provider_transport_defaults
from models.model_runtime import ModelTarget


# ---------------------------------------------------------------------------
# Stage names
# ---------------------------------------------------------------------------
#
# One entry per model-powered stage that actually exists in the conversational
# pipeline on current main. Deliberately not aspirational: a stage is listed
# here only because a real ``complete()``-equivalent call is made for it.

STAGE_SITUATION_ANALYZER = "situation_analyzer"
STAGE_WRITER_DEFAULT = "writer_default"
STAGE_WRITER_COMMERCIAL = "writer_commercial"
STAGE_WRITER_SAFETY = "writer_safety"
STAGE_FAN_INTELLIGENCE = "fan_intelligence"
STAGE_FAN_SUMMARY = "fan_summary"

STAGE_ORDER: tuple[str, ...] = (
    STAGE_SITUATION_ANALYZER,
    STAGE_WRITER_DEFAULT,
    STAGE_WRITER_COMMERCIAL,
    STAGE_WRITER_SAFETY,
    STAGE_FAN_INTELLIGENCE,
    STAGE_FAN_SUMMARY,
)

STAGE_LABELS: dict[str, str] = {
    STAGE_SITUATION_ANALYZER: "Situation analyzer",
    STAGE_WRITER_DEFAULT: "Writer — ordinary conversation",
    STAGE_WRITER_COMMERCIAL: "Writer — commercial expression",
    STAGE_WRITER_SAFETY: "Writer — safety-sensitive",
    STAGE_FAN_INTELLIGENCE: "Fan intelligence extraction",
    STAGE_FAN_SUMMARY: "Fan psychological summary",
}

# What the stage does with the model's text. Recorded because a parser mode is
# part of "what this stack does", and because reasoning-enabled models silently
# break ``json_array``.
OUTPUT_JSON_ARRAY = "json_array"
OUTPUT_JSON_OBJECT = "json_object"


@dataclass(frozen=True)
class StageSpec:
    """One model-powered stage, fully resolved."""

    stage: str
    provider: str
    model: str
    prompt_version: str
    reasoning: bool = False
    output_mode: str = OUTPUT_JSON_OBJECT
    max_tokens: int = 1000
    temperature: float | None = None
    fallback_provider: str | None = None
    fallback_model: str | None = None
    # Environment variables that may re-point this stage. Present on the frozen
    # legacy profile only; see the module docstring.
    provider_env: str | None = None
    model_env: str | None = None
    fallback_provider_env: str | None = None
    fallback_model_env: str | None = None
    max_tokens_env: str | None = None
    notes: str = ""

    # -- resolution ---------------------------------------------------------

    def _resolved_pair(
        self,
        provider: str | None,
        model: str | None,
        provider_env: str | None,
        model_env: str | None,
    ) -> tuple[str, str] | None:
        if not provider or not model:
            return None
        chosen_provider = (
            os.getenv(provider_env, provider) if provider_env else provider
        )
        chosen_provider = str(chosen_provider or provider).strip().lower()
        chosen_model = os.getenv(model_env, model) if model_env else model
        return chosen_provider, resolve_supported_model(
            chosen_provider, str(chosen_model or model)
        )

    def resolved_primary(self) -> tuple[str, str]:
        pair = self._resolved_pair(
            self.provider, self.model, self.provider_env, self.model_env
        )
        # provider/model are required fields, so this can only be None if a
        # future spec is written with empty strings.
        return pair or (self.provider, self.model)

    def resolved_fallback(self) -> tuple[str, str] | None:
        return self._resolved_pair(
            self.fallback_provider,
            self.fallback_model,
            self.fallback_provider_env,
            self.fallback_model_env,
        )

    def resolved_max_tokens(self) -> int:
        if not self.max_tokens_env:
            return int(self.max_tokens)
        try:
            return int(os.getenv(self.max_tokens_env, str(self.max_tokens)) or self.max_tokens)
        except (TypeError, ValueError):
            return int(self.max_tokens)

    def primary_target(self) -> ModelTarget:
        provider, model = self.resolved_primary()
        return _build_target(provider, model, reasoning=self.reasoning)

    def fallback_target(self) -> ModelTarget | None:
        pair = self.resolved_fallback()
        if pair is None:
            return None
        provider, model = pair
        return _build_target(provider, model, reasoning=self.reasoning)

    def describe(self) -> dict[str, Any]:
        """Read-only detail for the owner's profile inspector."""
        provider, model = self.resolved_primary()
        fallback = self.resolved_fallback()
        return {
            "stage": self.stage,
            "label": STAGE_LABELS.get(self.stage, self.stage),
            "provider": provider,
            "model": model,
            "fallback_provider": fallback[0] if fallback else None,
            "fallback_model": fallback[1] if fallback else None,
            "prompt_version": self.prompt_version,
            "reasoning": bool(self.reasoning),
            "output_mode": self.output_mode,
            "max_tokens": self.resolved_max_tokens(),
            "temperature": self.temperature,
            "env_overridable": bool(self.provider_env or self.model_env),
            "notes": self.notes,
        }


def _build_target(provider: str, model: str, *, reasoning: bool) -> ModelTarget:
    """One provider/model pair as a transport-ready target.

    The catalog wins when it knows the model, because it carries streaming,
    timeout, pricing and OpenRouter provider pinning. The profile's reasoning
    setting is then applied on top: reasoning is a property of how this *stack*
    uses the model, and a stack that asks for a JSON array from a reasoning
    model gets ``message.content = null``.
    """
    provider = provider.strip().lower()
    catalog = find_catalog_target(provider, model)
    if catalog is not None:
        metadata = {**(catalog.metadata or {}), "reasoning_enabled": bool(reasoning)}
        return replace(catalog, metadata=metadata)

    base_url, api_key_env = provider_transport_defaults(provider)
    return ModelTarget(
        name=f"{provider}:{model}",
        provider=provider,
        model=model,
        base_url=base_url,
        api_key_env=api_key_env,
        metadata={"reasoning_enabled": bool(reasoning)},
    )


@dataclass(frozen=True)
class AIStackProfile:
    profile_id: str
    label: str
    summary: str
    stages: dict[str, StageSpec] = field(default_factory=dict)

    def stage(self, name: str) -> StageSpec:
        try:
            return self.stages[name]
        except KeyError as error:  # pragma: no cover - defensive
            raise KeyError(
                f"AI stack profile {self.profile_id!r} has no stage {name!r}"
            ) from error

    def writer_prompt_version(self) -> str:
        return self.stage(STAGE_WRITER_DEFAULT).prompt_version

    def describe(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "label": self.label,
            "summary": self.summary,
            "stages": [self.stages[name].describe() for name in STAGE_ORDER if name in self.stages],
        }


# ---------------------------------------------------------------------------
# cleo_legacy_v1 — frozen snapshot of main before the V2 pass
# ---------------------------------------------------------------------------

_LEGACY_ANALYZER = StageSpec(
    stage=STAGE_SITUATION_ANALYZER,
    provider="anthropic",
    model="claude-haiku-4-5-20251001",
    prompt_version="analyzer_v1",
    reasoning=False,
    output_mode=OUTPUT_JSON_OBJECT,
    max_tokens=650,
    temperature=0.0,
    provider_env="ANALYZER_PROVIDER",
    model_env="ANALYZER_MODEL",
    notes="Observation only. Never decides a commercial action.",
)

_LEGACY_WRITER_DEFAULT = StageSpec(
    stage=STAGE_WRITER_DEFAULT,
    provider="openrouter",
    model="moonshotai/kimi-k2.6",
    prompt_version="writer_v1",
    reasoning=False,
    output_mode=OUTPUT_JSON_ARRAY,
    max_tokens=1000,
    temperature=None,
    fallback_provider="together",
    fallback_model="Qwen/Qwen3.7-Plus",
    provider_env="WRITER_DEFAULT_PROVIDER",
    model_env="WRITER_DEFAULT_MODEL",
    fallback_provider_env="WRITER_COMPLEX_PROVIDER",
    fallback_model_env="WRITER_COMPLEX_MODEL",
    notes="Kimi pinned to one OpenRouter upstream; Qwen on Together is the fallback.",
)

_LEGACY_WRITER_COMMERCIAL = StageSpec(
    stage=STAGE_WRITER_COMMERCIAL,
    provider="together",
    model="Qwen/Qwen3.7-Plus",
    prompt_version="writer_v1",
    reasoning=False,
    output_mode=OUTPUT_JSON_ARRAY,
    max_tokens=1000,
    temperature=None,
    provider_env="WRITER_COMPLEX_PROVIDER",
    model_env="WRITER_COMPLEX_MODEL",
    notes="Commercially complex, high-value, and session-active turns. No fallback.",
)

_LEGACY_WRITER_SAFETY = replace(
    _LEGACY_WRITER_COMMERCIAL,
    stage=STAGE_WRITER_SAFETY,
    notes="Crisis and human-review turns. Same target as the commercial writer.",
)

_LEGACY_FAN_INTELLIGENCE = StageSpec(
    stage=STAGE_FAN_INTELLIGENCE,
    provider="together",
    model="openai/gpt-oss-120b",
    prompt_version="fan_intelligence_v1",
    reasoning=False,
    output_mode=OUTPUT_JSON_OBJECT,
    max_tokens=700,
    temperature=0.0,
    provider_env="EXTRACTOR_PROVIDER",
    model_env="EXTRACTOR_MODEL",
    max_tokens_env="EXTRACTOR_MAX_TOKENS",
    notes="Durable fact extraction. Enrichment only; never blocks a reply.",
)

_LEGACY_FAN_SUMMARY = StageSpec(
    stage=STAGE_FAN_SUMMARY,
    provider="together",
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
    prompt_version="fan_summary_v1",
    reasoning=False,
    output_mode=OUTPUT_JSON_OBJECT,
    max_tokens=1000,
    temperature=0.3,
    notes="Periodic psychological profile refresh. Best-effort, out of band.",
)

CLEO_LEGACY_V1 = AIStackProfile(
    profile_id="cleo_legacy_v1",
    label="Cleo Legacy v1",
    summary=(
        "Frozen snapshot of the AI configuration that shipped before the V2 "
        "pass. Kimi writes ordinary conversation, Qwen writes every commercial "
        "and safety-sensitive turn, writer prompt writer_v1."
    ),
    stages={
        STAGE_SITUATION_ANALYZER: _LEGACY_ANALYZER,
        STAGE_WRITER_DEFAULT: _LEGACY_WRITER_DEFAULT,
        STAGE_WRITER_COMMERCIAL: _LEGACY_WRITER_COMMERCIAL,
        STAGE_WRITER_SAFETY: _LEGACY_WRITER_SAFETY,
        STAGE_FAN_INTELLIGENCE: _LEGACY_FAN_INTELLIGENCE,
        STAGE_FAN_SUMMARY: _LEGACY_FAN_SUMMARY,
    },
)


# ---------------------------------------------------------------------------
# cleo_v2 — one writer voice, new prompt
# ---------------------------------------------------------------------------
#
# The deterministic engine already decides what may be sold, at what price, from
# what inventory, under which session and lifecycle constraints. The writer only
# has to express that decision, so there is no longer a reason for a sale to
# arrive in a different model's voice than the conversation around it. Kimi is
# primary for both, with Qwen on Together as the deterministic fallback — still
# a different provider, so an OpenRouter incident does not silence the writer.
#
# Analyzer, extractor and summary are deliberately unchanged. Nothing about
# this pass gives a reason to re-point them, and changing a model for symmetry
# is how a comparison stops being a comparison.

_V2_WRITER_DEFAULT = StageSpec(
    stage=STAGE_WRITER_DEFAULT,
    provider="openrouter",
    model="moonshotai/kimi-k2.6",
    prompt_version="writer_v2",
    reasoning=False,
    output_mode=OUTPUT_JSON_ARRAY,
    max_tokens=1000,
    temperature=None,
    fallback_provider="together",
    fallback_model="Qwen/Qwen3.7-Plus",
    notes="Pinned. Same target for ordinary and commercial turns.",
)

_V2_WRITER_COMMERCIAL = replace(
    _V2_WRITER_DEFAULT,
    stage=STAGE_WRITER_COMMERCIAL,
    notes=(
        "Kimi primary so a sale does not arrive in a different voice than the "
        "conversation around it. Qwen on Together remains the fallback."
    ),
)

_V2_WRITER_SAFETY = StageSpec(
    stage=STAGE_WRITER_SAFETY,
    provider="together",
    model="Qwen/Qwen3.7-Plus",
    prompt_version="writer_v2",
    reasoning=False,
    output_mode=OUTPUT_JSON_ARRAY,
    max_tokens=1000,
    temperature=None,
    notes=(
        "Unchanged from legacy on purpose. A crisis turn is not a commercial "
        "expression turn, and keeping it off OpenRouter keeps one provider "
        "incident from taking every writer down at once."
    ),
)

CLEO_V2 = AIStackProfile(
    profile_id="cleo_v2",
    label="Cleo V2",
    summary=(
        "One writer voice: Kimi K2.6 writes both ordinary conversation and "
        "commercial expression, with Qwen3.7-Plus on Together as the "
        "deterministic fallback. New writer prompt (writer_v2): react before "
        "advancing, specific over generic, no invented current-life facts."
    ),
    stages={
        STAGE_SITUATION_ANALYZER: replace(
            _LEGACY_ANALYZER, provider_env=None, model_env=None
        ),
        STAGE_WRITER_DEFAULT: _V2_WRITER_DEFAULT,
        STAGE_WRITER_COMMERCIAL: _V2_WRITER_COMMERCIAL,
        STAGE_WRITER_SAFETY: _V2_WRITER_SAFETY,
        STAGE_FAN_INTELLIGENCE: replace(
            _LEGACY_FAN_INTELLIGENCE, provider_env=None, model_env=None
        ),
        STAGE_FAN_SUMMARY: _LEGACY_FAN_SUMMARY,
    },
)


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------

PROFILES: dict[str, AIStackProfile] = {
    CLEO_LEGACY_V1.profile_id: CLEO_LEGACY_V1,
    CLEO_V2.profile_id: CLEO_V2,
}

PROFILE_IDS: tuple[str, ...] = tuple(PROFILES)

# What an unconfigured deployment runs.
#
# Deliberately the frozen legacy profile rather than V2: an environment that has
# not been told which brain to run must keep behaving exactly as it did before
# this code shipped. Switching behaviour on deploy, silently, because a variable
# is absent is the one outcome a versioning system exists to prevent.
# Production sets AI_STACK_PROFILE=cleo_v2 explicitly.
DEFAULT_PROFILE_ID = CLEO_LEGACY_V1.profile_id

PROFILE_ENV_VAR = "AI_STACK_PROFILE"


def is_valid_profile_id(value: Any) -> bool:
    return str(value or "").strip() in PROFILES


def normalize_profile_id(value: Any) -> str | None:
    """A known profile id, or None. Never raises, never invents one."""
    text = str(value or "").strip()
    return text if text in PROFILES else None


def get_profile(profile_id: Any) -> AIStackProfile:
    """The profile for an id, falling back to the environment default.

    An unknown id is never an error here: this runs on the reply path, and a
    stale creator override must not be able to stop a fan being answered. The
    mutation endpoints reject unknown ids, which is where an operator finds out.
    """
    known = normalize_profile_id(profile_id)
    if known:
        return PROFILES[known]
    return PROFILES[environment_profile_id()]


def environment_profile_id() -> str:
    """The deployment-wide default, from ``AI_STACK_PROFILE``."""
    return normalize_profile_id(os.getenv(PROFILE_ENV_VAR)) or DEFAULT_PROFILE_ID


def describe_profiles() -> list[dict[str, Any]]:
    return [PROFILES[profile_id].describe() for profile_id in PROFILE_IDS]
