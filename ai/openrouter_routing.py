"""OpenRouter provider-routing, privacy, and cache-affinity request options.

Verified against the OpenRouter request schema published in the official
``openrouter`` Python SDK (Speakeasy-generated from their OpenAPI document):

``provider``
    ``only``            list of provider slugs the request may use.
    ``allow_fallbacks`` false means "use only the pinned provider and return
                        the upstream error if it is unavailable".
    ``data_collection`` ``"deny"`` restricts routing to providers that do not
                        collect user data; the request errors when no eligible
                        provider meets it.
    ``ignore``          list of provider slugs the request may NOT use. This
                        is what expresses "any eligible host EXCEPT the one
                        that just failed" without an operator maintaining a
                        list of every Kimi host by hand.
    ``zdr``             true restricts routing to Zero Data Retention
                        endpoints. Left off by default — see below.

``session_id``
    OpenRouter's documented sticky-routing key. Requests sharing it are routed
    to the same upstream provider, which is what makes provider-side prefix
    caching effective across the turns of one conversation.

``user``
    A stable pseudonymous end-user identifier used for abuse isolation.
    OpenRouter folds it into a hashed identity and never forwards it raw.

Deliberately NOT claimed here: this module does not assert that the configured
upstream provides Zero Data Retention or that prompts are never retained. It
configures the controls OpenRouter exposes; the actual guarantee depends on the
pinned provider's own policy, which operators must confirm on the model's
provider page before relying on it. ``OPENROUTER_ZDR=true`` is available for
deployments that have confirmed a ZDR endpoint exists for the pinned model.

TWO ROUTING MODES, AND WHY THERE IS NO THIRD
--------------------------------------------
Normal traffic is *pinned*: ``only=[Inceptron]``, ``allow_fallbacks=false``.
That is the whole of production's healthy behaviour and it is unchanged — one
provider, one warm prefix cache, one predictable price. OpenRouter can also
express ordered preference directly (``order=[Inceptron]`` with
``allow_fallbacks=true``), and that was deliberately NOT used: it switches
upstream on the first error, which would silently discard the cache-affine
retries that a passing rate limit needs, and would leave nothing to measure.

*Alternate* mode is the recovery mode, reached only after the pinned provider
has failed repeatedly within one turn. It asks OpenRouter to route the SAME
model anywhere eligible except the provider that just failed. Discovery is
OpenRouter's job — ``ignore=[preferred]`` with fallbacks allowed — so no
operator has to keep a list of twenty Kimi hosts current. ``OPENROUTER_FALLBACK_PROVIDERS``
narrows that to an explicit list for a deployment that wants one.

``data_collection`` and ``zdr`` are applied identically in BOTH modes. Provider
failover must never be a way to reach a host the privacy configuration excludes:
if ``deny`` leaves no alternate eligible, the alternate attempt fails and the
turn goes to the model fallback, which is the correct outcome.
"""

from __future__ import annotations

import os
from typing import Any

# OpenRouter's OpenAI-compatible endpoint.
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_API_KEY_ENV = "OPENROUTER_API_KEY"

# Initial pinned upstream for Kimi K2.6. "Inceptron" is a current provider name
# in OpenRouter's published ProviderName enum and a live provider for
# moonshotai/kimi-k2.6. Overridable without a deploy via OPENROUTER_PROVIDERS.
DEFAULT_PINNED_PROVIDERS = ("Inceptron",)

# How one request is routed among the eligible hosts for its model.
#
# PINNED is normal production traffic and is the default everywhere, so cache
# affinity is preserved unless a caller explicitly asks for recovery routing.
PROVIDER_MODE_PINNED = "pinned"
PROVIDER_MODE_ALTERNATE = "alternate"

_SESSION_ID_MAX_LENGTH = 256


def base_url() -> str:
    """Return the OpenRouter OpenAI-compatible endpoint for this deployment."""

    return (os.getenv("OPENROUTER_BASE_URL") or DEFAULT_BASE_URL).strip().rstrip("/")


def _flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _csv(name: str, default: tuple[str, ...]) -> list[str]:
    raw = os.getenv(name)
    if raw is None:
        return list(default)
    values = [part.strip() for part in raw.split(",")]
    return [value for value in values if value]


def pinned_providers(target_metadata: dict[str, Any] | None = None) -> list[str]:
    """Return the PREFERRED provider slugs for this route.

    This is what healthy traffic uses and is deliberately a pin, not a
    preference order: see the module docstring.
    """

    metadata = target_metadata or {}
    catalog_providers = metadata.get("openrouter_providers")
    if isinstance(catalog_providers, list):
        catalog_default = tuple(
            str(value).strip() for value in catalog_providers if str(value).strip()
        )
    else:
        catalog_default = DEFAULT_PINNED_PROVIDERS
    return _csv("OPENROUTER_PROVIDERS", catalog_default)


def alternate_providers() -> list[str]:
    """Explicit recovery hosts, when a deployment names them.

    Empty is the normal and preferred answer: an empty list means "let
    OpenRouter pick any eligible host that is not the preferred one", which is
    the only version of this that does not rot as providers come and go.
    """

    return _csv("OPENROUTER_FALLBACK_PROVIDERS", ())


def provider_failover_enabled() -> bool:
    """Whether a turn may leave the preferred provider after repeated failure.

    On by default. Turning it off restores the pre-failover behaviour exactly:
    the pinned attempts, then the model fallback.
    """

    return _flag("OPENROUTER_PROVIDER_FAILOVER", True)


def provider_preferences(
    target_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the ``provider`` routing object for one OpenRouter request.

    ``target_metadata["openrouter_provider_mode"]`` selects the mode. Absent
    means ``PROVIDER_MODE_PINNED``, so every existing caller — and every
    healthy production request — keeps exactly the routing it had.
    """

    metadata = target_metadata or {}
    mode = str(metadata.get("openrouter_provider_mode") or PROVIDER_MODE_PINNED)
    preferences: dict[str, Any] = {}

    providers = pinned_providers(metadata)

    if mode == PROVIDER_MODE_ALTERNATE:
        explicit = alternate_providers()
        if explicit:
            preferences["only"] = explicit
        elif providers:
            # Discovery rather than a hand-maintained list: anyone eligible
            # except the host this turn has already given up on.
            preferences["ignore"] = providers
        # The point of this attempt is to leave the failing provider, so
        # fallbacks are allowed regardless of OPENROUTER_ALLOW_FALLBACKS —
        # that variable governs NORMAL traffic, which is still pinned.
        preferences["allow_fallbacks"] = True
    elif providers:
        preferences["only"] = providers
        # Fail closed. Without this, an unavailable Inceptron would silently
        # become some other upstream with different behaviour, different
        # pricing, and a cold cache — on the very first attempt, which is
        # precisely what the recovery ladder exists to make deliberate.
        preferences["allow_fallbacks"] = _flag("OPENROUTER_ALLOW_FALLBACKS", False)

    # Privacy and eligibility are identical in both modes. Recovery is never a
    # reason to widen them.
    data_collection = (os.getenv("OPENROUTER_DATA_COLLECTION") or "deny").strip().lower()
    if data_collection in {"deny", "allow"}:
        preferences["data_collection"] = data_collection

    if _flag("OPENROUTER_ZDR", False):
        preferences["zdr"] = True

    return preferences


def request_options(
    *,
    target_metadata: dict[str, Any] | None = None,
    session_id: str | None = None,
    end_user_id: str | None = None,
) -> dict[str, Any]:
    """Build the OpenRouter-specific body fields for one chat completion."""

    options: dict[str, Any] = {}

    preferences = provider_preferences(target_metadata)
    if preferences:
        options["provider"] = preferences

    if session_id:
        options["session_id"] = str(session_id)[:_SESSION_ID_MAX_LENGTH]
    if end_user_id:
        options["user"] = str(end_user_id)[:_SESSION_ID_MAX_LENGTH]

    return options


def upstream_provider(response: Any) -> str | None:
    """Return the upstream provider OpenRouter actually used, when exposed.

    OpenRouter reports this either as a top-level ``provider`` on the
    completion or as the last entry of ``openrouter_metadata.attempts``.
    """

    direct = getattr(response, "provider", None)
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    metadata = getattr(response, "openrouter_metadata", None)
    attempts = getattr(metadata, "attempts", None)
    if attempts is None and isinstance(metadata, dict):
        attempts = metadata.get("attempts")
    if isinstance(attempts, list) and attempts:
        last = attempts[-1]
        provider = (
            last.get("provider")
            if isinstance(last, dict)
            else getattr(last, "provider", None)
        )
        if isinstance(provider, str) and provider.strip():
            return provider.strip()
    return None
