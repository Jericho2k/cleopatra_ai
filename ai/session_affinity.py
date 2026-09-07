"""Deterministic provider/session affinity keys for cached writer prompts.

OpenRouter uses ``session_id`` as its sticky-routing key: every request that
carries the same value is routed to the same upstream provider, which is what
lets that provider's prefix/KV cache survive between turns of one conversation.

The key must therefore be *stable for the life of a fan conversation*. It is
derived only from the creator and fan identifiers — never from a timestamp, a
message id, a counter, or a random value — so consecutive replies in the same
chat keep hitting the same warm cache.

The raw identifiers are hashed rather than sent verbatim: the affinity key
leaves our infrastructure, and an opaque digest gives the provider the grouping
signal it needs without handing it our internal creator/fan primary keys.
"""

from __future__ import annotations

from hashlib import sha256

# OpenRouter accepts up to 256 characters. A truncated SHA-256 is far shorter
# than the limit and still has no realistic collision risk at our scale.
_KEY_LENGTH = 32


def _normalized(value: object) -> str:
    return str(value or "").strip()


def writer_session_id(
    creator_id: object,
    fan_id: object,
    *,
    prefix: str = "cleo",
) -> str | None:
    """Return the stable affinity key for one creator/fan conversation.

    Returns ``None`` when either identifier is missing. A partial key would
    group unrelated conversations onto one cache lineage, which is worse than
    letting the request route normally.
    """

    creator = _normalized(creator_id)
    fan = _normalized(fan_id)
    if not creator or not fan:
        return None

    digest = sha256(f"{creator}\x1f{fan}".encode("utf-8")).hexdigest()
    return f"{prefix}-{digest[:_KEY_LENGTH]}"


def writer_end_user_id(creator_id: object, fan_id: object) -> str | None:
    """Return the pseudonymous end-user identifier for abuse isolation.

    OpenRouter documents ``user`` as "a stable ID, hash, or pseudonym" that it
    folds into a hashed upstream identity. Reusing the same deterministic
    digest keeps one fan conversation attributable across turns without
    exposing a real identifier.
    """

    return writer_session_id(creator_id, fan_id, prefix="fan")
