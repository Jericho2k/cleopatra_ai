"""Resolve the conversational runtime for one turn.

The runtime choice is deliberately separate from the AI Stack Profile.  A
profile chooses models and prompt versions; this choice decides which
application architecture owns the turn.

Resolution is persistent because Full Auto and scheduled actions run without a
browser session.  A fan-level override is honoured only for ``test_`` fans, so
side-by-side simulation can never switch a real customer.  Missing columns mean
"legacy" during a rolling migration.  A present but invalid value is an error:
silently crossing architectures in the middle of a turn would make an agency
evaluation meaningless.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from typing import Any

from core.simulation import is_simulatable_fan
from core.supabase import get_supabase

CORE_LEGACY = "legacy"
CORE_SEMANTIC_V1 = "semantic_v1"
CORE_SEMANTIC_V2 = "semantic_v2"
CORE_CONVERSATIONAL_V1 = "conversational_v1"
#: Session-aware alternative to v1 (not a layer on it): same GLM owner / Kimi
#: writer / deterministic authority, plus durable longer-interaction state.
CORE_CONVERSATIONAL_V2 = "conversational_v2"
CORE_IDS = (
    CORE_LEGACY,
    CORE_SEMANTIC_V1,
    CORE_SEMANTIC_V2,
    CORE_CONVERSATIONAL_V1,
    CORE_CONVERSATIONAL_V2,
)

#: Operator-facing names, served by ``GET /conversation-cores``.
CORE_LABELS = {
    CORE_LEGACY: "Legacy controller stack",
    CORE_SEMANTIC_V1: "Semantic owner v1",
    CORE_SEMANTIC_V2: "One-call conversational owner",
    CORE_CONVERSATIONAL_V1: "Conversational Core v1",
    CORE_CONVERSATIONAL_V2: "Conversational Core v2 — Session-aware",
}
CORE_ENV_VAR = "CONVERSATION_CORE"

SOURCE_SIMULATION_FAN = "simulation_fan"
SOURCE_CREATOR = "creator"
SOURCE_ENVIRONMENT = "environment"
SOURCE_BUILTIN = "builtin"

_UNKNOWN = object()


class InvalidConversationCoreConfiguration(RuntimeError):
    """The selected runtime is not a known, deployable architecture."""


@dataclass(frozen=True)
class ConversationCoreResolution:
    core_id: str
    source: str

    @property
    def is_semantic(self) -> bool:
        # Kept as the compatibility name used by the routing call sites. It
        # means "owned by live_orchestration rather than legacy choreography".
        return self.core_id in {
            CORE_SEMANTIC_V1,
            CORE_SEMANTIC_V2,
            CORE_CONVERSATIONAL_V1,
            CORE_CONVERSATIONAL_V2,
        }

    @property
    def is_semantic_v1(self) -> bool:
        return self.core_id == CORE_SEMANTIC_V1

    @property
    def is_semantic_v2(self) -> bool:
        return self.core_id == CORE_SEMANTIC_V2

    @property
    def is_conversational_v1(self) -> bool:
        return self.core_id == CORE_CONVERSATIONAL_V1

    @property
    def is_conversational_v2(self) -> bool:
        return self.core_id == CORE_CONVERSATIONAL_V2

    @property
    def is_conversational(self) -> bool:
        """GLM semantic owner + Kimi writer (v1 or v2)."""
        return self.core_id in {CORE_CONVERSATIONAL_V1, CORE_CONVERSATIONAL_V2}

    def to_dict(self) -> dict[str, str]:
        return {
            "conversation_core": self.core_id,
            "conversation_core_source": self.source,
        }


def normalize_core_id(value: object) -> str | None:
    raw = str(value or "").strip().lower()
    return raw if raw in CORE_IDS else None


def environment_core_id() -> str:
    raw = os.getenv(CORE_ENV_VAR, "").strip()
    if not raw:
        return CORE_LEGACY
    normalized = normalize_core_id(raw)
    if normalized is None:
        raise InvalidConversationCoreConfiguration(
            f"{CORE_ENV_VAR}={raw!r} is invalid; expected one of {', '.join(CORE_IDS)}"
        )
    return normalized


def _cache_seconds() -> float:
    try:
        return max(0.0, float(os.getenv("CONVERSATION_CORE_CACHE_SECONDS", "15") or 15))
    except (TypeError, ValueError):
        return 15.0


_creator_cache: dict[str, tuple[float, object]] = {}
_fan_cache: dict[str, tuple[float, object]] = {}


def clear_conversation_core_cache(
    creator_id: str | None = None,
    fan_id: str | None = None,
) -> None:
    if creator_id is None and fan_id is None:
        _creator_cache.clear()
        _fan_cache.clear()
        return
    if creator_id is not None:
        _creator_cache.pop(str(creator_id), None)
    if fan_id is not None:
        _fan_cache.pop(str(fan_id), None)


def _cached(cache: dict[str, tuple[float, object]], key: str) -> tuple[bool, object]:
    entry = cache.get(key)
    if entry and time.monotonic() - entry[0] < _cache_seconds():
        return True, entry[1]
    return False, None


def _remember(
    cache: dict[str, tuple[float, object]], key: str, value: object
) -> object:
    if _cache_seconds() > 0:
        cache[key] = (time.monotonic(), value)
    return value


def _missing_column(exc: Exception) -> bool:
    text = str(exc).lower()
    return "conversation_core" in text and (
        "42703" in text or "does not exist" in text or "could not find" in text
    )


async def _read(
    table: str,
    row_id: str,
    *,
    include_platform_id: bool = False,
    db: Any = None,
) -> dict | None:
    columns = "conversation_core"
    if include_platform_id:
        columns += ", platform_fan_id"

    def _get() -> dict | None:
        client = db or get_supabase()
        result = client.table(table).select(columns).eq("id", str(row_id)).execute()
        if isinstance(result.data, dict):
            return result.data
        rows = result.data or []
        return rows[0] if rows else None

    try:
        return await asyncio.to_thread(_get)
    except Exception as exc:
        if _missing_column(exc):
            return None
        raise


def _validated_stored(value: object, *, location: str) -> str | None:
    if value in (None, ""):
        return None
    normalized = normalize_core_id(value)
    if normalized is None:
        raise InvalidConversationCoreConfiguration(
            f"invalid conversation core {value!r} stored on {location}"
        )
    return normalized


async def creator_core_override(creator_id: str, *, db: Any = None) -> str | None:
    key = str(creator_id or "")
    if not key:
        return None
    hit, value = _cached(_creator_cache, key)
    if hit:
        return _validated_stored(value, location=f"creator {key}")
    row = await _read("creators", key, db=db)
    value = (row or {}).get("conversation_core")
    _remember(_creator_cache, key, value)
    return _validated_stored(value, location=f"creator {key}")


async def simulation_fan_core_override(fan_id: str, *, db: Any = None) -> str | None:
    key = str(fan_id or "")
    if not key:
        return None
    hit, value = _cached(_fan_cache, key)
    if hit:
        return _validated_stored(value, location=f"simulation fan {key}")
    row = await _read("fans", key, include_platform_id=True, db=db)
    if not row or not is_simulatable_fan(row.get("platform_fan_id")):
        _remember(_fan_cache, key, None)
        return None
    value = row.get("conversation_core")
    _remember(_fan_cache, key, value)
    return _validated_stored(value, location=f"simulation fan {key}")


async def resolve_conversation_core(
    *,
    creator_id: str | None,
    fan_id: str | None = None,
    platform_fan_id: object = _UNKNOWN,
    db: Any = None,
) -> ConversationCoreResolution:
    may_read_fan = platform_fan_id is _UNKNOWN or is_simulatable_fan(platform_fan_id)
    if fan_id and may_read_fan:
        override = await simulation_fan_core_override(str(fan_id), db=db)
        if override:
            return ConversationCoreResolution(override, SOURCE_SIMULATION_FAN)
    if creator_id:
        override = await creator_core_override(str(creator_id), db=db)
        if override:
            return ConversationCoreResolution(override, SOURCE_CREATOR)
    if os.getenv(CORE_ENV_VAR, "").strip():
        return ConversationCoreResolution(environment_core_id(), SOURCE_ENVIRONMENT)
    return ConversationCoreResolution(CORE_LEGACY, SOURCE_BUILTIN)


def log_effective_core(
    resolution: ConversationCoreResolution,
    *,
    creator_id: str | None,
    fan_id: str | None,
    trigger: str,
) -> None:
    print(
        f"[CONVERSATION CORE] trigger={trigger} creator={creator_id or '-'} "
        f"fan={fan_id or '-'} core={resolution.core_id} source={resolution.source}"
    )


async def set_creator_core_override(creator_id: str, core_id: str | None) -> str | None:
    value = normalize_core_id(core_id)
    if core_id not in (None, "") and value is None:
        raise ValueError(
            f"Unknown conversation core. Expected one of: {', '.join(CORE_IDS)}"
        )

    def _write() -> None:
        (
            get_supabase()
            .table("creators")
            .update({"conversation_core": value})
            .eq("id", str(creator_id))
            .execute()
        )

    await asyncio.to_thread(_write)
    clear_conversation_core_cache(creator_id=creator_id)
    return value


async def set_simulation_fan_core_override(
    fan_id: str, core_id: str | None
) -> str | None:
    value = normalize_core_id(core_id)
    if core_id not in (None, "") and value is None:
        raise ValueError(
            f"Unknown conversation core. Expected one of: {', '.join(CORE_IDS)}"
        )

    def _write() -> None:
        (
            get_supabase()
            .table("fans")
            .update({"conversation_core": value})
            .eq("id", str(fan_id))
            .execute()
        )

    await asyncio.to_thread(_write)
    clear_conversation_core_cache(fan_id=fan_id)
    return value
