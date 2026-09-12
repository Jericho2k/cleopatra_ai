"""Which AI Stack Profile answers this fan, and why.

Resolution order, most specific first:

1. **Simulation fan override** — ``fans.ai_stack_profile``, honoured only for a
   fan whose ``platform_fan_id`` starts with ``test_``. This is what lets the
   owner keep "Test Fan A -> cleo_legacy_v1" and "Test Fan B -> cleo_v2" under
   one creator and compare them turn for turn. A value on a real fan row is
   ignored outright, so a simulation setting can never reach a paying customer.
2. **Creator override** — ``creators.ai_stack_profile``. Persistent and
   creator-scoped rather than per-session, because Full Auto answers
   asynchronously from a worker where no browser session exists.
3. **Deployment default** — ``AI_STACK_PROFILE``.
4. **Built-in default** — ``cleo_legacy_v1`` (see ai/stack_profiles).

Reads are cached for a few seconds. An operator flipping a creator between
profiles is a rare event; an inbound message is not, and this resolution sits on
the reply path.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass

from ai.stack_profiles import (
    AIStackProfile,
    PROFILE_IDS,
    environment_profile_id,
    get_profile,
    normalize_profile_id,
)
from core.simulation import is_simulatable_fan
from core.supabase import get_supabase


# Sentinel for "the caller did not say what this fan's platform id is", which is
# different from "the fan has none". Only the first means we have to go and look.
_UNKNOWN = object()

SOURCE_SIMULATION_FAN = "simulation_fan"
SOURCE_CREATOR = "creator"
SOURCE_ENVIRONMENT = "environment"


@dataclass(frozen=True)
class ProfileResolution:
    profile_id: str
    source: str

    @property
    def profile(self) -> AIStackProfile:
        return get_profile(self.profile_id)

    def to_dict(self) -> dict:
        return {"ai_stack_profile": self.profile_id, "ai_stack_source": self.source}


def _may_have_fan_override(platform_fan_id: object) -> bool:
    """Whether a fan-level override could possibly apply, without reading."""
    if platform_fan_id is _UNKNOWN:
        return True
    return is_simulatable_fan(platform_fan_id)


def _cache_ttl_seconds() -> float:
    try:
        return max(0.0, float(os.getenv("AI_STACK_CACHE_SECONDS", "15") or 15))
    except (TypeError, ValueError):
        return 15.0


_creator_cache: dict[str, tuple[float, str | None]] = {}
_fan_cache: dict[str, tuple[float, str | None]] = {}


def clear_ai_stack_cache(creator_id: str | None = None, fan_id: str | None = None) -> None:
    """Drop cached overrides after an operator writes one."""
    if creator_id is None and fan_id is None:
        _creator_cache.clear()
        _fan_cache.clear()
        return
    if creator_id is not None:
        _creator_cache.pop(str(creator_id), None)
    if fan_id is not None:
        _fan_cache.pop(str(fan_id), None)


def _remember(cache: dict[str, tuple[float, str | None]], key: str, value: str | None) -> str | None:
    ttl = _cache_ttl_seconds()
    if ttl > 0:
        cache[key] = (time.monotonic(), value)
    return value


def _cached(cache: dict[str, tuple[float, str | None]], key: str) -> tuple[bool, str | None]:
    entry = cache.get(key)
    if entry and (time.monotonic() - entry[0]) < _cache_ttl_seconds():
        return True, entry[1]
    return False, None


def _missing_column(error: Exception) -> bool:
    """Tolerate a deployment that has not applied db/ai_stack_profile_v1.sql yet.

    PostgREST answers an unknown column with 42703 and fails the whole read. A
    backend that ships ahead of its migration must still answer fans; it simply
    has no override to honour, which is true by construction.
    """
    text = str(error).lower()
    return "ai_stack_profile" in text and (
        "42703" in text or "does not exist" in text or "could not find" in text
    )


async def _read_override(table: str, row_id: str, extra_columns: str = "") -> dict | None:
    columns = "ai_stack_profile" + (f", {extra_columns}" if extra_columns else "")

    def _get() -> dict | None:
        response = (
            get_supabase().table(table)
            .select(columns)
            .eq("id", row_id)
            .limit(1)
            .execute()
        )
        rows = response.data or []
        return rows[0] if rows else None

    try:
        return await asyncio.to_thread(_get)
    except Exception as exc:
        if not _missing_column(exc):
            print(f"[AI STACK] override read failed {table}={row_id}: {exc}")
        return None


async def creator_profile_override(creator_id: str) -> str | None:
    key = str(creator_id or "")
    if not key:
        return None
    hit, value = _cached(_creator_cache, key)
    if hit:
        return value
    row = await _read_override("creators", key)
    return _remember(_creator_cache, key, normalize_profile_id((row or {}).get("ai_stack_profile")))


async def simulation_fan_profile_override(fan_id: str) -> str | None:
    """A test fan's own override. Returns None for every real fan."""
    key = str(fan_id or "")
    if not key:
        return None
    hit, value = _cached(_fan_cache, key)
    if hit:
        return value
    row = await _read_override("fans", key, extra_columns="platform_fan_id")
    if not row:
        return _remember(_fan_cache, key, None)
    # The test-fan boundary, applied here and not at the call site: a fan row
    # carrying an override must not be able to use it merely by existing.
    if not is_simulatable_fan(row.get("platform_fan_id")):
        return _remember(_fan_cache, key, None)
    return _remember(_fan_cache, key, normalize_profile_id(row.get("ai_stack_profile")))


async def resolve_ai_stack(
    *,
    creator_id: str | None,
    fan_id: str | None = None,
    platform_fan_id: object = _UNKNOWN,
) -> ProfileResolution:
    """The effective profile for one turn.

    ``platform_fan_id`` is an optimisation, not a second boundary. A caller that
    has already loaded the fan can pass it so a real fan costs no read at all:
    the fan-level override exists only for the simulator, so there is nothing to
    look up for a fan that is not a ``test_`` fan. Omitting it is always safe —
    the read path checks the prefix itself either way.
    """
    if fan_id and _may_have_fan_override(platform_fan_id):
        fan_override = await simulation_fan_profile_override(str(fan_id))
        if fan_override:
            return ProfileResolution(fan_override, SOURCE_SIMULATION_FAN)
    if creator_id:
        creator_override = await creator_profile_override(str(creator_id))
        if creator_override:
            return ProfileResolution(creator_override, SOURCE_CREATOR)
    return ProfileResolution(environment_profile_id(), SOURCE_ENVIRONMENT)


def log_effective_stack(
    resolution: ProfileResolution,
    *,
    creator_id: str | None,
    fan_id: str | None,
    feature: str,
) -> None:
    """One line in Railway naming the brain that answered."""
    print(
        f"[AI STACK] feature={feature} creator={creator_id or '-'} fan={fan_id or '-'} "
        f"profile={resolution.profile_id} source={resolution.source}"
    )


async def set_creator_profile_override(creator_id: str, profile_id: str | None) -> str | None:
    """Persist (or clear) a creator's override. Returns what was stored."""
    value = normalize_profile_id(profile_id)
    if profile_id not in (None, "") and value is None:
        raise ValueError(
            f"Unknown AI stack profile. Expected one of: {', '.join(PROFILE_IDS)}"
        )

    def _write() -> None:
        (
            get_supabase().table("creators")
            .update({"ai_stack_profile": value})
            .eq("id", str(creator_id))
            .execute()
        )

    await asyncio.to_thread(_write)
    clear_ai_stack_cache(creator_id=creator_id)
    return value


async def set_simulation_fan_profile_override(
    fan_id: str,
    profile_id: str | None,
) -> str | None:
    """Persist (or clear) a TEST fan's override.

    The caller has already established that this fan belongs to the creator and
    is simulatable; the read path re-checks the ``test_`` prefix independently,
    so even a row written by some other means cannot take effect on a real fan.
    """
    value = normalize_profile_id(profile_id)
    if profile_id not in (None, "") and value is None:
        raise ValueError(
            f"Unknown AI stack profile. Expected one of: {', '.join(PROFILE_IDS)}"
        )

    def _write() -> None:
        (
            get_supabase().table("fans")
            .update({"ai_stack_profile": value})
            .eq("id", str(fan_id))
            .execute()
        )

    await asyncio.to_thread(_write)
    clear_ai_stack_cache(fan_id=fan_id)
    return value
