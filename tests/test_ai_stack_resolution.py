"""Which brain answers this fan, and which one must never answer a real one.

Resolution is simulation-fan override, then creator override, then
AI_STACK_PROFILE. The one rule that matters more than the order: a value on a
fan row is honoured ONLY for a ``test_`` fan. The read path re-checks the prefix
rather than trusting the column, so a simulation setting cannot reach a paying
customer even if something else wrote it there.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from services import ai_stack
from services.ai_stack import (
    SOURCE_CREATOR,
    SOURCE_ENVIRONMENT,
    SOURCE_SIMULATION_FAN,
    clear_ai_stack_cache,
    resolve_ai_stack,
    set_creator_profile_override,
    set_simulation_fan_profile_override,
)


CREATORS = {
    "creator-plain": {"id": "creator-plain", "ai_stack_profile": None},
    "creator-legacy": {"id": "creator-legacy", "ai_stack_profile": "cleo_legacy_v1"},
    "creator-v2": {"id": "creator-v2", "ai_stack_profile": "cleo_v2"},
}

FANS = {
    "fan-test-legacy": {
        "id": "fan-test-legacy",
        "platform_fan_id": "test_a1",
        "ai_stack_profile": "cleo_legacy_v1",
    },
    "fan-test-v2": {
        "id": "fan-test-v2",
        "platform_fan_id": "test_b2",
        "ai_stack_profile": "cleo_v2",
    },
    "fan-test-plain": {
        "id": "fan-test-plain",
        "platform_fan_id": "test_c3",
        "ai_stack_profile": None,
    },
    # A REAL fan whose row somehow carries an override. It must have no effect.
    "fan-real": {
        "id": "fan-real",
        "platform_fan_id": "884422113355",
        "ai_stack_profile": "cleo_legacy_v1",
    },
}


class _Table:
    def __init__(self, rows: dict, writes: list) -> None:
        self._rows = rows
        self._writes = writes
        self._filters: list[tuple[str, str]] = []
        self._payload: dict | None = None

    def select(self, *_a, **_k):
        return self

    def update(self, payload):
        self._payload = payload
        return self

    def eq(self, column, value):
        self._filters.append((column, str(value)))
        return self

    def limit(self, *_a, **_k):
        return self

    def execute(self):
        matched = [
            row
            for row in self._rows.values()
            if all(str(row.get(column)) == value for column, value in self._filters)
        ]
        if self._payload is not None:
            for row in matched:
                row.update(self._payload)
            self._writes.append((dict(self._payload), [row["id"] for row in matched]))
        return SimpleNamespace(data=[dict(row) for row in matched])


class _DB:
    def __init__(self) -> None:
        self.creators = {key: dict(value) for key, value in CREATORS.items()}
        self.fans = {key: dict(value) for key, value in FANS.items()}
        self.writes: list = []

    def table(self, name):
        rows = self.creators if name == "creators" else self.fans
        return _Table(rows, self.writes)


@pytest.fixture
def db(monkeypatch):
    monkeypatch.setenv("AI_STACK_PROFILE", "cleo_v2")
    # The cache is a per-process TTL; a test that wrote an override and then
    # read a stale cached value would be testing the cache, not resolution.
    monkeypatch.setenv("AI_STACK_CACHE_SECONDS", "0")
    clear_ai_stack_cache()
    store = _DB()
    monkeypatch.setattr(ai_stack, "get_supabase", lambda: store)
    return store


def _resolve(**kwargs):
    return asyncio.run(resolve_ai_stack(**kwargs))


def test_environment_answers_when_nothing_overrides_it(db):
    result = _resolve(creator_id="creator-plain", fan_id="fan-test-plain")

    assert result.profile_id == "cleo_v2"
    assert result.source == SOURCE_ENVIRONMENT


def test_a_creator_override_beats_the_production_default(db):
    result = _resolve(creator_id="creator-legacy", fan_id="fan-test-plain")

    assert result.profile_id == "cleo_legacy_v1"
    assert result.source == SOURCE_CREATOR


def test_a_test_fans_own_override_beats_the_creator(db):
    """Two test fans under one creator, compared turn for turn."""
    legacy = _resolve(creator_id="creator-v2", fan_id="fan-test-legacy")
    v2 = _resolve(creator_id="creator-legacy", fan_id="fan-test-v2")

    assert (legacy.profile_id, legacy.source) == ("cleo_legacy_v1", SOURCE_SIMULATION_FAN)
    assert (v2.profile_id, v2.source) == ("cleo_v2", SOURCE_SIMULATION_FAN)


def test_a_simulation_override_on_a_real_fan_is_ignored(db):
    """The boundary that matters. fan-real carries cleo_legacy_v1; the creator
    has no override and the deployment default is cleo_v2."""
    result = _resolve(creator_id="creator-plain", fan_id="fan-real")

    assert result.profile_id == "cleo_v2"
    assert result.source == SOURCE_ENVIRONMENT


def test_a_real_fan_still_follows_its_creators_override(db):
    result = _resolve(creator_id="creator-legacy", fan_id="fan-real")

    assert result.profile_id == "cleo_legacy_v1"
    assert result.source == SOURCE_CREATOR


def test_a_garbage_override_in_the_database_is_ignored(db):
    db.creators["creator-plain"]["ai_stack_profile"] = "gpt-9-ultra"
    clear_ai_stack_cache()

    result = _resolve(creator_id="creator-plain")

    assert result.profile_id == "cleo_v2"
    assert result.source == SOURCE_ENVIRONMENT


def test_a_read_failure_never_loses_the_turn(db, monkeypatch):
    """A missing ai_stack_profile column (migration not applied yet) must not
    stop a fan being answered."""

    class _Broken:
        def table(self, _name):
            raise RuntimeError(
                'column fans.ai_stack_profile does not exist (code 42703)'
            )

    monkeypatch.setattr(ai_stack, "get_supabase", _Broken)
    clear_ai_stack_cache()

    result = _resolve(creator_id="creator-legacy", fan_id="fan-test-legacy")

    assert result.profile_id == "cleo_v2"
    assert result.source == SOURCE_ENVIRONMENT


# --- writes -----------------------------------------------------------------


def test_writing_a_creator_override_persists_and_takes_effect(db):
    asyncio.run(set_creator_profile_override("creator-plain", "cleo_legacy_v1"))

    assert db.creators["creator-plain"]["ai_stack_profile"] == "cleo_legacy_v1"
    assert _resolve(creator_id="creator-plain").profile_id == "cleo_legacy_v1"


def test_clearing_a_creator_override_returns_it_to_the_default(db):
    asyncio.run(set_creator_profile_override("creator-legacy", None))

    assert db.creators["creator-legacy"]["ai_stack_profile"] is None
    assert _resolve(creator_id="creator-legacy").profile_id == "cleo_v2"


@pytest.mark.parametrize("value", ["cleo_v99", "moonshotai/kimi-k2.6", "openrouter"])
def test_an_arbitrary_profile_string_is_refused_by_the_writers(db, value):
    with pytest.raises(ValueError):
        asyncio.run(set_creator_profile_override("creator-plain", value))
    with pytest.raises(ValueError):
        asyncio.run(set_simulation_fan_profile_override("fan-test-plain", value))
    assert db.writes == []


def test_a_simulation_fan_write_cannot_change_how_a_real_fan_is_answered(db):
    """Even after writing onto the real fan's row directly, the read path
    refuses to honour it, because it re-checks the test_ prefix."""
    asyncio.run(set_simulation_fan_profile_override("fan-real", "cleo_legacy_v1"))

    assert _resolve(creator_id="creator-plain", fan_id="fan-real").profile_id == "cleo_v2"


# --- the read a real fan never has to pay for -------------------------------


def test_a_known_real_fan_costs_no_override_read(db, monkeypatch):
    """A fan-level override exists only for the simulator, so for a fan the
    caller already knows is real there is nothing to look up. This is an
    optimisation on the reply path, not a second boundary."""
    reads: list[str] = []
    original = ai_stack._read_override

    async def counting(table, row_id, extra_columns=""):
        reads.append(table)
        return await original(table, row_id, extra_columns)

    monkeypatch.setattr(ai_stack, "_read_override", counting)

    result = _resolve(
        creator_id="creator-legacy",
        fan_id="fan-real",
        platform_fan_id="884422113355",
    )

    assert reads == ["creators"]
    assert result.profile_id == "cleo_legacy_v1"


def test_a_known_test_fan_still_gets_its_override(db, monkeypatch):
    result = _resolve(
        creator_id="creator-v2",
        fan_id="fan-test-legacy",
        platform_fan_id="test_a1",
    )

    assert result.profile_id == "cleo_legacy_v1"
    assert result.source == SOURCE_SIMULATION_FAN


def test_omitting_the_platform_id_is_always_safe(db):
    """The read path checks the prefix itself, so a caller that does not know
    loses nothing but the saved read."""
    told = _resolve(
        creator_id="creator-v2", fan_id="fan-test-legacy", platform_fan_id="test_a1"
    )
    untold = _resolve(creator_id="creator-v2", fan_id="fan-test-legacy")

    assert told.profile_id == untold.profile_id == "cleo_legacy_v1"


def test_a_lying_platform_id_cannot_grant_an_override(db):
    """Claiming a real fan is a test fan does not help: the read re-checks the
    prefix against the row, not against what the caller said."""
    result = _resolve(
        creator_id="creator-plain", fan_id="fan-real", platform_fan_id="test_pretend"
    )

    assert result.profile_id == "cleo_v2"
    assert result.source == SOURCE_ENVIRONMENT
