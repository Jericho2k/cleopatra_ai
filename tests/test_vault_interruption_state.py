"""VAULT-003 — an interrupted vault sync says so instead of saying "idle".

Recovery was already correct and is unchanged: an interrupted sync never stamps
last_vault_sync_at, so the cooldown never starts and the autosync scheduler
picks the creator up again. What was wrong was the operator-facing state. Run
state lived only in a process dict, so after a restart the status endpoint
answered "idle" — the same answer it gives for a creator nobody has ever asked
to sync. An operator watching a large import saw the progress vanish with no way
to tell whether it finished or was cut off.

The distinguishing fact is which process owns the run, so that is what the
database now stores. These tests model a restart the way one actually happens:
the rows survive, the process identity does not.
"""

from __future__ import annotations

import asyncio

import pytest

import main
from tests.fake_supabase import FakeSupabase

CREATOR_ID = "creator-1"


@pytest.fixture
def db(monkeypatch):
    fake = FakeSupabase({
        "creators": [{
            "id": CREATOR_ID,
            "apifansly_account_id": "acct-1",
            "vault_sync_started_at": None,
            "vault_sync_finished_at": None,
            "vault_sync_owner": None,
        }],
    })
    monkeypatch.setattr(main, "get_supabase", lambda: fake)
    main._vault_sync_state.pop(CREATOR_ID, None)
    yield fake
    main._vault_sync_state.pop(CREATOR_ID, None)


def _status():
    return asyncio.run(main.sync_vault_status(CREATOR_ID))


def _restart(monkeypatch, new_process_id="a-different-process"):
    """A deploy: the rows stay, the process identity changes, memory is gone."""
    monkeypatch.setattr(main, "_PROCESS_ID", new_process_id)
    main._vault_sync_state.pop(CREATOR_ID, None)


def test_a_creator_that_never_synced_is_idle(db) -> None:
    assert _status()["status"] == "idle"


def test_a_run_owned_by_this_process_reports_from_memory(db) -> None:
    asyncio.run(main._mark_vault_run_started(CREATOR_ID))
    main._vault_sync_state[CREATOR_ID] = {
        "status": "running", "synced": 3, "total": 10, "album": "Beach",
    }

    state = _status()

    assert state["status"] == "running"
    assert state["synced"] == 3


def test_an_interrupted_run_is_reported_as_interrupted(db, monkeypatch) -> None:
    """The finding. This used to answer "idle"."""
    asyncio.run(main._mark_vault_run_started(CREATOR_ID))
    main._vault_sync_state[CREATOR_ID] = {"status": "running"}

    _restart(monkeypatch)
    state = _status()

    assert state["status"] == "interrupted"
    assert state["recoverable"] is True
    assert state["interrupted_at"]


def test_an_interrupted_run_is_not_reported_as_failed(db, monkeypatch) -> None:
    """A run cut off by a deploy has not failed, and saying so would invite an
    operator to intervene where nothing is wrong."""
    asyncio.run(main._mark_vault_run_started(CREATOR_ID))
    _restart(monkeypatch)

    state = _status()

    assert state["status"] != "error"
    assert state["status"] != "failed"
    assert state["recoverable"] is True


def test_a_completed_run_is_idle_after_a_restart(db, monkeypatch) -> None:
    """The marker must be released, or a finished sync would report interrupted
    forever."""
    asyncio.run(main._mark_vault_run_started(CREATOR_ID))
    asyncio.run(main._mark_vault_run_finished(CREATOR_ID))

    _restart(monkeypatch)

    assert _status()["status"] == "idle"


def test_a_failed_run_is_also_idle_after_a_restart(db, monkeypatch) -> None:
    """A run that errored has ended. The error is reported separately; what must
    not persist is the in-flight marker."""
    asyncio.run(main._mark_vault_run_started(CREATOR_ID))
    asyncio.run(main._mark_vault_run_finished(CREATOR_ID))
    _restart(monkeypatch)

    assert _status()["status"] == "idle"


def test_a_new_run_clears_a_previous_interruption(db, monkeypatch) -> None:
    """Nothing for an operator to clear by hand: starting again is what resets
    it."""
    asyncio.run(main._mark_vault_run_started(CREATOR_ID))
    _restart(monkeypatch)
    assert _status()["status"] == "interrupted"

    asyncio.run(main._mark_vault_run_started(CREATOR_ID))
    main._vault_sync_state.pop(CREATOR_ID, None)

    assert _status()["status"] == "idle"


def test_a_run_is_released_even_when_the_sync_raises(db, monkeypatch) -> None:
    """_run_vault_sync releases in a finally, so a crash does not leave the
    creator looking permanently interrupted."""

    async def boom(_creator_id, _db):
        raise RuntimeError("vault import exploded")

    monkeypatch.setattr(main, "_run_vault_sync_locked", boom)

    with pytest.raises(RuntimeError):
        asyncio.run(main._run_vault_sync(CREATOR_ID))

    _restart(monkeypatch)
    assert _status()["status"] == "idle"


def test_missing_columns_fall_back_to_idle(db, monkeypatch) -> None:
    """A deployment that has not applied the migration must keep working; only
    the interrupted/idle distinction is unavailable."""

    def explode(*_args, **_kwargs):
        raise RuntimeError('column "vault_sync_owner" does not exist')

    monkeypatch.setattr(db, "table", explode)

    assert _status()["status"] == "idle"
