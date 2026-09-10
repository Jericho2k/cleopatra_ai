"""API-001 end to end: what a deploy costs in provider calls.

Before this sprint the reconciliation checkpoint lived in a process dict. A
Railway restart emptied it, every known chat looked unreconciled, and the next
pass issued one list_chat_messages call per chat. At the beta target of 20
creators x 2,000 chats that is ~40,000 calls per deploy — the single largest
API Fansly cost driver identified in Sprint 3.

The checkpoint is now `fans.chat_last_message_id`, so it survives the restart.
These tests drive the real sync_chats against a PostgREST double and COUNT the
message-list calls, because that count is the finding. A test that only asserted
"the column is written" would not notice a regression that wrote it and then
ignored it.

A restart is modelled as: same database rows, brand new process state. That is
what a Railway deploy is, and after this change there is no process state left
for the decision to depend on.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

import main
from tests.fake_supabase import FakeSupabase

CREATOR_ID = "creator-1"
ACCOUNT_ID = "apifansly-account-1"
CREATOR_PLATFORM_ID = "creator-platform-1"
PAGE_SIZE = 50


def _chat(index: int, *, message_index: int | None = None) -> dict:
    return {
        "partnerAccountId": f"p-{index:06d}",
        "groupId": f"g-{index:06d}",
        "partnerUsername": f"Fan {index}",
        "lastMessageId": f"m-{index:06d}-{index if message_index is None else message_index}",
    }


def _account(index: int) -> dict:
    return {"id": f"p-{index:06d}", "displayName": f"Fan {index}"}


@pytest.fixture
def env(monkeypatch):
    """sync_chats wired to a fake database, chat list, and message importer.

    `state["message_syncs"]` is the number the finding is about.
    """
    state: dict = {
        "chats": [],
        "message_syncs": [],
        "fail_next_sync_for": set(),
    }

    db = FakeSupabase({
        "creators": [{
            "id": CREATOR_ID,
            "apifansly_account_id": ACCOUNT_ID,
            "fansly_account_id": CREATOR_PLATFORM_ID,
            "auto_mode": False,
        }],
        "fans": [],
    })
    db.rpc = lambda *_args, **_kwargs: SimpleNamespace(
        execute=lambda: SimpleNamespace(data=True)
    )
    state["db"] = db
    monkeypatch.setattr(main, "get_supabase", lambda: db)

    async def fake_list_chats(account_id, *, cursor=None, client=None, **_kwargs):
        start = int(cursor or 0)
        page = state["chats"][start:start + PAGE_SIZE]
        next_cursor = (
            str(start + PAGE_SIZE)
            if start + PAGE_SIZE < len(state["chats"])
            else None
        )
        indexes = [int(chat["partnerAccountId"].split("-")[1]) for chat in page]
        return page, [_account(index) for index in indexes], next_cursor

    monkeypatch.setattr(main, "apifansly_list_chats", fake_list_chats)

    created: list[str] = []

    async def fake_create_fan(creator_id, platform_fan_id, display_name):
        created.append(platform_fan_id)
        row = {
            "id": f"fan-{platform_fan_id}",
            "creator_id": creator_id,
            "platform_fan_id": platform_fan_id,
            "display_name": display_name,
            "fansly_group_id": None,
            "avatar_url": None,
            "chat_last_message_id": None,
            "chat_last_synced_at": None,
        }
        db.tables["fans"].append(row)
        return SimpleNamespace(id=row["id"])

    monkeypatch.setattr(main, "create_fan", fake_create_fan)
    state["created"] = created

    async def fake_sync_recent(*, group_id, **_kwargs):
        state["message_syncs"].append(group_id)
        if group_id in state["fail_next_sync_for"]:
            state["fail_next_sync_for"].discard(group_id)
            raise RuntimeError("transient API Fansly failure")
        return {"imported": 0, "inbound": 0, "media_updated": 0}

    monkeypatch.setattr(main, "_sync_recent_fan_messages", fake_sync_recent)

    async def noop(*_args, **_kwargs):
        return None

    monkeypatch.setattr(main, "_stamp_vault_op", noop)
    monkeypatch.setattr(main, "_sync_fansly_lists_if_due", noop)

    return state


def _pass(state) -> dict:
    """One incremental reconciliation pass."""
    state["message_syncs"].clear()
    return asyncio.run(main.sync_chats(CREATOR_ID, incremental=True, force=True))


def _restart(monkeypatch) -> None:
    """Model a deploy: throw away every piece of process state.

    The assertion this enables is that the call count after it is unchanged.
    Nothing has to be cleared for the checkpoint any more — that is the fix —
    but the reconcile scheduler's own dicts are cleared so the test cannot
    accidentally pass because of leftover process state.
    """
    main._chat_reconcile_due_at.clear()
    main._chat_reconcile_denied_bindings.clear()


# --------------------------------------------------------------------------
# The finding
# --------------------------------------------------------------------------


def test_first_sync_reconciles_every_chat(env) -> None:
    env["chats"] = [_chat(i) for i in range(20)]

    _pass(env)

    assert len(env["message_syncs"]) == 20
    assert len(env["created"]) == 20


def test_second_unchanged_pass_costs_no_message_calls(env) -> None:
    env["chats"] = [_chat(i) for i in range(20)]
    _pass(env)

    _pass(env)

    assert env["message_syncs"] == []


def test_restart_does_not_cold_resync_known_chats(env, monkeypatch) -> None:
    """The headline assertion: ZERO message-list calls after a deploy."""
    env["chats"] = [_chat(i) for i in range(20)]
    _pass(env)

    _restart(monkeypatch)
    _pass(env)

    assert env["message_syncs"] == []


def test_checkpoint_is_persisted_on_the_fan_row(env) -> None:
    env["chats"] = [_chat(0)]

    _pass(env)

    fan = env["db"].tables["fans"][0]
    assert fan["chat_last_message_id"] == env["chats"][0]["lastMessageId"]
    assert fan["chat_last_synced_at"]


def test_changed_marker_after_restart_reconciles_only_that_chat(
    env, monkeypatch
) -> None:
    env["chats"] = [_chat(i) for i in range(20)]
    _pass(env)
    _restart(monkeypatch)

    env["chats"][7] = _chat(7, message_index=999)
    _pass(env)

    assert env["message_syncs"] == ["g-000007"]


def test_new_chat_after_restart_is_imported(env, monkeypatch) -> None:
    env["chats"] = [_chat(i) for i in range(5)]
    _pass(env)
    _restart(monkeypatch)

    env["chats"].append(_chat(5))
    _pass(env)

    assert env["message_syncs"] == ["g-000005"]
    assert "p-000005" in env["created"]


def test_moved_group_binding_reconciles_the_new_conversation(env) -> None:
    env["chats"] = [_chat(0)]
    _pass(env)

    # Same fan, rebound to a different chat whose newest message happens to
    # carry the id we already checkpointed. Without the binding check the stale
    # marker would match and the new conversation would never be read.
    env["chats"] = [{
        "partnerAccountId": "p-000000",
        "groupId": "g-rebound",
        "partnerUsername": "Fan 0",
        "lastMessageId": "m-000000-0",
    }]
    _pass(env)

    assert env["message_syncs"] == ["g-rebound"]


def test_a_failed_sync_does_not_checkpoint_and_retries_next_pass(env) -> None:
    """Checkpoint persistence failure must not cause message loss."""
    env["chats"] = [_chat(0)]
    env["fail_next_sync_for"] = {"g-000000"}

    _pass(env)
    assert env["message_syncs"] == ["g-000000"]
    assert env["db"].tables["fans"][0]["chat_last_message_id"] is None

    _pass(env)
    assert env["message_syncs"] == ["g-000000"]


def test_missed_webhook_is_still_recovered_by_the_marker_moving(env) -> None:
    """Reconciliation is the safety net under a dropped webhook: the platform
    marker moves whether or not we were told about it."""
    env["chats"] = [_chat(0)]
    _pass(env)

    env["chats"] = [_chat(0, message_index=1)]  # a message we never received
    _pass(env)

    assert env["message_syncs"] == ["g-000000"]


# --------------------------------------------------------------------------
# Scale — the number in the report
# --------------------------------------------------------------------------


def test_two_thousand_chats_restart_costs_zero_message_calls(
    env, monkeypatch
) -> None:
    env["chats"] = [_chat(i) for i in range(2000)]

    _pass(env)
    assert len(env["message_syncs"]) == 2000, "cold start reconciles everything"

    _restart(monkeypatch)
    _pass(env)

    # BEFORE this change this pass cost 2,000 calls for this creator alone,
    # and ~40,000 across a 20-creator deployment.
    assert env["message_syncs"] == []


def test_twenty_creator_restart_model(env, monkeypatch) -> None:
    """The deployment-level number, modelled as 20 x 2,000 chats.

    Driving twenty real creators through sync_chats would be twenty times the
    same assertion; the per-creator count is what multiplies, so it is measured
    once and multiplied explicitly here.
    """
    creators = 20
    chats_per_creator = 2000

    env["chats"] = [_chat(i) for i in range(chats_per_creator)]
    _pass(env)
    _restart(monkeypatch)
    _pass(env)

    per_creator_after_restart = len(env["message_syncs"])
    deployment_after_restart = per_creator_after_restart * creators
    deployment_before_restart = chats_per_creator * creators

    assert deployment_before_restart == 40_000
    assert deployment_after_restart == 0
