import asyncio

import pytest
from fastapi import HTTPException

import main


@pytest.fixture(autouse=True)
def reset_denied_bindings():
    main._chat_reconcile_denied_bindings.clear()
    yield
    main._chat_reconcile_denied_bindings.clear()


def test_stale_account_binding_does_not_stop_other_creators(monkeypatch):
    calls = []

    async def fake_sync(creator_id, incremental=False):
        calls.append((creator_id, incremental))
        if creator_id == "stale":
            raise HTTPException(status_code=409, detail="access denied")
        return {"status": "ok", "new_chats": 2, "synced": 3}

    monkeypatch.setattr(main, "sync_chats", fake_sync)
    creators = [
        {"id": "stale", "apifansly_account_id": "fansly-old"},
        {"id": "healthy", "apifansly_account_id": "fansly-good"},
    ]
    result = asyncio.run(main._reconcile_chat_creators_once(creators))

    assert calls == [("stale", True), ("healthy", True)]
    assert result == {
        "processed": 1,
        "skipped_inaccessible": 0,
        "failed": 1,
    }
    assert ("stale", "fansly-old") in main._chat_reconcile_denied_bindings


def test_denied_binding_is_skipped_until_reconnect_changes_account_id(
    monkeypatch,
):
    calls = []

    async def fake_sync(creator_id, incremental=False):
        calls.append((creator_id, incremental))
        return {"status": "ok", "new_chats": 0, "synced": 0}

    monkeypatch.setattr(main, "sync_chats", fake_sync)
    main._chat_reconcile_denied_bindings.add(("creator", "fansly-old"))

    skipped = asyncio.run(main._reconcile_chat_creators_once([
        {"id": "creator", "apifansly_account_id": "fansly-old"},
    ]))
    assert calls == []
    assert skipped["skipped_inaccessible"] == 1

    resumed = asyncio.run(main._reconcile_chat_creators_once([
        {"id": "creator", "apifansly_account_id": "fansly-new"},
    ]))
    assert calls == [("creator", True)]
    assert resumed["processed"] == 1
    assert ("creator", "fansly-old") not in main._chat_reconcile_denied_bindings
