import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from services.vault_sync import (
    ordered_vault_albums,
    should_stop_album_scan,
    vault_sync_cooldown,
)

NOW = datetime(2026, 8, 10, 12, 0, tzinfo=timezone.utc)


def test_vault_sync_is_available_after_one_day():
    cooldown = vault_sync_cooldown(NOW - timedelta(hours=24), now=NOW)

    assert cooldown["allowed"] is True
    assert cooldown["hours_remaining"] == 0


def test_vault_sync_reports_hours_during_daily_cooldown():
    cooldown = vault_sync_cooldown(NOW - timedelta(hours=5), now=NOW)

    assert cooldown["allowed"] is False
    assert cooldown["hours_remaining"] == 19
    assert cooldown["days_remaining"] == 0.8


def test_custom_albums_are_scanned_before_overlapping_all_album():
    albums = [
        {"id": "all", "title": "All", "type": 38000, "itemCount": 100},
        {"id": "maid", "title": "Maid", "itemCount": 20},
        {"id": "bath", "title": "Bathroom", "itemCount": 30},
    ]

    ordered = ordered_vault_albums(albums)

    assert [album["id"] for album in ordered] == ["bath", "maid", "all"]


def test_confirmed_unchanged_latest_page_avoids_extra_api_pages():
    assert should_stop_album_scan(
        items=[{"id": "latest"}],
        next_cursor="older",
        is_first_page=True,
        last_item_id="latest",
        all_items_known=True,
        consecutive_known_batches=1,
    ) is True


def test_older_api_shape_keeps_conservative_duplicate_page_fallback():
    assert should_stop_album_scan(
        items=[{"id": "first"}],
        next_cursor="older",
        is_first_page=True,
        last_item_id=None,
        all_items_known=True,
        consecutive_known_batches=1,
    ) is False


def test_existing_media_scan_paginates_past_supabase_default(monkeypatch):
    import main

    ranges = []

    class Query:
        current_range = (0, 999)

        def table(self, _name):
            return self

        def select(self, _columns):
            return self

        def eq(self, _column, _value):
            return self

        def order(self, _column):
            return self

        def range(self, start, end):
            self.current_range = (start, end)
            ranges.append(self.current_range)
            return self

        def execute(self):
            if self.current_range == (0, 999):
                return SimpleNamespace(
                    data=[{"media_id": str(index)} for index in range(1000)]
                )
            return SimpleNamespace(data=[{"media_id": "1000"}])

    monkeypatch.setattr(main, "get_supabase", lambda: Query())

    media_ids = asyncio.run(main._vault_existing_media_ids("creator"))

    assert len(media_ids) == 1001
    assert ranges == [(0, 999), (1000, 1999)]


def test_sync_cannot_restart_while_new_media_is_being_categorized(monkeypatch):
    import main

    monkeypatch.setitem(
        main._vault_sync_state,
        "creator",
        {"status": "categorizing_new"},
    )

    result = asyncio.run(main.sync_vault_start("creator"))

    assert result == {"status": "already_running"}


def test_automatic_sync_backs_off_after_binding_access_denied(monkeypatch):
    import main

    monkeypatch.delitem(main._vault_sync_state, "creator-denied", raising=False)
    monkeypatch.setitem(
        main._vault_sync_retry_after,
        "creator-denied",
        main.time.time() + 3600,
    )

    result = asyncio.run(main.sync_vault_start("creator-denied"))

    assert result["status"] == "retry_backoff"
    assert result["requires_reconnect"] is True
    assert 3590 <= result["retry_after_seconds"] <= 3600


def test_operator_can_force_sync_during_binding_backoff(monkeypatch):
    import main

    monkeypatch.delitem(main._vault_sync_state, "creator-manual", raising=False)
    monkeypatch.setitem(
        main._vault_sync_retry_after,
        "creator-manual",
        main.time.time() + 3600,
    )

    def close_spawned_coroutine(coroutine, *, name):
        assert name == "run_vault_sync"
        coroutine.close()

    monkeypatch.setattr(main, "spawn", close_spawned_coroutine)

    result = asyncio.run(main.sync_vault_start("creator-manual", force=True))

    assert result == {"status": "started"}
    assert "creator-manual" not in main._vault_sync_retry_after


def test_autosync_checks_due_creators_before_its_first_sleep(monkeypatch):
    import main

    started = []

    class Query:
        def table(self, _name):
            return self

        def select(self, _columns):
            return self

        @property
        def not_(self):
            return self

        def is_(self, _column, _value):
            return self

        def execute(self):
            return SimpleNamespace(
                data=[
                    {
                        "id": "creator-due",
                        "last_vault_sync_at": (
                            datetime.now(timezone.utc) - timedelta(hours=25)
                        ).isoformat(),
                    },
                    {
                        "id": "creator-fresh",
                        "last_vault_sync_at": datetime.now(timezone.utc).isoformat(),
                    },
                ]
            )

    async def fake_start(creator_id):
        started.append(creator_id)
        return {"status": "started"}

    async def stop_at_first_sleep(_seconds):
        raise asyncio.CancelledError

    monkeypatch.setattr(main, "get_supabase", lambda: Query())
    monkeypatch.setattr(main, "sync_vault_start", fake_start)
    monkeypatch.setattr(main.asyncio, "sleep", stop_at_first_sleep)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(main.vault_autosync_scheduler())

    assert started == ["creator-due"]
