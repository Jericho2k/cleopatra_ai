"""Which vault rows an explicit video re-analysis targets.

The sampling curve itself — how many frames a duration deserves and where they
land — lives in tests/test_video_sampling_policy.py, which replaced the
fixed-four-frame assertions this file used to carry.
"""
from types import SimpleNamespace

import pytest

import main


@pytest.mark.asyncio
async def test_video_upgrade_scope_targets_only_frame_pending_rows(
    monkeypatch,
):
    class Query:
        def select(self, *_args):
            return self

        def eq(self, *_args):
            return self

        def single(self):
            return self

        def execute(self):
            return SimpleNamespace(
                data={"vault_initial_categorized_at": "2026-01-01T00:00:00Z"}
            )

    class Database:
        def table(self, name):
            assert name == "creators"
            return Query()

    captured = {}

    async def video_ids(_creator_id):
        return ["video-row-1", "video-row-2"]

    async def stamp(*_args):
        return None

    def spawn(coroutine, *, name):
        captured["name"] = name
        coroutine.close()

    monkeypatch.setattr(main, "get_supabase", lambda: Database())
    monkeypatch.setattr(main, "_video_frame_upgrade_media_ids", video_ids)
    monkeypatch.setattr(main, "_stamp_vault_op", stamp)
    monkeypatch.setattr(main, "spawn", spawn)
    main._categorize_state.pop("creator-1", None)

    result = await main.categorize_vault(
        "creator-1",
        mode="upgrade",
        confirm_upgrade=True,
        upgrade_scope="videos",
    )

    assert result["status"] == "started"
    assert result["items"] == 2
    assert result["upgrade_scope"] == "videos"
    assert captured["name"] == "run_vault_categorization:upgrade"
    main._categorize_state.pop("creator-1", None)
