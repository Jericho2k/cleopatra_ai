import io
from types import SimpleNamespace

import pytest
from PIL import Image

import main
from services.video_frames import (
    DEFAULT_FRAME_COUNT,
    FrameSettings,
    build_contact_sheet,
    frame_sample_offsets,
)


def jpeg_bytes(colour: tuple[int, int, int]) -> bytes:
    image = Image.new("RGB", (160, 240), colour)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def test_offsets_are_spread_across_the_clip_and_trim_the_edges():
    offsets = frame_sample_offsets(100.0, 4)
    assert len(offsets) == 4
    assert offsets == sorted(offsets)
    assert offsets[0] > 5.0
    assert offsets[-1] < 95.0


def test_unknown_duration_still_attempts_multiple_real_moments():
    assert frame_sample_offsets(0.0, 4) == [0.5, 2.0, 5.0, 10.0]
    assert frame_sample_offsets("not a number", 2) == [0.5, 2.0]


def test_short_clips_never_produce_duplicate_or_out_of_range_offsets():
    offsets = frame_sample_offsets(0.4, DEFAULT_FRAME_COUNT)
    assert len(offsets) == len(set(offsets))
    assert all(0.0 <= offset <= 0.4 for offset in offsets)


def test_contact_sheet_preserves_multiple_non_blank_frames():
    sheet, count = build_contact_sheet([
        jpeg_bytes((220, 40, 80)),
        jpeg_bytes((40, 180, 220)),
        jpeg_bytes((180, 120, 220)),
        jpeg_bytes((220, 180, 80)),
    ])
    image = Image.open(io.BytesIO(sheet))
    assert count == 4
    assert image.size == (896, 896)


def test_frame_settings_are_bounded(monkeypatch):
    monkeypatch.setenv("VIDEO_FRAME_COUNT", "999")
    monkeypatch.setenv("VIDEO_FRAME_TIMEOUT", "1")
    monkeypatch.setenv("VIDEO_FRAME_MAX_DIMENSION", "99999")
    settings = FrameSettings.from_env()
    assert settings.frame_count == 8
    assert settings.timeout_seconds == 10
    assert settings.max_dimension == 1440


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
