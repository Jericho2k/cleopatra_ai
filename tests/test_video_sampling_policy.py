"""Duration-aware sampling: how many frames, and where they land.

The old behaviour was four frames for every video. Four frames is generous for
a twenty-second tease and useless for a twelve-minute scene whose entire
commercial value is the progression through it.

Two properties are asserted for every representative duration:

* the COUNT scales at roughly one frame per minute, with a floor that keeps
  short clips useful and a hard ceiling that keeps a long one from becoming an
  unbounded VLM bill;
* the OFFSETS cover the whole clip chronologically, at segment midpoints, with
  no duplicates and nothing past the end.
"""
from __future__ import annotations

import io

import pytest
from PIL import Image

from services.video_frames import (
    DEFAULT_MAX_FRAME_COUNT,
    DEFAULT_MIN_FRAME_COUNT,
    MAX_FRAME_COUNT_CEILING,
    FrameSettings,
    build_contact_sheet,
    frame_sample_offsets,
    frames_for_duration,
)


def jpeg_bytes(colour: tuple[int, int, int]) -> bytes:
    image = Image.new("RGB", (160, 240), colour)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


MINUTE = 60.0

# The representative durations the sprint names, with the count each must
# produce under the default settings.
REPRESENTATIVE = [
    ("20 seconds", 20.0, 4),
    ("1 minute", 1 * MINUTE, 4),
    ("2 minutes", 2 * MINUTE, 4),
    ("5 minutes", 5 * MINUTE, 5),
    ("8 minutes", 8 * MINUTE, 8),
    ("15 minutes", 15 * MINUTE, 12),
    ("45 minutes", 45 * MINUTE, 12),
]


@pytest.mark.parametrize(
    "label,duration,expected",
    REPRESENTATIVE,
    ids=[row[0] for row in REPRESENTATIVE],
)
def test_frame_count_scales_with_duration(label, duration, expected):
    assert frames_for_duration(duration, settings=FrameSettings()) == expected, label


@pytest.mark.parametrize(
    "label,duration,expected",
    REPRESENTATIVE,
    ids=[row[0] for row in REPRESENTATIVE],
)
def test_offsets_cover_the_whole_clip_in_order(label, duration, expected):
    count = frames_for_duration(duration, settings=FrameSettings())
    offsets = frame_sample_offsets(duration, count)

    assert offsets == sorted(offsets), f"{label}: samples must be chronological"
    assert len(offsets) == len(set(offsets)), f"{label}: no duplicate offsets"
    assert all(0.0 <= offset < duration for offset in offsets), (
        f"{label}: no offset may fall outside the clip"
    )
    # Coverage: the first sample is inside the opening segment and the last is
    # inside the closing one, so no part of the clip is unrepresented.
    segment = duration / count
    assert offsets[0] < segment, f"{label}: the opening must be sampled"
    assert offsets[-1] > duration - segment, f"{label}: the ending must be sampled"


def test_a_short_clip_still_gets_minimum_useful_coverage():
    """Under a minute is 3-4 moments, not one poster frame."""
    for duration in (3.0, 10.0, 20.0, 45.0, 59.0):
        count = frames_for_duration(duration, settings=FrameSettings())
        assert count >= 3
        assert count == DEFAULT_MIN_FRAME_COUNT
        offsets = frame_sample_offsets(duration, count)
        assert len(offsets) == len(set(offsets))
        assert all(0.0 <= offset < duration for offset in offsets)


def test_a_thirty_minute_video_does_not_produce_thirty_frames():
    """The headline cost property. Thirty expensive VLM images is the bug."""
    assert frames_for_duration(30 * MINUTE, settings=FrameSettings()) == (
        DEFAULT_MAX_FRAME_COUNT
    )
    assert frames_for_duration(90 * MINUTE, settings=FrameSettings()) == (
        DEFAULT_MAX_FRAME_COUNT
    )
    assert frames_for_duration(8 * 60 * MINUTE, settings=FrameSettings()) == (
        DEFAULT_MAX_FRAME_COUNT
    )


def test_the_common_range_really_is_about_one_per_minute():
    settings = FrameSettings()
    for minutes in range(4, 13):
        assert frames_for_duration(minutes * MINUTE, settings=settings) == minutes


def test_the_eight_minute_example_lands_on_the_half_minutes():
    """The sprint's worked example: 0:30, 1:30, 2:30 ... 7:30."""
    count = frames_for_duration(8 * MINUTE, settings=FrameSettings())
    offsets = frame_sample_offsets(8 * MINUTE, count)

    assert offsets == [30.0, 90.0, 150.0, 210.0, 270.0, 330.0, 390.0, 450.0]


def test_samples_are_not_clustered_in_the_first_few_seconds():
    """The failure this replaces: describing a video from its intro."""
    duration = 10 * MINUTE
    offsets = frame_sample_offsets(
        duration, frames_for_duration(duration, settings=FrameSettings())
    )

    assert sum(1 for offset in offsets if offset < 60) <= 1
    assert max(offsets) > duration * 0.9


def test_an_unknown_duration_samples_conservatively():
    """ffprobe failing is not evidence that a clip is long.

    The floor, not the ceiling: guessing upward would spend the maximum on
    every asset whose metadata the CDN happens to withhold.
    """
    assert frames_for_duration(0.0, settings=FrameSettings()) == DEFAULT_MIN_FRAME_COUNT
    assert frames_for_duration(-5.0, settings=FrameSettings()) == DEFAULT_MIN_FRAME_COUNT
    assert frames_for_duration("nonsense", settings=FrameSettings()) == (
        DEFAULT_MIN_FRAME_COUNT
    )

    offsets = frame_sample_offsets(0.0, DEFAULT_MIN_FRAME_COUNT)
    assert offsets == [0.5, 2.0, 5.0, 10.0]
    assert len(offsets) == len(set(offsets))


def test_the_maximum_is_configurable_but_never_unbounded(monkeypatch):
    monkeypatch.setenv("VIDEO_FRAME_MAX_COUNT", "16")
    assert frames_for_duration(60 * MINUTE, settings=FrameSettings.from_env()) == 16

    monkeypatch.setenv("VIDEO_FRAME_MAX_COUNT", "9999")
    settings = FrameSettings.from_env()
    assert settings.max_frame_count == MAX_FRAME_COUNT_CEILING
    assert frames_for_duration(10 * 60 * MINUTE, settings=settings) == (
        MAX_FRAME_COUNT_CEILING
    )


def test_a_maximum_below_the_minimum_cannot_produce_an_empty_range(monkeypatch):
    monkeypatch.setenv("VIDEO_FRAME_MIN_COUNT", "6")
    monkeypatch.setenv("VIDEO_FRAME_MAX_COUNT", "2")
    settings = FrameSettings.from_env()

    assert settings.max_frame_count >= settings.min_frame_count
    assert frames_for_duration(30.0, settings=settings) >= 1


def test_the_rate_is_configurable(monkeypatch):
    monkeypatch.setenv("VIDEO_FRAMES_PER_MINUTE", "2")
    settings = FrameSettings.from_env()

    assert frames_for_duration(3 * MINUTE, settings=settings) == 6
    # ...and the ceiling still wins.
    assert frames_for_duration(60 * MINUTE, settings=settings) == (
        settings.max_frame_count
    )


def test_other_frame_settings_stay_bounded(monkeypatch):
    monkeypatch.setenv("VIDEO_FRAME_TIMEOUT", "1")
    monkeypatch.setenv("VIDEO_FRAME_MAX_DIMENSION", "99999")
    settings = FrameSettings.from_env()

    assert settings.timeout_seconds == 10
    assert settings.max_dimension == 1440


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
