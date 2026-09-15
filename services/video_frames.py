"""Duration-aware chronological keyframe sampling for vault classification.

Why this is not a fixed frame count
-----------------------------------
It used to be four frames, whatever the clip. Four frames is generous for a
twenty-second tease and close to useless for a twelve-minute scene, where the
whole commercial value is the PROGRESSION — clothed to lingerie to nude, one
room to another, a toy appearing halfway through. Four frames from that clip
describe a moment and call it the video.

So the count scales with duration, at roughly one representative frame per
minute, with a floor that keeps short clips useful and a hard ceiling that
keeps a long one from turning into an unbounded VLM bill:

    20 s   -> 4      (the floor)
    1 min  -> 4
    2 min  -> 4
    5 min  -> 5
    8 min  -> 8
    12 min -> 12
    15 min -> 12     (the ceiling)
    45 min -> 12     (the ceiling; a 45-minute clip is not 45 images)

``ceil`` rather than ``round`` on the minute, so a 4.2-minute clip samples five
times rather than four: the cost of one extra frame is trivial next to missing
the last minute of a scene.

Where the samples land
----------------------
The duration is divided into ``count`` equal segments and one frame is taken at
the MIDPOINT of each. For an eight-minute video that is exactly 0:30, 1:30,
2:30 … 7:30. Midpoints rather than boundaries because the boundaries are where
intros, title cards, outros and fade-to-black live, and a frame of black
teaches the classifier nothing. It also means coverage is complete by
construction: every part of the clip is represented by the segment that
contains it, and no region can be silently skipped.
"""
from __future__ import annotations

import asyncio
import io
import math
import os
import shutil
from dataclasses import dataclass
from typing import Any

from PIL import Image, ImageStat

# One frame per minute of runtime.
DEFAULT_FRAMES_PER_MINUTE = 1.0

# The floor. A clip shorter than this many frames' worth of minutes still gets
# this many samples, because "three or four moments" is the minimum that can
# show any change at all.
DEFAULT_MIN_FRAME_COUNT = 4

# The hard ceiling, and the reason a 30-minute video does not produce 30
# expensive VLM images. Configurable, but bounded below by the floor and above
# by ``MAX_FRAME_COUNT_CEILING`` so no configuration can make one asset
# unbounded.
DEFAULT_MAX_FRAME_COUNT = 12
MAX_FRAME_COUNT_CEILING = 16

DEFAULT_MAX_DIMENSION = 960
DEFAULT_TIMEOUT_SECONDS = 35.0

# Creator-level video jobs in flight. Unchanged.
_VIDEO_EXTRACTION_GATE = asyncio.Semaphore(2)

# How many ffmpeg processes one video may run at once. This did not need a
# bound while the count was fixed at four; at up to sixteen it does, or a
# single long clip forks sixteen decoders inside the container that also serves
# chat.
_FFMPEG_FANOUT = 4


def _bounded_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return min(max(parsed, minimum), maximum)


def _bounded_float(
    value: Any,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return min(max(parsed, minimum), maximum)


@dataclass(frozen=True)
class FrameSettings:
    """How many frames to take, how big, and how long to wait for them."""

    min_frame_count: int = DEFAULT_MIN_FRAME_COUNT
    max_frame_count: int = DEFAULT_MAX_FRAME_COUNT
    frames_per_minute: float = DEFAULT_FRAMES_PER_MINUTE
    max_dimension: int = DEFAULT_MAX_DIMENSION
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    enabled: bool = True

    @classmethod
    def from_env(cls) -> FrameSettings:
        minimum = _bounded_int(
            os.getenv("VIDEO_FRAME_MIN_COUNT"),
            DEFAULT_MIN_FRAME_COUNT,
            2,
            8,
        )
        maximum = _bounded_int(
            os.getenv("VIDEO_FRAME_MAX_COUNT"),
            DEFAULT_MAX_FRAME_COUNT,
            minimum,
            MAX_FRAME_COUNT_CEILING,
        )
        return cls(
            min_frame_count=minimum,
            # Clamped to at least the floor, so a maximum configured below the
            # minimum cannot produce an empty range.
            max_frame_count=max(maximum, minimum),
            frames_per_minute=_bounded_float(
                os.getenv("VIDEO_FRAMES_PER_MINUTE"),
                DEFAULT_FRAMES_PER_MINUTE,
                0.1,
                4.0,
            ),
            max_dimension=_bounded_int(
                os.getenv("VIDEO_FRAME_MAX_DIMENSION"),
                DEFAULT_MAX_DIMENSION,
                384,
                1440,
            ),
            timeout_seconds=_bounded_float(
                os.getenv("VIDEO_FRAME_TIMEOUT"),
                DEFAULT_TIMEOUT_SECONDS,
                10,
                120,
            ),
            enabled=os.getenv(
                "VIDEO_FRAME_ANALYSIS_ENABLED",
                "true",
            ).strip().lower() not in {"false", "0", "no", "off"},
        )


@dataclass(frozen=True)
class ExtractedFrames:
    frames: list[bytes]
    duration_seconds: float
    offsets_seconds: list[float]


def frames_for_duration(
    duration_seconds: float,
    *,
    settings: FrameSettings | None = None,
) -> int:
    """How many frames a clip of this length deserves.

    An unknown or unreadable duration returns the floor rather than the
    ceiling: ffprobe failing is not evidence that the clip is long, and
    guessing upward would spend the maximum on every asset whose metadata the
    CDN happens to withhold.
    """
    resolved = settings or FrameSettings.from_env()
    try:
        duration = float(duration_seconds)
    except (TypeError, ValueError):
        duration = 0.0
    if duration <= 0 or not math.isfinite(duration):
        return resolved.min_frame_count

    wanted = math.ceil((duration / 60.0) * resolved.frames_per_minute)
    return max(resolved.min_frame_count, min(wanted, resolved.max_frame_count))


def frame_sample_offsets(duration_seconds: float, count: int) -> list[float]:
    """Chronological segment midpoints covering the whole clip.

    ``count`` equal segments, one sample at the middle of each. Deterministic,
    sorted, and free of the intro/outro problem by construction.
    """
    safe_count = max(int(count), 1)
    try:
        duration = float(duration_seconds)
    except (TypeError, ValueError):
        duration = 0.0
    if duration <= 0 or not math.isfinite(duration):
        # ffprobe can fail on some protected or streaming sources even though
        # ffmpeg can decode them. Fixed early samples still inspect more than a
        # poster frame, and offsets past the end of a short clip simply fail to
        # grab rather than producing a wrong frame.
        defaults = [0.5, 2.0, 5.0, 10.0, 20.0, 35.0, 55.0, 80.0, 110.0,
                    145.0, 185.0, 230.0, 280.0, 335.0, 395.0, 460.0]
        return defaults[:safe_count]

    step = duration / safe_count
    unique: list[float] = []
    for index in range(safe_count):
        offset = round(step * (index + 0.5), 3)
        clamped = round(max(0.0, min(offset, max(duration - 0.05, 0.0))), 3)
        if clamped not in unique:
            unique.append(clamped)
    return unique or [0.0]


def ffmpeg_available() -> bool:
    return bool(shutil.which("ffmpeg"))


async def _communicate(
    process: asyncio.subprocess.Process,
    *,
    timeout_seconds: float,
) -> tuple[bytes, bytes]:
    try:
        return await asyncio.wait_for(
            process.communicate(),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        return b"", b""


async def probe_duration_seconds(
    source: str,
    *,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> float:
    ffprobe = shutil.which("ffprobe")
    if not source or not ffprobe:
        return 0.0
    process = await asyncio.create_subprocess_exec(
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        source,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await _communicate(
        process,
        timeout_seconds=timeout_seconds,
    )
    try:
        return max(float(stdout.decode().strip()), 0.0)
    except (TypeError, ValueError):
        return 0.0


async def _grab_frame(
    source: str,
    offset: float,
    *,
    max_dimension: int,
    timeout_seconds: float,
) -> bytes | None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None
    process = await asyncio.create_subprocess_exec(
        ffmpeg,
        "-nostdin",
        "-loglevel",
        "error",
        "-ss",
        f"{max(offset, 0.0):.3f}",
        "-i",
        source,
        "-frames:v",
        "1",
        "-vf",
        f"scale='min({max_dimension},iw)':-2",
        "-f",
        "image2",
        "-vcodec",
        "mjpeg",
        "-q:v",
        "4",
        "pipe:1",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await _communicate(
        process,
        timeout_seconds=timeout_seconds,
    )
    if process.returncode != 0 or len(stdout) < 1000:
        return None
    return stdout


async def extract_frames(
    source: str,
    *,
    settings: FrameSettings | None = None,
) -> ExtractedFrames:
    """Sample frames from a URL or local file without blocking other requests.

    The count is derived from the probed duration, so an HTTP(S) source is
    sampled by range request through ffmpeg rather than being downloaded: the
    decoder seeks to each offset and reads only what that frame needs.
    """
    resolved = settings or FrameSettings.from_env()
    if not source or not resolved.enabled or not ffmpeg_available():
        return ExtractedFrames([], 0.0, [])

    async with _VIDEO_EXTRACTION_GATE:
        duration = await probe_duration_seconds(
            source,
            timeout_seconds=resolved.timeout_seconds,
        )
        offsets = frame_sample_offsets(
            duration,
            frames_for_duration(duration, settings=resolved),
        )
        fanout = asyncio.Semaphore(_FFMPEG_FANOUT)

        async def grab(offset: float) -> bytes | None:
            async with fanout:
                return await _grab_frame(
                    source,
                    offset,
                    max_dimension=resolved.max_dimension,
                    timeout_seconds=resolved.timeout_seconds,
                )

        results = await asyncio.gather(
            *[grab(offset) for offset in offsets],
            return_exceptions=True,
        )

    # Offsets travel with their frames. A frame that failed to grab drops its
    # offset too, so position N in ``frames`` is always the moment at position
    # N in ``offsets_seconds`` — which is what makes chronological batching and
    # "this happened at 4:30" honest rather than approximate.
    frames: list[bytes] = []
    kept_offsets: list[float] = []
    for offset, frame in zip(offsets, results):
        if isinstance(frame, (bytes, bytearray)) and frame:
            frames.append(bytes(frame))
            kept_offsets.append(offset)
    return ExtractedFrames(frames, duration, kept_offsets)


def _usable_frame(frame: bytes) -> Image.Image | None:
    try:
        image = Image.open(io.BytesIO(frame))
        image.seek(0)
        image = image.convert("RGB")
    except (OSError, ValueError):
        return None
    brightness = sum(ImageStat.Stat(image.resize((32, 32))).mean) / 3
    return image if brightness >= 4 else None


def build_contact_sheet(
    frames: list[bytes],
    *,
    max_dimension: int = 896,
) -> tuple[bytes, int]:
    """Combine keyframes into one classifier image while preserving chronology.

    Kept deliberately small: a sheet is a 2xN grid of at most a handful of
    frames, read left to right and top to bottom. Squeezing twelve or sixteen
    frames into one 896px square would give each moment a ~200px thumbnail,
    which is how a "video description" becomes a paragraph about nothing.
    ``services.video_semantics`` is what splits a long sample into several such
    sheets.
    """
    images = [image for frame in frames if (image := _usable_frame(frame))]
    if not images:
        return b"", 0
    columns = min(2, len(images))
    rows = math.ceil(len(images) / columns)
    gap = 8
    cell_width = (max_dimension - gap * (columns - 1)) // columns
    cell_height = (max_dimension - gap * (rows - 1)) // rows
    sheet = Image.new("RGB", (max_dimension, max_dimension), (14, 14, 16))
    for index, image in enumerate(images):
        image.thumbnail(
            (cell_width, cell_height),
            Image.Resampling.LANCZOS,
        )
        column = index % columns
        row = index // columns
        x = column * (cell_width + gap) + (cell_width - image.width) // 2
        y = row * (cell_height + gap) + (cell_height - image.height) // 2
        sheet.paste(image, (x, y))
    buffer = io.BytesIO()
    sheet.save(buffer, format="JPEG", quality=86, optimize=True)
    return buffer.getvalue(), len(images)
