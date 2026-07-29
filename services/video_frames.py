"""Deterministic video keyframe sampling for vault classification."""
from __future__ import annotations

import asyncio
import io
import math
import os
import shutil
from dataclasses import dataclass
from typing import Any

from PIL import Image, ImageStat

DEFAULT_FRAME_COUNT = 4
DEFAULT_EDGE_TRIM_RATIO = 0.05
DEFAULT_MAX_DIMENSION = 960
DEFAULT_TIMEOUT_SECONDS = 35.0
_VIDEO_EXTRACTION_GATE = asyncio.Semaphore(2)


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
    frame_count: int = DEFAULT_FRAME_COUNT
    max_dimension: int = DEFAULT_MAX_DIMENSION
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    enabled: bool = True

    @classmethod
    def from_env(cls) -> FrameSettings:
        return cls(
            frame_count=_bounded_int(
                os.getenv("VIDEO_FRAME_COUNT"),
                DEFAULT_FRAME_COUNT,
                2,
                8,
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


def frame_sample_offsets(
    duration_seconds: float,
    count: int,
    *,
    edge_trim_ratio: float = DEFAULT_EDGE_TRIM_RATIO,
) -> list[float]:
    """Return deterministic midpoint samples away from intros and credits."""
    safe_count = max(int(count), 1)
    try:
        duration = float(duration_seconds)
    except (TypeError, ValueError):
        duration = 0.0
    if duration <= 0:
        # ffprobe can fail on some protected or streaming sources even though
        # ffmpeg can decode them. Fixed early samples still inspect more than a
        # poster frame and naturally drop offsets beyond a short clip.
        defaults = [0.5, 2.0, 5.0, 10.0, 20.0, 35.0, 55.0, 80.0]
        return defaults[:safe_count]

    trim = min(max(float(edge_trim_ratio), 0.0), 0.4) * duration
    start, end = trim, duration - trim
    if end <= start:
        start, end = 0.0, duration
    if safe_count == 1:
        return [round((start + end) / 2, 3)]

    step = (end - start) / safe_count
    offsets = [
        round(start + step * (index + 0.5), 3)
        for index in range(safe_count)
    ]
    unique: list[float] = []
    for offset in offsets:
        clamped = max(0.0, min(offset, max(duration - 0.05, 0.0)))
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
    """Sample frames from a URL or local file without blocking other requests."""
    resolved = settings or FrameSettings.from_env()
    if not source or not resolved.enabled or not ffmpeg_available():
        return ExtractedFrames([], 0.0, [])

    async with _VIDEO_EXTRACTION_GATE:
        duration = await probe_duration_seconds(
            source,
            timeout_seconds=resolved.timeout_seconds,
        )
        offsets = frame_sample_offsets(duration, resolved.frame_count)
        results = await asyncio.gather(
            *[
                _grab_frame(
                    source,
                    offset,
                    max_dimension=resolved.max_dimension,
                    timeout_seconds=resolved.timeout_seconds,
                )
                for offset in offsets
            ],
            return_exceptions=True,
        )
    frames = [
        bytes(frame)
        for frame in results
        if isinstance(frame, (bytes, bytearray)) and frame
    ]
    return ExtractedFrames(frames, duration, offsets)


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
    """Combine keyframes into one classifier image while preserving chronology."""
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
