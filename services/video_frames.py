"""Keyframe extraction so videos are classified on what they show.

Videos used to be categorised from their filename and album title alone,
which left the highest-priced tiers — ``solo_toy_video`` at $30-150 and
``bg_content`` at $50-300 — resting on whatever the creator happened to name
the file. Sampling real frames replaces that guess with evidence.

``ffmpeg`` reads the CDN URL directly and seeks before opening the input, so
only the sampled frames cross the network rather than the whole clip. When
``ffmpeg`` is absent the caller degrades to the old filename path instead of
failing the item.
"""
from __future__ import annotations

import asyncio
import os
import shutil
from dataclasses import dataclass


DEFAULT_FRAME_COUNT = 4
DEFAULT_EDGE_TRIM_RATIO = 0.05
DEFAULT_MAX_DIMENSION = 1024
DEFAULT_TIMEOUT_SECONDS = 45.0


@dataclass(frozen=True)
class FrameSettings:
    frame_count: int = DEFAULT_FRAME_COUNT
    max_dimension: int = DEFAULT_MAX_DIMENSION
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    enabled: bool = True

    @classmethod
    def from_env(cls) -> "FrameSettings":
        return cls(
            frame_count=int(os.getenv("VIDEO_FRAME_COUNT", str(DEFAULT_FRAME_COUNT)) or DEFAULT_FRAME_COUNT),
            max_dimension=int(
                os.getenv("VIDEO_FRAME_MAX_DIMENSION", str(DEFAULT_MAX_DIMENSION))
                or DEFAULT_MAX_DIMENSION
            ),
            timeout_seconds=float(
                os.getenv("VIDEO_FRAME_TIMEOUT", str(DEFAULT_TIMEOUT_SECONDS))
                or DEFAULT_TIMEOUT_SECONDS
            ),
            enabled=os.getenv("VIDEO_FRAME_ANALYSIS_ENABLED", "true").strip().lower()
            not in {"false", "0", "no", "off"},
        )


def frame_sample_offsets(
    duration_seconds: float,
    count: int,
    *,
    edge_trim_ratio: float = DEFAULT_EDGE_TRIM_RATIO,
) -> list[float]:
    """Evenly spaced sample points, with the credits and cold open trimmed off.

    Deterministic for a given duration so a re-analysis of the same clip reads
    the same frames. Returns ``[0.0]`` when the duration is unknown or too
    short to sample meaningfully.
    """
    safe_count = max(int(count), 1)

    try:
        duration = float(duration_seconds)
    except (TypeError, ValueError):
        duration = 0.0

    if duration <= 0.0:
        return [0.0]

    trim = max(min(float(edge_trim_ratio), 0.4), 0.0) * duration
    start = trim
    end = duration - trim
    if end <= start:
        start, end = 0.0, duration

    if safe_count == 1:
        return [round((start + end) / 2.0, 3)]

    span = end - start
    step = span / float(safe_count)
    # Sample at interval midpoints: no frame lands exactly on the first or
    # last frame, which are the most likely to be black or a title card.
    offsets = [round(start + step * (index + 0.5), 3) for index in range(safe_count)]

    unique: list[float] = []
    for offset in offsets:
        clamped = max(0.0, min(offset, max(duration - 0.05, 0.0)))
        if clamped not in unique:
            unique.append(clamped)
    return unique or [0.0]


def ffmpeg_available() -> bool:
    return bool(shutil.which("ffmpeg"))


async def probe_duration_seconds(
    url: str,
    *,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> float:
    """Best-effort duration lookup. Returns 0.0 when it cannot be determined."""
    if not url or not shutil.which("ffprobe"):
        return 0.0

    process = await asyncio.create_subprocess_exec(
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        url,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        stdout, _ = await asyncio.wait_for(process.communicate(), timeout=timeout_seconds)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        return 0.0

    try:
        return max(float(stdout.decode().strip()), 0.0)
    except (TypeError, ValueError):
        return 0.0


async def _grab_frame(
    url: str,
    offset: float,
    *,
    max_dimension: int,
    timeout_seconds: float,
) -> bytes | None:
    # -ss before -i seeks server-side, so a frame from the middle of a long
    # clip does not require downloading everything before it.
    process = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-nostdin",
        "-loglevel", "error",
        "-ss", f"{max(offset, 0.0):.3f}",
        "-i", url,
        "-frames:v", "1",
        "-vf", f"scale='min({max_dimension},iw)':-2",
        "-f", "image2",
        "-vcodec", "mjpeg",
        "-q:v", "4",
        "pipe:1",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        stdout, _ = await asyncio.wait_for(process.communicate(), timeout=timeout_seconds)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        return None

    if process.returncode != 0 or len(stdout) < 1000:
        return None
    return stdout


async def extract_frames(
    url: str,
    *,
    settings: FrameSettings | None = None,
) -> list[bytes]:
    """Sample JPEG frames from a video URL. Empty list means "fall back"."""
    resolved = settings or FrameSettings.from_env()
    if not url or not resolved.enabled or not ffmpeg_available():
        return []

    duration = await probe_duration_seconds(url, timeout_seconds=resolved.timeout_seconds)
    offsets = frame_sample_offsets(duration, resolved.frame_count)

    frames = await asyncio.gather(
        *[
            _grab_frame(
                url,
                offset,
                max_dimension=resolved.max_dimension,
                timeout_seconds=resolved.timeout_seconds,
            )
            for offset in offsets
        ],
        return_exceptions=True,
    )

    return [
        frame
        for frame in frames
        if isinstance(frame, (bytes, bytearray)) and frame
    ]
