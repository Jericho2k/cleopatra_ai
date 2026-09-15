"""What the classifier reads the pixels from, and how it is told apart.

The COST ordering of those routes — and the guard that keeps the billed media
proxy from being reached automatically — lives in
tests/test_vault_media_cost_guard.py.
"""

import asyncio

import httpx
import pytest

import main
from main import _download_visual_candidate, _vault_media_visual_urls
from services.video_frames import ExtractedFrames


def test_vault_media_visual_urls_keep_original_and_real_image_thumbnail():
    original, thumbnail = _vault_media_visual_urls(
        {
            "mimetype": "video/mp4",
            "locations": [
                {"location": "https://cdn3.fansly.com/account/video.mp4"}
            ],
            "variants": [
                {
                    "mimetype": "application/vnd.apple.mpegurl",
                    "locations": [
                        {"location": "https://cdn3.fansly.com/account/video.m3u8"}
                    ],
                },
                {
                    "mimetype": "image/jpeg",
                    "locations": [
                        {"location": "https://cdn3.fansly.com/account/poster.jpeg"}
                    ],
                },
            ],
        }
    )

    assert original.endswith("/video.mp4")
    assert thumbnail.endswith("/poster.jpeg")


def test_classifier_falls_back_to_protected_media_download(monkeypatch):
    """A protected PHOTO still reaches the proxy.

    Unchanged by the media-cost guard: a photo's worst case is single-digit
    megabytes, and refusing one on an unreadable size would leave ordinary
    photos unclassified to guard against a cost a photo cannot incur.
    """
    monkeypatch.setenv("APIFANSLY_API_KEY", "test-key")
    image_bytes = b"\xff\xd8" + (b"x" * 1200)
    requests = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append((request.method, request.url.path))
        if request.method in {"GET", "HEAD"}:
            return httpx.Response(
                403,
                request=request,
                content=b"expired or protected",
            )
        return httpx.Response(
            200,
            request=request,
            headers={"content-type": "image/jpeg"},
            content=image_bytes,
        )

    async def run():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            return await _download_visual_candidate(
                "https://cdn3.fansly.com/account/media.jpeg?Policy=signed",
                client=client,
                is_video=False,
            )

    content, method, billed = asyncio.run(run())
    assert content == image_bytes
    assert method == "apifansly_media_download"
    # The bytes that moved are recorded, so the credits they cost are visible.
    assert billed == len(image_bytes)
    assert ("POST", "/api/fansly/media/download") in requests


@pytest.mark.asyncio
async def test_video_visual_prefers_real_keyframes_over_the_thumbnail(
    monkeypatch,
):
    """Direct range sampling is both free and strictly more informative, so it
    is tried first and the thumbnail is never fetched when it succeeds."""
    async def keyframes(url):
        assert url == "https://cdn3.fansly.com/account/video.mp4"
        return ExtractedFrames(
            [b"\xff\xd8" + (b"x" * 1500)] * 8,
            480.0,
            [30.0 + 60 * i for i in range(8)],
        )

    monkeypatch.setattr(main, "_sample_video_frames", keyframes)
    visual = await main._load_vault_visual(
        {
            "url": "https://cdn3.fansly.com/account/video.mp4",
            "thumbnail_url": "https://cdn3.fansly.com/account/poster.jpeg",
        },
        is_video=True,
        client=object(),
    )

    assert visual.source == "video_frames"
    assert visual.retrieval_method == "direct_video_frames"
    assert len(visual.frames) == 8
    assert visual.duration_seconds == 480.0
    assert visual.status == "complete"
    # Free: nothing was transferred through the billed proxy.
    assert visual.media_bytes == 0
