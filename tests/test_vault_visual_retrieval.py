import asyncio

import httpx
import pytest

import main
from main import _download_visual_candidate, _vault_media_visual_urls


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
    monkeypatch.setenv("APIFANSLY_API_KEY", "test-key")
    image_bytes = b"\xff\xd8" + (b"x" * 1200)
    requests = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append((request.method, request.url.path))
        if request.method == "GET":
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
            )

    content, method = asyncio.run(run())
    assert content == image_bytes
    assert method == "apifansly_media_download"
    assert requests == [
        ("GET", "/account/media.jpeg"),
        ("POST", "/api/fansly/media/download"),
    ]


@pytest.mark.asyncio
async def test_video_visual_uses_real_keyframe_sheet_before_thumbnail(
    monkeypatch,
):
    async def keyframes(url, *, client):
        assert url == "https://cdn3.fansly.com/account/video.mp4"
        return b"\xff\xd8" + (b"x" * 1500), "video_frames_4_direct_cdn"

    monkeypatch.setattr(main, "_video_classifier_image", keyframes)
    content, source, method = await main._load_vault_visual(
        {
            "url": "https://cdn3.fansly.com/account/video.mp4",
            "thumbnail_url": "https://cdn3.fansly.com/account/poster.jpeg",
        },
        is_video=True,
        client=object(),
    )

    assert len(content) > 1000
    assert source == "video_frames"
    assert method == "video_frames_4_direct_cdn"
