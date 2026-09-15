"""The billed media proxy is never reached by accident.

API Fansly meters media transfer at 2 credits per megabyte. Classification used
to fall back to pulling an entire original video through that proxy whenever
direct frame extraction failed — silently, automatically, in a background job.
A 250 MB clip is ~500 credits, and a vault of them is a bill nobody approved.

These tests are the guarantee that it cannot happen again. They exercise the
real retrieval path in main.py with a mock transport, and assert on what
CROSSED THE WIRE, not on what a helper returned.
"""
from __future__ import annotations

import io

import httpx
import pytest
from PIL import Image

import main
from services.media_cost_guard import (
    REASON_OVER_SIZE,
    REASON_UNKNOWN_SIZE,
    auto_download_limits,
    estimated_credits_for_bytes,
    evaluate_download,
    manual_download_limits,
    parse_content_length,
)
from services.video_frames import ExtractedFrames


MEGABYTE = 1024 * 1024
CDN = "https://cdn3.fansly.com/account"
PROXY_PATH = "/api/fansly/media/download"


def jpeg_bytes(colour: tuple[int, int, int] = (200, 120, 90)) -> bytes:
    image = Image.new("RGB", (320, 240), colour)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


class Wire:
    """Records every request, and what each host was asked for."""

    def __init__(
        self,
        *,
        cdn_get_status: int = 200,
        cdn_head_status: int = 200,
        content_length: int | None = 4 * MEGABYTE,
        proxy_body: bytes | None = None,
    ) -> None:
        self.requests: list[tuple[str, str]] = []
        self.cdn_get_status = cdn_get_status
        self.cdn_head_status = cdn_head_status
        self.content_length = content_length
        self.proxy_body = proxy_body if proxy_body is not None else b"\xff\xd8" + b"v" * 5000

    @property
    def proxy_calls(self) -> int:
        return sum(1 for _, path in self.requests if path == PROXY_PATH)

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.requests.append((request.method, path))

        if path == PROXY_PATH:
            return httpx.Response(
                200,
                request=request,
                headers={"content-type": "video/mp4"},
                content=self.proxy_body,
            )

        headers: dict[str, str] = {}
        if request.method == "HEAD":
            if self.cdn_head_status >= 400 or self.content_length is None:
                return httpx.Response(
                    self.cdn_head_status if self.cdn_head_status >= 400 else 403,
                    request=request,
                )
            return httpx.Response(
                200,
                request=request,
                headers={"content-length": str(self.content_length)},
            )

        if request.headers.get("Range"):
            if self.content_length is None:
                return httpx.Response(403, request=request)
            return httpx.Response(
                206,
                request=request,
                headers={
                    "content-range": f"bytes 0-0/{self.content_length}",
                    "content-length": "1",
                },
                content=b"x",
            )

        if self.cdn_get_status >= 400:
            return httpx.Response(self.cdn_get_status, request=request, content=b"no")
        return httpx.Response(
            200,
            request=request,
            headers={"content-type": "image/jpeg", **headers},
            content=jpeg_bytes(),
        )


@pytest.fixture(autouse=True)
def _configured(monkeypatch):
    monkeypatch.setenv("APIFANSLY_API_KEY", "test-key")
    monkeypatch.setenv("APIFANSLY_ENABLED", "true")
    for name in (
        "VAULT_MAX_AUTO_MEDIA_DOWNLOAD_MB",
        "VAULT_MAX_AUTO_MEDIA_DOWNLOAD_CREDITS",
        "VAULT_MAX_MANUAL_MEDIA_DOWNLOAD_MB",
        "VAULT_MAX_MANUAL_MEDIA_DOWNLOAD_CREDITS",
    ):
        monkeypatch.delenv(name, raising=False)


def video_item(**overrides) -> dict:
    item = {
        "id": "row-1",
        "creator_id": "creator-1",
        "media_id": "m1",
        "fansly_media_id": "m1",
        "album_id": "a1",
        "url": f"{CDN}/clip.mp4",
        "thumbnail_url": f"{CDN}/poster.jpeg",
        "mimetype": "video/mp4",
        "filename": "clip.mp4",
        "album_title": "Bedroom",
    }
    item.update(overrides)
    return item


def frames(count: int = 8, duration: float = 480.0) -> ExtractedFrames:
    return ExtractedFrames(
        [jpeg_bytes((10 * i, 120, 200)) for i in range(count)],
        duration,
        [30.0 + 60.0 * i for i in range(count)],
    )


async def _load(item, wire, *, is_video=True, manual=False):
    transport = httpx.MockTransport(wire.handler)
    async with httpx.AsyncClient(transport=transport) as client:
        return await main._load_vault_visual(
            item,
            is_video=is_video,
            client=client,
            manual=manual,
        )


@pytest.fixture
def no_refresh(monkeypatch):
    async def refresh(_item):
        return None

    monkeypatch.setattr(main, "_refresh_vault_item_urls", refresh)


# ---------------------------------------------------------------------------
# 1. The pure policy
# ---------------------------------------------------------------------------


def test_credits_follow_the_published_two_per_megabyte_rule():
    assert estimated_credits_for_bytes(MEGABYTE) == pytest.approx(2.0)
    assert estimated_credits_for_bytes(250 * MEGABYTE) == pytest.approx(500.0)
    assert estimated_credits_for_bytes(0) == 0.0


def test_a_250mb_video_is_refused_automatically():
    """The exact scenario in the sprint: ~500 credits, silently, in a job."""
    decision = evaluate_download(content_length_bytes=250 * MEGABYTE, manual=False)

    assert decision.allowed is False
    assert decision.reason == REASON_OVER_SIZE
    assert decision.estimated_credits == pytest.approx(500.0)
    assert "500 credits" in decision.operator_message()


def test_an_unknown_size_is_refused_rather_than_attempted():
    """We cannot price it, so we do not buy it."""
    decision = evaluate_download(content_length_bytes=None, manual=False)

    assert decision.allowed is False
    assert decision.reason == REASON_UNKNOWN_SIZE


def test_a_still_image_may_proceed_on_an_unknown_size():
    """A photo's worst case is single-digit megabytes, and refusing one on an
    unreadable Content-Length would leave ordinary photos unclassified to guard
    against a cost a photo cannot incur."""
    decision = evaluate_download(
        content_length_bytes=None, manual=False, allow_unknown_size=True
    )

    assert decision.allowed is True


def test_a_known_oversize_asset_is_refused_even_as_an_image():
    decision = evaluate_download(
        content_length_bytes=300 * MEGABYTE, manual=False, allow_unknown_size=True
    )

    assert decision.allowed is False


def test_a_small_asset_is_allowed():
    decision = evaluate_download(content_length_bytes=8 * MEGABYTE, manual=False)

    assert decision.allowed is True
    assert decision.estimated_credits == pytest.approx(16.0)


def test_the_manual_tier_is_larger_but_still_bounded():
    assert manual_download_limits().max_megabytes > auto_download_limits().max_megabytes
    assert evaluate_download(
        content_length_bytes=200 * MEGABYTE, manual=True
    ).allowed is True
    assert evaluate_download(
        content_length_bytes=900 * MEGABYTE, manual=True
    ).allowed is False


def test_the_limits_are_configurable_and_zero_means_never(monkeypatch):
    # Both ceilings have to move: they are two views of one quantity and the
    # tighter one wins, so raising megabytes alone leaves credits binding.
    monkeypatch.setenv("VAULT_MAX_AUTO_MEDIA_DOWNLOAD_MB", "100")
    assert evaluate_download(content_length_bytes=60 * MEGABYTE).allowed is False

    monkeypatch.setenv("VAULT_MAX_AUTO_MEDIA_DOWNLOAD_CREDITS", "200")
    assert evaluate_download(content_length_bytes=60 * MEGABYTE).allowed is True

    monkeypatch.setenv("VAULT_MAX_AUTO_MEDIA_DOWNLOAD_MB", "0")
    refused = evaluate_download(content_length_bytes=1 * MEGABYTE)
    assert refused.allowed is False
    assert refused.reason == "download_disabled"


def test_the_credit_ceiling_binds_independently_of_the_size_ceiling(monkeypatch):
    monkeypatch.setenv("VAULT_MAX_AUTO_MEDIA_DOWNLOAD_MB", "1000")
    monkeypatch.setenv("VAULT_MAX_AUTO_MEDIA_DOWNLOAD_CREDITS", "20")
    decision = evaluate_download(content_length_bytes=50 * MEGABYTE)

    assert decision.allowed is False
    assert decision.reason == "exceeds_credit_limit"


def test_a_size_is_read_from_a_range_response():
    assert parse_content_length(
        {"content-range": "bytes 0-0/26214400", "content-length": "1"}
    ) == 26214400
    assert parse_content_length({"content-length": "4096"}) == 4096
    assert parse_content_length({}) is None


def test_the_operator_message_never_mentions_cdn_or_auth():
    """A protected asset is infrastructure behaviour, not something an agency
    configured or can fix. Nothing may suggest they change a platform setting."""
    for size in (250 * MEGABYTE, None):
        message = evaluate_download(content_length_bytes=size).operator_message()
        lowered = message.lower()
        for jargon in ("cdn", "signed", "token", "auth", "unprotect", "public", "403"):
            assert jargon not in lowered, message


# ---------------------------------------------------------------------------
# 2. The retrieval path, end to end
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_direct_sampling_succeeds_and_the_proxy_is_never_called(
    monkeypatch, no_refresh
):
    """The normal case: cost is compute, not transfer."""
    sampled = []

    async def sample(url):
        sampled.append(url)
        return frames(8)

    monkeypatch.setattr(main, "_sample_video_frames", sample)
    wire = Wire()
    visual = await _load(video_item(), wire)

    assert visual.source == "video_frames"
    assert visual.retrieval_method == "direct_video_frames"
    assert len(visual.frames) == 8
    assert visual.media_bytes == 0
    assert visual.estimated_credits == 0
    assert visual.status == "complete"
    assert wire.proxy_calls == 0, "the billed proxy must not be touched"
    assert sampled == [f"{CDN}/clip.mp4"]


@pytest.mark.asyncio
async def test_a_thumbnail_is_enough_and_the_full_video_is_never_downloaded(
    monkeypatch, no_refresh
):
    """Direct sampling failed, a thumbnail exists, the clip is large.

    A partial classification for nothing beats a complete one for hundreds of
    credits that no operator asked for.
    """
    async def sample(_url):
        raise RuntimeError("ffmpeg could not read the protected stream")

    monkeypatch.setattr(main, "_sample_video_frames", sample)
    wire = Wire(content_length=250 * MEGABYTE)
    visual = await _load(video_item(), wire)

    assert visual.source == "video_thumbnail"
    assert visual.retrieval_method == "platform_thumbnail"
    assert visual.status == "partial"
    assert visual.skip_reason == REASON_OVER_SIZE
    assert visual.media_bytes == 0
    assert wire.proxy_calls == 0, "a 250 MB download must never happen automatically"
    assert "high media-transfer cost" in visual.skip_message


@pytest.mark.asyncio
async def test_direct_failure_plus_a_small_asset_allows_the_bounded_fallback(
    monkeypatch, no_refresh
):
    """The fallback still works — for an asset whose cost is known and small."""
    calls = {"n": 0}

    async def sample(_url):
        calls["n"] += 1
        raise RuntimeError("protected stream")

    async def sample_downloaded(content):
        assert content
        return frames(4, duration=120.0)

    monkeypatch.setattr(main, "_sample_video_frames", sample)
    monkeypatch.setattr(main, "_sample_downloaded_video", sample_downloaded)
    # No thumbnail, so the guarded download is the remaining option.
    wire = Wire(content_length=6 * MEGABYTE)
    visual = await _load(video_item(thumbnail_url=""), wire)

    assert visual.source == "video_frames"
    assert visual.retrieval_method == "apifansly_media_download"
    assert wire.proxy_calls == 1
    assert visual.media_bytes > 0, "billed bytes must be recorded when spent"
    assert visual.estimated_credits > 0


@pytest.mark.asyncio
async def test_an_oversized_video_with_no_thumbnail_is_refused_and_left_pending(
    monkeypatch, no_refresh
):
    """Nothing free worked and the billed path is too expensive.

    A policy outcome with a number attached, not a fault: the item is recorded
    as pending with the reason, and no credits are spent.
    """
    async def sample(_url):
        raise RuntimeError("protected stream")

    monkeypatch.setattr(main, "_sample_video_frames", sample)
    wire = Wire(content_length=400 * MEGABYTE)

    with pytest.raises(main.VaultMediaCostRefusal) as caught:
        await _load(video_item(thumbnail_url=""), wire)

    assert caught.value.decision.allowed is False
    assert caught.value.decision.reason == REASON_OVER_SIZE
    assert wire.proxy_calls == 0


@pytest.mark.asyncio
async def test_an_unknown_size_video_is_refused_automatically(
    monkeypatch, no_refresh
):
    async def sample(_url):
        raise RuntimeError("protected stream")

    monkeypatch.setattr(main, "_sample_video_frames", sample)
    wire = Wire(content_length=None)

    with pytest.raises(main.VaultMediaCostRefusal) as caught:
        await _load(video_item(thumbnail_url=""), wire)

    assert caught.value.decision.reason == REASON_UNKNOWN_SIZE
    assert wire.proxy_calls == 0


@pytest.mark.asyncio
async def test_a_refusal_preserves_a_partial_classification(monkeypatch, no_refresh):
    """Refusing deep analysis must not throw away the thumbnail's metadata."""
    async def sample(_url):
        raise RuntimeError("protected stream")

    monkeypatch.setattr(main, "_sample_video_frames", sample)
    wire = Wire(content_length=250 * MEGABYTE)
    visual = await _load(video_item(), wire)

    assert visual.image, "the thumbnail's pixels are retained and classified"
    assert visual.status == "partial"
    assert visual.skip_reason


@pytest.mark.asyncio
async def test_a_pending_row_records_the_reason_and_writes_no_guessed_category():
    """What is persisted when a video is refused on cost."""
    from services.media_cost_guard import evaluate_download as _evaluate

    decision = _evaluate(content_length_bytes=400 * MEGABYTE)
    payload = main._pending_classification_payload(video_item(), decision)

    assert payload["classification_status"] == "pending"
    assert payload["classification_skip_reason"] == REASON_OVER_SIZE
    assert payload["classification_media_bytes"] == 0
    assert "content_category" not in payload
    assert "price_min" not in payload

    written = main._classification_update_payload(payload)
    assert set(written) == {
        "classification_status",
        "classification_skip_reason",
        "classification_media_key",
        "classification_retrieval_method",
        "classification_media_bytes",
        "classification_media_credits",
        "classification_frames_sampled",
        "classification_metadata",
    }
    assert "content_category" not in written, (
        "a cost decision must never overwrite real metadata with blanks"
    )


@pytest.mark.asyncio
async def test_an_image_still_falls_back_through_the_proxy(no_refresh):
    """Photos are unchanged: the proxy is what makes a protected photo
    classifiable, and a photo cannot incur a video-sized bill."""
    wire = Wire(cdn_get_status=403, content_length=None)
    visual = await _load(
        video_item(mimetype="image/jpeg", url=f"{CDN}/photo.jpeg"),
        wire,
        is_video=False,
    )

    assert visual.source == "image"
    assert visual.retrieval_method == "apifansly_media_download"
    assert wire.proxy_calls == 1
    assert visual.media_bytes > 0


@pytest.mark.asyncio
async def test_the_manual_tier_may_deep_scan_where_an_automatic_run_would_not(
    monkeypatch, no_refresh
):
    """An operator explicitly asked, and the manual ceiling is theirs to spend."""
    async def sample(_url):
        raise RuntimeError("protected stream")

    async def sample_downloaded(_content):
        return frames(6, duration=300.0)

    monkeypatch.setattr(main, "_sample_video_frames", sample)
    monkeypatch.setattr(main, "_sample_downloaded_video", sample_downloaded)
    wire = Wire(content_length=120 * MEGABYTE)

    automatic = await _load(video_item(), wire)
    assert automatic.status == "partial"
    assert wire.proxy_calls == 0

    manual_wire = Wire(content_length=120 * MEGABYTE)
    manual = await _load(video_item(), manual_wire, manual=True)
    assert manual.status == "complete"
    assert manual.source == "video_frames"
    assert manual_wire.proxy_calls == 1


@pytest.mark.asyncio
async def test_the_size_probe_costs_nothing_at_the_billed_proxy(
    monkeypatch, no_refresh
):
    """Learning what an asset weighs is a direct CDN request, never a proxy one."""
    async def sample(_url):
        raise RuntimeError("protected stream")

    monkeypatch.setattr(main, "_sample_video_frames", sample)
    wire = Wire(content_length=250 * MEGABYTE)
    await _load(video_item(), wire)

    probes = [r for r in wire.requests if r[1] != PROXY_PATH]
    assert probes, "the size must be probed"
    assert all(path != PROXY_PATH for _, path in wire.requests)
