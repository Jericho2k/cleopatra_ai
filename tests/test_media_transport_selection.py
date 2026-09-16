"""Which transport actually serves a protected vault asset, end to end.

The unit behaviour of each half lives in tests/test_transport_policy.py and
tests/test_fansly_direct_media.py. What is pinned here is the JOIN: that the
policy is consulted, that a direct success never reaches the metered proxy,
that a direct failure degrades into a larger invoice rather than a broken
classification, and that billed bytes keep meaning billed bytes.
"""
from __future__ import annotations

import pytest

import main
from core import transport_policy as tp
from services import fansly_direct as direct


CDN_URL = "https://cdn3.fansly.com/account/video.mp4"
ASSET = b"m" * 4000


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("FANSLY_TRANSPORT_DEFAULT", raising=False)
    monkeypatch.delenv("FANSLY_DIRECT_ACCOUNTS", raising=False)
    monkeypatch.delenv("FANSLY_DIRECT_FALLBACK", raising=False)
    for operation in tp.OPERATIONS:
        monkeypatch.delenv(f"FANSLY_TRANSPORT_{operation.upper()}", raising=False)
    direct.reset_savings_for_tests()
    yield
    direct.reset_savings_for_tests()


class ProxySpy:
    """Stands in for the metered provider download, and counts its use."""

    def __init__(self, content=ASSET):
        self.calls = 0
        self.content = content

    async def __call__(self, url, *, client=None, account_id=None, operation=""):
        self.calls += 1
        return self.content


def _install_direct(monkeypatch, *, result=None, error=None):
    calls: list[dict] = []

    async def fake_download(cdn_url, *, account_id, media_id="", operation="", **kw):
        calls.append(
            {"url": cdn_url, "account_id": account_id, "media_id": media_id}
        )
        if error is not None:
            raise error
        direct.record_saving(
            operation=operation,
            account_id=account_id,
            media_bytes=len(result),
            route=direct.ROUTE_AUTHENTICATED_CDN,
        )
        return result, direct.ROUTE_AUTHENTICATED_CDN

    monkeypatch.setattr(direct, "download_media", fake_download)
    return calls


@pytest.mark.asyncio
async def test_provider_still_serves_an_unconfigured_deployment(monkeypatch):
    """No environment change, no behaviour change. The migration is opt-in."""
    proxy = ProxySpy()
    monkeypatch.setattr(main, "apifansly_download_media", proxy)
    monkeypatch.setattr(main, "_probe_media_size", _size(1000))
    calls = _install_direct(monkeypatch, result=ASSET)

    fetch = await main._fetch_protected_media(
        CDN_URL, client=None, manual=False, is_video=False, account_id="acct-1"
    )

    assert proxy.calls == 1
    assert calls == []
    assert fetch.retrieval_method == main.RETRIEVAL_APIFANSLY_DOWNLOAD
    assert fetch.billed_bytes == len(ASSET)


@pytest.mark.asyncio
async def test_direct_success_never_reaches_the_metered_proxy(monkeypatch):
    monkeypatch.setenv("FANSLY_TRANSPORT_MEDIA_DOWNLOAD", "direct")
    proxy = ProxySpy()
    monkeypatch.setattr(main, "apifansly_download_media", proxy)
    calls = _install_direct(monkeypatch, result=ASSET)

    fetch = await main._fetch_protected_media(
        CDN_URL,
        client=None,
        manual=False,
        is_video=True,
        account_id="acct-1",
        media_id="m-9",
    )

    assert proxy.calls == 0
    assert calls == [{"url": CDN_URL, "account_id": "acct-1", "media_id": "m-9"}]
    assert fetch.content == ASSET
    assert fetch.retrieval_method == main.RETRIEVAL_DIRECT_SESSION
    # The telemetry field means "bytes the provider billed us for". A transfer
    # they did not bill must not inflate it.
    assert fetch.billed_bytes == 0
    assert direct.savings_snapshot()["credits_saved"] > 0


@pytest.mark.asyncio
async def test_direct_bypasses_a_credit_ceiling_that_no_longer_applies(monkeypatch):
    """The guard exists because of 2 credits/MB. Direct pays none of them.

    A 300 MB video is refused outright on the automatic provider path. Served
    directly it costs bandwidth, so refusing it would leave the asset
    unclassified to guard against a cost that is not being incurred.
    """
    monkeypatch.setenv("FANSLY_TRANSPORT_MEDIA_DOWNLOAD", "direct")
    proxy = ProxySpy()
    monkeypatch.setattr(main, "apifansly_download_media", proxy)
    monkeypatch.setattr(main, "_probe_media_size", _size(300 * 1024 * 1024))
    _install_direct(monkeypatch, result=ASSET)

    fetch = await main._fetch_protected_media(
        CDN_URL,
        client=None,
        manual=False,
        is_video=True,
        account_id="acct-1",
        media_id="m-9",
    )

    assert fetch.content == ASSET
    assert proxy.calls == 0


@pytest.mark.asyncio
async def test_direct_failure_falls_back_to_the_provider(monkeypatch):
    """A bad day on the new transport must cost money, not a fan's reply."""
    monkeypatch.setenv("FANSLY_TRANSPORT_MEDIA_DOWNLOAD", "direct")
    proxy = ProxySpy()
    monkeypatch.setattr(main, "apifansly_download_media", proxy)
    monkeypatch.setattr(main, "_probe_media_size", _size(1000))
    _install_direct(
        monkeypatch, error=direct.DirectTransportError("session expired")
    )

    fetch = await main._fetch_protected_media(
        CDN_URL, client=None, manual=False, is_video=False, account_id="acct-1"
    )

    assert proxy.calls == 1
    assert fetch.content == ASSET
    assert fetch.retrieval_method == main.RETRIEVAL_APIFANSLY_DOWNLOAD
    assert fetch.billed_bytes == len(ASSET)


@pytest.mark.asyncio
async def test_fallback_can_be_switched_off_so_a_regression_is_visible(monkeypatch):
    monkeypatch.setenv("FANSLY_TRANSPORT_MEDIA_DOWNLOAD", "direct")
    monkeypatch.setenv("FANSLY_DIRECT_FALLBACK", "false")
    proxy = ProxySpy()
    monkeypatch.setattr(main, "apifansly_download_media", proxy)
    _install_direct(
        monkeypatch, error=direct.DirectTransportError("session expired")
    )

    fetch = await main._fetch_protected_media(
        CDN_URL, client=None, manual=False, is_video=False, account_id="acct-1"
    )

    assert proxy.calls == 0
    assert fetch.content == b""
    assert fetch.decision is None
    assert "session expired" in fetch.direct_error
    # The caller still gets a sentence it can show an operator.
    assert fetch.refusal_message()


@pytest.mark.asyncio
async def test_an_account_outside_the_canary_keeps_using_the_provider(monkeypatch):
    monkeypatch.setenv("FANSLY_TRANSPORT_MEDIA_DOWNLOAD", "direct")
    monkeypatch.setenv("FANSLY_DIRECT_ACCOUNTS", "acct-canary")
    proxy = ProxySpy()
    monkeypatch.setattr(main, "apifansly_download_media", proxy)
    monkeypatch.setattr(main, "_probe_media_size", _size(1000))
    calls = _install_direct(monkeypatch, result=ASSET)

    await main._fetch_protected_media(
        CDN_URL, client=None, manual=False, is_video=False, account_id="acct-other"
    )

    assert calls == []
    assert proxy.calls == 1


@pytest.mark.asyncio
async def test_guard_refusal_is_still_reported_with_its_reason(monkeypatch):
    """The provider path's existing contract is unchanged by the switch."""
    proxy = ProxySpy()
    monkeypatch.setattr(main, "apifansly_download_media", proxy)
    monkeypatch.setattr(main, "_probe_media_size", _size(None))

    fetch = await main._fetch_protected_media(
        CDN_URL, client=None, manual=False, is_video=True, account_id="acct-1"
    )

    assert proxy.calls == 0
    assert fetch.content == b""
    assert fetch.decision is not None
    assert fetch.decision.reason == "unknown_size"
    assert fetch.refusal_message()


def test_item_media_id_reads_either_stored_key():
    assert main._item_media_id({"fansly_media_id": "a"}) == "a"
    assert main._item_media_id({"media_id": "b"}) == "b"
    assert main._item_media_id({}) == ""


def _size(value):
    async def probe(url, *, client):
        return value

    return probe
