"""Serving protected media from the creator's own session instead of buying it.

The behaviours worth pinning are the ones that cost money or leak credentials:
the cheap route is tried first, an expired signature is re-signed rather than
paid for, a failure is clean enough for the caller to fall back, session
headers never leave a Fansly host, and every saved byte is counted.
"""
from __future__ import annotations

import pytest

from services import fansly_direct as direct


CDN_URL = "https://cdn.fansly.com/media/abc123.mp4"
ASSET = b"x" * 5000


@pytest.fixture(autouse=True)
def _clean_ledger():
    direct.reset_savings_for_tests()
    direct.set_session_provider(None)
    yield
    direct.reset_savings_for_tests()
    direct.set_session_provider(None)


class FakeResponse:
    def __init__(self, *, status_code=200, content=b"", content_type="video/mp4"):
        self.status_code = status_code
        self.content = content
        self.headers = {"content-type": content_type}


class FakeClient:
    """A stand-in for one account's ``FanslyClient``.

    Records every call so a test can assert on the ROUTE taken, which is the
    thing that decides whether a credit was spent.
    """

    def __init__(self, *, responses=None, media_payload=None, media_error=None):
        self._responses = list(responses or [])
        self._media_payload = media_payload
        self._media_error = media_error
        self.downloads: list[tuple[str, bool]] = []
        self.media_lookups: list[list[str]] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def download_asset(self, url, *, timeout=45.0, authenticated=True):
        self.downloads.append((url, authenticated))
        if not self._responses:
            raise AssertionError("unexpected extra download")
        outcome = self._responses.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def get_account_media(self, media_ids):
        self.media_lookups.append(list(media_ids))
        if self._media_error is not None:
            raise self._media_error
        return self._media_payload


class FakeStore:
    def __init__(self, client, *, account="acct-1"):
        self._client = client
        self._account = account

    def get_client(self, account_id):
        if account_id != self._account:
            raise ValueError(f"No session for account {account_id}")
        return self._client


# --- The cheap route -------------------------------------------------------


@pytest.mark.asyncio
async def test_authenticated_read_serves_the_asset_without_a_refresh():
    client = FakeClient(responses=[FakeResponse(content=ASSET)])

    content, route = await direct.download_media(
        CDN_URL, account_id="acct-1", media_id="m-1", client=client
    )

    assert content == ASSET
    assert route == direct.ROUTE_AUTHENTICATED_CDN
    # The refresh is two requests instead of one, so it must not happen when
    # the first read already worked.
    assert client.media_lookups == []
    assert client.downloads == [(CDN_URL, True)]


@pytest.mark.asyncio
async def test_expired_signature_is_re_signed_rather_than_paid_for():
    """The route the metered proxy is really selling.

    A 403 from the CDN means the signature expired, not that the bytes are
    unreachable. Asking the API for a fresh location costs one ordinary
    request; buying the transfer through the provider costs 2 credits/MB.
    """
    fresh = "https://cdn.fansly.com/media/abc123.mp4?Signature=fresh"
    client = FakeClient(
        responses=[FakeResponse(status_code=403), FakeResponse(content=ASSET)],
        media_payload={"accountMedia": [{"media": {"location": fresh}}]},
    )

    content, route = await direct.download_media(
        CDN_URL, account_id="acct-1", media_id="m-1", client=client
    )

    assert content == ASSET
    assert route == direct.ROUTE_REFRESHED_URL
    assert client.media_lookups == [["m-1"]]
    # A freshly signed URL carries its own authorisation, so the second read
    # must NOT attach the session.
    assert client.downloads[1] == (fresh, False)


@pytest.mark.asyncio
async def test_without_a_media_id_there_is_nothing_to_re_sign():
    client = FakeClient(responses=[FakeResponse(status_code=403)])

    with pytest.raises(direct.DirectTransportError, match="no media id"):
        await direct.download_media(CDN_URL, account_id="acct-1", client=client)


# --- Failing cleanly, so the caller can fall back --------------------------


@pytest.mark.asyncio
async def test_an_error_page_is_not_mistaken_for_media():
    client = FakeClient(
        responses=[
            FakeResponse(content=b"<html>nope</html>" * 100, content_type="text/html")
        ]
    )

    with pytest.raises(direct.DirectTransportError, match="instead of media"):
        await direct.download_media(CDN_URL, account_id="acct-1", client=client)


@pytest.mark.asyncio
async def test_a_truncated_transfer_is_a_failure_not_a_tiny_asset():
    client = FakeClient(responses=[FakeResponse(content=b"tiny")])

    with pytest.raises(direct.DirectTransportError, match="empty or truncated"):
        await direct.download_media(CDN_URL, account_id="acct-1", client=client)


@pytest.mark.asyncio
async def test_in_process_size_ceiling_still_applies_without_credits():
    """Not paying per megabyte is not a reason to pull an unbounded file."""
    client = FakeClient(responses=[FakeResponse(content=b"y" * 200_000)])

    with pytest.raises(direct.DirectTransportError, match="ceiling"):
        await direct.download_media(
            CDN_URL, account_id="acct-1", max_megabytes=0.1, client=client
        )


@pytest.mark.asyncio
async def test_missing_session_is_distinguishable_from_a_failed_attempt():
    direct.set_session_provider(FakeStore(FakeClient(), account="acct-1"))

    with pytest.raises(direct.DirectTransportUnavailable):
        await direct.download_media(CDN_URL, account_id="acct-unknown")


@pytest.mark.asyncio
async def test_no_provider_installed_is_unavailable_not_a_crash():
    with pytest.raises(direct.DirectTransportUnavailable):
        await direct.download_media(CDN_URL, account_id="acct-1")


# --- Credentials -----------------------------------------------------------


@pytest.mark.asyncio
async def test_a_non_fansly_host_is_refused_before_any_request():
    client = FakeClient(responses=[FakeResponse(content=ASSET)])

    with pytest.raises(direct.DirectTransportError, match="Fansly CDN URL"):
        await direct.download_media(
            "https://evil.example.com/a.mp4", account_id="acct-1", client=client
        )

    assert client.downloads == []


@pytest.mark.asyncio
async def test_a_refreshed_location_on_a_foreign_host_is_ignored():
    """An API answer is not a licence to fetch whatever host it names."""
    client = FakeClient(
        responses=[FakeResponse(status_code=403)],
        media_payload={"accountMedia": [{"location": "https://evil.example.com/a"}]},
    )

    with pytest.raises(direct.DirectTransportError, match="no usable location"):
        await direct.download_media(
            CDN_URL, account_id="acct-1", media_id="m-1", client=client
        )


def test_host_check_accepts_fansly_subdomains_over_https_only():
    assert direct.is_fansly_host("https://cdn.fansly.com/x") is True
    assert direct.is_fansly_host("https://fansly.com/x") is True
    assert direct.is_fansly_host("http://cdn.fansly.com/x") is False
    assert direct.is_fansly_host("https://fansly.com.evil.net/x") is False
    assert direct.is_fansly_host("") is False


# --- Measuring the migration ------------------------------------------------


@pytest.mark.asyncio
async def test_every_saved_transfer_is_counted_at_the_provider_rate():
    client = FakeClient(responses=[FakeResponse(content=b"z" * (2 * 1024 * 1024))])

    await direct.download_media(
        CDN_URL, account_id="acct-1", media_id="m-1", client=client
    )

    snapshot = direct.savings_snapshot()
    assert snapshot["transfers"] == 1
    # 2 MB at the provider's 2 credits/MB.
    assert snapshot["credits_saved"] == pytest.approx(4.0, abs=0.01)
    assert snapshot["by_account"]["acct-1"] == pytest.approx(4.0, abs=0.01)
    assert snapshot["by_route"][direct.ROUTE_AUTHENTICATED_CDN] == pytest.approx(
        4.0, abs=0.01
    )


@pytest.mark.asyncio
async def test_a_failed_attempt_saves_nothing():
    client = FakeClient(responses=[FakeResponse(status_code=500)])

    with pytest.raises(direct.DirectTransportError):
        await direct.download_media(CDN_URL, account_id="acct-1", client=client)

    assert direct.savings_snapshot()["transfers"] == 0


def test_savings_use_the_same_rule_the_provider_bills_on():
    from services.media_cost_guard import estimated_credits_for_bytes

    for size in (1, 1024, 5 * 1024 * 1024, 250 * 1024 * 1024):
        assert direct.provider_credits_for_bytes(size) == pytest.approx(
            estimated_credits_for_bytes(size)
        )


@pytest.mark.asyncio
async def test_the_session_store_is_used_when_no_client_is_injected():
    client = FakeClient(responses=[FakeResponse(content=ASSET)])
    direct.set_session_provider(FakeStore(client, account="acct-1"))

    content, route = await direct.download_media(CDN_URL, account_id="acct-1")

    assert content == ASSET
    assert route == direct.ROUTE_AUTHENTICATED_CDN
