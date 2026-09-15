"""Vault classification spend is visible in the EXISTING usage telemetry.

Not a parallel accounting system: ``vault_classification`` is a view over the
same event stream every other figure in the snapshot is derived from. It exists
because the three questions an operator asks about vault spend are each one
filter away and nobody should do that arithmetic by hand:

    How many credits did vault sync and classification consume?
    How many of those were media downloads rather than metadata?
    Which creator/account caused them?
"""
from __future__ import annotations

import pytest

from services import apifansly
from services.apifansly import (
    CATEGORY_LIVE_CHAT,
    CATEGORY_VAULT,
    VAULT_MEDIA_DOWNLOAD_OPERATION,
    record_usage_event,
    reset_usage_for_tests,
    usage_snapshot,
)

MEGABYTE = 1024 * 1024


@pytest.fixture(autouse=True)
def _clean_usage():
    reset_usage_for_tests()
    yield
    reset_usage_for_tests()


def test_a_media_download_is_billed_at_two_credits_per_megabyte():
    record_usage_event(
        operation=VAULT_MEDIA_DOWNLOAD_OPERATION,
        account_id="acct-1",
        media_bytes=10 * MEGABYTE,
        category=CATEGORY_VAULT,
    )
    vault = usage_snapshot()["vault_classification"]

    assert vault["media_downloads"] == 1
    assert vault["media_megabytes"] == pytest.approx(10.0)
    assert vault["media_credits"] == pytest.approx(20.0)


def test_metadata_and_media_are_reported_separately():
    """The shape of a healthy deployment: mostly metadata, little transfer."""
    for _ in range(20):
        record_usage_event(
            operation="vault album media listing",
            account_id="acct-1",
            response_bytes=40_000,
            category=CATEGORY_VAULT,
        )
    record_usage_event(
        operation=VAULT_MEDIA_DOWNLOAD_OPERATION,
        account_id="acct-1",
        media_bytes=5 * MEGABYTE,
        category=CATEGORY_VAULT,
    )
    vault = usage_snapshot()["vault_classification"]

    assert vault["calls"] == 21
    assert vault["metadata_credits"] == pytest.approx(20.0)
    assert vault["media_credits"] == pytest.approx(10.0)
    assert vault["estimated_credits"] == pytest.approx(30.0)
    assert vault["media_share"] == pytest.approx(10.0 / 30.0, abs=0.001)


def test_spend_is_attributed_to_the_account_that_caused_it():
    record_usage_event(
        operation=VAULT_MEDIA_DOWNLOAD_OPERATION,
        account_id="acct-eliz",
        media_bytes=20 * MEGABYTE,
        category=CATEGORY_VAULT,
    )
    record_usage_event(
        operation=VAULT_MEDIA_DOWNLOAD_OPERATION,
        account_id="acct-sophia",
        media_bytes=2 * MEGABYTE,
        category=CATEGORY_VAULT,
    )
    vault = usage_snapshot()["vault_classification"]

    assert vault["media_by_account"]["acct-eliz"]["estimated_credits"] == (
        pytest.approx(40.0)
    )
    assert vault["media_by_account"]["acct-sophia"]["estimated_credits"] == (
        pytest.approx(4.0)
    )


def test_live_chat_spend_is_not_counted_as_vault_spend():
    record_usage_event(
        operation="message delivery",
        account_id="acct-1",
        response_bytes=2000,
        category=CATEGORY_LIVE_CHAT,
    )
    snapshot = usage_snapshot()

    assert snapshot["vault_classification"]["calls"] == 0
    assert snapshot["vault_classification"]["estimated_credits"] == 0
    assert snapshot["estimated_credits"] > 0


def test_the_vault_view_agrees_with_the_category_breakdown():
    """It is a VIEW, not a second ledger. The two must never disagree."""
    record_usage_event(
        operation=VAULT_MEDIA_DOWNLOAD_OPERATION,
        account_id="acct-1",
        media_bytes=3 * MEGABYTE,
        category=CATEGORY_VAULT,
    )
    record_usage_event(
        operation="vault albums",
        account_id="acct-1",
        response_bytes=1000,
        category=CATEGORY_VAULT,
    )
    snapshot = usage_snapshot()

    assert snapshot["vault_classification"]["estimated_credits"] == pytest.approx(
        snapshot["credits_by_category"][CATEGORY_VAULT]["estimated_credits"]
    )
    assert snapshot["vault_classification"]["media_bytes"] == (
        snapshot["credits_by_category"][CATEGORY_VAULT]["media_bytes"]
    )


def test_an_empty_window_reports_zeros_rather_than_failing():
    vault = usage_snapshot()["vault_classification"]

    assert vault["calls"] == 0
    assert vault["media_credits"] == 0
    assert vault["media_share"] == 0.0
    assert vault["by_account"] == {}


@pytest.mark.asyncio
async def test_a_vault_media_download_records_its_real_transferred_bytes(
    monkeypatch,
):
    """Estimated credits must reflect what actually moved, not the JSON
    envelope that came back with it."""
    import httpx

    monkeypatch.setenv("APIFANSLY_API_KEY", "test-key")
    monkeypatch.setenv("APIFANSLY_ENABLED", "true")
    payload = b"\xff\xd8" + b"v" * (3 * MEGABYTE)

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            request=request,
            headers={"content-type": "video/mp4"},
            content=payload,
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        await apifansly.download_media(
            "https://cdn3.fansly.com/account/clip.mp4",
            client=client,
            account_id="acct-1",
            operation=VAULT_MEDIA_DOWNLOAD_OPERATION,
        )

    vault = usage_snapshot()["vault_classification"]
    assert vault["media_downloads"] == 1
    assert vault["media_bytes"] == len(payload)
    assert vault["media_credits"] == pytest.approx(
        2.0 * len(payload) / MEGABYTE, abs=0.01
    )
    assert vault["media_by_account"]["acct-1"]["calls"] == 1
