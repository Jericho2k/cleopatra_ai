"""API Fansly credit observability.

Every figure here is an ESTIMATE — the provider's Usage dashboard is
authoritative — but an estimate that is wrong in a predictable direction is
worse than none, so the arithmetic, the instrumentation coverage and the
categorisation are all pinned.
"""

import httpx
import pytest

from services import apifansly
from services.apifansly import (
    CATEGORY_BACKGROUND_HISTORY,
    CATEGORY_LIVE_CHAT,
    CATEGORY_VAULT,
    CREDIT_RESPONSE_BYTES_PER_CREDIT,
    WEBHOOK_EVENTS_PER_CREDIT,
    background_history_allowed,
    background_history_budget_state,
    collect_usage,
    estimate_call_credits,
    estimate_webhook_credits,
    live_work_in_progress,
    raise_for_response,
    record_raw_call,
    record_usage_event,
    record_webhook_event,
    summarize_usage_events,
    usage_category,
    usage_snapshot,
)


def _response(status: int = 200, *, body: bytes = b"{}", path: str = "/api/fansly/x"):
    request = httpx.Request("GET", f"https://v1.apifansly.com{path}")
    return httpx.Response(status, request=request, content=body)


# --- the credit model -------------------------------------------------------


def test_ordinary_request_costs_one_credit():
    assert estimate_call_credits(response_bytes=0) == 1.0
    assert estimate_call_credits(response_bytes=1_000) == 1.0
    assert estimate_call_credits(response_bytes=CREDIT_RESPONSE_BYTES_PER_CREDIT) == 1.0


def test_responses_over_80kb_cost_proportionally_more():
    """The rule that makes history budgeting real.

    A chat page carries every attachment's accountMedia metadata, so a page of
    ten messages routinely blows past 80 KB. Assuming one page is one credit is
    how a 500-page import gets estimated at a third of what it bills.
    """
    assert estimate_call_credits(response_bytes=CREDIT_RESPONSE_BYTES_PER_CREDIT * 3) == 3.0
    assert estimate_call_credits(
        response_bytes=int(CREDIT_RESPONSE_BYTES_PER_CREDIT * 2.5)
    ) == pytest.approx(2.5)


def test_media_transfer_is_billed_per_megabyte_not_per_request():
    one_mb = 1024 * 1024
    assert estimate_call_credits(media_bytes=one_mb) == pytest.approx(2.0)
    assert estimate_call_credits(media_bytes=5 * one_mb) == pytest.approx(10.0)


def test_a_tiny_media_transfer_still_pays_the_request_floor():
    assert estimate_call_credits(media_bytes=1_000) == 1.0


def test_media_bytes_replace_rather_than_add_to_response_bytes():
    """A download's bytes ARE its response. Counting both double-bills it."""
    one_mb = 1024 * 1024
    assert estimate_call_credits(
        response_bytes=one_mb, media_bytes=one_mb
    ) == pytest.approx(2.0)


def test_webhook_events_are_eighty_to_the_credit():
    assert estimate_webhook_credits(0) == 0.0
    assert estimate_webhook_credits(WEBHOOK_EVENTS_PER_CREDIT) == 1.0
    assert estimate_webhook_credits(WEBHOOK_EVENTS_PER_CREDIT * 4) == 4.0


# --- instrumentation coverage ----------------------------------------------


def test_raise_for_response_accounts_even_when_the_call_failed():
    """A refused call still reached the provider and still cost a credit."""
    with pytest.raises(httpx.HTTPStatusError):
        raise_for_response(
            _response(404, body=b'{"error":"nope"}'),
            operation="chat listing",
            account_id="acct-1",
        )
    snapshot = usage_snapshot()
    assert snapshot["calls"] == 1
    assert snapshot["estimated_credits"] == 1.0


def test_a_response_is_never_billed_twice():
    """The dedupe that lets a call site both raise and report its media bytes."""
    response = _response(body=b"{}")
    raise_for_response(response, operation="vault media URL lookup", account_id="a")
    record_raw_call(response, operation="vault media URL lookup", account_id="a")
    assert usage_snapshot()["calls"] == 1


def test_raw_call_sites_that_bypass_request_are_still_counted():
    """Typing, mark-as-read, connect and 2FA never reach request()."""
    record_raw_call(
        _response(), operation="typing indicator", account_id="a",
        category=CATEGORY_LIVE_CHAT,
    )
    record_raw_call(
        _response(), operation="chat mark as read", account_id="a",
        category=CATEGORY_LIVE_CHAT,
    )
    record_raw_call(_response(), operation="account connect")
    snapshot = usage_snapshot()
    assert snapshot["calls"] == 3
    assert snapshot["by_operation"]["typing indicator"] == 1
    assert snapshot["by_operation"]["chat mark as read"] == 1


def test_media_upload_bills_the_bytes_that_were_sent():
    """Upload's cost is the payload, not the small JSON that comes back."""
    four_mb = 4 * 1024 * 1024
    record_raw_call(
        _response(body=b'{"data":{"jobId":"j"}}'),
        operation="media upload",
        account_id="a",
        media_bytes=four_mb,
        category=CATEGORY_VAULT,
    )
    snapshot = usage_snapshot()
    assert snapshot["media_bytes"] == four_mb
    assert snapshot["estimated_credits"] == pytest.approx(8.0)
    assert snapshot["credits_by_category"][CATEGORY_VAULT]["media_bytes"] == four_mb


def test_download_media_reports_transferred_bytes(monkeypatch):
    """The other half of media accounting, through the real helper."""
    import asyncio

    payload = b"x" * (2 * 1024 * 1024)

    class _Client:
        async def post(self, *args, **kwargs):
            request = httpx.Request("POST", "https://v1.apifansly.com/api/fansly/media/download")
            return httpx.Response(
                200,
                request=request,
                content=payload,
                headers={"content-type": "video/mp4"},
            )

    monkeypatch.setenv("APIFANSLY_API_KEY", "k")
    result = asyncio.run(
        apifansly.download_media(
            "https://cdn.fansly.com/asset.mp4", client=_Client()
        )
    )
    assert len(result) == len(payload)
    snapshot = usage_snapshot()
    assert snapshot["media_bytes"] == len(payload)
    assert snapshot["estimated_credits"] == pytest.approx(4.0)


def test_received_webhook_events_are_counted_separately():
    for index in range(WEBHOOK_EVENTS_PER_CREDIT * 2):
        record_webhook_event("ppv.purchased" if index % 2 else "message.new", account_id="a")
    snapshot = usage_snapshot()
    assert snapshot["webhook_events"] == WEBHOOK_EVENTS_PER_CREDIT * 2
    assert snapshot["estimated_webhook_credits"] == 2.0
    # Webhook credits are additive with call credits, not a replacement.
    assert snapshot["estimated_credits"] == pytest.approx(2.0)
    assert snapshot["webhook_events_by_type"]["message.new"] == WEBHOOK_EVENTS_PER_CREDIT


# --- categorisation and breakdown ------------------------------------------


def test_calls_are_attributed_to_the_scope_they_run_in():
    with usage_category(CATEGORY_BACKGROUND_HISTORY):
        record_raw_call(
            _response(body=b"x" * 1000),
            operation="chat message listing",
            account_id="acct-1",
        )
    record_raw_call(
        _response(), operation="message delivery", account_id="acct-2"
    )

    categories = usage_snapshot()["credits_by_category"]
    assert categories[CATEGORY_BACKGROUND_HISTORY]["calls"] == 1
    # Outside a scope the operation name classifies the call, so nothing lands
    # in an "unclassified" bucket that would make the breakdown useless.
    assert categories[CATEGORY_LIVE_CHAT]["calls"] == 1


def test_snapshot_breaks_down_by_operation_account_and_category():
    with usage_category(CATEGORY_BACKGROUND_HISTORY):
        record_usage_event(
            operation="chat message listing",
            account_id="acct-1",
            response_bytes=CREDIT_RESPONSE_BYTES_PER_CREDIT * 2,
        )
        record_usage_event(
            operation="chat message listing", account_id="acct-2", response_bytes=10
        )
    snapshot = usage_snapshot()
    assert snapshot["credits_by_operation"]["chat message listing"] == {
        "calls": 2,
        "response_bytes": CREDIT_RESPONSE_BYTES_PER_CREDIT * 2 + 10,
        "media_bytes": 0,
        "estimated_credits": 3.0,
    }
    assert snapshot["credits_by_account"]["acct-1"]["estimated_credits"] == 2.0
    assert snapshot["credits_by_account"]["acct-2"]["estimated_credits"] == 1.0


def test_snapshot_keeps_the_fields_the_dashboard_already_reads():
    record_usage_event(operation="chat listing", account_id="a", response_bytes=5)
    snapshot = usage_snapshot()
    for key in ("window_hours", "calls", "response_bytes", "by_operation", "by_account", "by_status", "note"):
        assert key in snapshot, key
    assert snapshot["by_status"]["0"] == 1


def test_monthly_run_rate_extrapolates_from_the_observed_window(monkeypatch):
    """A process that booted a minute ago has a minute of evidence, not a day."""
    import time

    monkeypatch.setattr(apifansly, "_USAGE_STARTED_AT", time.time() - 3600)
    for _ in range(10):
        record_usage_event(operation="chat listing", account_id="a")
    snapshot = usage_snapshot()
    # 10 credits in one observed hour -> 240/day -> 7,200/month.
    assert snapshot["estimated_daily_credits"] == pytest.approx(240.0, rel=0.05)
    assert snapshot["estimated_monthly_credits"] == pytest.approx(7200.0, rel=0.05)


def test_collect_usage_measures_one_unit_of_work():
    with collect_usage(CATEGORY_BACKGROUND_HISTORY) as spent:
        record_usage_event(
            operation="chat message listing",
            account_id="a",
            response_bytes=CREDIT_RESPONSE_BYTES_PER_CREDIT * 3,
        )
        record_usage_event(operation="chat message listing", account_id="a")
    assert summarize_usage_events(spent) == {
        "calls": 2,
        "response_bytes": CREDIT_RESPONSE_BYTES_PER_CREDIT * 3,
        "media_bytes": 0,
        "estimated_credits": 4.0,
    }


# --- the background budget --------------------------------------------------


def test_history_budget_is_unlimited_by_default(monkeypatch):
    monkeypatch.delenv("APIFANSLY_HISTORY_CREDIT_BUDGET_24H", raising=False)
    assert background_history_allowed() is True
    assert background_history_budget_state()["budget_credits"] is None


def test_history_budget_stops_background_history_when_spent(monkeypatch):
    monkeypatch.setenv("APIFANSLY_HISTORY_CREDIT_BUDGET_24H", "3")
    with usage_category(CATEGORY_BACKGROUND_HISTORY):
        for _ in range(3):
            record_usage_event(operation="chat message listing", account_id="a")
    state = background_history_budget_state()
    assert state["spent_credits"] == 3.0
    assert state["exhausted"] is True
    assert background_history_allowed() is False


def test_live_chat_spend_never_counts_against_the_history_budget(monkeypatch):
    """The invariant: live conversation cannot be stopped by a history budget."""
    monkeypatch.setenv("APIFANSLY_HISTORY_CREDIT_BUDGET_24H", "2")
    with usage_category(CATEGORY_LIVE_CHAT):
        for _ in range(50):
            record_usage_event(operation="message delivery", account_id="a")
    assert background_history_budget_state()["exhausted"] is False
    assert background_history_allowed() is True


# --- live priority ----------------------------------------------------------


def test_an_open_live_scope_makes_background_work_yield():
    assert live_work_in_progress() is False
    with usage_category(CATEGORY_LIVE_CHAT):
        assert live_work_in_progress() is True
    assert live_work_in_progress() is False


def test_a_recent_live_call_makes_background_work_yield():
    record_usage_event(operation="message delivery", account_id="a")
    assert live_work_in_progress() is True


def test_a_recent_background_call_does_not_make_background_work_yield():
    with usage_category(CATEGORY_BACKGROUND_HISTORY):
        record_usage_event(operation="chat message listing", account_id="a")
    assert live_work_in_progress() is False


def test_live_activity_expires(monkeypatch):
    import time

    record_usage_event(operation="message delivery", account_id="a")
    real_time = time.time
    monkeypatch.setattr(
        apifansly.time, "time", lambda: real_time() + apifansly.LIVE_ACTIVITY_WINDOW_SECONDS + 1
    )
    assert apifansly.live_activity_recent() is False
