"""API Fansly list transport: URLs, envelope unwrapping, pagination, auth errors.

Every call goes through the shared services/apifansly.py abstraction, so these
tests also pin that the list endpoints reuse its usage metering, headers, and
account-access error handling rather than issuing ad-hoc requests.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from services import apifansly, fansly_lists
from services.apifansly import (
    ApiFanslyAccountAccessError,
    ApiFanslyProtocolError,
    list_account_list_members,
    list_account_lists,
)


@pytest.fixture(autouse=True)
def api_env(monkeypatch):
    monkeypatch.setenv("APIFANSLY_API_KEY", "test-key")
    monkeypatch.delenv("APIFANSLY_BASE_URL", raising=False)
    monkeypatch.delenv("APIFANSLY_LISTS_PATH", raising=False)
    monkeypatch.delenv("APIFANSLY_LIST_ITEMS_PATH", raising=False)


def _envelope(response):
    return {"data": {"data": {"response": response}}}


def _client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_lists_use_the_documented_account_scoped_path_and_auth():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["api_key"] = request.headers.get("x-api-key")
        return httpx.Response(200, json=_envelope({"lists": [{"id": "1", "label": "VIP"}]}))

    async def run():
        async with _client(handler) as client:
            return await list_account_lists("acct-1", client=client)

    lists, cursor = asyncio.run(run())

    assert seen["url"].startswith(f"{apifansly.DEFAULT_BASE_URL}/acct-1/lists")
    assert seen["api_key"] == "test-key"
    assert lists == [{"external_list_id": "1", "name": "VIP", "item_count": 0}]
    assert cursor is None


def test_list_members_use_the_nested_items_path():
    seen = {}

    def handler(request):
        seen["path"] = request.url.path
        return httpx.Response(200, json=_envelope({"items": [{"accountId": "p-a"}]}))

    async def run():
        async with _client(handler) as client:
            return await list_account_list_members("acct-1", "1001", client=client)

    members, _ = asyncio.run(run())

    assert seen["path"].endswith("/acct-1/lists/1001/items")
    assert members == ["p-a"]


def test_endpoint_paths_are_overridable_without_a_deploy(monkeypatch):
    monkeypatch.setenv("APIFANSLY_LISTS_PATH", "account-lists")
    monkeypatch.setenv("APIFANSLY_LIST_ITEMS_PATH", "members")
    seen = {}

    def handler(request):
        seen["path"] = request.url.path
        return httpx.Response(200, json=_envelope({"items": []}))

    async def run():
        async with _client(handler) as client:
            await list_account_list_members("acct-1", "1001", client=client)

    asyncio.run(run())

    assert seen["path"].endswith("/acct-1/account-lists/1001/members")


def test_list_pagination_follows_the_cursor_to_the_last_page():
    pages = [
        (
            {"lists": [{"id": "1", "label": "VIP"}], "cursor": "c1"},
            None,
        ),
        (
            {"lists": [{"id": "2", "label": "Whales"}]},
            None,
        ),
    ]
    requested_cursors: list[str | None] = []

    def handler(request):
        cursor = request.url.params.get("cursor")
        requested_cursors.append(cursor)
        body, _ = pages[0] if cursor is None else pages[1]
        return httpx.Response(200, json=_envelope(body))

    async def run():
        async with _client(handler) as client:
            return await fansly_lists.fetch_remote_lists("acct-1", client=client)

    lists = asyncio.run(run())

    assert requested_cursors == [None, "c1"]
    assert [row["external_list_id"] for row in lists] == ["1", "2"]


def test_member_pagination_follows_the_cursor_and_deduplicates():
    def handler(request):
        cursor = request.url.params.get("cursor")
        if cursor is None:
            body = {"items": [{"accountId": "p-a"}, {"accountId": "p-b"}], "cursor": "c1"}
        else:
            # Overlapping page boundaries must not produce duplicate members.
            body = {"items": [{"accountId": "p-b"}, {"accountId": "p-c"}]}
        return httpx.Response(200, json=_envelope(body))

    async def run():
        async with _client(handler) as client:
            return await fansly_lists.fetch_remote_members("acct-1", "1001", client=client)

    assert asyncio.run(run()) == ["p-a", "p-b", "p-c"]


def test_pagination_stops_rather_than_looping_on_a_repeating_cursor(monkeypatch):
    monkeypatch.setattr(fansly_lists, "_MAX_LIST_PAGES", 3)
    calls = {"count": 0}

    def handler(request):
        calls["count"] += 1
        return httpx.Response(
            200,
            json=_envelope({"lists": [{"id": "1", "label": "VIP"}], "cursor": "always"}),
        )

    async def run():
        async with _client(handler) as client:
            return await fansly_lists.fetch_remote_lists("acct-1", client=client)

    lists = asyncio.run(run())

    assert calls["count"] == 3
    assert [row["external_list_id"] for row in lists] == ["1"]


def test_nextcursor_envelope_form_is_understood():
    def handler(request):
        cursor = request.url.params.get("cursor")
        if cursor is None:
            return httpx.Response(
                200,
                json={
                    "data": {
                        "data": {"response": {"lists": [{"id": "1", "label": "VIP"}]}},
                        "nextCursor": "c1",
                    }
                },
            )
        return httpx.Response(200, json=_envelope({"lists": [{"id": "2", "label": "W"}]}))

    async def run():
        async with _client(handler) as client:
            return await fansly_lists.fetch_remote_lists("acct-1", client=client)

    assert [row["external_list_id"] for row in asyncio.run(run())] == ["1", "2"]


@pytest.mark.parametrize("status", [401, 403])
def test_access_errors_reuse_the_shared_reconnect_exception(status):
    def handler(request):
        return httpx.Response(status, json={"error": "no access"})

    async def run():
        async with _client(handler) as client:
            await list_account_lists("acct-1", client=client)

    with pytest.raises(ApiFanslyAccountAccessError) as exc:
        asyncio.run(run())

    assert "acct-1" in str(exc.value)
    assert "Reconnect this creator" in str(exc.value)


def test_server_errors_surface_rather_than_returning_an_empty_list():
    def handler(request):
        return httpx.Response(500, json={"error": "boom"})

    async def run():
        async with _client(handler) as client:
            await list_account_lists("acct-1", client=client)

    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(run())


def test_a_missing_envelope_is_a_protocol_error_not_an_empty_result():
    def handler(request):
        return httpx.Response(200, json={"unexpected": True})

    async def run():
        async with _client(handler) as client:
            await list_account_lists("acct-1", client=client)

    with pytest.raises(ApiFanslyProtocolError):
        asyncio.run(run())


def test_list_calls_are_counted_by_the_shared_usage_meter():
    """List traffic shows up in the same operator-facing usage diagnostics."""

    def handler(request):
        return httpx.Response(200, json=_envelope({"lists": []}))

    before = apifansly.usage_snapshot()

    async def run():
        async with _client(handler) as client:
            await list_account_lists("acct-1", client=client)

    asyncio.run(run())
    after = apifansly.usage_snapshot()

    assert after["calls"] == before["calls"] + 1
    assert after["by_operation"]["account list listing"] == (
        before["by_operation"].get("account list listing", 0) + 1
    )
    assert after["by_account"]["acct-1"] == before["by_account"].get("acct-1", 0) + 1
