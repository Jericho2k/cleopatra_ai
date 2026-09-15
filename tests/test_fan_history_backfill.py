"""Historical conversation backfill: cursor resume, idempotency, priority, cost.

The system these tests describe exists because of one upstream fact: API Fansly
documents `limit min=1 max=10` on the chat-messages endpoint, so a
5,000-message fan is 500 provider round trips and nothing local can change
that. Everything else — the durable cursor, the bounded runs, the warm resume,
the yield to live work — follows from refusing to pay those 500 calls twice or
to put them in front of a fan waiting for a reply.
"""

import asyncio

import pytest

from services import apifansly, fan_history
from services.apifansly import CATEGORY_LIVE_CHAT, usage_category
from tests.fake_supabase import FakeSupabase


CREATOR = "creator-1"
FAN = "fan-1"
GROUP = "group-1"
ACCOUNT = "api-account-1"
CREATOR_PLATFORM = "999"


def _message(index: int, *, sender: str = "111", attachments=None) -> dict:
    return {
        "id": f"m{index}",
        "content": f"message {index}",
        "senderId": sender,
        # Descending, because the provider returns newest first.
        "createdAt": 1_700_000_000_000 - index * 1_000,
        "attachments": attachments or [],
    }


class FakeProvider:
    """A paginated chat-messages endpoint with a hard 10-per-page ceiling.

    Records every cursor it was asked for, which is how "did a restart refetch
    pages it had already paid for" becomes an assertion rather than a hope.
    """

    def __init__(self, total_messages: int, *, page_bytes: int = 2_000):
        self.messages = [_message(index) for index in range(total_messages)]
        self.page_bytes = page_bytes
        self.cursors_requested: list[str | None] = []
        self.account_media: list[dict] = []

    async def list_chat_messages(
        self, account_id, chat_id, *, cursor=None, limit=10, client=None
    ):
        assert limit <= 10, "the provider's documented maximum is 10"
        self.cursors_requested.append(cursor)
        start = int(cursor) if cursor else 0
        page = self.messages[start : start + limit]
        # Record the page's real size, exactly as the transport would.
        apifansly.record_usage_event(
            operation="chat message listing",
            account_id=account_id,
            response_bytes=self.page_bytes,
        )
        next_start = start + len(page)
        next_cursor = str(next_start) if next_start < len(self.messages) else None
        return page, list(self.account_media), next_cursor


@pytest.fixture
def db(monkeypatch):
    fake = FakeSupabase(
        {
            "fans": [
                {
                    "id": FAN,
                    "creator_id": CREATOR,
                    "fansly_group_id": GROUP,
                    "platform_fan_id": "111",
                }
            ],
            "creators": [
                {
                    "id": CREATOR,
                    "apifansly_account_id": ACCOUNT,
                    "fansly_account_id": CREATOR_PLATFORM,
                }
            ],
            "messages": [],
            "fan_history_backfill": [],
            "creator_vault_media": [],
        }
    )
    for module in (
        "services.fan_history",
        "db.fan_history_queries",
        "core.pagination",
    ):
        monkeypatch.setattr(
            __import__(module, fromlist=["get_supabase"]),
            "get_supabase",
            lambda: fake,
            raising=False,
        )
    return fake


@pytest.fixture
def provider(monkeypatch):
    fake = FakeProvider(total_messages=95)
    monkeypatch.setattr(fan_history, "list_chat_messages", fake.list_chat_messages)
    monkeypatch.setattr(
        fan_history, "client_scope", _null_client_scope, raising=True
    )
    return fake


def _null_client_scope():
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _scope():
        yield None

    return _scope()


def _state(db) -> dict:
    rows = db.tables["fan_history_backfill"]
    assert rows, "no backfill state was written"
    return rows[0]


# --- cursor resume and restart safety ---------------------------------------


def test_backfill_advances_the_cursor_page_by_page(db, provider):
    result = asyncio.run(
        fan_history.advance_backfill(
            creator_id=CREATOR, fan_id=FAN, max_pages=3, respect_live_priority=False
        )
    )
    assert result["pages"] == 3
    assert result["imported"] == 30
    assert provider.cursors_requested == [None, "10", "20"]
    assert _state(db)["page_cursor"] == "30"


def test_a_restart_resumes_from_the_stored_cursor(db, provider):
    """The whole point of HIST-001: a deploy must not repay for 30 pages."""
    asyncio.run(
        fan_history.advance_backfill(
            creator_id=CREATOR, fan_id=FAN, max_pages=3, respect_live_priority=False
        )
    )
    provider.cursors_requested.clear()

    # A "restart": nothing in memory, only the durable row.
    asyncio.run(
        fan_history.advance_backfill(
            creator_id=CREATOR, fan_id=FAN, max_pages=2, respect_live_priority=False
        )
    )
    assert provider.cursors_requested == ["30", "40"]
    assert _state(db)["pages_fetched"] == 5


def test_backfill_completes_when_the_provider_cursor_runs_out(db, provider):
    result = asyncio.run(
        fan_history.advance_backfill(
            creator_id=CREATOR, fan_id=FAN, max_pages=50, respect_live_priority=False
        )
    )
    assert result["pages"] == 10  # 95 messages at the provider's max of 10
    assert result["stop_reason"] == "exhausted"
    state = _state(db)
    assert state["exhausted"] is True
    assert state["status"] == "complete"
    assert state["messages_imported"] == 95


def test_a_completed_backfill_makes_no_further_provider_calls(db, provider):
    asyncio.run(
        fan_history.advance_backfill(
            creator_id=CREATOR, fan_id=FAN, max_pages=50, respect_live_priority=False
        )
    )
    provider.cursors_requested.clear()
    result = asyncio.run(
        fan_history.advance_backfill(
            creator_id=CREATOR, fan_id=FAN, respect_live_priority=False
        )
    )
    assert result["status"] == "complete"
    assert provider.cursors_requested == []


# --- idempotency ------------------------------------------------------------


def test_replaying_a_page_imports_nothing_twice(db, provider):
    """Persistence is idempotent on platform message identity."""
    asyncio.run(
        fan_history.advance_backfill(
            creator_id=CREATOR, fan_id=FAN, max_pages=2, respect_live_priority=False
        )
    )
    assert len(db.tables["messages"]) == 20

    # Rewind the durable cursor, as a crash between fetch and checkpoint would.
    _state(db)["page_cursor"] = None
    _state(db)["exhausted"] = False

    second = asyncio.run(
        fan_history.advance_backfill(
            creator_id=CREATOR, fan_id=FAN, max_pages=2, respect_live_priority=False
        )
    )
    assert second["imported"] == 0
    assert len(db.tables["messages"]) == 20


def test_messages_are_stored_with_platform_identity_and_role(db, provider):
    asyncio.run(
        fan_history.advance_backfill(
            creator_id=CREATOR, fan_id=FAN, max_pages=1, respect_live_priority=False
        )
    )
    row = db.tables["messages"][0]
    assert row["fansly_message_id"].startswith("m")
    assert row["creator_id"] == CREATOR
    assert row["fan_id"] == FAN
    # senderId 111 is the fan, not the creator's platform id.
    assert row["role"] == "fan"


def test_creator_messages_are_attributed_to_the_creator(db, provider):
    provider.messages = [_message(0, sender=CREATOR_PLATFORM)]
    asyncio.run(
        fan_history.advance_backfill(
            creator_id=CREATOR, fan_id=FAN, max_pages=1, respect_live_priority=False
        )
    )
    assert db.tables["messages"][0]["role"] == "creator"


# --- media: metadata only, never a download ---------------------------------


def test_history_persists_media_metadata_without_downloading_anything(db, provider, monkeypatch):
    """The invariant that keeps a backfill from costing 2 credits/MB.

    The page already carries accountMedia. History reads it and stores id,
    type, price, purchased and access — and makes no media call at all.
    """
    def _explode(*args, **kwargs):
        raise AssertionError("history backfill must never download media")

    monkeypatch.setattr(apifansly, "download_media", _explode)
    provider.messages = [
        _message(0, attachments=[{"contentId": "media-7", "contentType": 1}])
    ]
    provider.account_media = [
        {
            "id": "media-7",
            "price": 2500,
            "purchased": True,
            "access": "granted",
            "media": {
                "mimetype": "video/mp4",
                "locations": [{"location": "https://cdn.fansly.com/v.mp4"}],
            },
        }
    ]

    asyncio.run(
        fan_history.advance_backfill(
            creator_id=CREATOR, fan_id=FAN, max_pages=1, respect_live_priority=False
        )
    )
    attachment = db.tables["messages"][0]["media_context"]["attachments"][0]
    assert attachment["contentId"] == "media-7"
    assert attachment["is_ppv"] is True
    assert attachment["purchased"] is True
    assert attachment["access"] == "granted"
    assert attachment["mimetype"] == "video/mp4"
    # No media bytes were ever transferred, so nothing was billed at 2/MB.
    assert apifansly.usage_snapshot()["media_bytes"] == 0


def test_media_metadata_is_linked_to_local_vault_rows_when_we_have_them(db, provider):
    db.tables["creator_vault_media"].append(
        {
            "id": "vault-row-1",
            "creator_id": CREATOR,
            "fansly_media_id": "media-7",
            "content_category": "shower",
            "mimetype": "video/mp4",
        }
    )
    provider.messages = [
        _message(0, attachments=[{"contentId": "media-7", "contentType": 1}])
    ]
    provider.account_media = [{"id": "media-7", "price": 0, "media": {}}]

    asyncio.run(
        fan_history.advance_backfill(
            creator_id=CREATOR, fan_id=FAN, max_pages=1, respect_live_priority=False
        )
    )
    attachment = db.tables["messages"][0]["media_context"]["attachments"][0]
    assert attachment["vault_media_id"] == "vault-row-1"
    assert attachment["vault_content_category"] == "shower"


# --- warm resume ------------------------------------------------------------


def test_warm_resume_does_nothing_when_local_context_is_already_enough(db, provider):
    db.tables["messages"].extend(
        {
            "id": f"local-{index}",
            "fan_id": FAN,
            "creator_id": CREATOR,
            "role": "fan",
            "content": "hi",
            "fansly_message_id": f"existing-{index}",
            "sent_at": "2026-01-01T00:00:00+00:00",
        }
        for index in range(20)
    )
    result = asyncio.run(fan_history.warm_resume(creator_id=CREATOR, fan_id=FAN))
    assert result["status"] == "sufficient_context"
    assert result["pages"] == 0
    assert provider.cursors_requested == []


def test_warm_resume_fetches_only_the_newest_few_pages(db, provider):
    """A returning fan is answerable after three pages, not five hundred."""
    result = asyncio.run(fan_history.warm_resume(creator_id=CREATOR, fan_id=FAN))
    assert result["status"] == "resumed"
    assert result["pages"] == fan_history.warm_resume_max_pages()
    assert result["imported"] == 30
    assert len(provider.cursors_requested) == 3


def test_warm_resume_leaves_a_deep_cursor_alone(db, provider):
    """A backfill 400 pages in must never be rewound by a warm resume."""
    asyncio.run(
        fan_history.advance_backfill(
            creator_id=CREATOR, fan_id=FAN, max_pages=4, respect_live_priority=False
        )
    )
    db.tables["messages"].clear()  # thin local context, deep cursor intact
    deep_cursor = _state(db)["page_cursor"]

    asyncio.run(fan_history.warm_resume(creator_id=CREATOR, fan_id=FAN))
    assert _state(db)["page_cursor"] == deep_cursor


def test_warm_resume_still_records_what_it_spent(db, provider):
    asyncio.run(
        fan_history.advance_backfill(
            creator_id=CREATOR, fan_id=FAN, max_pages=1, respect_live_priority=False
        )
    )
    before = float(_state(db)["estimated_credits"])
    db.tables["messages"].clear()
    asyncio.run(fan_history.warm_resume(creator_id=CREATOR, fan_id=FAN))
    assert float(_state(db)["estimated_credits"]) > before


# --- live priority ----------------------------------------------------------


def test_deep_history_yields_to_an_active_conversation(db, provider):
    with usage_category(CATEGORY_LIVE_CHAT):
        result = asyncio.run(
            fan_history.advance_backfill(creator_id=CREATOR, fan_id=FAN)
        )
    assert result["status"] == "paused"
    assert result["reason"] == "live_work_in_progress"
    assert provider.cursors_requested == []


def test_deep_history_stops_mid_run_when_a_reply_arrives(db, provider):
    """Priority is re-checked before EVERY page, not once per batch.

    A fan whose reply lands after page two must not wait out pages three
    through twenty.
    """
    real = provider.list_chat_messages
    calls = {"n": 0}

    async def _with_interruption(*args, **kwargs):
        page = await real(*args, **kwargs)
        calls["n"] += 1
        if calls["n"] == 2:
            # Exactly what the real delivery path records: an explicit
            # live-chat category, because that call is made from another task
            # and is not inside this backfill's usage scope.
            apifansly.record_usage_event(
                operation="message delivery",
                account_id=ACCOUNT,
                category=CATEGORY_LIVE_CHAT,
            )
        return page

    fan_history.list_chat_messages = _with_interruption
    try:
        result = asyncio.run(
            fan_history.advance_backfill(creator_id=CREATOR, fan_id=FAN, max_pages=20)
        )
    finally:
        fan_history.list_chat_messages = real

    assert result["pages"] == 2
    assert result["stop_reason"] == "live_work_in_progress"
    assert _state(db)["status"] == "paused"
    # And the cursor kept everything already paid for.
    assert _state(db)["page_cursor"] == "20"


def test_the_scheduler_does_nothing_while_live_work_is_happening(db, provider, monkeypatch):
    monkeypatch.setenv("HISTORY_BACKFILL_ENABLED", "true")
    with usage_category(CATEGORY_LIVE_CHAT):
        result = asyncio.run(fan_history.backfill_scheduler_pass())
    assert result["status"] == "yielded"
    assert result["fans"] == 0


def test_the_scheduler_is_off_unless_explicitly_enabled(db, provider, monkeypatch):
    monkeypatch.delenv("HISTORY_BACKFILL_ENABLED", raising=False)
    assert asyncio.run(fan_history.backfill_scheduler_pass())["status"] == "disabled"


# --- credit budget ----------------------------------------------------------


def test_backfill_pauses_when_the_history_budget_is_spent(db, provider, monkeypatch):
    monkeypatch.setenv("APIFANSLY_HISTORY_CREDIT_BUDGET_24H", "3")
    result = asyncio.run(
        fan_history.advance_backfill(
            creator_id=CREATOR, fan_id=FAN, max_pages=20, respect_live_priority=False
        )
    )
    # Three pages at one credit each, then the budget stops it.
    assert result["pages"] == 3
    assert result["stop_reason"] == "history_credit_budget_exhausted"
    assert _state(db)["status"] == "paused"


def test_an_exhausted_history_budget_never_stops_live_work(db, provider, monkeypatch):
    """The safety invariant. A history budget is not a kill switch.

    Nothing on the live path consults the history budget, so an exhausted one
    leaves live conversation, delivery and reconciliation completely untouched.
    """
    monkeypatch.setenv("APIFANSLY_HISTORY_CREDIT_BUDGET_24H", "1")
    asyncio.run(
        fan_history.advance_backfill(
            creator_id=CREATOR, fan_id=FAN, max_pages=20, respect_live_priority=False
        )
    )
    assert apifansly.background_history_allowed() is False

    # A live send records normally and is refused by nothing.
    with usage_category(CATEGORY_LIVE_CHAT):
        apifansly.record_usage_event(operation="message delivery", account_id=ACCOUNT)
    assert apifansly.usage_snapshot()["credits_by_category"][CATEGORY_LIVE_CHAT]["calls"] == 1


def test_a_paused_backfill_resumes_once_the_budget_allows(db, provider, monkeypatch):
    monkeypatch.setenv("APIFANSLY_HISTORY_CREDIT_BUDGET_24H", "2")
    asyncio.run(
        fan_history.advance_backfill(
            creator_id=CREATOR, fan_id=FAN, max_pages=20, respect_live_priority=False
        )
    )
    monkeypatch.delenv("APIFANSLY_HISTORY_CREDIT_BUDGET_24H", raising=False)
    provider.cursors_requested.clear()
    asyncio.run(
        fan_history.advance_backfill(
            creator_id=CREATOR, fan_id=FAN, max_pages=2, respect_live_priority=False
        )
    )
    assert provider.cursors_requested == ["20", "30"]


# --- cost telemetry ---------------------------------------------------------


def test_large_pages_are_billed_from_their_real_size(db, monkeypatch):
    """An accountMedia-heavy page is not one credit, and history must say so."""
    big = FakeProvider(total_messages=30, page_bytes=240 * 1024)
    monkeypatch.setattr(fan_history, "list_chat_messages", big.list_chat_messages)
    monkeypatch.setattr(fan_history, "client_scope", _null_client_scope)

    result = asyncio.run(
        fan_history.advance_backfill(
            creator_id=CREATOR, fan_id=FAN, max_pages=3, respect_live_priority=False
        )
    )
    # 240 KB is three times the 80 KB threshold, so three credits a page.
    assert result["estimated_credits"] == pytest.approx(9.0)
    assert float(_state(db)["estimated_credits"]) == pytest.approx(9.0)


def test_progress_view_reports_cost_and_what_is_left(db, provider):
    asyncio.run(
        fan_history.advance_backfill(
            creator_id=CREATOR, fan_id=FAN, max_pages=3, respect_live_priority=False
        )
    )
    view = asyncio.run(fan_history.fan_history_status(FAN))["history"]
    assert view["pages_fetched"] == 3
    assert view["messages_imported"] == 30
    assert view["api_calls"] == 3
    assert view["estimated_credits_per_page"] == 1.0
    assert view["fully_paged"] is False
    assert view["messages_per_page_max"] == 10
    # Honest rather than invented: the provider does not say how long a
    # conversation is until the cursor runs out, so this is null, not a guess.
    assert view["remaining_pages"] is None
    assert view["remaining_pages_note"] == (
        "unknown until the provider cursor is exhausted"
    )


def test_progress_view_reports_completion(db, provider):
    asyncio.run(
        fan_history.advance_backfill(
            creator_id=CREATOR, fan_id=FAN, max_pages=50, respect_live_priority=False
        )
    )
    view = asyncio.run(fan_history.fan_history_status(FAN))["history"]
    assert view["fully_paged"] is True
    assert view["remaining_pages"] == 0


def test_five_thousand_messages_costs_five_hundred_pages(db):
    """The economics this design exists to make visible.

    Ten per page is the provider's documented ceiling, so this number is a
    property of the upstream API, not of our pagination.
    """
    from services.apifansly import CHAT_MESSAGE_PAGE_MAX

    assert CHAT_MESSAGE_PAGE_MAX == 10
    pages = -(-5_000 // CHAT_MESSAGE_PAGE_MAX)
    assert pages == 500
    # At one credit a page that is 500 credits; a media-heavy 240 KB page is
    # three, so the same import is 1,500. Estimating from bytes is what makes
    # the difference visible before it is billed.
    assert fan_history.estimate_page_credits(240 * 1024) == pytest.approx(3.0)
    assert fan_history.estimate_page_credits(2_000) == 1.0


def test_backfill_state_is_unavailable_rather_than_fatal_before_migration(db, provider, monkeypatch):
    """Shipping the code ahead of the migration disables history, not replies."""
    from db import fan_history_queries

    def _explode(*args, **kwargs):
        raise RuntimeError('relation "fan_history_backfill" does not exist')

    monkeypatch.setattr(fan_history_queries, "get_supabase", _explode)
    result = asyncio.run(
        fan_history.advance_backfill(creator_id=CREATOR, fan_id=FAN, respect_live_priority=False)
    )
    assert result["status"] == "unavailable"
    assert provider.cursors_requested == []


# --- isolation boundaries ---------------------------------------------------


def test_history_makes_no_provider_call_while_the_connector_is_off(
    db, provider, monkeypatch
):
    """The deployment-wide kill switch covers history like every other path."""
    monkeypatch.setenv("APIFANSLY_ENABLED", "false")
    warm = asyncio.run(fan_history.warm_resume(creator_id=CREATOR, fan_id=FAN))
    deep = asyncio.run(
        fan_history.advance_backfill(
            creator_id=CREATOR, fan_id=FAN, respect_live_priority=False
        )
    )
    assert warm["status"] == "connector_disabled"
    assert deep["reason"] == "connector_disabled"
    assert provider.cursors_requested == []


def test_a_simulator_test_fan_is_never_imported_from_the_platform(db, provider):
    """Requirement 6: a test fan must never reach the real platform.

    A test fan can carry a stale fansly_group_id, so the platform id — not the
    presence of a binding — is what decides. A "historical import" here would be
    a real, billed provider call against a conversation that does not exist.
    """
    db.tables["fans"][0]["platform_fan_id"] = "test_a1b2c3d4e5f6"
    warm = asyncio.run(fan_history.warm_resume(creator_id=CREATOR, fan_id=FAN))
    deep = asyncio.run(
        fan_history.advance_backfill(
            creator_id=CREATOR, fan_id=FAN, respect_live_priority=False
        )
    )
    assert warm["status"] == "simulation_fan"
    assert deep["reason"] == "simulation_fan"
    assert provider.cursors_requested == []


def test_a_refusal_does_not_disturb_an_existing_cursor(db, provider, monkeypatch):
    asyncio.run(
        fan_history.advance_backfill(
            creator_id=CREATOR, fan_id=FAN, max_pages=3, respect_live_priority=False
        )
    )
    cursor = _state(db)["page_cursor"]
    monkeypatch.setenv("APIFANSLY_ENABLED", "false")
    asyncio.run(
        fan_history.advance_backfill(
            creator_id=CREATOR, fan_id=FAN, respect_live_priority=False
        )
    )
    assert _state(db)["page_cursor"] == cursor
    assert _state(db)["status"] != "error"
