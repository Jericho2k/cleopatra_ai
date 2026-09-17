"""Sprint 1 — a customer who cannot reach what they paid for gets it back.

``docs/autonomy_architecture_review.md`` §3B. The baseline answered "I can't
open it" by calling the platform directly with the existing price, outside the
delivery service, after commercial state had already moved. PR #48 replaced that
with a review hold and said, in as many words, that it was containment and that
a safe verified access-recovery workflow was still needed.

These tests hold the properties that make the repair safe rather than merely
faster:

* the ledger decides whether anything is owed — a complaint is not proof of
  payment;
* the resend carries no price, so a repair cannot become a second charge;
* the media that goes out is the media the purchase recorded, never a
  re-planned selection;
* nothing is reported as repaired, and no hold is cleared, until the platform
  has returned a receipt.
"""

from __future__ import annotations

import asyncio

import pytest

from services import content_access
from tests.fake_supabase import FakeSupabase


def run(coro):
    return asyncio.run(coro)


def _world(
    *,
    status: str = "purchased",
    review_reason: str = content_access.REVIEW_REASON,
    media_ids: list[str] | None = None,
    platform_message_id: str = "platform-1",
) -> FakeSupabase:
    return FakeSupabase(
        {
            "fans": [
                {
                    "id": "fan-1",
                    "creator_id": "creator-1",
                    "platform_fan_id": "99",
                    "fansly_group_id": "group-1",
                    "needs_human_review": True,
                    "review_reason": review_reason,
                }
            ],
            "creators": [
                {"id": "creator-1", "apifansly_account_id": "account-1"}
            ],
            "ppv_deliveries": [
                {
                    "reference": "ref-1",
                    "creator_id": "creator-1",
                    "fan_id": "fan-1",
                    "status": status,
                    "media_ids": ["m1", "m2"] if media_ids is None else media_ids,
                    "price_cents": 2500,
                    "platform_message_id": platform_message_id,
                    "purchased_at": "2026-09-16T10:00:00+00:00",
                    "claimed_at": "2026-09-16T09:00:00+00:00",
                }
            ],
            "messages": [],
        }
    )


@pytest.fixture
def world(monkeypatch):
    db = _world()
    monkeypatch.setattr(content_access, "get_supabase", lambda: db)
    monkeypatch.setattr("services.ppv_delivery_ledger.get_supabase", lambda: db)
    monkeypatch.setattr("db.queries.get_supabase", lambda: db)
    monkeypatch.setattr(content_access, "apifansly_enabled", lambda: True)
    return db


def _stub_platform(monkeypatch, *, messages, account_media, sends: list | None = None):
    from services import apifansly

    async def fake_list(account_id, chat_id, **kwargs):
        return list(messages), list(account_media), None

    async def fake_send(account_id, chat_id, **kwargs):
        if sends is not None:
            sends.append({"account_id": account_id, "chat_id": chat_id, **kwargs})
        # The documented envelope: data.data.response (apifansly.response_data).
        return {"data": {"data": {"response": {"id": "platform-resend-1"}}}}

    monkeypatch.setattr(apifansly, "list_chat_messages", fake_list)
    monkeypatch.setattr(apifansly, "send_message", fake_send)


# --- the evidence half ------------------------------------------------------


def test_the_operator_sees_what_was_paid_for_before_deciding_anything(
    world, monkeypatch
):
    _stub_platform(
        monkeypatch,
        messages=[{"id": "platform-1", "attachments": [{"contentId": "a1"}]}],
        account_media=[{"id": "a1", "mediaId": "m1"}],
    )

    evidence = run(content_access.inspect_content_access("fan-1")).as_dict()

    assert evidence["has_paid_content"] is True
    assert evidence["paid_items"][0]["reference"] == "ref-1"
    assert evidence["paid_items"][0]["price_cents"] == 2500
    assert evidence["paid_items"][0]["platform_state"] == content_access.PLATFORM_VISIBLE
    assert evidence["frozen"] is True


def test_a_message_the_platform_no_longer_lists_is_reported_as_gone(
    world, monkeypatch
):
    _stub_platform(monkeypatch, messages=[], account_media=[])

    evidence = run(content_access.inspect_content_access("fan-1"))

    assert evidence.paid_items[0].platform_state == content_access.PLATFORM_MESSAGE_GONE


def test_a_message_that_lost_its_media_is_told_apart_from_one_that_is_gone(
    world, monkeypatch
):
    _stub_platform(
        monkeypatch,
        messages=[{"id": "platform-1", "attachments": []}],
        account_media=[],
    )

    evidence = run(content_access.inspect_content_access("fan-1"))

    assert evidence.paid_items[0].platform_state == content_access.PLATFORM_MEDIA_GONE


def test_an_item_older_than_the_page_is_unknown_rather_than_missing(
    world, monkeypatch
):
    """A full page may not reach back far enough, and that is not evidence."""
    from services.apifansly import CHAT_MESSAGE_PAGE_MAX

    _stub_platform(
        monkeypatch,
        messages=[
            {"id": f"other-{i}", "attachments": []}
            for i in range(CHAT_MESSAGE_PAGE_MAX)
        ],
        account_media=[],
    )

    evidence = run(content_access.inspect_content_access("fan-1"))

    assert evidence.paid_items[0].platform_state == content_access.PLATFORM_UNKNOWN
    assert "not checked" in evidence.paid_items[0].platform_detail


def test_a_platform_read_that_fails_says_so_instead_of_guessing(world, monkeypatch):
    from services import apifansly

    async def boom(*_a, **_k):
        raise RuntimeError("provider down")

    monkeypatch.setattr(apifansly, "list_chat_messages", boom)

    evidence = run(content_access.inspect_content_access("fan-1")).as_dict()

    assert evidence["platform_error"]
    assert evidence["paid_items"][0]["platform_state"] == content_access.PLATFORM_UNKNOWN


def test_inspection_sends_nothing_and_clears_nothing(world, monkeypatch):
    sends: list = []
    _stub_platform(monkeypatch, messages=[], account_media=[], sends=sends)

    run(content_access.inspect_content_access("fan-1"))

    assert sends == []
    assert world.tables["fans"][0]["needs_human_review"] is True
    assert world.tables["messages"] == []


# --- the repair half --------------------------------------------------------


def test_a_repair_resends_the_paid_media_with_no_price(world, monkeypatch):
    sends: list = []
    _stub_platform(monkeypatch, messages=[], account_media=[], sends=sends)

    result = run(content_access.resend_paid_content("fan-1"))

    assert result["status"] == "resent"
    assert sends[0]["media_ids"] == ["m1", "m2"], "exactly what the purchase recorded"
    assert "price_dollars" not in sends[0], (
        "an unpriced send is what stops the platform charging again"
    )


def test_a_complaint_without_a_purchase_is_refused(monkeypatch):
    """A complaint is not proof of payment."""
    db = _world(status="abandoned")
    monkeypatch.setattr(content_access, "get_supabase", lambda: db)
    monkeypatch.setattr("services.ppv_delivery_ledger.get_supabase", lambda: db)
    monkeypatch.setattr(content_access, "apifansly_enabled", lambda: True)
    sends: list = []
    _stub_platform(monkeypatch, messages=[], account_media=[], sends=sends)

    with pytest.raises(content_access.ContentAccessError, match="no confirmed purchase"):
        run(content_access.resend_paid_content("fan-1"))
    assert sends == []


def test_a_purchase_with_no_recorded_media_is_refused(monkeypatch):
    db = _world(media_ids=[])
    monkeypatch.setattr(content_access, "get_supabase", lambda: db)
    monkeypatch.setattr("services.ppv_delivery_ledger.get_supabase", lambda: db)
    monkeypatch.setattr(content_access, "apifansly_enabled", lambda: True)
    sends: list = []
    _stub_platform(monkeypatch, messages=[], account_media=[], sends=sends)

    with pytest.raises(content_access.ContentAccessError, match="nothing safe to resend"):
        run(content_access.resend_paid_content("fan-1"))
    assert sends == []


def test_a_resend_with_no_receipt_is_not_reported_as_repaired(world, monkeypatch):
    """Delivery claims are tied to the operation result, not to the attempt."""
    from services import apifansly

    async def accepts_but_returns_nothing(*_a, **_k):
        return {"data": {"data": {"response": {}}}}

    monkeypatch.setattr(apifansly, "send_message", accepts_but_returns_nothing)

    with pytest.raises(content_access.ContentAccessError, match="no message id"):
        run(content_access.resend_paid_content("fan-1"))
    assert world.tables["messages"] == []


def test_a_repair_is_recorded_as_a_repair_and_never_as_a_sale(world, monkeypatch):
    _stub_platform(monkeypatch, messages=[], account_media=[], sends=[])

    run(content_access.resend_paid_content("fan-1"))

    saved = world.tables["messages"][0]
    repair = saved["media_context"]["content_access_repair"]
    assert repair["price_cents"] == 0
    assert repair["original_price_cents"] == 2500
    assert repair["restored_reference"] == "ref-1"
    assert "ppv" not in saved["media_context"], (
        "a repair row must not read back as a second PPV against the same media"
    )


# --- resolving the hold -----------------------------------------------------


def test_a_verified_repair_clears_the_hold(world, monkeypatch):
    _stub_platform(monkeypatch, messages=[], account_media=[], sends=[])

    result = run(
        content_access.resolve_content_access(
            "fan-1", resolution=content_access.RESOLUTION_RESEND
        )
    )

    assert result["review_cleared"] is True
    assert world.tables["fans"][0]["needs_human_review"] is False


def test_a_failed_repair_leaves_the_conversation_frozen(monkeypatch):
    """An operator must be able to tell "repaired" from "could not repair"."""
    db = _world(status="abandoned")
    monkeypatch.setattr(content_access, "get_supabase", lambda: db)
    monkeypatch.setattr("services.ppv_delivery_ledger.get_supabase", lambda: db)
    monkeypatch.setattr("db.queries.get_supabase", lambda: db)
    monkeypatch.setattr(content_access, "apifansly_enabled", lambda: True)
    _stub_platform(monkeypatch, messages=[], account_media=[], sends=[])

    with pytest.raises(content_access.ContentAccessError):
        run(
            content_access.resolve_content_access(
                "fan-1", resolution=content_access.RESOLUTION_RESEND
            )
        )
    assert db.tables["fans"][0]["needs_human_review"] is True


def test_an_operator_may_record_that_they_restored_access_themselves(world):
    result = run(
        content_access.resolve_content_access(
            "fan-1", resolution=content_access.RESOLUTION_ACCESS_RESTORED
        )
    )

    assert result["status"] == "access_restored_by_operator"
    assert world.tables["fans"][0]["needs_human_review"] is False


def test_an_operator_may_record_that_the_analyzer_misread_it(world):
    result = run(
        content_access.resolve_content_access(
            "fan-1", resolution=content_access.RESOLUTION_NOT_AN_ACCESS_ISSUE
        )
    )

    assert result["status"] == "not_an_access_issue"
    assert world.tables["fans"][0]["needs_human_review"] is False


def test_these_resolutions_only_apply_to_an_access_hold(monkeypatch):
    db = _world(review_reason="ppv_send_failed")
    monkeypatch.setattr(content_access, "get_supabase", lambda: db)
    monkeypatch.setattr(content_access, "apifansly_enabled", lambda: True)

    with pytest.raises(content_access.ContentAccessError, match="not on hold"):
        run(
            content_access.resolve_content_access(
                "fan-1", resolution=content_access.RESOLUTION_ACCESS_RESTORED
            )
        )


def test_an_unsupported_resolution_is_refused(world):
    with pytest.raises(content_access.ContentAccessError, match="unsupported"):
        run(content_access.resolve_content_access("fan-1", resolution="send_more"))


# --- and plain "resume AI" no longer walks past the complaint ---------------


def test_resuming_ai_on_an_access_hold_is_refused(world, monkeypatch):
    """This is how the baseline came to answer a complaint with another sale."""
    from services import ppv_recovery

    monkeypatch.setattr(ppv_recovery, "get_supabase", lambda: world)

    with pytest.raises(ppv_recovery.PPVRecoveryError, match="cannot access paid content"):
        run(ppv_recovery.resolve_fan_review("fan-1", resolution="resume_ai"))
    assert world.tables["fans"][0]["needs_human_review"] is True


def test_the_access_resolutions_are_reachable_through_the_review_workflow(
    world, monkeypatch
):
    from services import ppv_recovery

    monkeypatch.setattr(ppv_recovery, "get_supabase", lambda: world)

    result = run(
        ppv_recovery.resolve_fan_review(
            "fan-1", resolution=content_access.RESOLUTION_NOT_AN_ACCESS_ISSUE
        )
    )

    assert result["review_cleared"] is True
    assert world.tables["fans"][0]["needs_human_review"] is False
