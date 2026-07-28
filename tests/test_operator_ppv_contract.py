from __future__ import annotations

import asyncio
import inspect
from pathlib import Path

import pytest

from main import read_operator_ppv_options
from models.commercial import CreatorPolicy
from services import ppv_delivery
from services.ppv_delivery import (
    PPVDeliveryError,
    send_locked_ppv,
    verify_locked_ppv,
)
from services.ppv_persistence import persist_ppv_reconciliation


def test_supervised_auto_is_opt_in():
    assert CreatorPolicy().require_operator_ppv_approval is False
    assert CreatorPolicy(require_operator_ppv_approval=True).require_operator_ppv_approval is True


def test_ppv_payment_hold_defaults_to_two_hours():
    assert CreatorPolicy().ppv_payment_window_hours == 2


def test_manual_ppv_is_not_blocked_by_an_existing_offer():
    source = inspect.getsource(send_locked_ppv)
    assert 'if source != "operator" and fan.get("pending_ppv_check")' in source
    assert 'if source != "operator" and session and step_index is None' in source

    migration = (
        Path(__file__).resolve().parents[1] / "db" / "operator_ppv_concurrency_v1.sql"
    ).read_text(encoding="utf-8")
    assert "drop index if exists public.ppv_deliveries_one_active_per_fan_idx" in migration
    assert "source <> 'operator'" in migration
    assert "return 'operator_tracked'" in migration


def test_locked_ppv_rejects_empty_media_before_any_delivery():
    with pytest.raises(PPVDeliveryError, match="at least one media"):
        asyncio.run(
            send_locked_ppv(
                creator_id="creator",
                fan_id="fan",
                media_ids=[],
                price_cents=3000,
                message_content="here",
                source="operator",
                was_ai_suggested=False,
            )
        )


def test_delivery_contract_persists_only_after_platform_acceptance():
    source = inspect.getsource(send_locked_ppv)
    persistence_source = inspect.getsource(persist_ppv_reconciliation)
    acceptance_check = "response_body = await send_apifansly_message("
    assert source.index(acceptance_check) < source.index("pending = {")
    assert "platform accepted PPV but did not return a message ID" in source
    assert source.index(acceptance_check) < source.index("save_ppv_message_receipt(")
    assert "FanStatus.PAYMENT_PENDING" in persistence_source
    assert "PPV_RECONCILE" in persistence_source
    assert "ppv_sent_but_reconciliation_not_persisted" in source


def test_operator_can_accept_delayed_account_media_as_provisional(monkeypatch):
    async def page(*_args, **_kwargs):
        return (
            [{
                "id": "message-1",
                "attachments": [{"contentId": "account-media-delayed"}],
            }],
            [],
            None,
        )

    async def no_sleep(_delay):
        return None

    monkeypatch.setattr(ppv_delivery, "list_chat_messages", page)
    monkeypatch.setattr(ppv_delivery.asyncio, "sleep", no_sleep)

    result = asyncio.run(verify_locked_ppv(
        account_id="account",
        group_id="group",
        platform_message_id="message-1",
        media_ids=["media-1"],
        price_cents=3500,
        allow_provisional=True,
    ))

    assert result["provisional"] is True
    assert result["reason"] == "account_media_not_visible"


def test_automated_ppv_still_rejects_delayed_account_media(monkeypatch):
    async def page(*_args, **_kwargs):
        return (
            [{
                "id": "message-1",
                "attachments": [{"contentId": "account-media-delayed"}],
            }],
            [],
            None,
        )

    async def no_sleep(_delay):
        return None

    monkeypatch.setattr(ppv_delivery, "list_chat_messages", page)
    monkeypatch.setattr(ppv_delivery.asyncio, "sleep", no_sleep)

    with pytest.raises(PPVDeliveryError, match="account_media_not_visible"):
        asyncio.run(verify_locked_ppv(
            account_id="account",
            group_id="group",
            platform_message_id="message-1",
            media_ids=["media-1"],
            price_cents=3500,
        ))


def test_operator_provisional_mode_requires_every_attachment(monkeypatch):
    async def page(*_args, **_kwargs):
        return (
            [{
                "id": "message-1",
                "attachments": [{"contentId": "only-one-attachment"}],
            }],
            [],
            None,
        )

    async def no_sleep(_delay):
        return None

    monkeypatch.setattr(ppv_delivery, "list_chat_messages", page)
    monkeypatch.setattr(ppv_delivery.asyncio, "sleep", no_sleep)

    with pytest.raises(PPVDeliveryError, match="account_media_not_visible"):
        asyncio.run(verify_locked_ppv(
            account_id="account",
            group_id="group",
            platform_message_id="message-1",
            media_ids=["media-1", "media-2"],
            price_cents=3500,
            allow_provisional=True,
        ))


def test_operator_provisional_mode_rejects_explicitly_free_media(monkeypatch):
    async def page(*_args, **_kwargs):
        return (
            [{
                "id": "message-1",
                "attachments": [{"contentId": "account-media-1"}],
            }],
            [{
                "id": "account-media-1",
                "mediaId": "media-1",
                "price": 0,
            }],
            None,
        )

    async def no_sleep(_delay):
        return None

    monkeypatch.setattr(ppv_delivery, "list_chat_messages", page)
    monkeypatch.setattr(ppv_delivery.asyncio, "sleep", no_sleep)

    with pytest.raises(PPVDeliveryError, match="media_is_not_payment_gated"):
        asyncio.run(verify_locked_ppv(
            account_id="account",
            group_id="group",
            platform_message_id="message-1",
            media_ids=["media-1"],
            price_cents=3500,
            allow_provisional=True,
        ))


def test_operator_provisional_mode_requires_the_sent_message(monkeypatch):
    async def page(*_args, **_kwargs):
        return [], [], None

    async def no_sleep(_delay):
        return None

    monkeypatch.setattr(ppv_delivery, "list_chat_messages", page)
    monkeypatch.setattr(ppv_delivery.asyncio, "sleep", no_sleep)

    with pytest.raises(PPVDeliveryError, match="message_not_visible"):
        asyncio.run(verify_locked_ppv(
            account_id="account",
            group_id="group",
            platform_message_id="message-1",
            media_ids=["media-1"],
            price_cents=3500,
            allow_provisional=True,
        ))


def test_approval_queue_is_single_pending_request_per_fan():
    migration = (
        Path(__file__).resolve().parents[1] / "db" / "agency_operability_v1.sql"
    ).read_text(encoding="utf-8")
    assert "ppv_approval_requests_one_pending_per_fan_idx" in migration
    assert "where status in ('pending', 'sending')" in migration
    assert "require_operator_ppv_approval" in migration


def test_auto_prepared_approval_keeps_exact_media_and_price():
    source = (
        Path(__file__).resolve().parents[1] / "services" / "suggestions.py"
    ).read_text(encoding="utf-8")
    assert "create_ppv_approval_request(" in source
    assert "media_ids=media_ids" in source
    assert "price_cents=int(round(price * 100))" in source
    assert "if approval_policy.require_operator_ppv_approval" in source


def test_operator_ppv_options_matches_production_vault_schema():
    source = inspect.getsource(read_operator_ppv_options)
    vault_query = source[
        source.index('lambda: db.table("creator_vault_media")'):
        source.index('lambda: db.table("vault_sets")')
    ]

    assert '.table("creator_vault_media")' in vault_query
    assert '.order("created_at"' not in vault_query
    assert "Could not load the creator vault for this PPV." in source
