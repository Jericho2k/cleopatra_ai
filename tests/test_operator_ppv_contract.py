from __future__ import annotations

import asyncio
import inspect
from pathlib import Path

import pytest

from main import read_operator_ppv_options
from models.commercial import CreatorPolicy
from services.ppv_delivery import PPVDeliveryError, send_locked_ppv
from services.ppv_persistence import persist_ppv_reconciliation


def test_supervised_auto_is_opt_in():
    assert CreatorPolicy().require_operator_ppv_approval is False
    assert CreatorPolicy(require_operator_ppv_approval=True).require_operator_ppv_approval is True


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
    acceptance_check = "raise_for_apifansly_response("
    assert source.index(acceptance_check) < source.index("pending = {")
    assert source.index(acceptance_check) < source.index("save_ppv_message_receipt(")
    assert "FanStatus.PAYMENT_PENDING" in persistence_source
    assert "PPV_RECONCILE" in persistence_source
    assert "ppv_sent_but_reconciliation_not_persisted" in source


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
