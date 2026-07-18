from __future__ import annotations

import asyncio
import inspect
from pathlib import Path

import pytest

from models.commercial import CreatorPolicy
from services.ppv_delivery import PPVDeliveryError, send_locked_ppv


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
    assert source.index("response.raise_for_status()") < source.index("pending = {")
    assert source.index("response.raise_for_status()") < source.index("FanStatus.PAYMENT_PENDING")
    assert "PPV_RECONCILE" in source
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

