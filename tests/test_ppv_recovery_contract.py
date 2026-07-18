from __future__ import annotations

import asyncio
import inspect
from datetime import datetime, timezone
from pathlib import Path

import pytest

from services.db_reliability import retry_transient_db_operation
from services.full_auto_operations import get_fan_full_auto_snapshot
from services.ppv_persistence import pending_from_message_receipt
from services.ppv_recovery import PPVRecoveryError, resolve_fan_review


def test_transient_database_write_is_retried_without_retrying_other_errors():
    class ConnectionTerminated(RuntimeError):
        pass

    calls = 0

    async def transient_operation():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ConnectionTerminated("connection terminated")
        return "persisted"

    assert asyncio.run(retry_transient_db_operation(
        transient_operation,
        label="PPV state",
        delay_seconds=0,
    )) == "persisted"
    assert calls == 2

    async def invalid_operation():
        raise ValueError("invalid state")

    with pytest.raises(ValueError, match="invalid state"):
        asyncio.run(retry_transient_db_operation(
            invalid_operation,
            label="PPV state",
            delay_seconds=0,
        ))


def test_pending_ppv_can_be_rebuilt_from_the_exact_local_receipt():
    sent_at = datetime(2026, 7, 19, tzinfo=timezone.utc)
    pending = pending_from_message_receipt(
        {
            "sent_at": sent_at.isoformat(),
            "fansly_message_id": None,
            "media_context": {
                "ppv": {
                    "payment_reference": "receipt-1",
                    "media_id": "one",
                    "media_ids": ["one", "two"],
                    "price": 30,
                    "price_cents": 3000,
                    "set_id": "set-1",
                    "step_index": 0,
                    "source": "auto",
                }
            },
        },
        payment_window_hours=24,
        local_test_fan=True,
    )

    assert pending["reference"] == "receipt-1"
    assert pending["media_id"] == "one"
    assert pending["media_ids"] == ["one", "two"]
    assert pending["price_cents"] == 3000
    assert pending["platform_message_id"] == "local-test:receipt-1"


def test_ambiguous_ppv_review_cannot_be_blindly_resumed(monkeypatch):
    async def load_fan(_fan_id: str):
        return {
            "creator_id": "creator",
            "review_reason": "ppv_sent_but_reconciliation_not_persisted",
        }

    monkeypatch.setattr("services.ppv_recovery._load_fan_review", load_fan)
    with pytest.raises(PPVRecoveryError, match="ambiguous delivery outcome"):
        asyncio.run(resolve_fan_review("fan", resolution="resume_ai"))


def test_status_snapshot_avoids_parallel_shared_client_reads():
    source = inspect.getsource(get_fan_full_auto_snapshot)
    assert "asyncio.gather" not in source
    assert "intentionally run sequentially" in source


def test_payment_enum_migration_covers_both_new_states():
    migration = (
        Path(__file__).resolve().parents[1] / "db" / "payment_pending_status_v1.sql"
    ).read_text(encoding="utf-8")
    assert "add value if not exists 'OFFER_SELECTED'" in migration
    assert "add value if not exists 'PAYMENT_PENDING'" in migration


def test_test_delivery_log_distinguishes_acceptance_from_persistence():
    source = (
        Path(__file__).resolve().parents[1] / "services" / "suggestions.py"
    ).read_text(encoding="utf-8")
    assert "accepted=true" in source
    assert "message_persisted=true" in source
    assert "persisted_locally=true" not in source
