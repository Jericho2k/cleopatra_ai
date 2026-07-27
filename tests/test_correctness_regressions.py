from __future__ import annotations

import inspect
from pathlib import Path

from main import handle_new_fan_message, read_operator_ppv_options, save_reply
from services import suggestions
from services.ppv_delivery import send_locked_ppv
from services.ppv_status import build_media_status_by_id


def test_operator_ppv_status_does_not_truncate_receipt_history():
    source = inspect.getsource(read_operator_ppv_options)
    assert '.not_.is_("media_context->ppv", "null")' in source
    assert ".limit(1000)" not in source
    assert "list_fan_deliveries" in source
    assert "not_sold_log" in source


def test_voided_legacy_receipt_is_not_marked_sent():
    statuses = build_media_status_by_id(
        deliveries=[],
        message_rows=[
            {
                "media_context": {
                    "ppv": {
                        "media_ids": ["media-1"],
                        "delivery_status": "voided",
                    }
                }
            }
        ],
        sales_log=[],
        not_sold_log=[],
        pending_ppv=None,
    )
    assert "media-1" not in statuses


def test_explicit_ppv_lifecycle_precedence():
    statuses = build_media_status_by_id(
        deliveries=[
            {"status": "voided", "media_ids": ["voided"]},
            {"status": "abandoned", "media_ids": ["abandoned"]},
            {"status": "delivered_pending", "media_ids": ["pending"]},
        ],
        message_rows=[],
        sales_log=[{"media_ids": ["sold", "abandoned"]}],
        not_sold_log=[{"media_ids": ["abandoned"]}],
        pending_ppv={"media_ids": ["pending"]},
    )
    assert statuses == {
        "voided": "voided",
        "abandoned": "sold",
        "pending": "payment_pending",
        "sold": "sold",
    }


def test_delivery_claim_is_database_atomic():
    migration = (
        Path(__file__).resolve().parents[1] / "db" / "ppv_delivery_ledger_v1.sql"
    ).read_text(encoding="utf-8")
    assert "ppv_deliveries_one_active_per_fan_idx" in migration
    assert "where status in ('claimed', 'delivered_pending')" in migration
    assert "function public.attach_pending_ppv" in migration
    assert "for update" in migration
    source = inspect.getsource(send_locked_ppv)
    assert source.index("await claim_delivery(") < source.index(
        "response_body = await send_apifansly_message("
    )
    assert '"delivered_pending"' in source


def test_poller_persists_before_processing():
    source = inspect.getsource(handle_new_fan_message)
    assert source.index("await save_message(") < source.index(
        "await process_incoming_fan_message("
    )


def test_manual_reply_reports_delivery_before_persisting_success():
    source = inspect.getsource(save_reply)
    assert source.index("response_body = await send_apifansly_message(") < source.index(
        "message_id = await save_message("
    )
    assert "Fansly did not accept the message" in source


def test_assisted_suggestions_use_internal_fan_id_and_direct_planner():
    source = inspect.getsource(suggestions.get_suggestions)
    assert "get_fan_by_id(fan_id)" in source
    assert "get_fan(creator_id, fan_id)" not in source
    assert "localhost:8080" not in source
    assert "await asyncio.gather(" in source
