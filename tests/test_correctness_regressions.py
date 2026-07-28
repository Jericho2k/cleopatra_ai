from __future__ import annotations

import inspect
from pathlib import Path

from main import (
    _apifansly_account_media_lookup,
    _apifansly_message_row,
    handle_new_fan_message,
    process_incoming_fan_message,
    read_operator_ppv_options,
    save_reply,
    update_creator_auto_mode,
    update_fan_auto_mode,
)
from services import suggestions
from services.ppv_delivery import send_locked_ppv, verify_locked_ppv
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
    assert "ppv_deliveries_one_active_automated_per_fan_idx" in migration
    assert "where status in ('claimed', 'delivered_pending')" in migration
    assert "source <> 'operator'" in migration
    assert "function public.attach_pending_ppv" in migration
    assert "for update" in migration
    source = inspect.getsource(send_locked_ppv)
    assert source.index("await claim_delivery(") < source.index(
        "response_body = await send_apifansly_message("
    )
    assert '"delivered_pending"' in source


def test_locked_ppv_is_verified_upstream_before_local_payment_state():
    source = inspect.getsource(send_locked_ppv)
    assert source.index("lock_evidence = await verify_locked_ppv(") < source.index(
        "pending = {"
    )
    assert "ppv_lock_verification_failed" in source
    assert "delete_apifansly_message" in source
    assert '"voided" if deleted else "delivered_pending"' in source
    assert "ppv_delivery_evidence" in inspect.getsource(verify_locked_ppv)


def test_recent_chat_import_preserves_inbound_media_and_role():
    row = _apifansly_message_row(
        {
            "id": "message-1",
            "senderId": "fan-platform-id",
            "content": "",
            "createdAt": 1_700_000_000,
            "attachments": [
                {"contentId": "account-media-1", "contentType": 1}
            ],
        },
        fan_id="fan",
        creator_id="creator",
        creator_platform_id="creator-platform-id",
        media_lookup={
            "account-media-1": {
                "url": "https://cdn3.fansly.com/fan/media.jpg",
                "price": 0,
                "is_ppv": False,
                "purchased": True,
                "access": True,
            }
        },
    )

    assert row is not None
    assert row["role"] == "fan"
    assert row["media_context"]["attachments"][0]["url"].endswith("media.jpg")


def test_inbound_media_lookup_uses_variants_and_preserves_type_metadata():
    lookup = _apifansly_account_media_lookup([{
        "id": "account-media-1",
        "mediaId": "media-1",
        "price": 0,
        "media": {
            "mimetype": "video/mp4",
            "filename": "fan-video.mp4",
            "variants": [{
                "locations": [{
                    "location": "https://cdn3.fansly.com/fan/video.mp4"
                }]
            }],
        },
    }])

    assert lookup["account-media-1"]["url"].endswith("video.mp4")
    assert lookup["account-media-1"]["mimetype"] == "video/mp4"
    assert lookup["account-media-1"]["filename"] == "fan-video.mp4"


def test_auto_mode_writes_are_backend_gated_by_approved_sets():
    creator_source = inspect.getsource(update_creator_auto_mode)
    fan_source = inspect.getsource(update_fan_auto_mode)
    processing_source = inspect.getsource(process_incoming_fan_message)
    assert "_creator_auto_availability" in creator_source
    assert "_creator_auto_availability" in fan_source
    assert "Auto mode is locked until at least one vault set is approved." in (
        creator_source + fan_source
    )
    assert "no_approved_sets" in processing_source


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
