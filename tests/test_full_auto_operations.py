import asyncio

import pytest

from services.full_auto_operations import (
    FullAutoStatusUnavailable,
    _retry_transient_operation,
    ppv_media_bundles,
    summarize_operation_rows,
)


def test_operational_summary_counts_only_actionable_states():
    summary = summarize_operation_rows(
        states=[
            {"status": "PAYMENT_PENDING", "next_followup_at": None},
            {"status": "IDLE", "next_followup_at": "2026-07-19T06:00:00Z"},
        ],
        fans=[
            {"needs_human_review": True},
            {"needs_human_review": False},
        ],
        actions=[
            {"status": "FAILED"},
            {"status": "PROCESSING"},
            {"status": "PENDING"},
        ],
    )
    assert summary == {
        "payment_pending": 1,
        "followups_pending": 1,
        "human_review": 1,
        "failed_actions": 1,
        "processing_actions": 1,
        "overdue_actions": 0,
    }


def test_ppv_media_bundles_indexes_every_item_to_the_full_bundle():
    bundles = ppv_media_bundles([
        {
            "media_context": {
                "ppv": {"media_id": "first", "media_ids": ["first", "second", "first"]}
            }
        },
        {"media_context": {"ppv": {"media_id": "solo"}}},
        {"media_context": None},
    ])
    assert bundles == {
        "first": ["first", "second"],
        "second": ["first", "second"],
        "solo": ["solo"],
    }


def test_operational_reads_retry_transient_protocol_disconnects():
    class RemoteProtocolError(RuntimeError):
        pass

    calls = 0

    async def operation():
        nonlocal calls
        calls += 1
        if calls < 3:
            raise RemoteProtocolError("connection terminated")
        return {"status": "ok"}

    result = asyncio.run(_retry_transient_operation(
        operation,
        label="test snapshot",
        delay_seconds=0,
    ))

    assert result == {"status": "ok"}
    assert calls == 3


def test_operational_reads_fail_cleanly_after_transient_retries():
    class RemoteProtocolError(RuntimeError):
        pass

    async def operation():
        raise RemoteProtocolError("server disconnected")

    with pytest.raises(FullAutoStatusUnavailable):
        asyncio.run(_retry_transient_operation(
            operation,
            label="test snapshot",
            attempts=2,
            delay_seconds=0,
        ))
