from services.full_auto_operations import summarize_operation_rows


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
    }
