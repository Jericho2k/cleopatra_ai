import main


def setup_function() -> None:
    main._chat_last_message_ids.clear()


def test_first_chat_observation_gets_safety_sync() -> None:
    assert main._chat_message_sync_needed(
        "creator",
        "group",
        "message-1",
        is_new_chat=False,
    )


def test_unchanged_chat_skips_message_history_call() -> None:
    main._remember_chat_message_id("creator", "group", "message-1")

    assert not main._chat_message_sync_needed(
        "creator",
        "group",
        "message-1",
        is_new_chat=False,
    )


def test_changed_or_new_chat_gets_message_history_call() -> None:
    main._remember_chat_message_id("creator", "group", "message-1")

    assert main._chat_message_sync_needed(
        "creator",
        "group",
        "message-2",
        is_new_chat=False,
    )
    assert main._chat_message_sync_needed(
        "creator",
        "other-group",
        "message-3",
        is_new_chat=True,
    )


def test_idle_and_active_reconciliation_intervals(monkeypatch) -> None:
    monkeypatch.delenv("CHAT_RECONCILE_ACTIVE_MINUTES", raising=False)
    monkeypatch.delenv("CHAT_RECONCILE_IDLE_MINUTES", raising=False)

    assert main._chat_reconcile_interval_seconds(
        creator_auto_mode=False,
        has_auto_fan=False,
    ) == 30 * 60
    assert main._chat_reconcile_interval_seconds(
        creator_auto_mode=True,
        has_auto_fan=False,
    ) == 10 * 60
    assert main._chat_reconcile_interval_seconds(
        creator_auto_mode=False,
        has_auto_fan=True,
    ) == 10 * 60
