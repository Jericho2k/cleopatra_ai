"""API-001 — the reconciliation checkpoint is durable, so a restart is cheap.

The unit of the decision is `_chat_message_sync_needed`. Everything it needs now
arrives as an argument (the remote marker, and the checkpoint read from the fan
row) rather than from process memory, which is exactly what makes the behaviour
survive a restart. The restart tests below therefore do not need to simulate a
process at all: "after a restart" is precisely "the same stored checkpoint, with
no in-memory state anywhere", and that is what they assert.

tests/test_chat_sync_restart.py drives the same behaviour end to end through
sync_chats with a counting fake client, and puts a number on the call reduction.
"""
import main


def test_first_ever_sync_of_a_known_chat_reconciles() -> None:
    """No checkpoint stored: nothing has ever been imported for this chat."""
    assert main._chat_message_sync_needed(
        "message-1",
        "",
        is_new_chat=False,
        group_binding_changed=False,
    )


def test_new_chat_always_reconciles() -> None:
    assert main._chat_message_sync_needed(
        "message-1",
        "",
        is_new_chat=True,
        group_binding_changed=False,
    )


def test_unchanged_chat_skips_the_message_history_call() -> None:
    assert not main._chat_message_sync_needed(
        "message-1",
        "message-1",
        is_new_chat=False,
        group_binding_changed=False,
    )


def test_unchanged_chat_still_skips_after_a_restart() -> None:
    """The whole point of API-001.

    A restart clears every dict in the process. The checkpoint is a column on
    the fan row, so the decision after a restart is made from the same two
    values as before it — and stays "skip".
    """
    stored = "message-1"  # what the last successful reconciliation persisted

    assert not main._chat_message_sync_needed(
        "message-1",
        stored,
        is_new_chat=False,
        group_binding_changed=False,
    )


def test_changed_remote_marker_triggers_reconciliation() -> None:
    assert main._chat_message_sync_needed(
        "message-2",
        "message-1",
        is_new_chat=False,
        group_binding_changed=False,
    )


def test_deleted_newest_message_moves_the_marker_and_reconciles() -> None:
    """A deleted newest message changes lastMessageId, so it is not silently
    skipped: the marker no longer matches and the chat is re-read."""
    assert main._chat_message_sync_needed(
        "message-0",  # platform now reports an older message as newest
        "message-1",
        is_new_chat=False,
        group_binding_changed=False,
    )


def test_moved_group_binding_ignores_the_old_checkpoint() -> None:
    """The stored marker describes the previous conversation. Trusting it here
    would suppress the first import of the new one."""
    assert main._chat_message_sync_needed(
        "message-1",
        "message-1",
        is_new_chat=False,
        group_binding_changed=True,
    )


def test_missing_remote_marker_fails_toward_synchronisation() -> None:
    """Cannot prove nothing changed, so reconcile rather than risk message
    loss."""
    assert main._chat_message_sync_needed(
        "",
        "message-1",
        is_new_chat=False,
        group_binding_changed=False,
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
