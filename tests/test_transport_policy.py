"""The transport switch: who serves each platform operation, and when."""
from __future__ import annotations

import pytest

from core import transport_policy as tp


@pytest.fixture(autouse=True)
def _clear_transport_env(monkeypatch):
    for name in (
        "FANSLY_TRANSPORT_DEFAULT",
        "FANSLY_DIRECT_ACCOUNTS",
        "FANSLY_DIRECT_FALLBACK",
    ):
        monkeypatch.delenv(name, raising=False)
    for operation in tp.OPERATIONS:
        monkeypatch.delenv(f"FANSLY_TRANSPORT_{operation.upper()}", raising=False)


def test_unconfigured_deployment_keeps_paying_the_provider():
    """The migration must be opt-in: an upgrade changes nothing by itself."""
    for operation in tp.OPERATIONS:
        assert tp.transport_for(operation) == tp.TRANSPORT_PROVIDER
    assert tp.direct_operations() == ()


def test_operation_specific_setting_beats_the_default(monkeypatch):
    monkeypatch.setenv("FANSLY_TRANSPORT_DEFAULT", "direct")
    monkeypatch.setenv("FANSLY_TRANSPORT_MESSAGE_SEND", "provider")

    assert tp.transport_for(tp.OP_MEDIA_DOWNLOAD) == tp.TRANSPORT_DIRECT
    assert tp.transport_for(tp.OP_MESSAGE_SEND) == tp.TRANSPORT_PROVIDER


def test_boolean_spellings_are_accepted(monkeypatch):
    monkeypatch.setenv("FANSLY_TRANSPORT_MEDIA_DOWNLOAD", "true")
    assert tp.transport_for(tp.OP_MEDIA_DOWNLOAD) == tp.TRANSPORT_DIRECT

    monkeypatch.setenv("FANSLY_TRANSPORT_MEDIA_DOWNLOAD", "off")
    assert tp.transport_for(tp.OP_MEDIA_DOWNLOAD) == tp.TRANSPORT_PROVIDER


def test_a_typo_falls_back_to_the_paid_but_working_transport(monkeypatch):
    """Unrecognised text must not read as "direct".

    A misspelled deployment variable should leave a creator on the transport
    that is known to work, not silently promote them onto an unproven one.
    """
    monkeypatch.setenv("FANSLY_TRANSPORT_MEDIA_DOWNLOAD", "drect")
    assert tp.transport_for(tp.OP_MEDIA_DOWNLOAD) == tp.TRANSPORT_PROVIDER


def test_allowlist_narrows_direct_to_named_accounts(monkeypatch):
    monkeypatch.setenv("FANSLY_TRANSPORT_MEDIA_DOWNLOAD", "direct")
    monkeypatch.setenv("FANSLY_DIRECT_ACCOUNTS", "acct-1, acct-2")

    assert tp.transport_for(tp.OP_MEDIA_DOWNLOAD, account_id="acct-1") == (
        tp.TRANSPORT_DIRECT
    )
    assert tp.transport_for(tp.OP_MEDIA_DOWNLOAD, account_id="acct-9") == (
        tp.TRANSPORT_PROVIDER
    )


def test_allowlist_refuses_calls_with_no_account(monkeypatch):
    """A canary that leaked to unattributed calls would not be a canary."""
    monkeypatch.setenv("FANSLY_TRANSPORT_MEDIA_DOWNLOAD", "direct")
    monkeypatch.setenv("FANSLY_DIRECT_ACCOUNTS", "acct-1")

    assert tp.transport_for(tp.OP_MEDIA_DOWNLOAD, account_id="") == (
        tp.TRANSPORT_PROVIDER
    )
    assert tp.transport_for(tp.OP_MEDIA_DOWNLOAD, account_id=None) == (
        tp.TRANSPORT_PROVIDER
    )


def test_empty_allowlist_means_every_account(monkeypatch):
    monkeypatch.setenv("FANSLY_TRANSPORT_MEDIA_DOWNLOAD", "direct")
    monkeypatch.setenv("FANSLY_DIRECT_ACCOUNTS", "  ")

    assert tp.transport_for(tp.OP_MEDIA_DOWNLOAD, account_id="anyone") == (
        tp.TRANSPORT_DIRECT
    )


def test_fallback_is_on_unless_explicitly_disabled(monkeypatch):
    assert tp.fallback_enabled() is True
    monkeypatch.setenv("FANSLY_DIRECT_FALLBACK", "false")
    assert tp.fallback_enabled() is False


def test_snapshot_reports_every_operation(monkeypatch):
    monkeypatch.setenv("FANSLY_TRANSPORT_MEDIA_DOWNLOAD", "direct")
    snapshot = tp.snapshot()

    assert set(snapshot["transports"]) == set(tp.OPERATIONS)
    assert snapshot["direct_operations"] == [tp.OP_MEDIA_DOWNLOAD]
    assert snapshot["fallback_enabled"] is True


def test_describe_names_the_migrated_operations(monkeypatch):
    assert "API Fansly" in tp.describe()

    monkeypatch.setenv("FANSLY_TRANSPORT_MEDIA_DOWNLOAD", "direct")
    monkeypatch.setenv("FANSLY_DIRECT_ACCOUNTS", "acct-1")
    line = tp.describe()

    assert tp.OP_MEDIA_DOWNLOAD in line
    assert "1 allowlisted account(s)" in line
