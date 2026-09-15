"""Safe import-time configuration for the isolated test environment."""
from __future__ import annotations

import os


_TEST_ENV = {
    "SUPABASE_URL": "https://example.supabase.co",
    "SUPABASE_SERVICE_KEY": "test-service-key",
    "TOGETHER_API_KEY": "test-together-key",
    "UPSTASH_REDIS_URL": "https://example.upstash.io",
    "UPSTASH_REDIS_TOKEN": "test-redis-token",
    "OPENAI_API_KEY": "test-openai-key",
    "ANTHROPIC_API_KEY": "test-anthropic-key",
    "APIFANSLY_API_KEY": "test-apifansly-key",
    "FANSLY_SESSION_KEY": "test-session-key",
    "DASHBOARD_API_SECRET": "test-dashboard-secret",
    "WEBHOOK_SECRET": "test-webhook-secret",
    "APP_ENV": "test",
    "MODEL_TELEMETRY_ENABLED": "false",
}

for key, value in _TEST_ENV.items():
    os.environ.setdefault(key, value)


import pytest


@pytest.fixture(autouse=True)
def _isolate_process_global_health_signals():
    """Keep deliberately process-global operator signals from leaking between tests.

    Two pieces of state are global on purpose, because they describe the
    *process* rather than a request: whether ingestion has had to run without
    the platform-identity unique index, and the database health confirmation
    ladder. Both are read by ``services.operational_health.evaluate``.

    That is what made CI fail while a local run was green: with
    ``TEST_DATABASE_URL`` set, ``test_message_ingestion_idempotency`` stops being
    skipped, deliberately drives the missing-index fallback, and every later
    health assertion in the session then saw ``degraded`` instead of ``ok``.
    Without the variable those tests skip and the leak never happens — so the
    suite was only green because part of it was not running.

    Resetting on the way in AND out, so neither a test that sets these nor a
    test that merely runs after one can be affected.

    The API Fansly usage ledger and the fan_history_backfill availability flag
    are process-global for the same reason and are reset here too.
    """
    from core.db_health_state import DATABASE_HEALTH
    from db.fan_history_queries import reset_backfill_table_state
    from db.queries import reset_message_identity_index_state
    from services.apifansly import reset_usage_for_tests

    def _reset() -> None:
        reset_message_identity_index_state()
        reset_backfill_table_state()
        # API Fansly usage is a rolling process-global ledger, and it now
        # carries a live/background priority signal that history tests read.
        # One test's recorded call must not make the next test believe a
        # conversation is in progress.
        reset_usage_for_tests()
        DATABASE_HEALTH.reset()

    _reset()
    try:
        yield
    finally:
        _reset()
