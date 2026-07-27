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
