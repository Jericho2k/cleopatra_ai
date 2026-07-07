"""Application settings — single source of truth for all env vars."""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

# Switch to Anthropic
USE_ANTHROPIC = True
PRIMARY_MODEL = "claude-sonnet-4-20250514"
FALLBACK_MODEL = "openai/gpt-oss-20b"


class Settings(BaseSettings):
    SUPABASE_URL: str
    SUPABASE_SERVICE_KEY: str
    TOGETHER_API_KEY: str
    ANTHROPIC_API_KEY: str
    UPSTASH_REDIS_URL: str
    UPSTASH_REDIS_TOKEN: str
    OPENAI_API_KEY: str
    APP_ENV: str = "development"
    DASHBOARD_API_SECRET: str = ""
    WEBHOOK_SECRET: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()