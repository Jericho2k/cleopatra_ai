"""Supabase client — single instance, cached."""
from functools import lru_cache

from supabase import Client, create_client

from core.config import get_settings


@lru_cache
def get_supabase() -> Client:
    settings = get_settings()
    return create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_KEY)
