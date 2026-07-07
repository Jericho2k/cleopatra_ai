"""Lightweight API authentication.

Two independent secrets guard two kinds of caller:

- DASHBOARD_API_SECRET  — the Next.js dashboard sends this on every backend call
  (header: X-API-Key). Protects the ~33 operator/CRUD/admin endpoints from the
  open internet. The dashboard itself is already auth-gated (Supabase session +
  middleware), so this secret simply proves "the caller is our dashboard".

- WEBHOOK_SECRET — external services (ApiFansly / the message webhook) send this
  (header: X-Webhook-Secret). Prevents anyone from injecting fake fan messages.

If a secret is not configured in the environment, the corresponding guard fails
CLOSED in production (APP_ENV != 'development') and OPEN in development, so local
work isn't blocked but a misconfigured prod deploy can't silently run wide open.
"""
import os

from fastapi import Header, HTTPException, status


def _is_dev() -> bool:
    return os.environ.get("APP_ENV", "development") == "development"


async def require_dashboard(x_api_key: str | None = Header(default=None)) -> None:
    expected = os.environ.get("DASHBOARD_API_SECRET")
    if not expected:
        if _is_dev():
            return  # unconfigured + dev: allow, so local dev isn't blocked
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server auth is not configured",
        )
    if not x_api_key or not _consteq(x_api_key, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API key",
        )


async def require_webhook(x_webhook_secret: str | None = Header(default=None)) -> None:
    expected = os.environ.get("WEBHOOK_SECRET")
    if not expected:
        if _is_dev():
            return
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Webhook auth is not configured",
        )
    if not x_webhook_secret or not _consteq(x_webhook_secret, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid webhook secret",
        )


def _consteq(a: str, b: str) -> bool:
    """Constant-time comparison to avoid leaking secret length/prefix via timing."""
    import hmac
    return hmac.compare_digest(a, b)