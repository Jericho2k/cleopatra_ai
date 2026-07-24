"""Dashboard and webhook authentication.

Two independent secrets guard two kinds of caller:

- DASHBOARD_API_SECRET — identifies the Cleopatra dashboard deployment.
- Supabase access token — identifies the signed-in operator. Production
  dashboard calls require both. Route-level tenancy checks then verify that the
  operator is assigned to the creator whose data is being requested.

- WEBHOOK_SECRET — external services (ApiFansly / the message webhook) send this
  (header: X-Webhook-Secret). Prevents anyone from injecting fake fan messages.

If a secret is not configured in the environment, the corresponding guard fails
CLOSED in production (APP_ENV != 'development') and OPEN in development, so local
work isn't blocked but a misconfigured prod deploy can't silently run wide open.
"""
import os
import time
from hashlib import sha256

from fastapi import Header, HTTPException, Request, status

from core.supabase import get_supabase


_USER_CACHE: dict[str, tuple[float, str]] = {}
_USER_CACHE_TTL_SECONDS = 60.0


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


def _bearer_token(authorization: str | None) -> str | None:
    scheme, _, value = str(authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not value.strip():
        return None
    return value.strip()


async def authenticated_dashboard_user(
    authorization: str | None,
) -> str | None:
    """Validate a Supabase access token and return its user ID.

    Development keeps the historical open-local behavior when no token is
    supplied. Production always fails closed.
    """
    token = _bearer_token(authorization)
    if not token:
        if _is_dev():
            return None
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing dashboard user session",
        )

    cache_key = sha256(token.encode("utf-8")).hexdigest()
    cached = _USER_CACHE.get(cache_key)
    now = time.monotonic()
    if cached and cached[0] > now:
        return cached[1]

    try:
        import asyncio

        result = await asyncio.to_thread(get_supabase().auth.get_user, token)
        user = getattr(result, "user", None)
        user_id = str(getattr(user, "id", "") or "")
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired dashboard user session",
        ) from exc
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired dashboard user session",
        )

    _USER_CACHE[cache_key] = (now + _USER_CACHE_TTL_SECONDS, user_id)
    if len(_USER_CACHE) > 512:
        for key, (expires_at, _) in list(_USER_CACHE.items()):
            if expires_at <= now:
                _USER_CACHE.pop(key, None)
    return user_id


def dashboard_user_id(request: Request) -> str | None:
    """Return the authenticated operator stored by the API middleware."""
    return getattr(request.state, "dashboard_user_id", None)


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
