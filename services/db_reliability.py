"""Retry helpers for short-lived database transport failures.

Only idempotent reads and writes belong here. External delivery calls must never
be retried through this module because their outcome may already be visible to a
fan even when the HTTP response was lost.
"""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any


TRANSIENT_ERROR_NAMES = {
    "ConnectError",
    "ConnectTimeout",
    "ConnectionTerminated",
    "PoolTimeout",
    "ReadError",
    "ReadTimeout",
    "RemoteProtocolError",
    "WriteError",
    "WriteTimeout",
}
TRANSIENT_ERROR_MARKERS = (
    "connection terminated",
    "connection reset",
    "connection refused",
    "connection closed",
    "server disconnected",
    "temporarily unavailable",
)


def is_transient_db_error(exc: BaseException) -> bool:
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if type(current).__name__ in TRANSIENT_ERROR_NAMES:
            return True
        message = str(current).lower()
        if any(marker in message for marker in TRANSIENT_ERROR_MARKERS):
            return True
        current = current.__cause__ or current.__context__
    return False


async def retry_transient_db_operation(
    operation: Callable[[], Awaitable[Any]],
    *,
    label: str,
    attempts: int = 3,
    delay_seconds: float = 0.15,
    log_prefix: str = "DB RETRY",
) -> Any:
    """Retry an idempotent database operation after transport disconnects."""
    for attempt in range(1, attempts + 1):
        try:
            return await operation()
        except Exception as exc:
            if not is_transient_db_error(exc) or attempt >= attempts:
                raise
            print(
                f"[{log_prefix}] transient failure label={label} "
                f"attempt={attempt}/{attempts}: {exc}"
            )
            await asyncio.sleep(delay_seconds * attempt)
    raise RuntimeError(f"{label} unavailable")
