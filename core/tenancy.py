"""Request-level creator ownership checks for dashboard routes."""
from __future__ import annotations

import asyncio

from fastapi import HTTPException, Request, status

from core.auth import _is_dev, dashboard_user_id
from core.supabase import get_supabase


def _forbidden() -> HTTPException:
    # Deliberately do not reveal whether another agency's resource exists.
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Resource not found",
    )


async def _creator_ids_for_user(user_id: str) -> set[str]:
    result = await asyncio.to_thread(
        lambda: get_supabase()
        .table("chatter_creators")
        .select("creator_id")
        .eq("chatter_id", user_id)
        .execute()
    )
    return {
        str(row["creator_id"])
        for row in (result.data or [])
        if row.get("creator_id")
    }


async def require_creator_access(
    request: Request,
    creator_id: str,
) -> None:
    user_id = dashboard_user_id(request)
    if not user_id and _is_dev():
        return
    if not user_id:
        raise _forbidden()

    cached = getattr(request.state, "allowed_creator_ids", None)
    if cached is None:
        cached = await _creator_ids_for_user(user_id)
        request.state.allowed_creator_ids = cached
    if str(creator_id) not in cached:
        raise _forbidden()


async def require_creator_path_access(
    request: Request,
    creator_id: str,
) -> None:
    await require_creator_access(request, creator_id)


async def _fan_creator_id(fan_id: str) -> str | None:
    result = await asyncio.to_thread(
        lambda: get_supabase()
        .table("fans")
        .select("creator_id")
        .eq("id", fan_id)
        .limit(1)
        .execute()
    )
    rows = result.data or []
    return str(rows[0]["creator_id"]) if rows and rows[0].get("creator_id") else None


async def require_fan_access(request: Request, fan_id: str) -> str:
    creator_id = await _fan_creator_id(fan_id)
    if not creator_id:
        raise _forbidden()
    await require_creator_access(request, creator_id)
    return creator_id


async def require_fan_path_access(request: Request, fan_id: str) -> None:
    await require_fan_access(request, fan_id)


async def require_creator_fan_access(
    request: Request,
    creator_id: str,
    fan_id: str,
) -> None:
    await require_creator_access(request, creator_id)
    actual_creator_id = await _fan_creator_id(fan_id)
    if actual_creator_id != str(creator_id):
        raise _forbidden()


async def require_account_access(request: Request, account_id: str) -> None:
    def _find() -> list[dict]:
        db = get_supabase()
        for column in ("apifansly_account_id", "fansly_account_id"):
            result = (
                db.table("creators")
                .select("id")
                .eq(column, account_id)
                .limit(1)
                .execute()
            )
            if result.data:
                return result.data
        return []

    rows = await asyncio.to_thread(_find)
    if not rows:
        raise _forbidden()
    await require_creator_access(request, str(rows[0]["id"]))


async def require_account_path_access(
    request: Request,
    account_id: str,
) -> None:
    await require_account_access(request, account_id)


async def require_vault_item_path_access(
    request: Request,
    item_id: str,
) -> None:
    result = await asyncio.to_thread(
        lambda: get_supabase()
        .table("creator_vault_media")
        .select("creator_id")
        .eq("id", item_id)
        .limit(1)
        .execute()
    )
    rows = result.data or []
    if not rows or not rows[0].get("creator_id"):
        raise _forbidden()
    await require_creator_access(request, str(rows[0]["creator_id"]))


async def require_ppv_approval_path_access(
    request: Request,
    request_id: str,
) -> None:
    result = await asyncio.to_thread(
        lambda: get_supabase()
        .table("ppv_approval_requests")
        .select("creator_id")
        .eq("id", request_id)
        .limit(1)
        .execute()
    )
    rows = result.data or []
    if not rows or not rows[0].get("creator_id"):
        raise _forbidden()
    await require_creator_access(request, str(rows[0]["creator_id"]))
