"""Synchronize Fansly follower, subscriber, and supporter context."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

import httpx

from core.supabase import get_supabase
from services.apifansly import list_followers, list_subscribers, top_supporters


def _accounts_from_followers(response: Any) -> tuple[set[str], dict[str, dict]]:
    if not isinstance(response, dict):
        return set(), {}
    followers = response.get("followers")
    aggregation = response.get("aggregationData")
    accounts = (
        aggregation.get("accounts", [])
        if isinstance(aggregation, dict)
        else []
    )
    account_lookup = {
        str(row.get("id")): row
        for row in accounts
        if isinstance(row, dict) and row.get("id")
    }
    follower_ids = {
        str(row.get("followerId"))
        for row in (followers if isinstance(followers, list) else [])
        if isinstance(row, dict) and row.get("followerId")
    }
    return follower_ids, account_lookup


def _subscription_rows(response: Any) -> list[dict]:
    if not isinstance(response, dict):
        return []
    rows = response.get("subscriptions")
    return rows if isinstance(rows, list) else []


def _timestamp(value: Any) -> str | None:
    try:
        raw = float(value)
    except (TypeError, ValueError):
        return None
    if raw <= 0:
        return None
    if raw > 1e12:
        raw /= 1000
    return datetime.fromtimestamp(raw, tz=timezone.utc).isoformat()


async def sync_fansly_audience(
    creator_id: str,
    account_id: str,
) -> dict[str, int]:
    """Refresh platform attributes for fans that already have Cleopatra chats."""
    follower_ids: set[str] = set()
    follower_accounts: dict[str, dict] = {}
    subscriptions: dict[str, dict] = {}

    async with httpx.AsyncClient() as client:
        cursor: str | None = None
        while True:
            response, cursor = await list_followers(
                account_id,
                cursor=cursor,
                limit=100,
                client=client,
            )
            page_ids, page_accounts = _accounts_from_followers(response)
            follower_ids.update(page_ids)
            follower_accounts.update(page_accounts)
            if not cursor:
                break

        cursor = None
        while True:
            response, cursor = await list_subscribers(
                account_id,
                status="all",
                cursor=cursor,
                limit=100,
                client=client,
            )
            for row in _subscription_rows(response):
                fan_id = str(row.get("subscriberId") or "")
                if not fan_id:
                    continue
                current = subscriptions.get(fan_id)
                # Active status 3 wins over expired status 4; otherwise keep the
                # most recently updated relationship.
                if (
                    current is None
                    or int(row.get("status") or 0) == 3
                    or float(row.get("updatedAt") or 0)
                    > float(current.get("updatedAt") or 0)
                ):
                    subscriptions[fan_id] = row
            if not cursor:
                break

        supporters_response = await top_supporters(account_id, client=client)

    supporter_spend: dict[str, int] = {}
    for row in supporters_response if isinstance(supporters_response, list) else []:
        if not isinstance(row, dict):
            continue
        platform_fan_id = str(row.get("correlationAccountId") or "")
        if platform_fan_id:
            supporter_spend[platform_fan_id] = int(row.get("totalGross") or 0)

    db = get_supabase()
    existing_result = await asyncio.to_thread(
        lambda: db.table("fans")
        .select("id, platform_fan_id, total_spent")
        .eq("creator_id", creator_id)
        .execute()
    )
    now = datetime.now(timezone.utc).isoformat()
    updated = 0
    for fan in existing_result.data or []:
        platform_fan_id = str(fan.get("platform_fan_id") or "")
        if not platform_fan_id:
            continue
        subscription = subscriptions.get(platform_fan_id) or {}
        status_code = int(subscription.get("status") or 0)
        platform_spend_cents = supporter_spend.get(platform_fan_id)
        patch: dict[str, Any] = {
            "is_follower": platform_fan_id in follower_ids,
            "subscription_status": (
                "active" if status_code == 3
                else "expired" if status_code == 4
                else "none"
            ),
            "subscription_tier_id": subscription.get("subscriptionTierId"),
            "subscription_tier_name": subscription.get("subscriptionTierName"),
            "subscription_ends_at": _timestamp(subscription.get("endsAt")),
            "fansly_lifetime_spend_cents": platform_spend_cents,
            "fansly_audience_synced_at": now,
        }
        account = follower_accounts.get(platform_fan_id) or {}
        display_name = account.get("displayName") or account.get("username")
        if display_name:
            patch["display_name"] = str(display_name)
        avatar = account.get("avatar") or {}
        locations = avatar.get("locations") if isinstance(avatar, dict) else []
        if isinstance(locations, list) and locations:
            location = locations[0].get("location")
            if location:
                patch["avatar_url"] = location
        if platform_spend_cents is not None:
            patch["total_spent"] = max(
                int(fan.get("total_spent") or 0),
                int(round(platform_spend_cents / 100)),
            )
        await asyncio.to_thread(
            lambda fid=str(fan["id"]), values=patch: db.table("fans")
            .update(values)
            .eq("id", fid)
            .eq("creator_id", creator_id)
            .execute()
        )
        updated += 1

    await asyncio.to_thread(
        lambda: db.table("creators")
        .update({"last_fansly_audience_sync_at": now})
        .eq("id", creator_id)
        .execute()
    )
    return {
        "followers": len(follower_ids),
        "subscriptions": len(subscriptions),
        "supporters": len(supporter_spend),
        "updated_fans": updated,
    }
