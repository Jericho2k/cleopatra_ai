"""Mirror the Lists an agency already maintains on a creator's Fansly account.

Agencies build VIP / Whales / Buyers / Re-engage lists directly on Fansly. This
pulls them into Cleopatra so those lists can drive Auto Audience and
re-engagement targeting without being recreated by hand.

Direction of travel is one way. Nothing here creates, renames, or deletes a
remote list, and nothing here modifies a locally created Cleopatra list.

Reconciliation rules:

* A remote list maps to exactly one local mirror, keyed on the remote list ID.
  A repeated sync upserts that same row, so a rename updates the mirror in place
  and two remote lists sharing a name stay distinct.
* Membership is mapped by ``fans.platform_fan_id`` only. Usernames and display
  names are mutable and not unique, so they are never used.
* A remote member Cleopatra has not imported yet is counted and skipped. No
  placeholder fan row is invented to satisfy a membership.
* A member removed remotely has its mirrored membership removed. Memberships an
  operator created by hand are never touched, even on the same fan.
* A mirror that disappears remotely is archived, not deleted. Auto Audience and
  re-engagement rules reference ``fan_lists.id``; deleting the row would silently
  change which fans those rules select.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from typing import Any

import httpx

from core.supabase import get_supabase
from services.apifansly import (
    ApiFanslyAccountAccessError,
    list_account_list_members,
    list_account_lists,
)

# A creator with hundreds of lists, or a list with a very long tail, must not be
# able to spin the sync forever on a broken cursor.
_MAX_LIST_PAGES = 50
_MAX_MEMBER_PAGES = 200

FANSLY_SOURCE = "fansly"
LOCAL_SOURCE = "local"


def lists_sync_enabled() -> bool:
    """Fansly list mirroring is opt-in until db/fansly_lists_v1.sql is applied.

    Defaults off, matching FAN_INTELLIGENCE_ENABLED and FAN_LIFECYCLE_ENABLED:
    running the sync against a database that lacks the source/external_list_id
    columns would fail every pass.
    """
    return os.getenv("FANSLY_LISTS_SYNC_ENABLED", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def fetch_remote_lists(
    account_id: str,
    *,
    client: httpx.AsyncClient,
) -> list[dict[str, Any]]:
    """Page through every list on the remote account."""
    lists: list[dict[str, Any]] = []
    seen: set[str] = set()
    cursor: str | None = None
    for _ in range(_MAX_LIST_PAGES):
        page, cursor = await list_account_lists(
            account_id,
            cursor=cursor,
            client=client,
        )
        for row in page:
            external_id = row["external_list_id"]
            if external_id not in seen:
                seen.add(external_id)
                lists.append(row)
        if not cursor:
            break
    return lists


async def fetch_remote_members(
    account_id: str,
    external_list_id: str,
    *,
    client: httpx.AsyncClient,
) -> list[str]:
    """Page through every member of one remote list."""
    members: list[str] = []
    seen: set[str] = set()
    cursor: str | None = None
    for _ in range(_MAX_MEMBER_PAGES):
        page, cursor = await list_account_list_members(
            account_id,
            external_list_id,
            cursor=cursor,
            client=client,
        )
        for platform_fan_id in page:
            if platform_fan_id not in seen:
                seen.add(platform_fan_id)
                members.append(platform_fan_id)
        if not cursor:
            break
    return members


def _load_state(creator_id: str) -> tuple[list[dict], dict[str, str], dict[str, set[str]]]:
    """Read the creator's current mirrors, fan id map, and mirrored memberships."""
    db = get_supabase()
    mirrors = (
        db.table("fan_lists")
        .select("id, name, source, external_list_id, external_archived_at")
        .eq("creator_id", creator_id)
        .eq("source", FANSLY_SOURCE)
        .execute()
    ).data or []
    fans = (
        db.table("fans")
        .select("id, platform_fan_id")
        .eq("creator_id", creator_id)
        .execute()
    ).data or []
    fan_by_platform_id = {
        str(row["platform_fan_id"]): str(row["id"])
        for row in fans
        if row.get("platform_fan_id") and row.get("id")
    }

    memberships: dict[str, set[str]] = {}
    mirror_ids = [str(row["id"]) for row in mirrors if row.get("id")]
    if mirror_ids:
        rows = (
            db.table("fan_list_members")
            .select("list_id, fan_id, source")
            .in_("list_id", mirror_ids)
            .eq("source", FANSLY_SOURCE)
            .execute()
        ).data or []
        for row in rows:
            list_id = str(row.get("list_id") or "")
            fan_id = str(row.get("fan_id") or "")
            if list_id and fan_id:
                memberships.setdefault(list_id, set()).add(fan_id)
    return mirrors, fan_by_platform_id, memberships


def _reconcile(
    creator_id: str,
    remote_lists: list[dict[str, Any]],
    remote_members: dict[str, list[str]],
) -> dict[str, int]:
    """Apply one full remote snapshot to the local mirrors."""
    db = get_supabase()
    now = _now()
    mirrors, fan_by_platform_id, mirrored_memberships = _load_state(creator_id)
    mirror_by_external_id = {
        str(row["external_list_id"]): row
        for row in mirrors
        if row.get("external_list_id")
    }

    counters = {
        "remote_lists": len(remote_lists),
        "created_lists": 0,
        "renamed_lists": 0,
        "archived_lists": 0,
        "restored_lists": 0,
        "added_members": 0,
        "removed_members": 0,
        "unmapped_members": 0,
    }

    seen_external_ids: set[str] = set()

    for remote in remote_lists:
        external_id = remote["external_list_id"]
        seen_external_ids.add(external_id)
        existing = mirror_by_external_id.get(external_id)

        values: dict[str, Any] = {
            "name": remote["name"],
            "external_synced_at": now,
            "external_item_count": remote["item_count"],
            # A list that came back is no longer stale.
            "external_archived_at": None,
        }

        if existing is None:
            created = (
                db.table("fan_lists")
                .insert(
                    {
                        "creator_id": creator_id,
                        "source": FANSLY_SOURCE,
                        "external_list_id": external_id,
                        **values,
                    }
                )
                .execute()
            ).data or []
            if not created:
                continue
            list_id = str(created[0]["id"])
            counters["created_lists"] += 1
        else:
            list_id = str(existing["id"])
            if str(existing.get("name") or "") != remote["name"]:
                counters["renamed_lists"] += 1
            if existing.get("external_archived_at"):
                counters["restored_lists"] += 1
            db.table("fan_lists").update(values).eq("id", list_id).eq(
                "creator_id", creator_id
            ).execute()

        desired_fan_ids: set[str] = set()
        for platform_fan_id in remote_members.get(external_id, []):
            fan_id = fan_by_platform_id.get(platform_fan_id)
            if fan_id is None:
                # Cleopatra has not imported this fan yet. Deferring is correct:
                # a fabricated fan row would corrupt every downstream count.
                counters["unmapped_members"] += 1
                continue
            desired_fan_ids.add(fan_id)

        current_fan_ids = mirrored_memberships.get(list_id, set())

        for fan_id in sorted(desired_fan_ids - current_fan_ids):
            db.table("fan_list_members").upsert(
                {
                    "list_id": list_id,
                    "fan_id": fan_id,
                    "source": FANSLY_SOURCE,
                    "external_synced_at": now,
                },
                on_conflict="list_id,fan_id",
            ).execute()
            counters["added_members"] += 1

        for fan_id in sorted(current_fan_ids - desired_fan_ids):
            # Scoped to this mirror and to fansly-sourced rows, so an operator's
            # own membership on a local list is never affected.
            db.table("fan_list_members").delete().eq("list_id", list_id).eq(
                "fan_id", fan_id
            ).eq("source", FANSLY_SOURCE).execute()
            counters["removed_members"] += 1

    for external_id, mirror in mirror_by_external_id.items():
        if external_id in seen_external_ids or mirror.get("external_archived_at"):
            continue
        # Archive rather than delete: Auto Audience rules point at this row.
        db.table("fan_lists").update(
            {"external_archived_at": now, "external_synced_at": now}
        ).eq("id", str(mirror["id"])).eq("creator_id", creator_id).execute()
        counters["archived_lists"] += 1

    db.table("creators").update(
        {
            "last_fansly_lists_sync_at": now,
            "fansly_lists_sync_error": None,
            "fansly_lists_sync_failed_at": None,
        }
    ).eq("id", creator_id).execute()

    return counters


def _record_failure(creator_id: str, error: str) -> None:
    try:
        get_supabase().table("creators").update(
            {
                "fansly_lists_sync_error": error[:500],
                "fansly_lists_sync_failed_at": _now(),
            }
        ).eq("id", creator_id).execute()
    except Exception as exc:
        print(f"[FANSLY LISTS] failure state not persisted creator={creator_id}: {exc}")


async def sync_fansly_lists(
    creator_id: str,
    account_id: str,
    *,
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """Pull the creator's Fansly lists and reconcile the local mirrors.

    Raises ApiFanslyAccountAccessError on 401/403 so callers reuse the existing
    reconnect/backoff semantics instead of retrying a binding that will keep
    failing.
    """
    if not lists_sync_enabled():
        return {"status": "disabled"}

    owns_client = client is None
    active_client = client or httpx.AsyncClient()
    try:
        remote_lists = await fetch_remote_lists(account_id, client=active_client)
        remote_members: dict[str, list[str]] = {}
        for remote in remote_lists:
            remote_members[remote["external_list_id"]] = await fetch_remote_members(
                account_id,
                remote["external_list_id"],
                client=active_client,
            )
    except ApiFanslyAccountAccessError as exc:
        # The creator binding needs reconnecting. Record it and re-raise so the
        # caller applies the same backoff it uses for every other API Fansly
        # access failure.
        await asyncio.to_thread(_record_failure, creator_id, str(exc))
        raise
    except Exception as exc:
        await asyncio.to_thread(_record_failure, creator_id, str(exc))
        raise
    finally:
        if owns_client:
            await active_client.aclose()

    counters = await asyncio.to_thread(
        _reconcile,
        creator_id,
        remote_lists,
        remote_members,
    )
    print(
        f"[FANSLY LISTS] creator={creator_id} remote={counters['remote_lists']} "
        f"created={counters['created_lists']} renamed={counters['renamed_lists']} "
        f"archived={counters['archived_lists']} "
        f"members+{counters['added_members']}/-{counters['removed_members']} "
        f"unmapped={counters['unmapped_members']}"
    )
    return {"status": "ok", **counters}


async def read_lists_sync_state(creator_id: str) -> dict[str, Any]:
    """Return the creator's mirrored lists plus last sync/failure state."""

    def _load() -> dict[str, Any]:
        db = get_supabase()
        creator = (
            db.table("creators")
            .select(
                "last_fansly_lists_sync_at, "
                "fansly_lists_sync_error, "
                "fansly_lists_sync_failed_at"
            )
            .eq("id", creator_id)
            .limit(1)
            .execute()
        ).data or []
        lists = (
            db.table("fan_lists")
            .select(
                "id, name, source, external_list_id, external_synced_at, "
                "external_archived_at, external_item_count"
            )
            .eq("creator_id", creator_id)
            .eq("source", FANSLY_SOURCE)
            .execute()
        ).data or []
        state = creator[0] if creator else {}
        return {
            "last_synced_at": state.get("last_fansly_lists_sync_at"),
            "last_error": state.get("fansly_lists_sync_error"),
            "last_failed_at": state.get("fansly_lists_sync_failed_at"),
            "lists": lists,
        }

    return await asyncio.to_thread(_load)
