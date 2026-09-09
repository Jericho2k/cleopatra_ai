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
* Removal requires POSITIVE evidence that the fan is gone remotely (SCALE-003).
  If any input to that judgement is incomplete — a truncated local read, a
  database error, a remote member listing that hit its page cap — the removal
  phase is skipped and the sync reports a degraded result. A stale membership
  that survives one cycle is recoverable; a correct membership deleted because
  our snapshot was short is not.
* A mirror that disappears remotely is archived, not deleted. Auto Audience and
  re-engagement rules reference ``fan_lists.id``; deleting the row would silently
  change which fans those rules select.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import httpx

from core.pagination import fetch_all_rows
from core.supabase import get_supabase
from services.apifansly import (
    ApiFanslyAccountAccessError,
    shared_client as apifansly_shared_client,
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
) -> tuple[list[str], bool]:
    """Page through every member of one remote list.

    Returns the members and whether the listing reached the end. A run that stops
    at _MAX_MEMBER_PAGES with a live cursor holds only part of the remote list,
    and a partial remote snapshot must never drive deletions (SCALE-003).
    """
    members: list[str] = []
    seen: set[str] = set()
    cursor: str | None = None
    complete = False
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
            complete = True
            break
    return members, complete


@dataclass
class _LocalState:
    """The local snapshot reconciliation compares the remote list against.

    ``complete`` is the safety interlock for SCALE-003. Every read below feeds
    the decision "this fan is no longer on the remote list, delete its mirrored
    membership". If any of them returned only part of the truth, absence from
    the snapshot is not evidence of absence remotely.
    """

    mirrors: list[dict] = field(default_factory=list)
    fan_by_platform_id: dict[str, str] = field(default_factory=dict)
    memberships: dict[str, set[str]] = field(default_factory=dict)
    complete: bool = True
    incomplete_reason: str = ""

    def degrade(self, reason: str) -> None:
        self.complete = False
        if not self.incomplete_reason:
            self.incomplete_reason = reason


def _load_state(creator_id: str) -> _LocalState:
    """Read the creator's current mirrors, fan id map, and mirrored memberships.

    Every read is paginated with a deterministic order. Previously ``fans`` and
    ``fan_list_members`` were unranged and unordered, so PostgREST returned an
    arbitrary 1,000-row prefix: a fan inside the prefix on one sync and outside
    it on the next could not be mapped, landed in ``current - desired``, and had
    its membership deleted — then re-added on the following run. That flapped
    VIP/Whale targeting for every creator with more than 1,000 fans (SCALE-003).
    """
    db = get_supabase()
    state = _LocalState()

    try:
        state.mirrors = fetch_all_rows(
            lambda start, end: db.table("fan_lists")
            .select("id, name, source, external_list_id, external_archived_at")
            .eq("creator_id", creator_id)
            .eq("source", FANSLY_SOURCE)
            .order("id")
            .range(start, end)
            .execute()
        )
    except Exception as exc:
        # Without the mirror list there is nothing to reconcile at all.
        # PaginationIncompleteError lands here too: a partial mirror list is as
        # unusable as a failed read.
        state.degrade(f"fan_lists read failed: {exc}")
        return state

    try:
        fans = fetch_all_rows(
            lambda start, end: db.table("fans")
            .select("id, platform_fan_id")
            .eq("creator_id", creator_id)
            .order("id")
            .range(start, end)
            .execute()
        )
    except Exception as exc:
        state.degrade(f"fans read failed: {exc}")
        fans = []
    state.fan_by_platform_id = {
        str(row["platform_fan_id"]): str(row["id"])
        for row in fans
        if row.get("platform_fan_id") and row.get("id")
    }

    mirror_ids = [str(row["id"]) for row in state.mirrors if row.get("id")]
    if mirror_ids:
        try:
            rows = fetch_all_rows(
                lambda start, end: db.table("fan_list_members")
                .select("list_id, fan_id, source")
                .in_("list_id", mirror_ids)
                .eq("source", FANSLY_SOURCE)
                .order("list_id")
                .order("fan_id")
                .range(start, end)
                .execute()
            )
        except Exception as exc:
            state.degrade(f"fan_list_members read failed: {exc}")
            rows = []
        for row in rows:
            list_id = str(row.get("list_id") or "")
            fan_id = str(row.get("fan_id") or "")
            if list_id and fan_id:
                state.memberships.setdefault(list_id, set()).add(fan_id)
    return state


def _reconcile(
    creator_id: str,
    remote_lists: list[dict[str, Any]],
    remote_members: dict[str, list[str]],
    incomplete_remote_lists: set[str] | None = None,
) -> dict[str, int]:
    """Apply one remote snapshot to the local mirrors.

    ``incomplete_remote_lists`` names remote lists whose member listing did not
    reach the end. Those lists still gain additions, but never lose memberships.
    """
    db = get_supabase()
    now = _now()
    truncated_remote = set(incomplete_remote_lists or set())
    state = _load_state(creator_id)
    mirrors = state.mirrors
    fan_by_platform_id = state.fan_by_platform_id
    # Reverse direction, used to justify each individual removal below.
    platform_id_by_fan_id = {
        fan_id: platform_id for platform_id, fan_id in fan_by_platform_id.items()
    }
    mirrored_memberships = state.memberships
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
        # SCALE-003 observability: how many removals were withheld because the
        # snapshot could not justify them.
        "skipped_removals": 0,
        "degraded": 0,
    }
    degraded_reasons: list[str] = []
    if not state.complete:
        degraded_reasons.append(state.incomplete_reason or "local snapshot incomplete")

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

        remote_platform_ids = set(remote_members.get(external_id, []))
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

        # SCALE-003 — removal requires positive evidence that the fan is gone
        # from the remote list, not merely its absence from `desired_fan_ids`.
        #
        # A fan lands in `current - desired` for two indistinguishable reasons:
        # it really was removed remotely, or we could not map it locally. The
        # old code deleted in both cases, so a fan dropped by the 1,000-row cap
        # lost its membership and got it back on the next run.
        #
        # Two interlocks now stand between a difference and a delete. First, the
        # phase is skipped entirely unless both snapshots are known-complete.
        # Second, each candidate is individually justified: we resolve the fan's
        # own platform id and require it to be genuinely absent remotely.
        removal_blocker = ""
        if not state.complete:
            removal_blocker = state.incomplete_reason or "local snapshot incomplete"
        elif external_id in truncated_remote:
            removal_blocker = f"remote member listing truncated for list {external_id}"

        stale_fan_ids = sorted(current_fan_ids - desired_fan_ids)
        if removal_blocker and stale_fan_ids:
            counters["skipped_removals"] += len(stale_fan_ids)
            degraded_reasons.append(removal_blocker)
            print(
                f"[FANSLY LISTS] withheld {len(stale_fan_ids)} membership "
                f"removal(s) creator={creator_id} list={external_id}: "
                f"{removal_blocker}"
            )
            stale_fan_ids = []

        for fan_id in stale_fan_ids:
            platform_fan_id = platform_id_by_fan_id.get(fan_id)
            if platform_fan_id is None:
                # The membership names a fan absent from our own fans read. That
                # is the exact SCALE-003 signature: the row was truncated away,
                # so its absence is evidence about our snapshot, not about
                # Fansly. Leave the membership alone.
                counters["skipped_removals"] += 1
                degraded_reasons.append(
                    f"membership on list {external_id} references fan {fan_id} "
                    "missing from the local fans snapshot"
                )
                continue
            if platform_fan_id in remote_platform_ids:
                # Present remotely but not in desired_fan_ids — the two maps
                # disagree. Never resolve that disagreement by deleting.
                counters["skipped_removals"] += 1
                degraded_reasons.append(
                    f"fan {fan_id} is still on remote list {external_id} but did "
                    "not map; membership kept"
                )
                continue
            # Positive evidence: the fan exists locally, both snapshots are
            # complete, and its platform id is not on the remote list.
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

    counters["degraded"] = 1 if degraded_reasons else 0
    if degraded_reasons:
        # A degraded run did real work (creates, renames, additions) but could
        # not be trusted to remove. Record it rather than reporting success, so
        # a persistently short snapshot is visible instead of looking healthy.
        summary = "; ".join(dict.fromkeys(degraded_reasons))[:500]
        db.table("creators").update(
            {
                "last_fansly_lists_sync_at": now,
                "fansly_lists_sync_error": f"degraded: {summary}",
                "fansly_lists_sync_failed_at": now,
            }
        ).eq("id", creator_id).execute()
    else:
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

    active_client = client if client is not None else apifansly_shared_client()
    try:
        remote_lists = await fetch_remote_lists(account_id, client=active_client)
        remote_members: dict[str, list[str]] = {}
        incomplete_remote_lists: set[str] = set()
        for remote in remote_lists:
            external_list_id = remote["external_list_id"]
            members, complete = await fetch_remote_members(
                account_id,
                external_list_id,
                client=active_client,
            )
            remote_members[external_list_id] = members
            if not complete:
                incomplete_remote_lists.add(external_list_id)
    except ApiFanslyAccountAccessError as exc:
        # The creator binding needs reconnecting. Record it and re-raise so the
        # caller applies the same backoff it uses for every other API Fansly
        # access failure.
        await asyncio.to_thread(_record_failure, creator_id, str(exc))
        raise
    except Exception as exc:
        await asyncio.to_thread(_record_failure, creator_id, str(exc))
        raise

    counters = await asyncio.to_thread(
        _reconcile,
        creator_id,
        remote_lists,
        remote_members,
        incomplete_remote_lists,
    )
    print(
        f"[FANSLY LISTS] creator={creator_id} remote={counters['remote_lists']} "
        f"created={counters['created_lists']} renamed={counters['renamed_lists']} "
        f"archived={counters['archived_lists']} "
        f"members+{counters['added_members']}/-{counters['removed_members']} "
        f"unmapped={counters['unmapped_members']} "
        f"skipped_removals={counters['skipped_removals']}"
    )
    return {"status": "degraded" if counters["degraded"] else "ok", **counters}


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
