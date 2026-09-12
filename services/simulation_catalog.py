"""Mirror one creator's vault metadata into another creator's TEST catalog.

The problem
-----------
The owner's admin/testing creator has a thin vault, so simulated Full Auto turns
exercise almost none of the commercial logic that production hits: coherence
grouping, explicitness escalation, photo/video mixes, multi-step allocation,
price probing inside approved bounds. All of that is a function of catalog
variety. The owner wants a realistic catalog on the test creator, built from a
real creator's vault.

Why this is not a copy
----------------------
``creator_vault_media.media_id`` is a Fansly media id belonging to the SOURCE
creator's account. Duplicating those rows under the target creator's id would
produce rows that every planner and every delivery path reads as ordinary,
sellable, deliverable inventory — pointing at another account's media. Sending
one would be an attempt to deliver content the target creator does not own.

So the mirror is deliberately NOT a faithful duplicate. Two things are changed,
and each on its own would be sufficient:

``simulation_only = true``
    Live planning filters these rows out (``core.simulation_catalog``); the
    owner simulator, which runs inside ``simulation_scope()``, includes them.

a rewritten media id
    ``sim:<source>:<media_id>``. Not a platform id, and cannot become one. Even
    a code path that forgot the flag cannot deliver it: the id is refused by
    ``services.ppv_delivery.send_locked_ppv`` and by the Auto delivery branch,
    and the platform would reject it regardless.

Everything else — descriptions, tags, categories, explicitness, scene metadata,
approved price bounds, set composition — is preserved exactly, because that is
the whole point: the simulator must plan against realistic content.

Safety properties
-----------------
* **The source is never written to.** Every statement here is scoped to the
  target creator, and deletions additionally require ``simulation_only = true``
  and a matching ``source_creator_id``. Rebuilding or deleting a mirror cannot
  touch the source creator's vault.
* **Idempotent.** Provenance columns carry a unique index, so re-running the
  mirror refreshes rows rather than duplicating them. A refresh also removes
  mirrored rows whose source has since disappeared.
* **Owner-only.** Authorization lives in the route, reusing the existing
  simulator allowlist; nothing here is reachable by an agency tenant.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from core.simulation_catalog import (
    exclude_simulation_only,
    run_live_catalog_query,
    simulation_media_id,
)
from core.supabase import get_supabase

# Columns copied verbatim from a source media row. Deliberately explicit: a
# future column is NOT mirrored until somebody decides it should be, and ``url``
# / ``fansly_media_id`` are deliberately absent — a mirrored row must not carry
# a live location or a real platform identity.
_MEDIA_COLUMNS = (
    "media_id",
    "album_title",
    "mimetype",
    "content_category",
    "ai_description",
    "price_min",
    "price_max",
    "scene_location",
    "scene_outfit",
    "scene_lighting",
    "scene_id",
)

_SET_COLUMNS = (
    "title",
    "description",
    "location",
    "outfit",
    "explicit_min",
    "explicit_max",
    "preview_media_id",
    "suggested_price",
    "tags",
    "base_price_cents",
    "min_price_cents",
    "max_price_cents",
    "dynamic_pricing_enabled",
    "metadata_version",
    "status",
)

_INSERT_CHUNK = 100


class SimulationCatalogError(RuntimeError):
    """A mirror operation could not be completed."""


@dataclass(frozen=True)
class MirrorResult:
    source_creator_id: str
    target_creator_id: str
    media_mirrored: int
    sets_mirrored: int
    media_removed: int
    sets_removed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_creator_id": self.source_creator_id,
            "target_creator_id": self.target_creator_id,
            "media_mirrored": self.media_mirrored,
            "sets_mirrored": self.sets_mirrored,
            "media_removed": self.media_removed,
            "sets_removed": self.sets_removed,
            "simulation_only": True,
        }


def _mirrored_media_row(
    row: dict[str, Any],
    *,
    source_creator_id: str,
    target_creator_id: str,
) -> dict[str, Any] | None:
    source_media_id = str(row.get("media_id") or "").strip()
    if not source_media_id:
        return None
    mirrored: dict[str, Any] = {
        key: row.get(key) for key in _MEDIA_COLUMNS if key in row
    }
    mirrored["creator_id"] = target_creator_id
    mirrored["media_id"] = simulation_media_id(source_creator_id, source_media_id)
    mirrored["source_creator_id"] = source_creator_id
    mirrored["source_media_id"] = source_media_id
    mirrored["simulation_only"] = True
    # No live location and no platform identity travel with a mirrored row.
    mirrored["url"] = None
    mirrored["fansly_media_id"] = None
    return mirrored


def _mirrored_set_row(
    row: dict[str, Any],
    *,
    source_creator_id: str,
    target_creator_id: str,
) -> dict[str, Any] | None:
    source_set_id = str(row.get("id") or "").strip()
    if not source_set_id:
        return None
    media_ids = [
        simulation_media_id(source_creator_id, str(value))
        for value in (row.get("media_ids") or [])
        if str(value or "").strip()
    ]
    if not media_ids:
        # A set with no media cannot be planned, so mirroring it would only
        # create a row the simulator has to filter out again.
        return None
    mirrored: dict[str, Any] = {key: row.get(key) for key in _SET_COLUMNS if key in row}
    mirrored["creator_id"] = target_creator_id
    mirrored["media_ids"] = media_ids
    preview = str(row.get("preview_media_id") or "").strip()
    mirrored["preview_media_id"] = (
        simulation_media_id(source_creator_id, preview) if preview else None
    )
    mirrored["source_creator_id"] = source_creator_id
    mirrored["source_set_id"] = source_set_id
    mirrored["simulation_only"] = True
    # Mirrored sets keep the source's approval state so the simulator plans
    # against the same inventory the source creator actually sells.
    mirrored["status"] = row.get("status") or "approved"
    mirrored["source"] = "simulation_mirror"
    return mirrored


async def mirror_creator_catalog(
    *,
    source_creator_id: str,
    target_creator_id: str,
) -> MirrorResult:
    """Refresh the target creator's simulation catalog from the source vault.

    Idempotent: a second run against unchanged source data produces the same
    rows. Rows whose source has disappeared are removed, so a refresh converges
    rather than accumulating.
    """
    source_creator_id = str(source_creator_id).strip()
    target_creator_id = str(target_creator_id).strip()
    if not source_creator_id or not target_creator_id:
        raise SimulationCatalogError("both creator ids are required")
    if source_creator_id == target_creator_id:
        # Mirroring a creator onto itself would mark its own vault
        # simulation-only and take it out of live planning.
        raise SimulationCatalogError(
            "a creator cannot mirror its own catalog onto itself"
        )

    def _run() -> MirrorResult:
        db = get_supabase()

        source_media = (
            db.table("creator_vault_media")
            .select(", ".join(("id", *_MEDIA_COLUMNS)))
            .eq("creator_id", source_creator_id)
            .execute()
        ).data or []
        source_sets = (
            db.table("vault_sets")
            .select(", ".join(("id", "media_ids", *_SET_COLUMNS)))
            .eq("creator_id", source_creator_id)
            .execute()
        ).data or []

        media_rows = [
            mirrored
            for row in source_media
            if (
                mirrored := _mirrored_media_row(
                    row,
                    source_creator_id=source_creator_id,
                    target_creator_id=target_creator_id,
                )
            )
        ]
        set_rows = [
            mirrored
            for row in source_sets
            if (
                mirrored := _mirrored_set_row(
                    row,
                    source_creator_id=source_creator_id,
                    target_creator_id=target_creator_id,
                )
            )
        ]

        # Delete-then-insert, scoped three ways: the target creator, the
        # simulation flag, and this source. Nothing else can be in range.
        removed_media = _delete_mirror(
            db, "creator_vault_media", target_creator_id, source_creator_id
        )
        removed_sets = _delete_mirror(
            db, "vault_sets", target_creator_id, source_creator_id
        )

        for chunk_start in range(0, len(media_rows), _INSERT_CHUNK):
            db.table("creator_vault_media").insert(
                media_rows[chunk_start : chunk_start + _INSERT_CHUNK]
            ).execute()
        for chunk_start in range(0, len(set_rows), _INSERT_CHUNK):
            db.table("vault_sets").insert(
                set_rows[chunk_start : chunk_start + _INSERT_CHUNK]
            ).execute()

        return MirrorResult(
            source_creator_id=source_creator_id,
            target_creator_id=target_creator_id,
            media_mirrored=len(media_rows),
            sets_mirrored=len(set_rows),
            media_removed=removed_media,
            sets_removed=removed_sets,
        )

    result = await asyncio.to_thread(_run)
    print(
        f"[SIMULATION CATALOG] mirrored source={source_creator_id} "
        f"target={target_creator_id} media={result.media_mirrored} "
        f"sets={result.sets_mirrored} "
        f"replaced_media={result.media_removed} replaced_sets={result.sets_removed}"
    )
    return result


async def delete_creator_catalog_mirror(
    *,
    source_creator_id: str,
    target_creator_id: str,
) -> MirrorResult:
    """Remove a mirror from the target creator. The source is never touched."""
    source_creator_id = str(source_creator_id).strip()
    target_creator_id = str(target_creator_id).strip()
    if not source_creator_id or not target_creator_id:
        raise SimulationCatalogError("both creator ids are required")

    def _run() -> MirrorResult:
        db = get_supabase()
        removed_sets = _delete_mirror(
            db, "vault_sets", target_creator_id, source_creator_id
        )
        removed_media = _delete_mirror(
            db, "creator_vault_media", target_creator_id, source_creator_id
        )
        return MirrorResult(
            source_creator_id=source_creator_id,
            target_creator_id=target_creator_id,
            media_mirrored=0,
            sets_mirrored=0,
            media_removed=removed_media,
            sets_removed=removed_sets,
        )

    result = await asyncio.to_thread(_run)
    print(
        f"[SIMULATION CATALOG] mirror deleted source={source_creator_id} "
        f"target={target_creator_id} media={result.media_removed} "
        f"sets={result.sets_removed}"
    )
    return result


def _delete_mirror(
    db: Any,
    table: str,
    target_creator_id: str,
    source_creator_id: str,
) -> int:
    """Delete only mirrored rows, only on the target, only from this source.

    All three predicates are required. ``creator_id`` alone would wipe the
    target's own vault; without ``simulation_only`` a future non-mirrored row
    carrying provenance could be caught; without ``source_creator_id`` one
    source's refresh would delete another's mirror.
    """
    response = (
        db.table(table)
        .delete()
        .eq("creator_id", target_creator_id)
        .eq("simulation_only", True)
        .eq("source_creator_id", source_creator_id)
        .execute()
    )
    return len(response.data or [])


# ---------------------------------------------------------------------------
# Owner-only visual preview of mirrored test media
# ---------------------------------------------------------------------------
#
# A mirrored row deliberately carries no ``url`` and no ``fansly_media_id``, so
# the simulator can plan a PPV against realistic content but cannot render it.
# Showing the owner a grey box where the fan would see a locked photo defeats
# the point of a full-fidelity workspace.
#
# The resolution below is a PREVIEW, and the distinction from delivery is the
# whole design:
#
# * it goes through PROVENANCE (``source_creator_id`` + ``source_media_id``),
#   reading the SOURCE creator's own row for a display URL;
# * it never writes anything. The source's platform media id is not copied onto
#   the simulation creator's catalog, so nothing here can make a mirrored row
#   deliverable. ``send_locked_ppv`` and the Auto delivery branch still refuse a
#   ``sim:`` id, and the simulation creator still has no platform identity for
#   the source's media;
# * it is owner-only, enforced at the route with the same allowlist the rest of
#   the simulator uses.
#
# In other words: simulation preview access and live delivery authority stay
# separate, which is exactly the property the mirror was built to guarantee.


async def resolve_simulation_media_previews(
    *,
    creator_id: str,
    media_ids: list[str],
) -> dict[str, dict[str, Any]]:
    """Display URLs for mirrored test media, resolved through provenance.

    Returns one entry per requested id. An id that is not a mirrored row of this
    creator resolves to nulls rather than raising: the caller is rendering a
    chat, and one unresolvable thumbnail must not fail the request.
    """
    wanted = [str(value).strip() for value in media_ids if str(value or "").strip()]
    wanted = list(dict.fromkeys(wanted))
    if not wanted:
        return {}

    from core.simulation_catalog import is_simulation_media_id

    simulated = [value for value in wanted if is_simulation_media_id(value)]
    empty = {"url": None, "thumbnail_url": None, "mimetype": None, "source": None}
    resolved: dict[str, dict[str, Any]] = {value: dict(empty) for value in wanted}
    if not simulated:
        return resolved

    def _load_mirrors() -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        db = get_supabase()
        for start in range(0, len(simulated), 200):
            chunk = simulated[start : start + 200]
            response = (
                db.table("creator_vault_media")
                .select(
                    "media_id, source_creator_id, source_media_id, mimetype, "
                    "simulation_only"
                )
                .eq("creator_id", str(creator_id))
                .in_("media_id", chunk)
                .execute()
            )
            rows.extend(response.data or [])
        return rows

    try:
        mirrors = await asyncio.to_thread(_load_mirrors)
    except Exception as exc:
        print(f"[SIMULATION PREVIEW] mirror read failed creator={creator_id}: {exc}")
        return resolved

    # Group the source lookups by source creator: one query per source, not one
    # per media item.
    by_source: dict[str, dict[str, str]] = {}
    for row in mirrors:
        # Only a genuinely mirrored row is previewable. A row that somehow
        # carries a sim: id without the flag and without provenance is not one.
        if not row.get("simulation_only"):
            continue
        source_creator = str(row.get("source_creator_id") or "").strip()
        source_media = str(row.get("source_media_id") or "").strip()
        mirrored_id = str(row.get("media_id") or "").strip()
        if not (source_creator and source_media and mirrored_id):
            continue
        by_source.setdefault(source_creator, {})[source_media] = mirrored_id
        resolved[mirrored_id]["mimetype"] = row.get("mimetype")

    def _load_source(source_creator: str, source_media_ids: list[str]) -> list[dict]:
        rows: list[dict] = []
        db = get_supabase()
        for start in range(0, len(source_media_ids), 200):
            response = (
                db.table("creator_vault_media")
                .select("media_id, url, thumbnail_url, mimetype")
                .eq("creator_id", source_creator)
                .in_("media_id", source_media_ids[start : start + 200])
                .execute()
            )
            rows.extend(response.data or [])
        return rows

    for source_creator, mapping in by_source.items():
        try:
            source_rows = await asyncio.to_thread(
                _load_source, source_creator, list(mapping)
            )
        except Exception as exc:
            print(
                f"[SIMULATION PREVIEW] source read failed "
                f"creator={source_creator}: {exc}"
            )
            continue
        for row in source_rows:
            mirrored_id = mapping.get(str(row.get("media_id") or ""))
            if not mirrored_id:
                continue
            resolved[mirrored_id] = {
                "url": row.get("url"),
                "thumbnail_url": row.get("thumbnail_url"),
                "mimetype": row.get("mimetype") or resolved[mirrored_id].get("mimetype"),
                # Named so the operator can see this pixel came from another
                # creator's vault and is a preview, not this creator's content.
                "source": "simulation_mirror",
            }
    return resolved


# ---------------------------------------------------------------------------
# Owner-only mirror SOURCE discovery
# ---------------------------------------------------------------------------
#
# The mirror exists so the platform owner can build a realistic test catalog on
# their own simulation creator from a real creator's vault. That real creator is
# typically AGENCY-OWNED and deliberately NOT assigned to the owner's dashboard
# user — granting ordinary tenancy over it just to copy metadata would hand the
# owner that agency's chats, fans and revenue, which is far more access than the
# job needs.
#
# So source discovery is its own thing, and it is deliberately NOT the simulator
# creator list:
#
#   /simulation/creators          WHO the owner may simulate AS. Tenancy-scoped.
#                                 Unchanged by this function.
#   /simulation/catalog/sources   WHOSE vault may be COPIED FROM. Owner-gated,
#                                 cross-tenant, read-only, and metadata only.
#
# Appearing in this list grants nothing. It does not put a creator in the
# simulator selector, does not create an assignment, and does not widen any
# other route: the only thing it enables is being named as the SOURCE of a
# mirror, whose target must still be a creator the caller ordinarily holds.
#
# The columns returned are the minimum the picker needs — id, display name, and
# whether there is usable content — and deliberately exclude everything about
# the account itself (no platform account id, no session state, no settings).


# What the picker is told about each candidate source. Anything not on this list
# is not read, so a future column cannot start leaking across tenants because
# somebody widened a select.
_SOURCE_COLUMNS = ("id", "platform_username")


@dataclass(frozen=True)
class MirrorSource:
    creator_id: str
    name: str
    approved_sets: int
    media_items: int

    @property
    def usable(self) -> bool:
        """Whether mirroring this creator would produce a catalog worth testing.

        A vault with no approved sets can be mirrored, but the simulator would
        plan against nothing, so the picker says so rather than letting the
        owner discover it after the fact.
        """
        return self.approved_sets > 0 and self.media_items > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "creator_id": self.creator_id,
            "name": self.name,
            "approved_sets": self.approved_sets,
            "media_items": self.media_items,
            "usable": self.usable,
        }


def _row_count(response: Any) -> int:
    """Total matching rows from a PostgREST response.

    The queries below pair ``count="exact"`` with ``limit(1)``, so ``count`` is
    the whole match while ``data`` is one row. Reading ``data`` instead would
    report every non-empty vault as having exactly one item. The length fallback
    is only for a transport that returned no count at all.
    """
    exact = getattr(response, "count", None)
    if isinstance(exact, int) and exact >= 0:
        return exact
    return len(getattr(response, "data", None) or [])


async def list_mirror_source_creators() -> list[MirrorSource]:
    """Every creator whose vault may be mirrored, with its content counts.

    Cross-tenant by design and by necessity — see the note above. Authorization
    is the caller's job and lives in the route: this function must only ever be
    reached through the platform-owner simulator allowlist.

    Counts describe REAL content. Mirrored rows are excluded, so a creator that
    is itself a simulation target does not advertise somebody else's catalog
    back as if it were its own.
    """

    def _run() -> list[MirrorSource]:
        db = get_supabase()
        creators = (
            db.table("creators")
            .select(", ".join(_SOURCE_COLUMNS))
            .order("platform_username")
            .limit(500)
            .execute()
        ).data or []

        sources: list[MirrorSource] = []
        for creator in creators:
            creator_id = str(creator.get("id") or "").strip()
            if not creator_id:
                continue

            def _sets(apply_filter: bool, _id: str = creator_id) -> Any:
                query = (
                    db.table("vault_sets")
                    .select("id", count="exact")
                    .eq("creator_id", _id)
                    .eq("status", "approved")
                )
                if apply_filter:
                    query = exclude_simulation_only(query)
                # count + limit(1): PostgREST reports the total in a header and
                # sends one row, so counting a 10,000-item vault does not drag
                # 10,000 ids across the wire for a picker that shows a number.
                return query.limit(1).execute()

            def _media(apply_filter: bool, _id: str = creator_id) -> Any:
                query = (
                    db.table("creator_vault_media")
                    .select("id", count="exact")
                    .eq("creator_id", _id)
                )
                if apply_filter:
                    query = exclude_simulation_only(query)
                return query.limit(1).execute()

            approved_sets = _row_count(
                run_live_catalog_query(
                    _sets,
                    label="simulation.mirror_sources.sets",
                    include_simulation=False,
                )
            )
            media_items = _row_count(
                run_live_catalog_query(
                    _media,
                    label="simulation.mirror_sources.media",
                    include_simulation=False,
                )
            )
            sources.append(
                MirrorSource(
                    creator_id=creator_id,
                    name=str(creator.get("platform_username") or creator_id),
                    approved_sets=approved_sets,
                    media_items=media_items,
                )
            )
        return sources

    return await asyncio.to_thread(_run)


async def mirror_source_exists(creator_id: str) -> bool:
    """Whether a creator id names a real creator.

    Used by the mirror route so a mistyped source id is reported as such rather
    than silently mirroring an empty vault and looking like a working no-op.
    Only ever called after the owner allowlist has passed, and it reveals one
    boolean about a creator the caller may already enumerate.
    """
    key = str(creator_id or "").strip()
    if not key:
        return False

    def _run() -> bool:
        rows = (
            get_supabase().table("creators")
            .select("id")
            .eq("id", key)
            .limit(1)
            .execute()
        ).data or []
        return bool(rows)

    try:
        return await asyncio.to_thread(_run)
    except Exception as exc:
        print(f"[SIMULATION CATALOG] source existence check failed {key}: {exc}")
        # Fail open into the mirror itself, which is scoped to the target and
        # cannot damage a source that does not exist. A transient read failure
        # must not be reported to the owner as "no such creator".
        return True
