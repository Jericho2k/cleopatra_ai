-- Owner-only simulation catalog: test content that can never be delivered.
--
-- WHY THIS EXISTS
--
-- The owner-only Full Auto simulator (core/simulation.py) runs the real
-- pipeline against a ``test_`` fan. Its value depends entirely on the catalog
-- behind it: simulating against a creator with three sets exercises none of the
-- coherence, escalation, media-type or allocation logic that production hits.
-- The owner wants the admin/testing creator to carry a realistic catalog
-- mirrored from a real creator's vault.
--
-- WHY IT IS NOT A PLAIN COPY
--
-- creator_vault_media.media_id is a FANSLY media id belonging to the SOURCE
-- creator's account. Copying those rows under a different creator_id would
-- create rows that look, to every planner and delivery path, exactly like
-- deliverable inventory — while pointing at another account's media. That is
-- the one outcome this feature must make impossible.
--
-- So the mirror is marked, not disguised, and it is marked twice over:
--
--   1. ``simulation_only = true``. Live planning filters it out; the owner
--      simulator includes it. One boolean, one filter, one meaning.
--   2. The mirrored rows carry a REWRITTEN media id (``sim:<...>``), applied by
--      services/simulation_catalog.py. A rewritten id is not a valid platform
--      media id, so even a code path that forgot the boolean cannot hand one to
--      the platform and have it deliver.
--
-- Provenance is preserved (``source_creator_id`` / ``source_set_id`` /
-- ``source_media_id``) so a mirror can be refreshed and rebuilt idempotently,
-- and so an operator can always see where a test row came from.
--
-- DESTRUCTIVENESS: none. This migration only ADDS columns and indexes. Deleting
-- a mirror deletes rows on the TARGET creator that carry simulation_only=true
-- AND a source_creator_id; the source creator's own rows are never in scope.
--
-- Idempotent: safe to re-run.

-- ---------------------------------------------------------------------------
-- vault_sets
-- ---------------------------------------------------------------------------
alter table public.vault_sets
    add column if not exists simulation_only boolean not null default false,
    add column if not exists source_creator_id uuid null
        references public.creators(id) on delete set null,
    add column if not exists source_set_id uuid null;

comment on column public.vault_sets.simulation_only is
'True only for owner-mirrored test content. Excluded from live package planning and never eligible for platform delivery.';
comment on column public.vault_sets.source_creator_id is
'Creator this mirrored test set was copied from. Null for ordinary sets.';
comment on column public.vault_sets.source_set_id is
'vault_sets.id this mirrored test set was copied from. Null for ordinary sets.';

-- The mirror is refreshable: re-running it must update, not duplicate.
create unique index if not exists vault_sets_simulation_mirror_idx
    on public.vault_sets (creator_id, source_set_id)
    where source_set_id is not null;

-- The filter every live planning read applies.
create index if not exists vault_sets_live_catalog_idx
    on public.vault_sets (creator_id, status)
    where simulation_only = false;

-- ---------------------------------------------------------------------------
-- creator_vault_media
-- ---------------------------------------------------------------------------
alter table public.creator_vault_media
    add column if not exists simulation_only boolean not null default false,
    add column if not exists source_creator_id uuid null
        references public.creators(id) on delete set null,
    add column if not exists source_media_id text null;

comment on column public.creator_vault_media.simulation_only is
'True only for owner-mirrored test media. Never eligible for platform delivery; its media_id is a rewritten sim: id, not a platform id.';
comment on column public.creator_vault_media.source_creator_id is
'Creator this mirrored test media was copied from. Null for ordinary media.';
comment on column public.creator_vault_media.source_media_id is
'The source creator''s media_id this row mirrors. Null for ordinary media.';

create unique index if not exists creator_vault_media_simulation_mirror_idx
    on public.creator_vault_media (creator_id, source_media_id)
    where source_media_id is not null;

create index if not exists creator_vault_media_live_catalog_idx
    on public.creator_vault_media (creator_id)
    where simulation_only = false;

-- ---------------------------------------------------------------------------
-- The invariant, stated in the database as well as in code.
--
-- A mirrored row must be marked as simulation-only. Without this a future
-- writer could copy provenance across without the flag and produce rows that
-- live planning would happily sell.
-- ---------------------------------------------------------------------------
do $$
begin
    if not exists (
        select 1 from pg_constraint
         where conname = 'vault_sets_mirror_is_simulation_only'
    ) then
        alter table public.vault_sets
            add constraint vault_sets_mirror_is_simulation_only
            check (source_creator_id is null or simulation_only = true)
            not valid;
    end if;

    if not exists (
        select 1 from pg_constraint
         where conname = 'creator_vault_media_mirror_is_simulation_only'
    ) then
        alter table public.creator_vault_media
            add constraint creator_vault_media_mirror_is_simulation_only
            check (source_creator_id is null or simulation_only = true)
            not valid;
    end if;
end
$$;

-- NOT VALID on purpose: the constraint applies to every future write without
-- taking a full table scan on a large production vault at migration time. Run
-- the validation separately, out of hours, when convenient:
--
--   alter table public.vault_sets
--     validate constraint vault_sets_mirror_is_simulation_only;
--   alter table public.creator_vault_media
--     validate constraint creator_vault_media_mirror_is_simulation_only;
--
-- Both are no-ops on a database that has never had a mirror.
