-- Let mirrored test media exist without a platform identity it must not have.
--
-- THE BUG THIS FIXES
--
-- Mirroring a real creator's vault into the owner's simulation creator failed
-- in production with:
--
--   null value in column "fansly_media_id" of relation "creator_vault_media"
--   violates not-null constraint
--
-- services/simulation_catalog.py deliberately writes a mirrored media row as:
--
--   media_id          = 'sim:<source>:<source_media_id>'   (rewritten)
--   fansly_media_id   = NULL                               (no platform identity)
--   simulation_only   = true
--
-- The NULL is the point, not an oversight. fansly_media_id is the SOURCE
-- creator's real Fansly media id. Copying it onto the simulation creator would
-- produce a row that every delivery path reads as ordinary, sendable inventory
-- pointing at another account's media — the single outcome this whole feature
-- is built to make impossible. So the row cannot carry it, and production's
-- NOT NULL meant no mirrored media could be inserted at all.
--
-- WHY THE CONSTRAINT IS NOT SIMPLY DROPPED
--
-- fansly_media_id is the platform's record that a real media item exists, and a
-- live row without one is meaningless: /vault-media-urls resolves thumbnails by
-- it, the operator PPV composer identifies sendable media by it, set generation
-- reads it, and SEC-001 deliberately withholds UPDATE on it because editing it
-- locally would desynchronise the mirror without changing anything on Fansly.
-- Relaxing it for everything would let a live row lose its identity silently.
--
-- So the requirement becomes conditional, which is what it always meant:
--
--   simulation_only = true  OR  fansly_media_id IS NOT NULL
--
-- Live media must still have a platform identity. Simulation media must still
-- not. Both are now stated in the database rather than in one code path.
--
-- WHY PLANNING IS UNAFFECTED
--
-- Commercial planning reads vault_sets.media_ids, which carry the rewritten
-- sim: ids, never creator_vault_media.fansly_media_id. The one query that maps
-- fansly_media_id into a planning media_id (db/queries.get_vault_for_session)
-- is reached only by the /debug-scenes endpoint. Mirrored media is therefore
-- fully plannable with a NULL platform identity, which is the behaviour the
-- simulator has always assumed.
--
-- DEPENDS ON: db/simulation_catalog_v1.sql, which adds simulation_only. Applied
-- in the order db/migration_order.txt declares.
--
-- DESTRUCTIVENESS: none. It removes one NOT NULL and adds one CHECK that is
-- strictly weaker on simulation rows and exactly equivalent on live rows. No
-- data is read, moved or rewritten.
--
-- Idempotent: safe to re-run.

-- The CHECK goes on FIRST, deliberately. Dropping NOT NULL before the check
-- exists would leave a window in which an unmarked live row could be inserted
-- with no platform identity. Every existing row satisfies it already, because
-- fansly_media_id is NOT NULL at this point.
do $$
begin
    if not exists (
        select 1 from pg_constraint
         where conname = 'creator_vault_media_live_has_platform_identity'
    ) then
        alter table public.creator_vault_media
            add constraint creator_vault_media_live_has_platform_identity
            check (simulation_only = true or fansly_media_id is not null)
            not valid;
    end if;
end
$$;

alter table public.creator_vault_media
    alter column fansly_media_id drop not null;

comment on column public.creator_vault_media.fansly_media_id is
'The platform''s record of a real media item. Required for live rows and enforced by creator_vault_media_live_has_platform_identity; NULL only on simulation_only mirrored rows, which carry a rewritten sim: media_id instead and must never hold the source creator''s real Fansly id.';

-- NOT VALID above skips re-checking rows that already exist; it does NOT skip
-- enforcement on new writes, so the invariant holds from the moment this runs.
-- Validation is a no-op on any database whose rows all predate the relaxation,
-- and can be done whenever convenient:
--
--   alter table public.creator_vault_media
--     validate constraint creator_vault_media_live_has_platform_identity;
