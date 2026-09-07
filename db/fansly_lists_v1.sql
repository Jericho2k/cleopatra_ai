-- Mirror of the Lists an agency already maintains on the creator's Fansly
-- account, alongside Cleopatra's own locally created lists.
--
-- Read-only in one direction: Cleopatra pulls from Fansly and never creates,
-- renames, or deletes a remote list. Locally created lists are never touched by
-- the sync.
--
-- Idempotent. Safe to re-run.

alter table public.fan_lists
    -- 'local'  — created in Cleopatra by an agency operator (the existing rows).
    -- 'fansly' — mirrored from the creator's Fansly account.
    add column if not exists source text not null default 'local',
    -- The remote Fansly list ID. This is the ONLY stable mapping key: a repeated
    -- sync upserts on it, so a remote rename updates the mirror in place instead
    -- of creating a second list, and two remote lists that happen to share a
    -- name stay distinct.
    add column if not exists external_list_id text null,
    add column if not exists external_synced_at timestamptz null,
    -- Set when a previously mirrored list is no longer returned by Fansly.
    -- Deliberately not a hard delete: Auto Audience include/exclude rules and
    -- re-engagement settings reference fan_lists.id, and dropping the row would
    -- silently change which fans those rules select. An archived mirror stays
    -- resolvable and stops being offered for new targeting.
    add column if not exists external_archived_at timestamptz null,
    add column if not exists external_item_count integer null;

alter table public.fan_lists
    drop constraint if exists fan_lists_source_check;
alter table public.fan_lists
    add constraint fan_lists_source_check
    check (source in ('local', 'fansly'));

-- A Fansly-sourced list must carry its remote identifier, and a local list must
-- not. Without this, a mirror with a null external id would duplicate on every
-- sync because there would be nothing to upsert against.
alter table public.fan_lists
    drop constraint if exists fan_lists_external_id_required;
alter table public.fan_lists
    add constraint fan_lists_external_id_required
    check (
        (source = 'fansly' and external_list_id is not null)
        or (source = 'local' and external_list_id is null)
    );

-- The invariant the whole reconciliation depends on: one mirror per remote list
-- per creator. Scoped to the creator so two agencies cannot collide, and
-- partial so it never constrains local lists.
create unique index if not exists fan_lists_creator_external_id_key
    on public.fan_lists (creator_id, external_list_id)
    where external_list_id is not null;

create index if not exists fan_lists_creator_source_idx
    on public.fan_lists (creator_id, source);

-- Membership rows carry the same provenance so reconciliation can remove a
-- fan from a Fansly mirror without ever touching a membership an operator
-- created by hand — including on a fan who is in both kinds of list.
alter table public.fan_list_members
    add column if not exists source text not null default 'local',
    add column if not exists external_synced_at timestamptz null;

alter table public.fan_list_members
    drop constraint if exists fan_list_members_source_check;
alter table public.fan_list_members
    add constraint fan_list_members_source_check
    check (source in ('local', 'fansly'));

create index if not exists fan_list_members_list_source_idx
    on public.fan_list_members (list_id, source);

-- Per-creator sync state for the dashboard: when the last refresh ran, whether
-- it succeeded, and why not. Mirrors the existing last_fansly_audience_sync_at
-- convention rather than introducing a separate status table.
alter table public.creators
    add column if not exists last_fansly_lists_sync_at timestamptz null,
    add column if not exists fansly_lists_sync_error text null,
    add column if not exists fansly_lists_sync_failed_at timestamptz null;

-- Existing rows predate the source column and are all operator-created.
update public.fan_lists
   set source = 'local'
 where source is null;

update public.fan_list_members
   set source = 'local'
 where source is null;

-- Row level security is applied by db/tenant_isolation_v1.sql, which enumerates
-- every creator-owned table rather than a hand-maintained list. fan_lists and
-- fan_list_members are already covered there and these columns inherit those
-- policies; re-run that migration after this one if any new table is ever added.
