-- Durable webhook ingestion: one processing obligation per platform event.
--
-- Audit reference: REL-006 (depends on DB-000 and on
-- db/message_platform_identity_v1.sql).
--
-- Apply BEFORE deploying the matching backend commit. Idempotent.
--
-- The webhook now returns HTTP 2xx as soon as the fan message is persisted and
-- a PROCESS_INBOUND_MESSAGE action exists, so the platform is free to redeliver
-- the same event. Two guarantees make that safe, and they live in two places:
--
--   * one platform message, one row — (creator_id, fansly_message_id), created
--     by db/message_platform_identity_v1.sql (REL-002). Deliberately NOT
--     duplicated here with a global key on fansly_message_id alone: that would
--     reject a second creator's legitimately distinct message if platform ids
--     turn out to be unique only per account. See that file for the reasoning.
--   * one event, one obligation — the scheduled-action dedupe key, below.

-- ---------------------------------------------------------------------------
-- scheduled_actions.dedupe_key is the authority for "one effective obligation".
--
-- schedule_action() already upserts on it, and the ingestion path relies on the
-- conflict being DETECTED rather than inserting a second row: an inbound event
-- is written with ignore_duplicates so a redelivery cannot reset an action that
-- is already PENDING (it would run twice), PROCESSING (it would race an
-- in-flight run), or COMPLETED (it would reprocess an answered message). Every
-- one of those depends on this index existing.
--
-- Refuse rather than destroy: if duplicates already exist, report them instead
-- of merging or deleting durable work. Run with -v ON_ERROR_STOP=1.
-- ---------------------------------------------------------------------------
do $$
declare
    duplicate_groups bigint;
    sample text;
begin
    if to_regclass(current_schema() || '.scheduled_actions') is null then
        raise warning 'scheduled_actions table not found in %; skipping', current_schema();
        return;
    end if;

    execute format(
        'select count(*) from ('
        '  select dedupe_key from %I.scheduled_actions'
        '   where dedupe_key is not null'
        '   group by dedupe_key having count(*) > 1'
        ') d',
        current_schema()
    ) into duplicate_groups;

    if duplicate_groups > 0 then
        execute format(
            'select string_agg(format(%L, dedupe_key, n), '', '') from ('
            '  select dedupe_key, count(*) as n from %I.scheduled_actions'
            '   where dedupe_key is not null'
            '   group by dedupe_key having count(*) > 1'
            '   order by count(*) desc limit 5) s',
            'key=%s copies=%s',
            current_schema()
        ) into sample;
        raise exception
            'scheduled_actions has % duplicated dedupe_key group(s): %. '
            'Resolve them before applying this migration; this file will not '
            'merge or delete durable work.',
            duplicate_groups, coalesce(sample, 'n/a');
    end if;

    -- Only create it if the column is not already unique. Some deployments
    -- carry the constraint inline on the table (dedupe_key text unique); adding
    -- a second index on the same column there would cost writes and buy
    -- nothing.
    if exists (
        select 1
          from pg_index i
          join pg_class t on t.oid = i.indrelid
          join pg_namespace n on n.oid = t.relnamespace
         where n.nspname = current_schema()
           and t.relname = 'scheduled_actions'
           and i.indisunique
           and i.indnatts = 1
           and i.indkey[0] = (
               select attnum from pg_attribute
                where attrelid = t.oid and attname = 'dedupe_key'
           )
    ) then
        raise notice 'scheduled_actions.dedupe_key is already unique; nothing to do';
        return;
    end if;

    execute format(
        'create unique index if not exists scheduled_actions_dedupe_key_key '
        'on %I.scheduled_actions (dedupe_key)',
        current_schema()
    );
end
$$;

-- The health surface reads pending queue depth and the oldest due action on
-- every probe, and the worker claims by (status, execute_at) far more often now
-- that a full batch re-polls immediately instead of sleeping 60 seconds.
create index if not exists scheduled_actions_status_execute_at_idx
    on public.scheduled_actions (status, execute_at);
