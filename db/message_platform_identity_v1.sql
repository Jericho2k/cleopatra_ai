-- Make platform message identity authoritative in the database.
--
-- Audit reference: REL-002 (depends on DB-000).
--
-- Four ingestion paths write fan messages — the webhook, the poller,
-- reconciliation, and save_message — and every one of them did a read followed
-- by a non-atomic write. Whether concurrent writers could actually create
-- duplicate rows depended entirely on a unique constraint that no migration in
-- this repository creates.
--
-- CHOICE OF KEY: (creator_id, fansly_message_id).
--
-- Not fansly_message_id alone. Fansly ids look globally unique — the webhook
-- has always looked them up with no creator filter (main.py fansly_webhook) and
-- production has not mis-deduplicated — but "looks globally unique" is not the
-- same as "is guaranteed globally unique per the platform contract", and the
-- two choices fail very differently:
--
--   * If ids are globally unique, (creator_id, fansly_message_id) still catches
--     every real race. Every writer racing on one message carries the same
--     creator_id, because they are all ingesting the same creator's inbox.
--   * If ids are only unique per account or per group, a global unique index
--     REJECTS a second creator's legitimately distinct message — silent,
--     permanent message loss.
--
-- So the composite key is correct under both hypotheses and lossy under
-- neither. It is also exactly what save_message already checked, minus fan_id;
-- dropping fan_id tightens the key, which is safe because a platform message
-- belongs to one conversation and therefore one fan.
--
-- Partial index: rows with a null fansly_message_id are locally originated
-- (operator sends before platform confirmation) and must not collide.
--
-- Idempotent: safe to re-run.

-- ---------------------------------------------------------------------------
-- Refuse rather than destroy, and do it in ONE block.
--
-- CREATE UNIQUE INDEX would fail by itself on duplicate data, but with an error
-- that says nothing about how many or which rows. Production history must never
-- be silently merged or deleted to make a migration pass.
--
-- The check and the index creation share a single DO block deliberately: as two
-- top-level statements, a psql run without ON_ERROR_STOP would report the
-- refusal and then attempt the index anyway, producing a second, more confusing
-- error. Here the RAISE aborts both. (Still run with -v ON_ERROR_STOP=1.)
-- ---------------------------------------------------------------------------
do $$
declare
    duplicate_groups bigint;
    excess_rows bigint;
    sample text;
begin
    if to_regclass(current_schema() || '.messages') is null then
        raise warning 'messages table not found in %; skipping', current_schema();
        return;
    end if;

    execute format(
        'select count(*), coalesce(sum(n) - count(*), 0) from ('
        '  select creator_id, fansly_message_id, count(*) as n'
        '    from %I.messages'
        '   where fansly_message_id is not null'
        '   group by creator_id, fansly_message_id'
        '  having count(*) > 1'
        ') d',
        current_schema()
    ) into duplicate_groups, excess_rows;

    if duplicate_groups > 0 then
        execute format(
            'select string_agg(format(%L, creator_id, fansly_message_id, n), '', '') '
            'from (select creator_id, fansly_message_id, count(*) as n '
            '        from %I.messages where fansly_message_id is not null '
            '       group by creator_id, fansly_message_id having count(*) > 1 '
            '       order by count(*) desc limit 5) s',
            'creator=%s message=%s copies=%s',
            current_schema()
        ) into sample;

        raise exception
            'REL-002: % duplicate (creator_id, fansly_message_id) group(s) '
            'covering % excess row(s) already exist. The unique index was NOT '
            'created and NO rows were changed. Investigate before proceeding — '
            'do not delete production history to make this migration pass. '
            'Worst offenders: %',
            duplicate_groups, excess_rows, coalesce(sample, 'n/a');
    end if;

    -- The constraint itself. This is what makes ingestion idempotent:
    -- save_message upserts on it, so two concurrent writers produce exactly one
    -- row regardless of who read what first.
    execute format(
        'create unique index if not exists messages_creator_platform_identity_idx '
        'on %I.messages (creator_id, fansly_message_id) '
        'where fansly_message_id is not null',
        current_schema()
    );
end
$$;

-- Supporting index for the hottest read in the product — conversation history,
-- run several times per inbound message. Created only if an equivalent leading
-- (fan_id, sent_at) index is absent, so this cannot duplicate one that the
-- production schema already has.
do $$
begin
    if to_regclass(current_schema() || '.messages') is null then
        return;
    end if;
    if not exists (
        select 1
          from pg_index i
          join pg_class t on t.oid = i.indrelid
          join pg_namespace n on n.oid = t.relnamespace
         where n.nspname = current_schema()
           and t.relname = 'messages'
           and array_to_string(i.indkey, ' ') like (
               (select attnum::text from pg_attribute
                 where attrelid = t.oid and attname = 'fan_id') || ' ' ||
               (select attnum::text from pg_attribute
                 where attrelid = t.oid and attname = 'sent_at') || '%'
           )
    ) then
        execute format(
            'create index if not exists messages_fan_recent_idx '
            'on %I.messages (fan_id, sent_at desc)',
            current_schema()
        );
        raise notice 'created messages_fan_recent_idx';
    else
        raise notice 'a leading (fan_id, sent_at) index already exists; skipped';
    end if;
end
$$;

-- Duplicate diagnostic to run against production BEFORE applying this file:
--
--   select creator_id, fansly_message_id, count(*) as copies,
--          min(sent_at) as first_seen, max(sent_at) as last_seen
--     from public.messages
--    where fansly_message_id is not null
--    group by creator_id, fansly_message_id
--   having count(*) > 1
--    order by copies desc;
--
-- An empty result means this migration will apply cleanly.
