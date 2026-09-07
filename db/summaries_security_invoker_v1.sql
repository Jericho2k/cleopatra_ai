-- fan_conversation_summaries must respect the CALLER's row-level security.
--
-- Audit reference: SEC-003.
--
-- The dashboard reads this view directly from the browser
-- (cleopatra-dashboard/app/page.tsx) — fan names, spend, last message content,
-- notes. tenant_isolation_v1.sql discovers objects to police with
-- `t.table_type = 'BASE TABLE'`, so no policy was ever created for it.
--
-- PostgreSQL does not support row-level security policies ON a view. Creating
-- one would fail, and pretending otherwise would be worse than doing nothing.
-- The correct mechanism is security_invoker: with it on, the view executes with
-- the querying role's privileges, so the RLS already enforced on the underlying
-- fans and messages tables applies. Without it the view runs with its owner's
-- privileges and RLS on those tables is bypassed — meaning an authenticated
-- operator who removes the client-side .eq('creator_id', ...) reads every
-- agency's conversations.
--
-- This migration does NOT redefine the view. Its real definition lives in
-- Supabase and is not in version control (DB-000), so redefining it here would
-- mean inventing it. ALTER VIEW ... SET changes only the setting and leaves the
-- definition untouched, which is exactly what is needed.
--
-- Requires PostgreSQL 15+. Supabase has been on 15+ since 2023.
-- Idempotent: safe to re-run.

do $$
declare
    view_oid oid;
    was_invoker boolean;
begin
    select c.oid into view_oid
      from pg_class c
      join pg_namespace n on n.oid = c.relnamespace
     where n.nspname = current_schema()
       and c.relname = 'fan_conversation_summaries'
       and c.relkind = 'v';

    if view_oid is null then
        -- Not an error: a fresh environment may not have created it yet, and a
        -- migration that hard-failed here could not be run before the view
        -- exists. It IS a problem in production — see the verification query at
        -- the bottom of this file.
        raise warning
            'fan_conversation_summaries not found in %; nothing to harden (SEC-003)',
            current_schema();
        return;
    end if;

    -- Read the option VALUE rather than string-matching the whole entry.
    -- Postgres stores what you wrote: `set (security_invoker = on)` yields
    -- 'security_invoker=on' and `= true` yields 'security_invoker=true'. A check
    -- that compares against one spelling reports a correctly configured view as
    -- unprotected — a trap worth avoiding in the manual verification too.
    was_invoker := coalesce(
        (
            select lower(o.option_value) in ('on', 'true', 'yes', '1')
              from pg_class c,
                   pg_options_to_table(c.reloptions) o
             where c.oid = view_oid
               and lower(o.option_name) = 'security_invoker'
        ),
        false
    );

    if was_invoker then
        raise notice
            'fan_conversation_summaries was already security_invoker (SEC-003)';
    else
        raise warning
            'fan_conversation_summaries was NOT security_invoker — it was '
            'running with definer privileges and bypassing RLS (SEC-003)';
    end if;

    execute format(
        'alter view %I.fan_conversation_summaries set (security_invoker = on)',
        current_schema()
    );
end
$$;

-- Fail loudly if the setting did not take. A migration that silently does
-- nothing is how this defect survived in the first place.
do $$
declare ok boolean;
begin
    select coalesce(
        (
            select lower(o.option_value) in ('on', 'true', 'yes', '1')
              from pg_class c
              join pg_namespace n on n.oid = c.relnamespace,
                   pg_options_to_table(c.reloptions) o
             where n.nspname = current_schema()
               and c.relname = 'fan_conversation_summaries'
               and c.relkind = 'v'
               and lower(o.option_name) = 'security_invoker'
        ),
        -- No row: either the view is absent (the warning above covered it) or
        -- the option is unset, which the ALTER above should have fixed.
        not exists (
            select 1
              from pg_class c
              join pg_namespace n on n.oid = c.relnamespace
             where n.nspname = current_schema()
               and c.relname = 'fan_conversation_summaries'
               and c.relkind = 'v'
        )
    ) into ok;

    if not ok then
        raise exception
            'fan_conversation_summaries is still not security_invoker (SEC-003)';
    end if;
end
$$;

-- Verification query for the live database:
--
--   select c.relname, o.option_name, o.option_value
--     from pg_class c
--     join pg_namespace n on n.oid = c.relnamespace
--     left join lateral pg_options_to_table(c.reloptions) o on true
--    where n.nspname = 'public'
--      and c.relname = 'fan_conversation_summaries';
--
-- Expect security_invoker with value on/true. NOTE: do not grep reloptions for
-- the literal 'security_invoker=true' — a view configured with `= on` stores
-- 'security_invoker=on' and such a check reports a false negative.
