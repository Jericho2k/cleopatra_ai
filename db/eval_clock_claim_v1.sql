-- claim_due_actions can be told what "now" is.
--
-- WHY
-- ---
-- The evaluation harness needs to simulate a customer returning a day or a
-- week later (docs/continuation_brief_2026-09-17.md, Phase B; review §5's
-- "return after one day and one week"). core/clock.py moves the application's
-- clock for that, but claiming a due scheduled action is decided inside this
-- function, by SQL's own now() — so a Python-side clock moved everything
-- EXCEPT the one thing "a queued follow-up becomes due" is about.
--
-- The function therefore takes an optional p_now. The application passes it
-- only while the eval clock is actually offset (core.clock.simulated_now_for_sql
-- returns None otherwise), so a normal deployment sends exactly the arguments
-- it sent before and this function behaves exactly as it did.
--
-- SAFETY
-- ------
-- p_now defaults to null and coalesces to now(), so nothing changes for any
-- caller that does not pass it. Execute stays revoked from public, anon and
-- authenticated and granted only to service_role: an operator holding a
-- browser Supabase client cannot call this at all, let alone tell it that it
-- is next Tuesday.
--
-- The application-side guard is the one that matters, and it is in
-- core/clock.py: the clock refuses to move unless APP_ENV is not production
-- AND EVAL_CLOCK_ENABLED is explicitly set. This file cannot enforce that —
-- it can only decline to be the reason the guard is bypassed.
--
-- Idempotent. Safe to re-run. Order relative to tenant_isolation_v1 does not
-- matter (it creates no table), but it must run AFTER
-- db/scheduled_action_claim_v1.sql, whose function it replaces.

create or replace function public.claim_due_actions(
    p_limit integer default 20,
    p_stale_minutes integer default 10,
    p_now timestamptz default null
)
returns setof public.scheduled_actions
language plpgsql
security definer
set search_path = public
as $$
declare
    v_limit integer := greatest(coalesce(p_limit, 20), 1);
    v_stale integer := greatest(coalesce(p_stale_minutes, 10), 1);
    -- The only line that differs from scheduled_action_claim_v1. Every use of
    -- now() below became v_now, so due-ness and staleness are decided against
    -- one consistent instant rather than three separate calls.
    v_now timestamptz := coalesce(p_now, now());
begin
    return query
    with due as (
        select id
          from public.scheduled_actions
         where status = 'PENDING'
           and execute_at <= v_now
         order by execute_at
         limit v_limit
           for update skip locked
    ),
    stale as (
        select id
          from public.scheduled_actions
         where status = 'PROCESSING'
           and locked_at is not null
           and locked_at < v_now - make_interval(mins => v_stale)
         order by locked_at
         limit v_limit
           for update skip locked
    ),
    claimable as (
        select id from due
        union
        select id from stale
    )
    update public.scheduled_actions as a
       set status = 'PROCESSING',
           locked_at = v_now
      from claimable as c
     where a.id = c.id
    returning a.*;
end;
$$;

-- The two-argument signature from scheduled_action_claim_v1 is now a distinct
-- overload that PostgreSQL would keep alongside this one, and an ambiguous
-- overload set is worse than either member of it. Dropped explicitly so there
-- is exactly one claim_due_actions.
drop function if exists public.claim_due_actions(integer, integer);

revoke all on function public.claim_due_actions(integer, integer, timestamptz)
    from public, anon, authenticated;
grant execute on function public.claim_due_actions(integer, integer, timestamptz)
    to service_role;
