-- Atomic batch claim for the scheduled-action worker.
--
-- Audit reference: the claim_due_actions round-trip count (2 selects plus one
-- CAS UPDATE per claimed row — 22 round trips for a batch of 20, MEASURED).
--
-- The client-side compare-and-swap it replaces was correct: it re-asserted the
-- observed status, and for a stale reclaim it also re-asserted the observed
-- locked_at, so two workers racing on one row could not both win. What it could
-- not do is avoid the read-then-write window, and it paid a round trip per row
-- to close it.
--
-- Doing the whole thing in one statement is both cheaper and strictly safer:
--
--   * FOR UPDATE SKIP LOCKED means two workers never even select the same row,
--     so the race the CAS existed to lose is not entered at all;
--   * the UPDATE and the selection commit together, so there is no window in
--     which a row is chosen but not yet owned;
--   * RETURNING hands back exactly the rows this caller now owns.
--
-- Semantics preserved from the Python implementation:
--
--   * a PENDING row is claimable once execute_at has passed;
--   * a PROCESSING row is reclaimable once locked_at is older than the stale
--     window — that is the crash-recovery path, and it stays keyed on
--     locked_at, never on wall-clock age of the row;
--   * locked_at is stamped at claim time;
--   * the limit applies to each of the two sets, as it did before, so one call
--     can return up to 2 * p_limit rows.
--
-- Apply before deploying the worker change. The application falls back to the
-- previous per-row CAS when this function is absent, so a rolling deploy in
-- either order is safe.

create or replace function public.claim_due_actions(
    p_limit integer default 20,
    p_stale_minutes integer default 10
)
returns setof public.scheduled_actions
language plpgsql
security definer
set search_path = public
as $$
declare
    v_limit integer := greatest(coalesce(p_limit, 20), 1);
    v_stale integer := greatest(coalesce(p_stale_minutes, 10), 1);
begin
    return query
    with due as (
        select id
          from public.scheduled_actions
         where status = 'PENDING'
           and execute_at <= now()
         order by execute_at
         limit v_limit
           for update skip locked
    ),
    stale as (
        select id
          from public.scheduled_actions
         where status = 'PROCESSING'
           and locked_at is not null
           and locked_at < now() - make_interval(mins => v_stale)
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
           locked_at = now()
      from claimable as c
     where a.id = c.id
    returning a.*;
end;
$$;

revoke all on function public.claim_due_actions(integer, integer)
    from public, anon, authenticated;
grant execute on function public.claim_due_actions(integer, integer) to service_role;

-- The worker polls this constantly; both branches of the claim need an index.
create index if not exists scheduled_actions_due_claim_idx
    on public.scheduled_actions (execute_at)
    where status = 'PENDING';

create index if not exists scheduled_actions_stale_claim_idx
    on public.scheduled_actions (locked_at)
    where status = 'PROCESSING';
