-- One Fansly Lists reconciliation per creator at a time.
--
-- Audit reference: Sprint 3 remaining work. Additive and idempotent.
--
-- The problem
-- -----------
-- sync_fansly_lists had no guard at all. The staleness check it did have
-- (_fansly_lists_sync_due, reading last_fansly_lists_sync_at) is a read
-- followed by an act, which answers "should this run" and not "am I the one
-- running it". Two operators pressing Refresh, or a manual refresh landing
-- while the chat-sync pass fires the automatic one, produced two concurrent
-- reconciliations of the same membership: both diff the same remote lists
-- against the same local mirror, both add and remove rows, and each sees the
-- other's half-applied state.
--
-- The claim
-- ---------
-- The same shape as claim_chat_reconciliation, for the same reason: a
-- conditional UPDATE ... RETURNING is atomic, so exactly one caller can observe
-- the transition from unclaimed to claimed. No advisory lock, nothing to leak.
--
-- Per creator, deliberately. Different creators hit different Fansly accounts
-- and have no shared state to corrupt; serialising them globally would make one
-- agency's large refresh block everyone else's for no correctness benefit.
--
-- Crash safety
-- ------------
-- A claim is a timestamp, not a lock, so a process that dies holding one cannot
-- wedge the creator forever: the claim is reclaimable once it is older than the
-- stale window. That is the same crash-recovery shape the scheduled-action
-- worker uses for PROCESSING rows, and it is why the window is a parameter
-- rather than a constant — a legitimately long sync must not be reclaimed
-- underneath itself.

alter table public.creators
    -- When the in-flight (or most recent) list reconciliation was claimed.
    -- Distinct from last_fansly_lists_sync_at, which records when one last
    -- COMPLETED and drives staleness. Conflating the two would make a claim
    -- look like a successful sync and suppress the next real one.
    add column if not exists fansly_lists_sync_claimed_at timestamptz null;

comment on column public.creators.fansly_lists_sync_claimed_at is
    'Single-flight claim for Fansly list reconciliation. Set when a run starts, '
    'cleared when it ends, reclaimable after the stale window so a crashed '
    'process cannot wedge the creator.';

-- ---------------------------------------------------------------------------
-- Win the right to reconcile this creator's lists.
--
-- Returns true to exactly one caller. A second concurrent caller gets false and
-- is expected to report "already syncing" rather than starting a second run.
-- ---------------------------------------------------------------------------
create or replace function public.claim_fansly_lists_sync(
    p_creator_id uuid,
    p_stale_minutes integer default 15
)
returns boolean
language plpgsql
security definer
set search_path = public
as $$
declare
    v_claimed uuid;
begin
    update public.creators
       set fansly_lists_sync_claimed_at = now()
     where id = p_creator_id
       and (
           fansly_lists_sync_claimed_at is null
           or fansly_lists_sync_claimed_at
              <= now() - make_interval(mins => greatest(coalesce(p_stale_minutes, 15), 1))
       )
    returning id into v_claimed;
    return v_claimed is not null;
end;
$$;

-- ---------------------------------------------------------------------------
-- Release the claim. Called on success AND on failure: a failed sync must not
-- keep the creator claimed until the stale window expires, or a retry after a
-- transient API error would be refused for a quarter of an hour.
-- ---------------------------------------------------------------------------
create or replace function public.release_fansly_lists_sync(
    p_creator_id uuid
)
returns void
language sql
security definer
set search_path = public
as $$
    update public.creators
       set fansly_lists_sync_claimed_at = null
     where id = p_creator_id;
$$;

revoke all on function public.claim_fansly_lists_sync(uuid, integer)
    from public, anon, authenticated;
revoke all on function public.release_fansly_lists_sync(uuid)
    from public, anon, authenticated;
grant execute on function public.claim_fansly_lists_sync(uuid, integer) to service_role;
grant execute on function public.release_fansly_lists_sync(uuid) to service_role;
