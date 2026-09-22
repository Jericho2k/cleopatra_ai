-- Durable conversation supersession, outbound delivery sequences, and the
-- cross-process per-fan execution lease.
--
-- WHY
-- ---
-- Full Auto's interruption story used to be a process-local dictionary
-- (``services.suggestions._pending_auto_replies``) plus a 0.5 s poll inside a
-- sleeping coroutine. That is correct only while every event for one fan lands
-- in one Python process, and it pins a live coroutine for the whole of a
-- deliberate human-like pause. Neither property survives a second worker or a
-- thousand simultaneously active conversations.
--
-- Three additive objects replace it, and all three are DURABLE STATE rather
-- than process memory:
--
--   1. ``fans.conversation_generation`` — a monotonically increasing revision
--      bumped by a genuinely new fan message or a human creator reply, and by
--      nothing else. Every planned outbound sequence is bound to the exact
--      generation that produced it, and every externally visible send boundary
--      revalidates that binding.
--
--   2. ``outbound_sequences`` / ``outbound_sequence_parts`` — the multi-bubble
--      reply as durable rows with their own due times, so a composition pause
--      or an inter-bubble pause occupies a row in an indexed queue rather than
--      a scarce worker slot. A restart between bubble 1 and bubble 2 leaves
--      bubble 2 durably represented and still revalidated before it sends.
--
--   3. ``fan_execution_leases`` — the per-fan mutual exclusion that the old
--      in-process ``group_actions_by_fan`` grouping could only provide within
--      one worker. Two workers can now be told apart by the database.
--
-- Everything here is additive. A deployment that applies this file and keeps
-- running the previous code is unaffected: the column defaults to 0, the tables
-- stay empty, and the functions are never called.
--
-- Rolling-deploy order: apply this BEFORE deploying the code that uses it. New
-- code against an un-migrated database falls back (see db/outbound_queries.py
-- and services/conversation_generation.py) rather than failing a turn, but the
-- fallbacks are rollout affordances, not the intended path.

-- ---------------------------------------------------------------------------
-- 1. Conversation generation
-- ---------------------------------------------------------------------------

alter table public.fans
    add column if not exists conversation_generation bigint not null default 0;

comment on column public.fans.conversation_generation is
    'Monotonic conversation revision. Bumped by a newly inserted fan message or '
    'a human creator reply; never by an automated bubble, a retry, or a '
    'duplicate platform delivery. Planned outbound work is bound to the exact '
    'value that produced it.';

-- One atomic statement, so two concurrent inbound messages cannot both read the
-- same value and write the same successor.
create or replace function public.bump_conversation_generation(p_fan_id uuid)
returns bigint
language plpgsql
security definer
set search_path = public
as $$
declare
    v_next bigint;
begin
    update public.fans
       set conversation_generation = coalesce(conversation_generation, 0) + 1
     where id = p_fan_id
    returning conversation_generation into v_next;
    return v_next;
end;
$$;

revoke all on function public.bump_conversation_generation(uuid)
    from public, anon, authenticated;
grant execute on function public.bump_conversation_generation(uuid) to service_role;

-- ---------------------------------------------------------------------------
-- 2. Durable outbound sequences
-- ---------------------------------------------------------------------------

create table if not exists public.outbound_sequences (
    id uuid primary key default gen_random_uuid(),
    creator_id uuid not null references public.creators(id) on delete cascade,
    fan_id uuid not null references public.fans(id) on delete cascade,
    -- The generation this wording was produced from. Revalidated before every
    -- externally visible send.
    conversation_generation bigint not null,
    -- What caused the turn: a fan message id, or a scheduled action id. Unique
    -- per fan, so a retried action adopts the existing sequence instead of
    -- planning a second one.
    trigger_identity text not null,
    turn_id text not null default '',
    status text not null default 'PLANNED',
    cancel_reason text not null default '',
    planned_timing jsonb not null default '{}'::jsonb,
    -- The reply-provenance record this turn built, minus the per-part fields
    -- the sender fills in. Carried here so a bubble delivered minutes later by
    -- another process lands on the message row with the same provenance the
    -- first bubble did.
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    unique (fan_id, trigger_identity),
    constraint outbound_sequences_status_known check (
        status in ('PLANNED', 'SENDING', 'COMPLETED', 'SUPERSEDED', 'CANCELLED')
    )
);

create index if not exists outbound_sequences_fan_active_idx
    on public.outbound_sequences (fan_id, created_at desc)
    where status in ('PLANNED', 'SENDING');

comment on table public.outbound_sequences is
    'One planned multi-bubble creator reply, bound to the conversation '
    'generation that produced it. Not transaction authority and never a '
    'substitute for the messages table.';

create table if not exists public.outbound_sequence_parts (
    id uuid primary key default gen_random_uuid(),
    sequence_id uuid not null
        references public.outbound_sequences(id) on delete cascade,
    creator_id uuid not null references public.creators(id) on delete cascade,
    fan_id uuid not null references public.fans(id) on delete cascade,
    part_index integer not null check (part_index >= 0),
    body text not null,
    -- Planned wall-clock time for this bubble. The whole point: a nine second
    -- inter-bubble pause is a row in an indexed queue, not a sleeping task.
    due_at timestamptz not null,
    planned_delay_seconds double precision not null default 0,
    status text not null default 'PENDING',
    platform_message_id text null,
    message_id uuid null,
    sent_at timestamptz null,
    created_at timestamptz not null default now(),
    unique (sequence_id, part_index),
    constraint outbound_sequence_parts_status_known check (
        status in ('PENDING', 'SENT', 'SUPERSEDED', 'FAILED')
    )
);

create index if not exists outbound_sequence_parts_due_idx
    on public.outbound_sequence_parts (due_at)
    where status = 'PENDING';

create index if not exists outbound_sequence_parts_sequence_idx
    on public.outbound_sequence_parts (sequence_id, part_index);

comment on table public.outbound_sequence_parts is
    'One planned chat bubble with its own due time and delivery receipt. A '
    'restart between two bubbles leaves the later one durably represented and '
    'still subject to generation revalidation before it may send.';

-- ---------------------------------------------------------------------------
-- 3. Per-fan execution lease
-- ---------------------------------------------------------------------------

create table if not exists public.fan_execution_leases (
    fan_id uuid primary key references public.fans(id) on delete cascade,
    creator_id uuid not null references public.creators(id) on delete cascade,
    owner_token text not null,
    purpose text not null default '',
    acquired_at timestamptz not null default now(),
    expires_at timestamptz not null
);

create index if not exists fan_execution_leases_expiry_idx
    on public.fan_execution_leases (expires_at);

comment on table public.fan_execution_leases is
    'Cross-process mutual exclusion for conflicting outbound work on one fan. '
    'Held only while work that is actually due is executing, never across a '
    'future scheduled action. Expiry is the crash-recovery path.';

create or replace function public.acquire_fan_execution_lease(
    p_fan_id uuid,
    p_creator_id uuid,
    p_owner text,
    p_ttl_seconds integer default 180,
    p_purpose text default ''
)
returns boolean
language plpgsql
security definer
set search_path = public
as $$
declare
    v_ttl integer := greatest(coalesce(p_ttl_seconds, 180), 5);
    v_owner uuid;
begin
    insert into public.fan_execution_leases as l
        (fan_id, creator_id, owner_token, purpose, acquired_at, expires_at)
    values
        (p_fan_id, p_creator_id, p_owner, coalesce(p_purpose, ''), now(),
         now() + make_interval(secs => v_ttl))
    on conflict (fan_id) do update
        set owner_token = excluded.owner_token,
            creator_id  = excluded.creator_id,
            purpose     = excluded.purpose,
            acquired_at = now(),
            expires_at  = excluded.expires_at
        -- Take it only when nobody holds it, it has expired, or this caller
        -- already owns it (re-entrant renewal inside one bounded operation).
        where l.expires_at <= now()
           or l.owner_token = excluded.owner_token
    returning l.fan_id into v_owner;
    return v_owner is not null;
end;
$$;

create or replace function public.release_fan_execution_lease(
    p_fan_id uuid,
    p_owner text
)
returns boolean
language plpgsql
security definer
set search_path = public
as $$
declare
    v_released uuid;
begin
    delete from public.fan_execution_leases
     where fan_id = p_fan_id
       and owner_token = p_owner
    returning fan_id into v_released;
    return v_released is not null;
end;
$$;

revoke all on function public.acquire_fan_execution_lease(uuid, uuid, text, integer, text)
    from public, anon, authenticated;
grant execute on function public.acquire_fan_execution_lease(uuid, uuid, text, integer, text)
    to service_role;
revoke all on function public.release_fan_execution_lease(uuid, text)
    from public, anon, authenticated;
grant execute on function public.release_fan_execution_lease(uuid, text) to service_role;

-- ---------------------------------------------------------------------------
-- 4. Browser boundary
--
-- All three tables are internal delivery machinery. They carry creator_id
-- and/or fan_id, so the two discovery migrations would otherwise hand
-- `authenticated` a FOR ALL policy and a SELECT grant on unsent creator copy
-- and on the lease that arbitrates sending. Registering them keeps both
-- discovery loops away; RLS is enabled here with no policy, so the browser
-- roles get nothing and the service role (which bypasses RLS) is unaffected.
-- ---------------------------------------------------------------------------

do $$
begin
    if to_regclass('public.owner_only_tables') is not null then
        insert into public.owner_only_tables (table_name, reason)
        values
            ('outbound_sequences',
             'Unsent creator copy and its supersession state. Internal delivery '
             'machinery; the dashboard reads delivered messages, never the plan.'),
            ('outbound_sequence_parts',
             'Unsent creator copy with per-bubble due times. Internal delivery '
             'machinery; the dashboard reads delivered messages, never the plan.'),
            ('fan_execution_leases',
             'Cross-worker send arbitration. A browser client able to write this '
             'could stall or duplicate a conversation.')
        on conflict (table_name) do update set reason = excluded.reason;
    end if;
end
$$;

alter table public.outbound_sequences enable row level security;
alter table public.outbound_sequence_parts enable row level security;
alter table public.fan_execution_leases enable row level security;

revoke all on table public.outbound_sequences from anon, authenticated;
revoke all on table public.outbound_sequence_parts from anon, authenticated;
revoke all on table public.fan_execution_leases from anon, authenticated;
