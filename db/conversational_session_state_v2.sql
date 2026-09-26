-- Conversational Core v2 (session-aware): selectable runtime id plus its own
-- versioned session document.
--
-- Additive and independent of Core v1's table: v1 keeps reading and writing
-- public.conversational_core_states, v2 reads and writes only
-- public.conversational_session_states, so switching a test fan between the two
-- never lets one runtime consume the other's interpretation.
--
-- The document is interpretation plus an application-rebuilt MIRROR of content
-- facts. Raw messages, creator configuration, commercial state and the
-- purchase/delivery ledgers remain authoritative. A planned trajectory beat is
-- never an offer, reservation, send or sale.
--
-- Rolling-deploy safety: apply BEFORE selecting conversational_v2 anywhere.
-- Old code ignores the table; new code only touches it on a v2-selected turn,
-- and a v2 turn against an unmigrated database fails visibly (it never falls
-- through to another runtime). Rollback is selecting another core; the rows
-- are kept so a later re-selection resumes where it stopped.

create table if not exists public.conversational_session_states (
    id uuid primary key default gen_random_uuid(),
    creator_id uuid not null references public.creators(id) on delete cascade,
    fan_id uuid not null references public.fans(id) on delete cascade,
    schema_version text not null
        check (schema_version = 'conversational_session_v2'),
    revision bigint not null default 0 check (revision >= 0),
    state jsonb not null default '{}'::jsonb check (jsonb_typeof(state) = 'object'),
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    unique (creator_id, fan_id)
);

create index if not exists conversational_session_states_fan_idx
    on public.conversational_session_states (fan_id, updated_at desc);

comment on table public.conversational_session_states is
    'Versioned session-aware interaction state for conversational_v2. Interpretation plus an application-rebuilt mirror of content facts; never transaction authority.';
comment on column public.conversational_session_states.state is
    'JSON session document without hidden reasoning: premise, goal, evidence-only constraints, provisional trajectory, append-only completed beats, and ledger-derived content lifecycle.';

-- Internal runtime state: registered owner-only so the tenant-isolation /
-- least-privilege discovery pair never grants the browser a policy on it. The
-- service role bypasses RLS and is unaffected.
do $$
begin
    if to_regclass('public.owner_only_tables') is not null then
        insert into public.owner_only_tables (table_name, reason)
        values (
            'conversational_session_states',
            'Core v2 session planning state (candidate content references, '
            'provisional trajectory). Internal runtime interpretation; the '
            'dashboard reads it through the owner-only simulator endpoint.'
        )
        on conflict (table_name) do update set reason = excluded.reason;
    end if;
end
$$;

alter table public.conversational_session_states enable row level security;

-- Widen runtime validation. conversational_working_state_v1.sql set the
-- previous list; this migration supersedes it without editing it.
alter table public.creators
    drop constraint if exists creators_conversation_core_known;
alter table public.creators
    add constraint creators_conversation_core_known
    check (conversation_core is null or conversation_core in (
        'legacy', 'semantic_v1', 'semantic_v2', 'conversational_v1',
        'conversational_v2'
    )) not valid;

alter table public.fans
    drop constraint if exists fans_conversation_core_known;
alter table public.fans
    add constraint fans_conversation_core_known
    check (conversation_core is null or conversation_core in (
        'legacy', 'semantic_v1', 'semantic_v2', 'conversational_v1',
        'conversational_v2'
    )) not valid;
