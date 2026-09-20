-- Conversational Core v1: selectable runtime plus versioned interaction state.
--
-- The state is an interpretation of the current interaction. Raw messages,
-- creator configuration, and transaction/delivery tables remain authoritative.
-- A creator/fan pair has exactly one current versioned document; superseded
-- beliefs remain inside that document with their provenance links.

create table if not exists public.conversational_core_states (
    id uuid primary key default gen_random_uuid(),
    creator_id uuid not null references public.creators(id) on delete cascade,
    fan_id uuid not null references public.fans(id) on delete cascade,
    schema_version text not null check (schema_version = 'conversational_core_v1'),
    revision bigint not null default 0 check (revision >= 0),
    state jsonb not null default '{}'::jsonb check (jsonb_typeof(state) = 'object'),
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    unique (creator_id, fan_id)
);

create index if not exists conversational_core_states_fan_idx
    on public.conversational_core_states (fan_id, updated_at desc);

alter table public.conversational_core_states enable row level security;

comment on table public.conversational_core_states is
    'Versioned evidence-grounded interaction state for conversational_v1. Not raw history and never transaction authority.';
comment on column public.conversational_core_states.state is
    'JSON working state without hidden reasoning. Established elements carry immutable epistemic types and evidence refs.';

-- Keep every deployed/runtime identifier valid. conversation_core_v1.sql
-- predates semantic_v2; this migration is the compatibility point that brings
-- database validation in line with the application registry.
alter table public.creators
    drop constraint if exists creators_conversation_core_known;
alter table public.creators
    add constraint creators_conversation_core_known
    check (conversation_core is null or conversation_core in (
        'legacy', 'semantic_v1', 'semantic_v2', 'conversational_v1'
    )) not valid;

alter table public.fans
    drop constraint if exists fans_conversation_core_known;
alter table public.fans
    add constraint fans_conversation_core_known
    check (conversation_core is null or conversation_core in (
        'legacy', 'semantic_v1', 'semantic_v2', 'conversational_v1'
    )) not valid;
