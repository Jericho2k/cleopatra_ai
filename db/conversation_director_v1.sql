-- Cleopatra Phase 2: persistent multi-turn conversation director.
-- Run after adaptive_planning_v1.sql.

create extension if not exists pgcrypto;

create table if not exists public.fan_conversation_directors (
    fan_id uuid primary key references public.fans(id) on delete cascade,
    creator_id uuid not null references public.creators(id) on delete cascade,
    phase text not null,
    previous_phase text null,
    action text not null,
    fan_turn_count integer not null default 0,
    creator_turn_count integer not null default 0,
    turns_in_phase integer not null default 1,
    same_action_streak integer not null default 1,
    recent_actions text[] not null default '{}'::text[],
    engagement_score integer not null default 0,
    qualification_complete boolean not null default false,
    offer_eligible boolean not null default false,
    question_due boolean not null default false,
    must_not_ask_question boolean not null default false,
    transition_reason text not null,
    director_version integer not null default 1,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create table if not exists public.fan_conversation_director_audits (
    id uuid primary key default gen_random_uuid(),
    creator_id uuid not null references public.creators(id) on delete cascade,
    fan_id uuid not null references public.fans(id) on delete cascade,
    trigger_type text not null,
    phase text not null,
    action text not null,
    state jsonb not null,
    dedupe_key text not null unique,
    created_at timestamptz not null default now()
);

create index if not exists fan_conversation_director_creator_idx
    on public.fan_conversation_directors
    (creator_id, phase, action, updated_at desc);

create index if not exists fan_conversation_director_audit_fan_idx
    on public.fan_conversation_director_audits
    (fan_id, created_at desc);

alter table public.fan_conversation_directors enable row level security;
alter table public.fan_conversation_director_audits enable row level security;

comment on table public.fan_conversation_directors is
    'Current persistent conversation progression. Commercial policy remains authoritative.';

comment on table public.fan_conversation_director_audits is
    'Immutable phase/action transition history for debugging and future UI.';
