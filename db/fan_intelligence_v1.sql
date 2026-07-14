-- Cleopatra Phase 2 / Sprint 1: passive learned fan intelligence
-- Run once in Supabase SQL Editor before enabling FAN_INTELLIGENCE_ENABLED.

create extension if not exists pgcrypto;

create table if not exists public.fan_fact_observations (
    id uuid primary key default gen_random_uuid(),
    creator_id uuid not null references public.creators(id) on delete cascade,
    fan_id uuid not null references public.fans(id) on delete cascade,
    source_message_id text null,
    category text not null check (category in (
        'identity', 'availability', 'preference', 'boundary', 'commercial', 'behavior'
    )),
    fact_key text not null,
    observed_value_json jsonb not null,
    normalized_value text not null,
    confidence numeric(5,4) not null check (confidence >= 0 and confidence <= 1),
    certainty text not null check (certainty in ('explicit', 'strong_inference')),
    source_type text not null default 'fan_message',
    evidence_text text not null,
    extraction_provider text not null,
    extraction_model text not null,
    dedupe_key text not null unique,
    created_at timestamptz not null default now()
);

create table if not exists public.fan_facts (
    id uuid primary key default gen_random_uuid(),
    creator_id uuid not null references public.creators(id) on delete cascade,
    fan_id uuid not null references public.fans(id) on delete cascade,
    category text not null check (category in (
        'identity', 'availability', 'preference', 'boundary', 'commercial', 'behavior'
    )),
    fact_key text not null,
    value_json jsonb not null,
    normalized_value text not null,
    confidence numeric(5,4) not null check (confidence >= 0 and confidence <= 1),
    status text not null check (status in ('inferred', 'explicit', 'confirmed', 'contradicted')),
    source_type text not null default 'fan_message',
    first_observed_at timestamptz not null default now(),
    last_observed_at timestamptz not null default now(),
    confirmation_count integer not null default 1 check (confirmation_count >= 1),
    first_evidence_message_id text null,
    last_evidence_message_id text null,
    first_evidence_text text null,
    last_evidence_text text null,
    is_active boolean not null default true,
    contradicted_at timestamptz null,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    unique (fan_id, fact_key, normalized_value)
);

create index if not exists fan_fact_observations_fan_created_idx
    on public.fan_fact_observations (fan_id, created_at desc);
create index if not exists fan_fact_observations_key_idx
    on public.fan_fact_observations (fan_id, fact_key, created_at desc);
create index if not exists fan_facts_active_idx
    on public.fan_facts (fan_id, is_active, category, fact_key);
create index if not exists fan_facts_creator_idx
    on public.fan_facts (creator_id, updated_at desc);

alter table public.fan_fact_observations enable row level security;
alter table public.fan_facts enable row level security;

comment on table public.fan_fact_observations is
    'Immutable evidence proposed by the passive fan-intelligence extractor.';
comment on table public.fan_facts is
    'Current conservatively merged fan knowledge. Payment events remain authoritative for purchases.';
