-- Cleopatra Phase 2 / Sprint 2: deterministic buyer lifecycle.
-- Run once in Supabase SQL Editor before enabling FAN_LIFECYCLE_ENABLED.

create extension if not exists pgcrypto;

create table if not exists public.creator_lifecycle_policies (
    creator_id uuid primary key references public.creators(id) on delete cascade,
    vip_spend_cents integer not null default 50000 check (vip_spend_cents >= 0),
    vip_purchase_count integer not null default 5 check (vip_purchase_count >= 1),
    repeat_buyer_purchase_count integer not null default 2 check (repeat_buyer_purchase_count >= 2),
    first_purchase_intent_ttl_hours integer not null default 72
        check (first_purchase_intent_ttl_hours between 1 and 720),
    updated_at timestamptz not null default now()
);

create table if not exists public.fan_lifecycle_states (
    fan_id uuid primary key references public.fans(id) on delete cascade,
    creator_id uuid not null references public.creators(id) on delete cascade,
    stage text not null check (stage in (
        'PROSPECT',
        'FIRST_PURCHASE_PROSPECT',
        'FIRST_TIME_BUYER',
        'REPEAT_BUYER',
        'VIP'
    )),
    purchase_count integer not null default 0 check (purchase_count >= 0),
    purchase_revenue_cents integer not null default 0 check (purchase_revenue_cents >= 0),
    total_spent_cents integer not null default 0 check (total_spent_cents >= 0),
    first_purchase_at timestamptz null,
    last_purchase_at timestamptz null,
    intent_expires_at timestamptz null,
    flags jsonb not null default '{}'::jsonb,
    reason_codes text[] not null default '{}'::text[],
    state_version integer not null default 1,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create table if not exists public.fan_lifecycle_transitions (
    id uuid primary key default gen_random_uuid(),
    creator_id uuid not null references public.creators(id) on delete cascade,
    fan_id uuid not null references public.fans(id) on delete cascade,
    from_stage text null,
    to_stage text not null,
    trigger_type text not null,
    reason_codes text[] not null default '{}'::text[],
    purchase_count integer not null default 0,
    total_spent_cents integer not null default 0,
    metadata jsonb not null default '{}'::jsonb,
    dedupe_key text not null unique,
    created_at timestamptz not null default now()
);

create index if not exists fan_lifecycle_creator_stage_idx
    on public.fan_lifecycle_states (creator_id, stage, updated_at desc);
create index if not exists fan_lifecycle_transition_fan_idx
    on public.fan_lifecycle_transitions (fan_id, created_at desc);

alter table public.creator_lifecycle_policies enable row level security;
alter table public.fan_lifecycle_states enable row level security;
alter table public.fan_lifecycle_transitions enable row level security;

comment on table public.fan_lifecycle_states is
    'Current deterministic buyer lifecycle, suitable for routing, prompts, and future UI.';
comment on table public.fan_lifecycle_transitions is
    'Immutable buyer-stage transition audit history.';
