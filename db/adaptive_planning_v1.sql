-- Cleopatra Phase 2: adaptive session planning + scoped pricing + vault boundaries.
-- Run after price_learning_v1.sql.

create extension if not exists pgcrypto;

create table if not exists public.price_learning_policy_scopes (
    scope_type text not null check (scope_type in ('AGENCY', 'CREATOR')),
    scope_id text not null,
    settings jsonb not null default '{}'::jsonb,
    updated_at timestamptz not null default now(),
    primary key (scope_type, scope_id)
);

create table if not exists public.creator_pricing_scope_memberships (
    creator_id uuid primary key references public.creators(id) on delete cascade,
    agency_scope_id text null,
    updated_at timestamptz not null default now()
);

alter table public.vault_sets
    add column if not exists base_price_cents integer,
    add column if not exists min_price_cents integer,
    add column if not exists max_price_cents integer,
    add column if not exists dynamic_pricing_enabled boolean not null default true;

update public.vault_sets
set
    base_price_cents = coalesce(base_price_cents, greatest(0, round(coalesce(suggested_price, 0) * 100)::integer)),
    min_price_cents = coalesce(min_price_cents, greatest(0, round(coalesce(suggested_price, 0) * 100)::integer)),
    max_price_cents = coalesce(max_price_cents, greatest(0, round(coalesce(suggested_price, 0) * 100)::integer))
where base_price_cents is null or min_price_cents is null or max_price_cents is null;

alter table public.vault_sets
    alter column base_price_cents set default 0,
    alter column min_price_cents set default 0,
    alter column max_price_cents set default 0;

create table if not exists public.fan_session_strategies (
    fan_id uuid primary key references public.fans(id) on delete cascade,
    creator_id uuid not null references public.creators(id) on delete cascade,
    goal text not null,
    phase text not null,
    next_action text not null,
    writer_goal text not null,
    writer_avoid text[] not null default '{}'::text[],
    must_ask_question boolean not null default false,
    must_not_ask_question boolean not null default false,
    max_messages integer null,
    approved_offer_ids text[] not null default '{}'::text[],
    approved_offer_prices_cents integer[] not null default '{}'::integer[],
    selected_offer_price_cents integer null,
    route_hint text not null default 'default',
    reason_codes text[] not null default '{}'::text[],
    planner_version integer not null default 1,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create table if not exists public.fan_session_strategy_audits (
    id uuid primary key default gen_random_uuid(),
    creator_id uuid not null references public.creators(id) on delete cascade,
    fan_id uuid not null references public.fans(id) on delete cascade,
    trigger_type text not null,
    goal text not null,
    phase text not null,
    next_action text not null,
    strategy jsonb not null,
    dedupe_key text not null unique,
    created_at timestamptz not null default now()
);

create index if not exists fan_session_strategy_creator_idx
    on public.fan_session_strategies (creator_id, goal, phase, updated_at desc);
create index if not exists fan_session_strategy_audit_fan_idx
    on public.fan_session_strategy_audits (fan_id, created_at desc);

alter table public.price_learning_policy_scopes enable row level security;
alter table public.creator_pricing_scope_memberships enable row level security;
alter table public.fan_session_strategies enable row level security;
alter table public.fan_session_strategy_audits enable row level security;

comment on table public.price_learning_policy_scopes is
'Agency defaults and creator overrides. Railway values remain emergency fallbacks.';
comment on column public.vault_sets.dynamic_pricing_enabled is
'When false, the approved set is offered only at base_price_cents.';
comment on table public.fan_session_strategies is
'Current deterministic next-best-action guidance. Commercial policy remains authoritative.';
