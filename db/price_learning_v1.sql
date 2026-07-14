-- Cleopatra Phase 2: deterministic, evidence-backed price learning.
-- Run after affordability_v1.sql and before PRICE_LEARNING_ENABLED=true.

create extension if not exists pgcrypto;

create table if not exists public.creator_price_learning_policies (
    creator_id uuid primary key references public.creators(id) on delete cascade,
    min_offer_cents integer not null default 500 check (min_offer_cents >= 0),
    max_offer_cents integer not null default 50000 check (max_offer_cents >= min_offer_cents),
    first_purchase_target_cents integer not null default 2500 check (first_purchase_target_cents >= 0),
    repeat_buyer_uplift_bps integer not null default 1000 check (repeat_buyer_uplift_bps between 0 and 5000),
    vip_uplift_bps integer not null default 1500 check (vip_uplift_bps between 0 and 7500),
    max_step_up_bps integer not null default 2500 check (max_step_up_bps between 0 and 10000),
    range_width_bps integer not null default 2000 check (range_width_bps between 0 and 7500),
    price_step_cents integer not null default 500 check (price_step_cents > 0),
    evidence_lookback_days integer not null default 365 check (evidence_lookback_days between 1 and 3650),
    updated_at timestamptz not null default now()
);

create table if not exists public.fan_price_learning_profiles (
    fan_id uuid primary key references public.fans(id) on delete cascade,
    creator_id uuid not null references public.creators(id) on delete cascade,
    mode text not null check (mode in ('NO_OFFER', 'DISCOVERY', 'RANGE', 'EXACT')),
    confidence text not null check (confidence in ('NONE', 'LOW', 'MEDIUM', 'HIGH')),
    lifecycle_stage text not null default 'PROSPECT',
    recommended_floor_cents integer null check (recommended_floor_cents is null or recommended_floor_cents >= 0),
    recommended_target_cents integer null check (recommended_target_cents is null or recommended_target_cents >= 0),
    recommended_ceiling_cents integer null check (recommended_ceiling_cents is null or recommended_ceiling_cents >= 0),
    anchor_cents integer null check (anchor_cents is null or anchor_cents >= 0),
    confirmed_purchase_count integer not null default 0 check (confirmed_purchase_count >= 0),
    positive_signal_count integer not null default 0 check (positive_signal_count >= 0),
    resistance_signal_count integer not null default 0 check (resistance_signal_count >= 0),
    evidence_score double precision not null default 0 check (evidence_score >= 0),
    evidence_summary jsonb not null default '{}'::jsonb,
    reason_codes text[] not null default '{}'::text[],
    state_version integer not null default 1,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create table if not exists public.fan_price_learning_audits (
    id uuid primary key default gen_random_uuid(),
    creator_id uuid not null references public.creators(id) on delete cascade,
    fan_id uuid not null references public.fans(id) on delete cascade,
    trigger_type text not null,
    mode text not null,
    confidence text not null,
    recommended_floor_cents integer null,
    recommended_target_cents integer null,
    recommended_ceiling_cents integer null,
    reason_codes text[] not null default '{}'::text[],
    evidence_summary jsonb not null default '{}'::jsonb,
    dedupe_key text not null unique,
    created_at timestamptz not null default now()
);

create index if not exists fan_price_profiles_creator_idx
    on public.fan_price_learning_profiles (creator_id, lifecycle_stage, confidence, updated_at desc);
create index if not exists fan_price_audits_fan_idx
    on public.fan_price_learning_audits (fan_id, created_at desc);

alter table public.creator_price_learning_policies enable row level security;
alter table public.fan_price_learning_profiles enable row level security;
alter table public.fan_price_learning_audits enable row level security;

comment on table public.fan_price_learning_profiles is
'Current deterministic price recommendation over approved packages. Never estimated wealth.';
comment on table public.fan_price_learning_audits is
'Immutable audit trail of materially changed price recommendations.';
