-- Cleopatra Phase 2 / Sprint 3: commercial affordability ledger.
-- Run once in Supabase SQL Editor before enabling AFFORDABILITY_ENABLED.

create extension if not exists pgcrypto;

create table if not exists public.fan_affordability_events (
    id uuid primary key default gen_random_uuid(),
    creator_id uuid not null references public.creators(id) on delete cascade,
    fan_id uuid not null references public.fans(id) on delete cascade,
    event_type text not null check (event_type in (
        'CURRENT_AMOUNT_STATED',
        'CURRENT_LIMIT_STATED',
        'OFFER_SELECTED',
        'COUNTEROFFER_STATED',
        'OFFER_DECLINED',
        'MONEY_UNAVAILABLE',
        'MONEY_AVAILABLE',
        'PAYDAY_MENTIONED',
        'PURCHASE_CONFIRMED'
    )),
    authority text not null check (authority in (
        'CHAT_EXPLICIT',
        'PAYMENT_CONFIRMED',
        'SYSTEM_DERIVED'
    )),
    amount_cents integer null check (amount_cents is null or amount_cents >= 0),
    raw_expression text null,
    confidence double precision not null default 1.0 check (confidence between 0 and 1),
    occurred_at timestamptz not null default now(),
    expires_at timestamptz null,
    source_message_id text null,
    source_ref text null,
    metadata jsonb not null default '{}'::jsonb,
    dedupe_key text not null unique,
    created_at timestamptz not null default now()
);

create table if not exists public.fan_affordability_states (
    fan_id uuid primary key references public.fans(id) on delete cascade,
    creator_id uuid not null references public.creators(id) on delete cascade,
    status text not null default 'UNKNOWN' check (status in (
        'UNKNOWN',
        'AVAILABLE_NOW',
        'LIMITED_NOW',
        'TEMPORARILY_UNAVAILABLE'
    )),
    current_available_cents integer null check (
        current_available_cents is null or current_available_cents >= 0
    ),
    current_limit_cents integer null check (
        current_limit_cents is null or current_limit_cents >= 0
    ),
    current_signal_expires_at timestamptz null,
    temporary_constraint boolean not null default false,
    constraint_until timestamptz null,
    payday_raw text null,
    payday_at timestamptz null,
    payday_confidence double precision null check (
        payday_confidence is null or payday_confidence between 0 and 1
    ),
    latest_offer_selected_cents integer null check (
        latest_offer_selected_cents is null or latest_offer_selected_cents >= 0
    ),
    latest_counteroffer_cents integer null check (
        latest_counteroffer_cents is null or latest_counteroffer_cents >= 0
    ),
    latest_rejected_price_cents integer null check (
        latest_rejected_price_cents is null or latest_rejected_price_cents >= 0
    ),
    last_confirmed_purchase_cents integer null check (
        last_confirmed_purchase_cents is null or last_confirmed_purchase_cents >= 0
    ),
    highest_confirmed_purchase_cents integer null check (
        highest_confirmed_purchase_cents is null or highest_confirmed_purchase_cents >= 0
    ),
    confirmed_purchase_count integer not null default 0 check (
        confirmed_purchase_count >= 0
    ),
    confirmed_purchase_total_cents integer not null default 0 check (
        confirmed_purchase_total_cents >= 0
    ),
    last_confirmed_purchase_at timestamptz null,
    reason_codes text[] not null default '{}'::text[],
    state_version integer not null default 1,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create index if not exists fan_affordability_events_fan_idx
    on public.fan_affordability_events (fan_id, occurred_at desc);
create index if not exists fan_affordability_events_creator_idx
    on public.fan_affordability_events (creator_id, event_type, occurred_at desc);
create index if not exists fan_affordability_states_creator_idx
    on public.fan_affordability_states (creator_id, status, updated_at desc);

alter table public.fan_affordability_events enable row level security;
alter table public.fan_affordability_states enable row level security;

comment on table public.fan_affordability_events is
    'Immutable evidence ledger for explicit money signals and confirmed purchases.';
comment on table public.fan_affordability_states is
    'Current affordability snapshot. Never an estimated wealth or permanent budget score.';
