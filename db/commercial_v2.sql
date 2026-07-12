-- Cleopatra AI — Commercial v2
-- Idempotent Supabase/PostgreSQL migration.
--
-- Purpose:
--   * persist exact package options shown to each fan;
--   * distinguish selected package / current budget / future payday;
--   * keep one replaceable payday follow-up per fan;
--   * expose per-creator quick/full package price targets.
--
-- Safe to run more than once. Run this before enabling COMMERCIAL_LAYER_ENABLED.

create extension if not exists pgcrypto;

create table if not exists public.creator_commercial_policies (
    creator_id uuid primary key references public.creators(id) on delete cascade,
    sexting_mode text not null default 'HYBRID_TEASER',
    teaser_max_messages integer not null default 4,
    free_text_max_messages integer not null default 20,
    free_session_cooldown_hours integer not null default 24,
    media_always_paid boolean not null default true,
    payday_reengagement_enabled boolean not null default true,
    payday_send_hour_local integer not null default 18,
    timezone text not null default 'UTC',
    offer_two_packages boolean not null default true,
    quick_package_target_cents integer not null default 2500,
    full_package_target_cents integer not null default 6000,
    updated_at timestamptz not null default now()
);

alter table public.creator_commercial_policies
    add column if not exists quick_package_target_cents integer not null default 2500,
    add column if not exists full_package_target_cents integer not null default 6000;

create table if not exists public.fan_commercial_states (
    fan_id uuid primary key references public.fans(id) on delete cascade,
    creator_id uuid not null references public.creators(id) on delete cascade,
    status text not null default 'IDLE',
    desired_experience text,
    preferences_snapshot jsonb not null default '{}'::jsonb,
    confirmed_budget_cents integer,
    budget_source text,
    offered_packages jsonb not null default '[]'::jsonb,
    selected_package_id text,
    selected_package_set_id text,
    selected_package_label text,
    selected_package_price_cents integer,
    last_offer_at timestamptz,
    payday_raw text,
    payday_at timestamptz,
    payday_confidence double precision,
    last_declined_price_cents integer,
    teaser_messages_used integer not null default 0,
    free_session_started_at timestamptz,
    updated_at timestamptz not null default now()
);

alter table public.fan_commercial_states
    add column if not exists offered_packages jsonb not null default '[]'::jsonb,
    add column if not exists selected_package_id text,
    add column if not exists selected_package_set_id text,
    add column if not exists selected_package_label text,
    add column if not exists selected_package_price_cents integer,
    add column if not exists last_offer_at timestamptz;

create index if not exists fan_commercial_states_creator_idx
    on public.fan_commercial_states (creator_id);

create index if not exists fan_commercial_states_status_idx
    on public.fan_commercial_states (status);

create table if not exists public.scheduled_actions (
    id uuid primary key default gen_random_uuid(),
    creator_id uuid not null references public.creators(id) on delete cascade,
    fan_id uuid not null references public.fans(id) on delete cascade,
    action_type text not null,
    execute_at timestamptz not null,
    payload jsonb not null default '{}'::jsonb,
    status text not null default 'PENDING',
    attempts integer not null default 0,
    locked_at timestamptz,
    last_error text,
    dedupe_key text not null,
    created_at timestamptz not null default now()
);

create unique index if not exists scheduled_actions_dedupe_key_uidx
    on public.scheduled_actions (dedupe_key);

create index if not exists scheduled_actions_due_idx
    on public.scheduled_actions (status, execute_at);

-- Guardrails remain text-based so existing rows are not broken by a PostgreSQL
-- enum migration. The Python layer is the typed source of truth.
do $$
begin
    if not exists (
        select 1 from pg_constraint
        where conname = 'creator_commercial_policies_sexting_mode_check'
    ) then
        alter table public.creator_commercial_policies
            add constraint creator_commercial_policies_sexting_mode_check
            check (sexting_mode in ('PAID_ONLY', 'HYBRID_TEASER', 'FREE_TEXT_ALLOWED'))
            not valid;
    end if;

    if not exists (
        select 1 from pg_constraint
        where conname = 'creator_commercial_policies_send_hour_check'
    ) then
        alter table public.creator_commercial_policies
            add constraint creator_commercial_policies_send_hour_check
            check (payday_send_hour_local between 0 and 23)
            not valid;
    end if;
end $$;

comment on column public.fan_commercial_states.offered_packages is
    'Exact set-backed options most recently shown to this fan; required to resolve “first one” / “$28 one” safely.';

comment on column public.fan_commercial_states.selected_package_set_id is
    'Vault set selected by the fan. Stored as text so historical references survive package-system changes.';

comment on column public.scheduled_actions.dedupe_key is
    'One logical action key. Payday uses payday:<fan_id>, so a corrected date replaces the old one.';
