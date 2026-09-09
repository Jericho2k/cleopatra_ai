-- CI BASELINE SCHEMA — NOT THE PRODUCTION SCHEMA DUMP.
--
-- Audit reference: DB-000.
--
-- READ THIS BEFORE USING THIS FILE FOR ANYTHING.
--
-- The authoritative definitions of the objects below live only in the live
-- Supabase project. They were created out of band, so their real primary keys,
-- foreign keys, unique constraints, cascade behaviour, nullability, defaults,
-- and indexes are NOT knowable from this repository.
--
-- This file is a TEST FIXTURE. It reproduces enough of the shape for a fresh
-- PostgreSQL to accept every db/*.sql migration in order and for the schema
-- tests to run, and nothing more. It is deliberately NOT named
-- 000_base_schema.sql, because it must never be mistaken for — or applied as —
-- the real base schema.
--
-- NEVER APPLY THIS TO A REAL DATABASE.
--
-- To produce the authoritative baseline, run scripts/dump_base_schema.sh
-- against the production database with read credentials. That writes
-- db/000_base_schema.sql, and db/MIGRATIONS.md describes how CI switches over
-- to it. Until that dump exists, every index and constraint question this audit
-- had to mark UNKNOWN stays UNKNOWN — see db/MIGRATIONS.md § Open questions.

create extension if not exists pgcrypto;

-- --------------------------------------------------------------------------
-- Tenancy
-- --------------------------------------------------------------------------

create table public.chatter_creators (
    id uuid primary key default gen_random_uuid(),
    chatter_id uuid not null,
    creator_id uuid not null
);

create table public.creators (
    id uuid primary key default gen_random_uuid(),
    name text null,
    platform text null,
    platform_username text null,
    fansly_account_id text null,
    apifansly_account_id text null,
    auto_mode boolean not null default false,
    auto_audience_policy jsonb null,
    persona jsonb null,
    created_at timestamptz not null default now()
);

create table public.fans (
    id uuid primary key default gen_random_uuid(),
    creator_id uuid not null references public.creators(id) on delete cascade,
    platform_fan_id text null,
    display_name text null,
    avatar_url text null,
    fansly_group_id text null,
    auto_mode boolean null,
    total_spent integer not null default 0,
    spend_tier text not null default 'cold',
    needs_human_review boolean not null default false,
    review_reason text null,
    sale_paused_at timestamptz null,
    pending_ppv_check jsonb null,
    pending_tip jsonb null,
    sales_log jsonb null,
    ai_summary jsonb null,
    active_session jsonb null,
    notes text null,
    member_note text null,
    model_note text null,
    last_active timestamptz null,
    created_at timestamptz not null default now()
);

-- --------------------------------------------------------------------------
-- Conversation
-- --------------------------------------------------------------------------

create table public.messages (
    id uuid primary key default gen_random_uuid(),
    fan_id uuid not null references public.fans(id) on delete cascade,
    creator_id uuid not null references public.creators(id) on delete cascade,
    role text not null,
    content text not null default '',
    was_ai_suggested boolean not null default false,
    fansly_message_id text null,
    media_context jsonb null,
    sent_at timestamptz not null default now()
);

create table public.suggestions (
    id uuid primary key default gen_random_uuid(),
    fan_id uuid not null references public.fans(id) on delete cascade,
    creator_id uuid not null references public.creators(id) on delete cascade,
    suggestions jsonb not null default '[]'::jsonb,
    stage text null,
    created_at timestamptz not null default now()
);

-- --------------------------------------------------------------------------
-- Lists
-- --------------------------------------------------------------------------

create table public.fan_lists (
    id uuid primary key default gen_random_uuid(),
    creator_id uuid not null references public.creators(id) on delete cascade,
    name text not null,
    color text null,
    exclude_from_auto boolean not null default false,
    created_at timestamptz not null default now()
);

create table public.fan_list_members (
    list_id uuid not null references public.fan_lists(id) on delete cascade,
    fan_id uuid not null references public.fans(id) on delete cascade,
    primary key (list_id, fan_id)
);

-- --------------------------------------------------------------------------
-- Commercial state and the durable action queue
-- --------------------------------------------------------------------------

create table public.creator_commercial_policies (
    creator_id uuid primary key references public.creators(id) on delete cascade,
    updated_at timestamptz not null default now()
);

-- payment_pending_status_v1.sql runs `alter type public.fan_commercial_status
-- add value ...`, so the type must pre-exist. The REAL value set is UNKNOWN —
-- it lives in Supabase. These are the values evidenced in application code;
-- treat any absence here as a gap in this fixture, not in production.
create type public.fan_commercial_status as enum ('IDLE');

create table public.fan_commercial_states (
    fan_id uuid primary key references public.fans(id) on delete cascade,
    creator_id uuid not null references public.creators(id) on delete cascade,
    status public.fan_commercial_status not null default 'IDLE',
    state jsonb not null default '{}'::jsonb,
    updated_at timestamptz not null default now()
);

create table public.scheduled_actions (
    id uuid primary key default gen_random_uuid(),
    creator_id uuid not null references public.creators(id) on delete cascade,
    fan_id uuid not null references public.fans(id) on delete cascade,
    action_type text not null,
    execute_at timestamptz not null default now(),
    status text not null default 'PENDING',
    attempts integer not null default 0,
    last_error text null,
    locked_at timestamptz null,
    dedupe_key text null unique,
    payload jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now()
);

-- --------------------------------------------------------------------------
-- Vault
-- --------------------------------------------------------------------------

create table public.vault_sets (
    id uuid primary key default gen_random_uuid(),
    creator_id uuid not null references public.creators(id) on delete cascade,
    name text null,
    -- The pre-cents price column adaptive_planning_v1 backfills base/min/max
    -- from. Numeric dollars, per that migration's round(... * 100) conversion.
    suggested_price numeric null,
    created_at timestamptz not null default now()
);

create table public.creator_vault_media (
    id uuid primary key default gen_random_uuid(),
    creator_id uuid not null references public.creators(id) on delete cascade,
    media_id text not null,
    created_at timestamptz not null default now()
);

create table public.vault_recategorization_usage (
    creator_id uuid primary key references public.creators(id) on delete cascade,
    updated_at timestamptz not null default now()
);

-- --------------------------------------------------------------------------
-- Miscellaneous creator-owned tables the application reads
-- --------------------------------------------------------------------------

create table public.ppv_offers (
    id uuid primary key default gen_random_uuid(),
    creator_id uuid not null references public.creators(id) on delete cascade,
    price integer not null default 0,
    created_at timestamptz not null default now()
);

create table public.scripts (
    id uuid primary key default gen_random_uuid(),
    creator_id uuid not null references public.creators(id) on delete cascade,
    body text null
);

create table public.blocked_words (
    id uuid primary key default gen_random_uuid(),
    creator_id uuid not null references public.creators(id) on delete cascade,
    word text not null
);

create table public.reengagement_settings (
    creator_id uuid primary key references public.creators(id) on delete cascade,
    enabled boolean not null default false
);

create table public.reengagement_log (
    id uuid primary key default gen_random_uuid(),
    creator_id uuid not null references public.creators(id) on delete cascade,
    fan_id uuid not null references public.fans(id) on delete cascade,
    sent_at timestamptz not null default now()
);

create table public.fansly_sessions (
    id uuid primary key default gen_random_uuid(),
    creator_id uuid not null references public.creators(id) on delete cascade,
    encrypted_payload text null
);

create table public.message_embeddings (
    id uuid primary key default gen_random_uuid(),
    creator_id uuid not null references public.creators(id) on delete cascade,
    message_id uuid null references public.messages(id) on delete cascade
);

-- --------------------------------------------------------------------------
-- Placeholder routines the migrations replace.
--
-- The real bodies live in Supabase. These exist only so a migration that does
-- `create or replace function` or references the name applies cleanly.
-- --------------------------------------------------------------------------

create or replace function public.attach_pending_ppv() returns void
language sql as $$ select null::void $$;

create or replace function public.claim_chat_reconciliation() returns void
language sql as $$ select null::void $$;

create or replace function public.claim_vault_recategorization() returns void
language sql as $$ select null::void $$;

-- --------------------------------------------------------------------------
-- fan_conversation_summaries (SEC-003)
--
-- The dashboard reads this view directly from the browser, so it must respect
-- the caller's RLS rather than the definer's privileges. security_invoker is
-- declared here so the property is version-controlled and cannot silently
-- disappear; db/summaries_security_invoker_v1.sql enforces the same thing on
-- the live database.
--
-- The column list is the one the dashboard consumes (see rowToFan in
-- cleopatra-dashboard/app/page.tsx). The production view may select more; this
-- definition is a CI fixture, not a replacement for it.
-- --------------------------------------------------------------------------

create view public.fan_conversation_summaries
with (security_invoker = on) as
select
    f.id,
    f.creator_id,
    f.display_name,
    f.platform_fan_id,
    f.auto_mode,
    f.total_spent,
    f.spend_tier,
    f.needs_human_review,
    f.notes,
    f.member_note,
    f.model_note,
    f.ai_summary,
    f.last_active,
    latest.content as last_message,
    latest.sent_at as last_message_time
from public.fans f
left join lateral (
    select m.content, m.sent_at
      from public.messages m
     where m.fan_id = f.id
     order by m.sent_at desc
     limit 1
) latest on true;
