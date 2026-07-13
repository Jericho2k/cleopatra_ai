-- Cleopatra Model Lab v1
-- Safe to run after the commercial migrations.

create extension if not exists pgcrypto;

create table if not exists public.model_usage_events (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz not null default now(),
  creator_id uuid null,
  fan_id uuid null,
  feature text not null,
  provider text not null,
  model text not null,
  input_tokens integer not null default 0,
  output_tokens integer not null default 0,
  cache_read_tokens integer not null default 0,
  cache_write_tokens integer not null default 0,
  latency_ms integer null,
  retry_count integer not null default 0,
  success boolean not null default true,
  parse_valid boolean null,
  estimated_cost_usd numeric(14,8) not null default 0,
  error text null,
  raw_response_id text null,
  evaluation_run_id uuid null,
  scenario_id text null,
  metadata jsonb not null default '{}'::jsonb
);

create index if not exists model_usage_events_created_at_idx
  on public.model_usage_events (created_at desc);
create index if not exists model_usage_events_creator_idx
  on public.model_usage_events (creator_id, created_at desc);
create index if not exists model_usage_events_model_idx
  on public.model_usage_events (provider, model, created_at desc);
create index if not exists model_usage_events_feature_idx
  on public.model_usage_events (feature, created_at desc);

create table if not exists public.model_evaluation_runs (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz not null default now(),
  finished_at timestamptz null,
  status text not null default 'running',
  scenario_count integer not null default 0,
  model_count integer not null default 0,
  config jsonb not null default '{}'::jsonb,
  summary jsonb not null default '{}'::jsonb
);

create table if not exists public.model_evaluation_outputs (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz not null default now(),
  run_id uuid not null references public.model_evaluation_runs(id) on delete cascade,
  scenario_id text not null,
  candidate_name text not null,
  provider text not null,
  model text not null,
  skipped boolean not null default false,
  skip_reason text null,
  replies jsonb not null default '[]'::jsonb,
  automatic_checks jsonb not null default '{}'::jsonb,
  latency_ms integer null,
  estimated_cost_usd numeric(14,8) not null default 0,
  input_tokens integer not null default 0,
  output_tokens integer not null default 0,
  cache_read_tokens integer not null default 0,
  raw_text text null,
  error text null,
  unique (run_id, scenario_id, candidate_name)
);

create index if not exists model_evaluation_outputs_run_idx
  on public.model_evaluation_outputs (run_id, scenario_id);

-- These tables are internal-only. The service role bypasses RLS.
alter table public.model_usage_events enable row level security;
alter table public.model_evaluation_runs enable row level security;
alter table public.model_evaluation_outputs enable row level security;
