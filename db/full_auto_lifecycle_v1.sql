-- Full-auto lifecycle and follow-up policy controls.
-- Apply before deploying the matching backend commit.

alter table public.creator_commercial_policies
    add column if not exists ppv_recheck_minutes integer not null default 20,
    add column if not exists ppv_payment_window_hours integer not null default 2,
    add column if not exists abandoned_ppv_followup_enabled boolean not null default true,
    add column if not exists abandoned_ppv_followup_delay_hours integer not null default 18,
    add column if not exists post_session_followup_enabled boolean not null default true,
    add column if not exists post_session_followup_delay_hours integer not null default 18,
    add column if not exists followup_recent_activity_suppression_hours integer not null default 6;

alter table public.fan_commercial_states
    add column if not exists last_session_package_id text null,
    add column if not exists last_session_set_ids jsonb not null default '[]'::jsonb,
    add column if not exists last_session_experience text null,
    add column if not exists last_abandoned_ppv_at timestamptz null,
    add column if not exists last_abandoned_media_id text null,
    add column if not exists next_followup_at timestamptz null,
    add column if not exists next_followup_type text null,
    add column if not exists next_followup_payload jsonb not null default '{}'::jsonb,
    add column if not exists next_followup_dedupe_key text null,
    add column if not exists last_followup_at timestamptz null;

create index if not exists scheduled_actions_due_idx
    on public.scheduled_actions (status, execute_at);

create index if not exists scheduled_actions_fan_status_idx
    on public.scheduled_actions (fan_id, status);
