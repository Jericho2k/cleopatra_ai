-- Conservative, durable re-engagement for otherwise idle Full Auto chats.
-- Apply after abandoned_offer_lifecycle_v1.sql and before deploying matching code.

alter table public.creator_commercial_policies
    add column if not exists inactivity_reengagement_enabled boolean not null default false,
    add column if not exists inactivity_reengagement_delay_hours integer not null default 48,
    add column if not exists inactivity_reengagement_cooldown_hours integer not null default 168,
    add column if not exists inactivity_reengagement_max_per_30_days integer not null default 2;

alter table public.fan_commercial_states
    add column if not exists last_inactivity_reengagement_at timestamptz null,
    add column if not exists inactivity_reengagement_window_started_at timestamptz null,
    add column if not exists inactivity_reengagement_count integer not null default 0;
