-- Opt-in, durable callbacks for explicitly dated personal fan events.
-- Apply before enabling the matching controls in Settings.

alter table public.creator_commercial_policies
    add column if not exists personal_event_callbacks_enabled boolean not null default false,
    add column if not exists personal_event_callback_send_hour_local integer not null default 18,
    add column if not exists personal_event_callback_max_per_30_days integer not null default 3;

alter table public.creator_commercial_policies
    drop constraint if exists personal_event_callback_send_hour_local_range;

alter table public.creator_commercial_policies
    add constraint personal_event_callback_send_hour_local_range
    check (personal_event_callback_send_hour_local between 0 and 23);

alter table public.creator_commercial_policies
    drop constraint if exists personal_event_callback_max_per_30_days_range;

alter table public.creator_commercial_policies
    add constraint personal_event_callback_max_per_30_days_range
    check (personal_event_callback_max_per_30_days between 1 and 10);
