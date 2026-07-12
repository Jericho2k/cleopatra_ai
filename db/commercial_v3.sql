-- Cleopatra AI — Auto-mode readiness / Commercial v3
-- Run AFTER Commercial v2. Idempotent.

alter table public.creator_commercial_policies
    add column if not exists session_min_steps integer not null default 2,
    add column if not exists session_max_steps integer not null default 4,
    add column if not exists post_purchase_cooldown_messages integer not null default 2,
    add column if not exists require_purchase_before_next_step boolean not null default true;

alter table public.fan_commercial_states
    add column if not exists selected_package_set_ids jsonb not null default '[]'::jsonb,
    add column if not exists free_session_ended_at timestamptz,
    add column if not exists last_session_completed_at timestamptz,
    add column if not exists last_session_revenue_cents integer not null default 0;

-- Keep invalid settings from reaching the policy engine. NOT VALID avoids
-- blocking the migration if old manually-edited rows need correction first.
do $$
begin
    if not exists (
        select 1 from pg_constraint
        where conname = 'creator_commercial_policies_session_steps_check'
    ) then
        alter table public.creator_commercial_policies
            add constraint creator_commercial_policies_session_steps_check
            check (
                session_min_steps between 1 and 8
                and session_max_steps between session_min_steps and 8
            ) not valid;
    end if;

    if not exists (
        select 1 from pg_constraint
        where conname = 'creator_commercial_policies_package_prices_check'
    ) then
        alter table public.creator_commercial_policies
            add constraint creator_commercial_policies_package_prices_check
            check (
                quick_package_target_cents > 0
                and full_package_target_cents > 0
            ) not valid;
    end if;

    if not exists (
        select 1 from pg_constraint
        where conname = 'creator_commercial_policies_cooldowns_check'
    ) then
        alter table public.creator_commercial_policies
            add constraint creator_commercial_policies_cooldowns_check
            check (
                free_session_cooldown_hours >= 0
                and post_purchase_cooldown_messages >= 0
            ) not valid;
    end if;
end $$;

comment on column public.creator_commercial_policies.session_min_steps is
    'Minimum approved coherent vault sets used in a paid-session package when enough sets exist.';
comment on column public.creator_commercial_policies.require_purchase_before_next_step is
    'When true, sending a PPV creates a purchase gate and the planner advances only after a confirmed buy.';
comment on column public.fan_commercial_states.selected_package_set_ids is
    'Ordered approved vault-set IDs backing the exact package selected by the fan.';
comment on column public.fan_commercial_states.free_session_ended_at is
    'Starts the configured free-session cooldown; allowance resets only after it expires.';
