-- Durable expiry and follow-up controls for offers that were presented but
-- never selected. Apply after full_auto_lifecycle_v1.sql.

alter table public.creator_commercial_policies
    add column if not exists pending_offer_expiry_hours integer not null default 24,
    add column if not exists abandoned_offer_followup_enabled boolean not null default true,
    add column if not exists abandoned_offer_followup_delay_hours integer not null default 18;
