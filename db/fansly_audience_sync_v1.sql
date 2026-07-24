-- Fansly-native membership and spend context for existing chat profiles.

alter table public.fans
    add column if not exists is_follower boolean not null default false,
    add column if not exists subscription_status text not null default 'none',
    add column if not exists subscription_tier_id text null,
    add column if not exists subscription_tier_name text null,
    add column if not exists subscription_ends_at timestamptz null,
    add column if not exists fansly_lifetime_spend_cents bigint null,
    add column if not exists fansly_audience_synced_at timestamptz null;

alter table public.creators
    add column if not exists last_fansly_audience_sync_at timestamptz null;

create index if not exists fans_creator_membership_idx
    on public.fans (
        creator_id,
        subscription_status,
        is_follower,
        fansly_lifetime_spend_cents desc
    );
