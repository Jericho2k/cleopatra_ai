-- One platform purchase, processed once.
--
-- Audit reference: REL-003. Additive and idempotent; safe to re-run.
--
-- The problem
-- -----------
-- The ppv.purchased webhook decided whether it had already handled an order by
-- scanning fans.sales_log — a jsonb array — in Python, and then writing. That
-- is read-then-write across a network boundary with no lock. Two concurrent
-- deliveries of the same order both read a log that did not contain it, both
-- concluded "new", and both applied the downstream effects: spend, lifecycle
-- transition, PPV purchase, follow-up cancellation.
--
-- Sprint 1 fixed message identity the same way and for the same reason
-- (db/message_platform_identity_v1.sql). This is that argument applied to
-- commercial identity, where a duplicate is worse: a double-counted sale
-- corrupts spend tiers, affordability and price learning.
--
-- The fix is to let PostgreSQL decide who is first. An INSERT with a unique
-- index cannot be lost to a race: exactly one of two concurrent transactions
-- commits the row, the other gets a conflict. No advisory lock, no SELECT ...
-- FOR UPDATE, no application-level coordination.
--
-- Identity scope
-- --------------
-- (creator_id, platform_order_id), not platform_order_id alone.
--
-- Identical reasoning to message_platform_identity_v1: the two choices fail
-- asymmetrically. If Fansly order ids really are globally unique, the composite
-- key still catches every race, because two deliveries of ONE order always
-- carry the same creator. If they turn out to be unique only per account, a
-- global key would silently reject a second creator's legitimately distinct
-- order — a lost sale, and a lost sale that looks like correct deduplication.
-- Composite is correct under both, and the API contract does not settle which
-- one is true.
--
-- Why not extend ppv_deliveries
-- -----------------------------
-- ppv_deliveries is keyed by `reference`, an identifier WE mint when an offer
-- is claimed. It answers "which offer did we send". This table answers "which
-- platform order have we ingested", which is a different question with a
-- different lifetime: an order can arrive with no matching delivery (an
-- operator-sent offer, a purchase from a post), and a delivery can exist with
-- no order (never bought). Overloading one on the other would make both
-- ambiguous. This table stays deliberately thin and records only identity.
--
-- sales_log is untouched
-- ----------------------
-- It remains the fan-facing purchase history the dashboard renders, and this
-- migration neither rewrites nor deletes any of it. It simply stops being the
-- concurrency authority, which it was never able to be.

create table if not exists public.platform_purchase_events (
    id uuid primary key default gen_random_uuid(),
    creator_id uuid not null references public.creators(id) on delete cascade,
    -- The platform's own order identifier. Text: a foreign identifier whose
    -- shape is not ours to constrain.
    platform_order_id text not null,
    -- Resolved at claim time. Nullable and ON DELETE SET NULL so that removing
    -- a fan never destroys the record that an order was already ingested —
    -- losing that would make a redelivery look new.
    fan_id uuid null references public.fans(id) on delete set null,
    event_type text not null default 'ppv.purchased',
    account_media_id text null,
    price_cents integer null,
    status text not null default 'claimed'
        check (status in ('claimed', 'processed')),
    claimed_at timestamptz not null default now(),
    processed_at timestamptz null,
    metadata jsonb not null default '{}'::jsonb
);

-- The whole point of the table. Partial on nothing — platform_order_id is NOT
-- NULL — so every ingested order is covered.
create unique index if not exists platform_purchase_events_identity_key
    on public.platform_purchase_events (creator_id, platform_order_id);

-- Operator queries: what did this creator ingest recently, and is anything
-- stuck in 'claimed' because a handler died mid-flight.
create index if not exists platform_purchase_events_creator_claimed_idx
    on public.platform_purchase_events (creator_id, claimed_at desc);

create index if not exists platform_purchase_events_status_claimed_idx
    on public.platform_purchase_events (status, claimed_at)
    where status = 'claimed';

alter table public.platform_purchase_events enable row level security;

comment on table public.platform_purchase_events is
    'REL-003: relational identity for platform purchase events. One row per '
    '(creator_id, platform_order_id); the unique index is what makes two '
    'concurrent webhook deliveries of one order resolve to a single effective '
    'purchase. Not a replacement for fans.sales_log, which remains history.';

-- ---------------------------------------------------------------------------
-- Claim the right to process one platform order.
--
-- Returns 'claimed' to exactly one caller and 'duplicate' to every other,
-- including concurrent ones. The INSERT ... ON CONFLICT DO NOTHING is the
-- decision; `found` after it is how we learn which side we were on.
--
-- security definer + service_role only: this is backend ingestion, never
-- something a browser session may assert.
-- ---------------------------------------------------------------------------
create or replace function public.claim_platform_purchase(
    p_creator_id uuid,
    p_platform_order_id text,
    p_fan_id uuid default null,
    p_event_type text default 'ppv.purchased',
    p_account_media_id text default null,
    p_price_cents integer default null
)
returns text
language plpgsql
security definer
set search_path = public
as $$
declare
    v_id uuid;
begin
    if p_platform_order_id is null or length(trim(p_platform_order_id)) = 0 then
        -- No platform identity to deduplicate on. Say so explicitly rather
        -- than inventing a key: the caller decides what to do with an event
        -- the platform did not identify.
        return 'no_identity';
    end if;

    insert into public.platform_purchase_events (
        creator_id, platform_order_id, fan_id,
        event_type, account_media_id, price_cents
    )
    values (
        p_creator_id, trim(p_platform_order_id), p_fan_id,
        coalesce(p_event_type, 'ppv.purchased'), p_account_media_id, p_price_cents
    )
    on conflict (creator_id, platform_order_id) do nothing
    returning id into v_id;

    if v_id is null then
        return 'duplicate';
    end if;
    return 'claimed';
end;
$$;

-- ---------------------------------------------------------------------------
-- Mark a claimed order as fully applied.
-- ---------------------------------------------------------------------------
create or replace function public.complete_platform_purchase(
    p_creator_id uuid,
    p_platform_order_id text
)
returns void
language sql
security definer
set search_path = public
as $$
    update public.platform_purchase_events
       set status = 'processed',
           processed_at = now()
     where creator_id = p_creator_id
       and platform_order_id = trim(p_platform_order_id);
$$;

-- ---------------------------------------------------------------------------
-- Release a claim that did not result in a recorded purchase.
--
-- This is what keeps the guarantee one-directional. A claim is taken BEFORE the
-- work, so if the work does not happen — the price did not match, the media did
-- not match, the handler raised — the claim must go away, or a later legitimate
-- redelivery of that same order would be rejected as a duplicate and the sale
-- would be lost.
--
-- Deleting rather than marking failed is deliberate: the row's only job is to
-- answer "has this order been applied", and a released claim has not been. It
-- refuses to release a row already marked processed, so a late release from a
-- crashed handler cannot un-record a completed purchase.
-- ---------------------------------------------------------------------------
create or replace function public.release_platform_purchase(
    p_creator_id uuid,
    p_platform_order_id text
)
returns void
language sql
security definer
set search_path = public
as $$
    delete from public.platform_purchase_events
     where creator_id = p_creator_id
       and platform_order_id = trim(p_platform_order_id)
       and status = 'claimed';
$$;

revoke all on function public.claim_platform_purchase(uuid, text, uuid, text, text, integer)
    from public, anon, authenticated;
revoke all on function public.complete_platform_purchase(uuid, text)
    from public, anon, authenticated;
revoke all on function public.release_platform_purchase(uuid, text)
    from public, anon, authenticated;
grant execute on function public.claim_platform_purchase(uuid, text, uuid, text, text, integer)
    to service_role;
grant execute on function public.complete_platform_purchase(uuid, text) to service_role;
grant execute on function public.release_platform_purchase(uuid, text) to service_role;
