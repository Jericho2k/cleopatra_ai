-- Durable locked-PPV lifecycle and atomic delivery claims.
-- Apply after agency_operability_v1.sql.

create table if not exists public.ppv_deliveries (
    id uuid primary key default gen_random_uuid(),
    reference text not null unique,
    creator_id uuid not null references public.creators(id) on delete cascade,
    fan_id uuid not null references public.fans(id) on delete cascade,
    status text not null default 'claimed'
        check (status in (
            'claimed',
            'delivered_pending',
            'purchased',
            'abandoned',
            'voided',
            'failed'
        )),
    media_ids jsonb not null default '[]'::jsonb,
    price_cents integer not null check (price_cents > 0),
    source text not null default 'auto',
    set_id text null,
    step_index integer null,
    platform_message_id text null,
    amount_paid_cents integer null,
    claimed_at timestamptz not null default now(),
    delivered_at timestamptz null,
    purchased_at timestamptz null,
    abandoned_at timestamptz null,
    voided_at timestamptz null,
    failed_at timestamptz null,
    updated_at timestamptz not null default now(),
    last_error text null,
    metadata jsonb not null default '{}'::jsonb,
    check (jsonb_typeof(media_ids) = 'array' and jsonb_array_length(media_ids) > 0)
);

-- This index is the delivery lock. It makes two workers/tabs racing to send a
-- PPV to the same fan mutually exclusive before either reaches the platform.
create unique index if not exists ppv_deliveries_one_active_per_fan_idx
    on public.ppv_deliveries (fan_id)
    where status in ('claimed', 'delivered_pending');

create index if not exists ppv_deliveries_fan_created_idx
    on public.ppv_deliveries (fan_id, claimed_at desc);

create index if not exists ppv_deliveries_creator_status_created_idx
    on public.ppv_deliveries (creator_id, status, claimed_at desc);

create index if not exists ppv_deliveries_media_ids_gin_idx
    on public.ppv_deliveries using gin (media_ids);

alter table public.ppv_deliveries enable row level security;

-- Atomically attach the legacy fan snapshot only while this exact delivery is
-- still pending. A purchase webhook and this function serialize on the ledger
-- row, preventing a fast purchase from being overwritten back to pending.
create or replace function public.attach_pending_ppv(
    p_fan_id uuid,
    p_reference text,
    p_pending jsonb
)
returns text
language plpgsql
security definer
set search_path = public
as $$
declare
    v_status text;
begin
    select status
      into v_status
      from public.ppv_deliveries
     where fan_id = p_fan_id
       and reference = p_reference
     for update;

    if not found then
        return 'missing';
    end if;
    if v_status <> 'delivered_pending' then
        return v_status;
    end if;

    update public.fans
       set pending_ppv_check = p_pending
     where id = p_fan_id;
    return 'attached';
end;
$$;

revoke all on function public.attach_pending_ppv(uuid, text, jsonb)
    from public, anon, authenticated;
grant execute on function public.attach_pending_ppv(uuid, text, jsonb)
    to service_role;
