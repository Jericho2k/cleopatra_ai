-- Allow agencies to send multiple manual PPVs while preserving the atomic
-- single-flight guard for AI/session delivery. Apply after
-- ppv_delivery_ledger_v1.sql.

drop index if exists public.ppv_deliveries_one_active_per_fan_idx;

create unique index if not exists ppv_deliveries_one_active_automated_per_fan_idx
    on public.ppv_deliveries (fan_id)
    where status in ('claimed', 'delivered_pending')
      and source <> 'operator';

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
    v_source text;
begin
    select status, source
      into v_status, v_source
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
    if v_source = 'operator' then
        return 'operator_tracked';
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

alter table public.creator_commercial_policies
    alter column ppv_payment_window_hours set default 2;

update public.creator_commercial_policies
   set ppv_payment_window_hours = 2,
       updated_at = now()
 where ppv_payment_window_hours > 2;
