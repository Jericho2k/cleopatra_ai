-- Atomically consume one Assisted attribution token across backend replicas.
--
-- assisted_provenance_v1.sql originally redeemed with SELECT followed by
-- DELETE.  Two transactions could both finish the SELECT before either DELETE
-- committed and both attribute a send to the same generated turn.  The DELETE
-- is now the read, and its RETURNING row is the one durable claim.
--
-- Additive and idempotent.  MUST run after assisted_provenance_v1.sql.

create or replace function public.consume_assisted_provenance(
    p_token text,
    p_creator_id text default '',
    p_fan_id text default '',
    p_now timestamptz default now()
)
returns table (
    record jsonb,
    created_at timestamptz,
    expired boolean
)
language sql
security definer
set search_path = ''
as $$
    delete from public.assisted_provenance as provenance
    where provenance.token = p_token
      and (coalesce(p_creator_id, '') = '' or provenance.creator_id::text = p_creator_id)
      and (coalesce(p_fan_id, '') = '' or provenance.fan_id::text = p_fan_id)
    returning
        provenance.record,
        provenance.created_at,
        provenance.created_at < (p_now - interval '30 minutes');
$$;

revoke all on function public.consume_assisted_provenance(
    text, text, text, timestamptz
) from public, anon, authenticated;
grant execute on function public.consume_assisted_provenance(
    text, text, text, timestamptz
) to service_role;
