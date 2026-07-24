-- Multi-tenant browser access boundary.
--
-- The backend uses the Supabase service role and remains able to run cross-
-- creator schedulers. Authenticated dashboard users can only access rows owned
-- by creators assigned to them through chatter_creators.

create or replace function public.can_access_creator(p_creator_id text)
returns boolean
language sql
stable
security definer
set search_path = public
as $$
    select exists (
        select 1
          from public.chatter_creators cc
         where cc.chatter_id::text = auth.uid()::text
           and cc.creator_id::text = p_creator_id
    );
$$;

create or replace function public.can_access_fan(p_fan_id text)
returns boolean
language sql
stable
security definer
set search_path = public
as $$
    select exists (
        select 1
          from public.fans f
          join public.chatter_creators cc
            on cc.creator_id::text = f.creator_id::text
         where f.id::text = p_fan_id
           and cc.chatter_id::text = auth.uid()::text
    );
$$;

create or replace function public.can_access_fan_list(p_list_id text)
returns boolean
language sql
stable
security definer
set search_path = public
as $$
    select exists (
        select 1
          from public.fan_lists fl
          join public.chatter_creators cc
            on cc.creator_id::text = fl.creator_id::text
         where fl.id::text = p_list_id
           and cc.chatter_id::text = auth.uid()::text
    );
$$;

revoke all on function public.can_access_creator(text) from public;
revoke all on function public.can_access_fan(text) from public;
revoke all on function public.can_access_fan_list(text) from public;
grant execute on function public.can_access_creator(text) to authenticated, service_role;
grant execute on function public.can_access_fan(text) to authenticated, service_role;
grant execute on function public.can_access_fan_list(text) to authenticated, service_role;

alter table public.chatter_creators enable row level security;
do $$
declare existing_policy record;
begin
    for existing_policy in
        select policyname
          from pg_policies
         where schemaname = 'public'
           and tablename = 'chatter_creators'
    loop
        execute format(
            'drop policy %I on public.chatter_creators',
            existing_policy.policyname
        );
    end loop;
end
$$;
create policy tenant_own_assignments
on public.chatter_creators
for select
to authenticated
using (chatter_id = auth.uid());

alter table public.creators enable row level security;
do $$
declare existing_policy record;
begin
    for existing_policy in
        select policyname
          from pg_policies
         where schemaname = 'public'
           and tablename = 'creators'
    loop
        execute format(
            'drop policy %I on public.creators',
            existing_policy.policyname
        );
    end loop;
end
$$;
create policy tenant_creator_membership
on public.creators
for all
to authenticated
using (public.can_access_creator(id::text))
with check (public.can_access_creator(id::text));

-- Cover every current creator-owned base table without relying on a manually
-- maintained list. Re-run this idempotent migration after adding a new
-- creator-owned table. Service-role background jobs bypass RLS.
do $$
declare
    target_table text;
    existing_policy record;
begin
    for target_table in
        select c.table_name
          from information_schema.columns c
          join information_schema.tables t
            on t.table_schema = c.table_schema
           and t.table_name = c.table_name
         where c.table_schema = 'public'
           and t.table_type = 'BASE TABLE'
           and c.column_name = 'creator_id'
           and c.table_name not in ('chatter_creators')
    loop
        execute format(
            'alter table public.%I enable row level security',
            target_table
        );
        for existing_policy in
            select policyname
              from pg_policies
             where schemaname = 'public'
               and tablename = target_table
        loop
            execute format(
                'drop policy %I on public.%I',
                existing_policy.policyname,
                target_table
            );
        end loop;
        execute format(
            'create policy tenant_creator_membership on public.%I '
            'for all to authenticated '
            'using (public.can_access_creator(creator_id::text)) '
            'with check (public.can_access_creator(creator_id::text))',
            target_table
        );
    end loop;
end
$$;

-- Some child tables only carry fan_id. Bind those to the fan's creator.
do $$
declare
    target_table text;
    existing_policy record;
begin
    for target_table in
        select c.table_name
          from information_schema.columns c
          join information_schema.tables t
            on t.table_schema = c.table_schema
           and t.table_name = c.table_name
         where c.table_schema = 'public'
           and t.table_type = 'BASE TABLE'
           and c.column_name = 'fan_id'
           and not exists (
               select 1
                 from information_schema.columns creator_column
                where creator_column.table_schema = 'public'
                  and creator_column.table_name = c.table_name
                  and creator_column.column_name = 'creator_id'
           )
    loop
        execute format(
            'alter table public.%I enable row level security',
            target_table
        );
        for existing_policy in
            select policyname
              from pg_policies
             where schemaname = 'public'
               and tablename = target_table
        loop
            execute format(
                'drop policy %I on public.%I',
                existing_policy.policyname,
                target_table
            );
        end loop;
        execute format(
            'create policy tenant_fan_membership on public.%I '
            'for all to authenticated '
            'using (public.can_access_fan(fan_id::text)) '
            'with check (public.can_access_fan(fan_id::text))',
            target_table
        );
    end loop;
end
$$;

-- A list-member row belongs to both a fan and a list. Requiring both prevents
-- cross-creator joins even if a caller guesses another agency's list UUID.
do $$
declare existing_policy record;
begin
    if to_regclass('public.fan_list_members') is not null then
        for existing_policy in
            select policyname
              from pg_policies
             where schemaname = 'public'
               and tablename = 'fan_list_members'
        loop
            execute format(
                'drop policy %I on public.fan_list_members',
                existing_policy.policyname
            );
        end loop;
        create policy tenant_list_membership
        on public.fan_list_members
        for all
        to authenticated
        using (
            public.can_access_fan(fan_id::text)
            and public.can_access_fan_list(list_id::text)
        )
        with check (
            public.can_access_fan(fan_id::text)
            and public.can_access_fan_list(list_id::text)
        );
    end if;
end
$$;
