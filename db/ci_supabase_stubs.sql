-- CI-ONLY Supabase platform stubs. NEVER APPLY THIS TO A REAL DATABASE.
--
-- Audit reference: DB-000.
--
-- The application migrations grant to Supabase's managed roles and call
-- auth.uid(). Those objects are created by the Supabase platform, not by this
-- repository, and dumping the real auth/storage schemas into an application
-- baseline would be both noisy and wrong.
--
-- This file creates the minimum stand-ins so a plain PostgreSQL container can
-- apply db/*.sql end to end. On the real database these already exist and this
-- file must never run.

do $$
begin
    if not exists (select 1 from pg_roles where rolname = 'anon') then
        create role anon nologin;
    end if;
    if not exists (select 1 from pg_roles where rolname = 'authenticated') then
        create role authenticated nologin;
    end if;
    if not exists (select 1 from pg_roles where rolname = 'service_role') then
        create role service_role nologin bypassrls;
    end if;
end
$$;

create schema if not exists auth;

-- Supabase resolves auth.uid() from the request JWT. The stub reads a GUC so a
-- schema test can impersonate an operator:
--     set local request.jwt.claim.sub = '<uuid>';
create or replace function auth.uid() returns uuid
language sql
stable
as $$
    select nullif(current_setting('request.jwt.claim.sub', true), '')::uuid
$$;

grant usage on schema auth to anon, authenticated, service_role;
grant usage on schema public to anon, authenticated, service_role;
