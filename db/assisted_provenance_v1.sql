-- An Assisted reply stays attributable across a restart.
--
-- THE PROBLEM
-- -----------
-- Full Auto generates and delivers inside one function, so its provenance
-- recorder is a local variable. Assisted does not: `get_suggestions` produces
-- candidates, a person reads them, and some time later `POST /reply` sends one.
-- Two HTTP requests with a human in between.
--
-- services/reply_provenance.py bridged that with SUGGESTION_PROVENANCE, an
-- in-process OrderedDict. Its own comment says what that costs:
--
--     The process-wide store. One per backend replica, which is why a miss is
--     an ordinary outcome rather than an error.
--
-- So a deploy, a crash, an autoscale event, or simply the second request
-- landing on a different replica loses the record — and the message is then
-- saved with NO provenance at all. The reply is unattributable and nothing
-- anywhere says so, which is the exact failure the provenance work exists to
-- remove: a reply that cannot name the commit, the flags or the model that
-- produced it is not evidence about any particular piece of code.
--
-- WHY A TABLE AND NOT A LONGER TTL
-- --------------------------------
-- The window is not the problem. A thirty-minute TTL does not survive a deploy
-- that takes thirty seconds, and a multi-replica deployment loses records with
-- no time passing at all.
--
-- OWNER-ONLY, NECESSARILY
-- -----------------------
-- The record carries which model actually answered and the fallback ladder it
-- took. That is the same content db/owner_only_diagnostics_v1.sql moved out of
-- messages.media_context for, so this table is registered owner-only by the
-- same mechanism and the browser roles hold no grant on it. Storing attribution
-- durably in a place an agency could read would trade one disclosure for
-- another.
--
-- Idempotent and additive. Safe to re-run. MUST precede tenant_isolation_v1.

create extension if not exists pgcrypto;

create table if not exists public.assisted_provenance (
    -- The opaque handle the dashboard returns on POST /reply. Not a secret and
    -- not an authorization: creator_id and fan_id are checked on redemption,
    -- because a record belonging to another conversation is simply the wrong
    -- record and attaching it would be worse than attaching nothing.
    token text primary key,
    creator_id uuid not null references public.creators(id) on delete cascade,
    fan_id uuid not null references public.fans(id) on delete cascade,
    -- The recorder's state, as services/reply_provenance.py serialises it.
    record jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now()
);

-- Redemption deletes, so the common read is by primary key. This index is for
-- the sweep: one generated turn becomes at most one sent message, and the
-- turns nobody acted on have to age out rather than accumulate forever.
create index if not exists assisted_provenance_age_idx
    on public.assisted_provenance (created_at);

insert into public.owner_only_tables (table_name, reason)
values (
    'assisted_provenance',
    'Carries which model actually answered and the fallback ladder it took, '
    'for a reply an operator has not sent yet. Same content, and therefore the '
    'same rule, as message_diagnostics.'
)
on conflict (table_name) do update set reason = excluded.reason;

alter table public.assisted_provenance enable row level security;
revoke all on public.assisted_provenance from anon, authenticated;
grant all on public.assisted_provenance to service_role;
