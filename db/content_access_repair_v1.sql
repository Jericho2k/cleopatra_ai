-- A free access repair happens at most once, and clears only its own hold.
--
-- THE THREE FAILURES THIS CLOSES
-- ------------------------------
-- services/content_access.py called the platform adapter directly and then
-- wrote the receipt, with nothing durable in between. Reproduced against
-- backend 4a1683a with the existing FakeSupabase and platform fixtures:
--
--   1. Platform accepts, save_message then fails. The operator clicks Resend
--      again. TWO provider sends, and the review hold is still set — so the
--      operator is invited to click a third time. Two free copies, not two
--      charges, but the customer sees the same media twice and nothing
--      records that it happened.
--
--   2. Two operators click Resend at the same moment. TWO provider sends, and
--      BOTH calls report success.
--
--   3. A crisis hold is raised while the platform call is in flight.
--      clear_fan_review is an unconditional update by fan id, so the repair
--      clears the NEWER hold on its way out. A conversation frozen for crisis
--      language silently resumes.
--
-- WHY A TABLE AND NOT A LOCK
-- --------------------------
-- The dangerous window spans a slow external call. An advisory lock or a row
-- lock held across it would serialise operators at the cost of pinning a
-- database connection to a provider's latency, and it would still lose the
-- claim if the process died mid-send — which is the case that matters, because
-- that is the one where nobody knows whether the customer got the media.
--
-- So the claim is a row, written and committed BEFORE the send. It survives a
-- worker restart, and it is what a retry reconciles against.
--
-- The unique index IS the claim. A second operator's insert conflicts, and
-- PostgREST's ignore-duplicates returns them nothing — which they read as "a
-- repair for this purchase already exists" and reconcile against, rather than
-- as permission to send.
--
-- FOUR OUTCOMES, BECAUSE THREE IS NOT ENOUGH
-- ------------------------------------------
-- The review is explicit that a delivery claim must be tied to the operation
-- result. That needs `unknown` as a first-class outcome and not a synonym for
-- `failed`:
--
--   claimed    this repair is in flight. Nobody else sends.
--   confirmed  the platform returned a receipt and it is recorded. Done.
--   failed     the platform refused BEFORE sending anything. Safe to retry.
--   unknown    the send may have happened and cannot be proven — a receipt
--              that never came back, a transport error after the request left,
--              a crash between the send and the write. NEVER retried
--              automatically. An operator decides, with the evidence in front
--              of them.
--
-- Exactly-once delivery is not claimed. The platform does not offer it, so
-- what is claimed is exactly-once ATTEMPT: at most one send per purchase per
-- review case, and an honest record of every outcome including the one where
-- the answer is "we do not know".
--
-- Idempotent and additive. Safe to re-run. Must precede tenant_isolation_v1.

create extension if not exists pgcrypto;

-- ---------------------------------------------------------------------------
-- 1. Which hold is this?
--
-- `needs_human_review` plus `review_reason` cannot tell two holds apart: a
-- content-access hold that is cleared and re-raised looks identical, and a
-- DIFFERENT hold raised meanwhile looks like the same row to an unconditional
-- update. `review_case_id` names the individual hold, so clearing one can be a
-- compare-and-set against the hold that was actually observed.
--
-- Text rather than uuid, and '' rather than null for "no case", so the unique
-- index below needs neither NULLS NOT DISTINCT (PostgreSQL 15+) nor a
-- coalescing expression PostgREST could not name as a conflict target.
-- ---------------------------------------------------------------------------
alter table public.fans
    add column if not exists review_case_id text not null default '';

-- Existing holds predate the column and would all share ''. Give each one an
-- identity, so a repair resolving a hold raised before this migration is
-- compare-and-set against that hold rather than against every empty string.
update public.fans
   set review_case_id = gen_random_uuid()::text
 where needs_human_review is true
   and coalesce(review_case_id, '') = '';

-- ---------------------------------------------------------------------------
-- 2. The claim
-- ---------------------------------------------------------------------------
create table if not exists public.content_access_repairs (
    id uuid primary key default gen_random_uuid(),
    creator_id uuid not null references public.creators(id) on delete cascade,
    fan_id uuid not null references public.fans(id) on delete cascade,
    -- The exact purchased delivery this repairs. ppv_deliveries.reference,
    -- never a re-planned selection: a repair that picks its own media is a new
    -- offer wearing a repair's clothes.
    reference text not null,
    -- The hold it answers, or '' for a repair performed outside one.
    review_case_id text not null default '',
    status text not null default 'claimed',
    -- What was actually sent, frozen at claim time. Read back on reconcile so
    -- a retry can prove it would have sent the same thing.
    media_ids jsonb not null default '[]'::jsonb,
    platform_message_id text null,
    -- Who asked. An auditable resolution needs an actor, and "the backend did
    -- it" is not one.
    claimed_by text not null default '',
    -- Why it ended where it did, for the outcomes that need a sentence.
    detail text not null default '',
    claimed_at timestamptz not null default now(),
    resolved_at timestamptz null,
    constraint content_access_repairs_status_check check (
        status in ('claimed', 'confirmed', 'failed', 'unknown')
    )
);

-- THE claim. One repair per purchase per review case.
--
-- A new hold raised later carries a new review_case_id, so a genuinely new
-- complaint about the same purchase can be repaired again — which is correct:
-- the media went missing twice.
create unique index if not exists content_access_repairs_claim_idx
    on public.content_access_repairs (creator_id, fan_id, reference, review_case_id);

-- The operator panel reads a fan's repair history newest first.
create index if not exists content_access_repairs_fan_idx
    on public.content_access_repairs (creator_id, fan_id, claimed_at desc);

-- Finding the repairs nobody has resolved, which is the operational question
-- "is anything stuck?".
create index if not exists content_access_repairs_open_idx
    on public.content_access_repairs (status, claimed_at)
    where status in ('claimed', 'unknown');

-- ---------------------------------------------------------------------------
-- 3. Privileges
--
-- Deliberately NOT owner-only: the operator panel has to show the real state
-- of the operation, including a repair that is in flight or whose outcome is
-- unknown. Hiding that is what makes someone click Resend again.
--
-- tenant_isolation_v1 and browser_least_privilege_v1 discover this table by
-- its creator_id and give it a tenancy-scoped SELECT and no write policy,
-- which is exactly right: an operator reads their own repairs and cannot forge
-- one. Stated here because "why does this table have no explicit grant" is the
-- first question a reader will have.
-- ---------------------------------------------------------------------------
grant all on public.content_access_repairs to service_role;
