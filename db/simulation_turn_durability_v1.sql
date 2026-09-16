-- One simulated fan turn, durable and pollable.
--
-- Additive and idempotent; safe to re-run.
--
-- The problem
-- -----------
-- POST /creator/{id}/fan/{id}/simulate-inbound ran the ENTIRE Full Auto
-- pipeline inside one browser request. That was survivable while the writer
-- gave up quickly. It stopped being survivable the moment the writer started
-- pursuing its own model properly:
--
--   browser POSTs  ->  writer retries a rate-limited provider  ->  browser
--   reaches its 180s ceiling and reports "Timeout: the simulated turn did not
--   finish"  ->  backend keeps working  ->  the fallback answers  ->  the reply
--   is persisted anyway.
--
-- The UI said the turn failed. The database said it succeeded. Both were
-- reporting honestly about different things, which is the worst possible
-- outcome: an operator presses Send again, the first turn wakes up, and the
-- fan gets two contradictory creator replies — after commercial state has
-- already moved.
--
-- The fix is not a longer browser timeout. It is to stop the lifetime of model
-- recovery being the lifetime of one HTTP request. The POST now records a turn
-- and returns; the pipeline runs in the background against THIS row; the UI
-- polls it. A browser that times out, reloads, or closes entirely changes
-- nothing about what the backend does or what it reports afterwards.
--
-- Why a table and not a job framework
-- -----------------------------------
-- There is already a durable-work table here (scheduled_actions) and it is the
-- wrong shape for this: it schedules FUTURE work claimed by a worker loop,
-- whereas this records work that started immediately and whose only consumer
-- is the operator watching it. One small record with one status column is the
-- whole requirement, and inventing a general job runner to hold it would be a
-- much larger thing to get right for no additional capability.
--
-- The invariants this schema enforces, rather than hopes for
-- ----------------------------------------------------------
--   * ONE turn per submission. `(fan_id, idempotency_key)` is unique, so a
--     double-clicked Send, a retried POST and a browser that reconnects and
--     resubmits all resolve to the same row and one generation.
--   * ONE active turn per simulated fan. The partial unique index means a
--     second concurrent submission is rejected by the database, not by a
--     read-then-act check that two requests can both pass.
--   * Terminal is terminal. `completed_at`/`failed` are written once, by a
--     conditional update that only matches a still-active row, so a task
--     abandoned at the deadline cannot later flip a failed turn to succeeded.

create table if not exists public.simulation_turns (
    id uuid primary key default gen_random_uuid(),
    creator_id uuid not null references public.creators(id) on delete cascade,
    fan_id uuid not null references public.fans(id) on delete cascade,

    -- Supplied by the client, one per Send press. Not a nonce the server
    -- invents: the point is that the SAME submission retried is recognised as
    -- the same submission, which only the client can assert.
    idempotency_key text not null,

    -- accepted   : recorded, the pipeline has not started yet
    -- processing : the Full Auto turn is running
    -- completed  : the turn finished; creator_message_ids is what it produced
    --              (possibly empty — Full Auto deciding to send nothing is a
    --              successful turn, and `outcome` says which happened)
    -- failed     : terminal failure. No reply from this turn will ever appear.
    status text not null default 'accepted' check (status in (
        'accepted', 'processing', 'completed', 'failed'
    )),

    -- What the operator typed as the fan, kept so a reconnecting browser can
    -- render the turn it is watching without re-reading the message table.
    fan_message text not null default '',
    fast boolean not null default true,
    include_mirrored_catalog boolean not null default false,

    fan_message_id uuid null,
    -- The creator messages THIS turn produced, in production order. Ids only:
    -- the transcript itself is read from public.messages, which stays the one
    -- source of truth for what was said.
    creator_message_ids jsonb not null default '[]'::jsonb,

    -- The real Full Auto outcome vocabulary (replied / no_send /
    -- analyzer_degraded / writer_failed / plan_unrecoverable /
    -- inventory_unsafe), plus deadline_exceeded for a turn the backend
    -- abandoned. Deliberately not constrained here: the vocabulary belongs to
    -- the pipeline, and a CHECK would turn adding an outcome into a migration.
    outcome text null,
    analysis_degraded boolean not null default false,

    -- Operator-facing failure detail. Never rendered to an agency verbatim —
    -- see services/simulation_turns.py, which reduces it to a product-level
    -- status for anyone who is not the platform owner.
    error text null,
    -- Correlates with the [SIM TURN] / traceback lines in the server log.
    error_id text null,

    -- The backend-owned ceiling this turn was started under, in seconds. Stored
    -- rather than recomputed so a turn that expired can be explained with the
    -- deadline it actually had, not the one configured today.
    deadline_seconds numeric(10,2) null,

    created_at timestamptz not null default now(),
    started_at timestamptz null,
    finished_at timestamptz null,
    updated_at timestamptz not null default now(),

    unique (fan_id, idempotency_key)
);

-- One active turn per simulated fan.
--
-- Partial, so completed history accumulates freely. This is what makes "Send
-- is disabled while a turn is running" a fact rather than a UI courtesy: a
-- client that ignores the disabled button is refused here.
create unique index if not exists simulation_turns_one_active_per_fan_idx
    on public.simulation_turns (fan_id)
    where status in ('accepted', 'processing');

-- The poll and the resume-after-reload read: newest turn for this conversation.
create index if not exists simulation_turns_fan_recent_idx
    on public.simulation_turns (fan_id, created_at desc);

create index if not exists simulation_turns_creator_recent_idx
    on public.simulation_turns (creator_id, created_at desc);

alter table public.simulation_turns enable row level security;

comment on table public.simulation_turns is
    'One durable, pollable simulated Full Auto turn. Decouples model recovery '
    'from the lifetime of a browser request so the UI and the backend can no '
    'longer disagree about whether a turn happened.';
comment on column public.simulation_turns.idempotency_key is
    'Client-supplied, one per Send press. Unique with fan_id, so a duplicate '
    'POST resolves to the existing turn instead of starting a second one.';
comment on column public.simulation_turns.status is
    'accepted/processing are active; completed/failed are terminal and are '
    'written once by a conditional update that only matches an active row.';
comment on column public.simulation_turns.creator_message_ids is
    'Ids of the creator messages this turn produced. The messages themselves '
    'live in public.messages, which remains the only transcript.';

-- Row level security policies are applied by db/tenant_isolation_v1.sql, which
-- discovers creator-owned tables at run time, and narrowed to SELECT by
-- db/browser_least_privilege_v1.sql. Both run after this file. SELECT-only is
-- correct: every write here happens on the backend, which owns the status
-- machine, and a browser that could write it could forge a completed turn.
