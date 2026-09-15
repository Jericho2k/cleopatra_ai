-- Restart-safe historical conversation backfill and compaction state.
--
-- Audit reference: HIST-001. Additive and idempotent; safe to re-run.
--
-- The problem
-- -----------
-- /load-history paginated a whole conversation in one request, in memory, with
-- no durable cursor. Three things followed from that:
--
--   * A 5,000-message fan is 500 provider pages (API Fansly documents
--     `limit min=1 max=10` on the chat-messages endpoint, so ten per page is
--     the ceiling, not a setting). Any interruption — a deploy, a timeout, an
--     operator closing the tab — threw away every page already paid for and
--     the next attempt started at page one.
--   * Nothing recorded what had already been imported beyond the messages
--     themselves, so "resume" meant "refetch".
--   * Nothing bounded the work, so an import could sit in front of live
--     conversation for hundreds of round trips.
--
-- The state
-- ---------
-- One row per fan, holding the paging cursor, the compaction cursor, and the
-- provider cost already spent. Both cursors are durable, so a process that
-- stops after page 300 resumes at page 301 rather than at page one.
--
-- Deliberately a TABLE and not more columns on `fans`:
--
--   * fans is read on the hot reply path and on every reconciliation pass. It
--     is the wrong place for a dozen columns that only a background job reads.
--   * The two model-authored jsonb documents on fans (ai_summary,
--     active_session) are overwritten wholesale, which is exactly how a
--     checkpoint hidden inside one would be silently lost — the same reasoning
--     db/chat_sync_checkpoint_v1.sql gives for NOT putting the reconciliation
--     marker there.
--
-- It carries creator_id as well as fan_id so tenant_isolation_v1 discovers it
-- and gives it a policy automatically, and so per-creator cost reporting is one
-- indexed read rather than a join.
--
-- Failure direction
-- -----------------
-- A missing row means "never backfilled", which resolves toward doing the work.
-- A stale cursor can only ever cause a page to be RE-fetched, never skipped:
-- the cursor advances after the import of that page succeeds, and imports are
-- idempotent on (creator_id, fansly_message_id), so a repeated page writes
-- nothing. The system fails toward paying one extra credit, never toward
-- losing history.

create table if not exists public.fan_history_backfill (
    id uuid primary key default gen_random_uuid(),
    creator_id uuid not null references public.creators(id) on delete cascade,
    -- One backfill per fan. The unique constraint is what makes "start the
    -- backfill" safe to call from a dashboard button, a scheduler and a warm
    -- resume at the same time.
    fan_id uuid not null references public.fans(id) on delete cascade,

    -- pending  : known to need work, nothing fetched yet
    -- running  : a worker is actively paging
    -- paused   : yielded to live work, or the credit budget is spent
    -- complete : the provider reported no further cursor
    -- error    : the last attempt failed; last_error says how
    status text not null default 'pending' check (status in (
        'pending', 'running', 'paused', 'complete', 'error'
    )),

    -- The provider cursor for the NEXT page. Null with exhausted=false means
    -- "start from the newest page"; exhausted=true means there is no next page.
    page_cursor text null,
    exhausted boolean not null default false,

    -- Paging progress.
    pages_fetched integer not null default 0 check (pages_fetched >= 0),
    messages_seen integer not null default 0 check (messages_seen >= 0),
    messages_imported integer not null default 0 check (messages_imported >= 0),
    oldest_message_id text null,
    oldest_sent_at timestamptz null,
    newest_message_id text null,
    newest_sent_at timestamptz null,

    -- Compaction progress, which trails paging. Messages are extracted
    -- chronologically in bounded chunks, so this is the sent_at of the last
    -- message already turned into durable facts. Separate from the paging
    -- cursor because the two advance in OPPOSITE directions: paging walks
    -- backwards from newest, extraction walks forwards from oldest.
    extraction_cursor_sent_at timestamptz null,
    extraction_cursor_message_id text null,
    chunks_extracted integer not null default 0 check (chunks_extracted >= 0),
    messages_extracted integer not null default 0 check (messages_extracted >= 0),
    facts_proposed integer not null default 0 check (facts_proposed >= 0),
    facts_accepted integer not null default 0 check (facts_accepted >= 0),

    -- Compact durable continuity state: ongoing topics, established personal
    -- context, previous commercial context. Small by construction and NEVER
    -- authoritative over public.fan_facts — see services/fan_history_memory.py.
    continuity jsonb null,

    -- Provider cost already spent on THIS fan's history, so an operator can see
    -- what an import cost and what finishing it would cost. Credits are
    -- estimates; API Fansly's Usage dashboard is authoritative.
    api_calls integer not null default 0 check (api_calls >= 0),
    response_bytes bigint not null default 0 check (response_bytes >= 0),
    estimated_credits numeric(12,3) not null default 0 check (estimated_credits >= 0),

    -- Media metadata observed while paging. Counted, never downloaded: history
    -- backfill makes no media call of any kind.
    media_references_seen integer not null default 0 check (media_references_seen >= 0),

    last_error text null,
    started_at timestamptz null,
    last_page_at timestamptz null,
    completed_at timestamptz null,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),

    unique (fan_id)
);

create index if not exists fan_history_backfill_creator_idx
    on public.fan_history_backfill (creator_id, updated_at desc);

-- The scheduler's claim query: rows with history left to fetch or compact,
-- oldest attempt first. Partial, because a finished backfill is the common case
-- once a creator has been onboarded for a while and should not be scanned.
create index if not exists fan_history_backfill_pending_idx
    on public.fan_history_backfill (updated_at)
    where status in ('pending', 'running', 'paused', 'error');

alter table public.fan_history_backfill enable row level security;

comment on table public.fan_history_backfill is
    'HIST-001: restart-safe cursor and cost state for historical chat backfill '
    'and its compaction into fan_facts. One row per fan.';
comment on column public.fan_history_backfill.page_cursor is
    'Provider cursor for the NEXT page. Advanced only after that page imported '
    'successfully, so a stale value re-fetches rather than skips.';
comment on column public.fan_history_backfill.continuity is
    'Compact historical continuity state. Never overrides public.fan_facts.';

-- Historical facts are ordinary fan_facts with a distinguishing source_type, so
-- there is ONE fan memory rather than two. The existing check constraint only
-- ever defaulted this column, so no constraint has to be widened; this comment
-- records the vocabulary so the next reader does not invent a second table.
comment on column public.fan_facts.source_type is
    'fan_message (live extraction), historical_message (compacted backfill), '
    'or a payment-event source. Historical evidence never outranks live '
    'explicit or confirmed facts.';

-- Row level security policies are applied by db/tenant_isolation_v1.sql, which
-- discovers creator-owned tables at run time, and narrowed to SELECT by
-- db/browser_least_privilege_v1.sql. Both run after this file. No policy is
-- declared here, deliberately: a hand-written one would diverge from the
-- discovered predicate the moment tenancy rules change.
