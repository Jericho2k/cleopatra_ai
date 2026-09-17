-- Sprint 2: the part of a conversation that is not a fact and not a message.
--
-- docs/autonomy_architecture_review.md §3E and §4. The review is explicit that
-- it would be inaccurate to say Cleopatra has no memory: fan_facts already
-- holds evidence-validated knowledge with merge rules, fan_history_memory
-- compacts history, creator facts and scene state persist. What none of them
-- hold is the STATE OF THE INTERACTION:
--
--     Long-term quality is not simply recall of names and favorite things. It
--     includes the state of the interaction: why a topic matters, which
--     question remains unanswered, whether a misunderstanding was repaired,
--     and whether a prior invitation to continue is still current.
--
-- fan_facts uses an enumerated CRM vocabulary (preferred_name, payday,
-- content_interest); the scene stores one interaction's progression. Neither is
-- a ledger of multiple unfinished topics and their resolution conditions, and
-- compacting history into the same fact schema does not create one.
--
-- Two tables, for two different things.
--
-- `conversation_open_threads` is what is UNFINISHED. A question nobody
-- answered, a promise nobody kept, a topic put off, a complaint nobody
-- resolved. Each persists until it is fulfilled, cancelled, superseded or
-- expired — never merely because the recent-message window rolled over it.
-- This is the record behind the review's requirement to "reserve context space
-- for unresolved obligations".
--
-- `conversation_episodes` is what a completed stretch of conversation WAS
-- ABOUT and how it ended. Compact summaries with source ranges, so an episode
-- can be read back to the messages it describes. The review states the
-- constraint this table must never violate: an episode is "never proof of
-- payment". It therefore carries no amount, no price and no purchase flag;
-- ppv_deliveries and fan_facts remain the only authorities on money.
--
-- Both are scoped by creator AND fan, because §4 requires it: "Retrieval must
-- be scoped by creator and customer." One customer talking to two creators has
-- two sets of open threads and they must not see each other.
--
-- Idempotent and additive. Safe to re-run. Must precede tenant_isolation_v1.

create extension if not exists pgcrypto;

-- ---------------------------------------------------------------------------
-- Open threads: the obligations a conversation is carrying
-- ---------------------------------------------------------------------------

create table if not exists public.conversation_open_threads (
    id uuid primary key default gen_random_uuid(),
    creator_id uuid not null references public.creators(id) on delete cascade,
    fan_id uuid not null references public.fans(id) on delete cascade,

    -- What kind of unfinished business this is. The five the review names, and
    -- no more: a wider vocabulary would invite the same drift that made
    -- fan_facts a CRM schema rather than a conversation one.
    kind text not null check (kind in (
        'question',        -- asked and not answered
        'promise',         -- said we would do something
        'deferred_topic',  -- explicitly put off until later
        'complaint',       -- raised a problem that is not resolved
        'correction'       -- corrected something we had wrong
    )),

    -- Who is waiting on whom. A question the fan asked and a question the
    -- creator asked are different obligations, and answering the wrong one is
    -- one of the failures §1 asks a test to catch.
    raised_by text not null check (raised_by in ('fan', 'creator')),

    -- One line, in plain language, of what is unfinished. Deliberately short:
    -- this is rendered into a prompt alongside others, and a paragraph here
    -- would spend the budget the thread exists to be given.
    summary text not null check (length(summary) between 1 and 400),
    -- What would close it. Written when the thread is recorded, so resolution
    -- is a test against a stated condition rather than a later judgement call.
    resolution_condition text null check (resolution_condition is null or length(resolution_condition) <= 400),

    status text not null default 'open' check (status in (
        'open', 'fulfilled', 'cancelled', 'superseded', 'expired'
    )),

    -- Evidence discipline (§4): every memory needs a source, a timestamp and an
    -- evidence type. "The customer said payment succeeded" and "the platform
    -- confirmed order X" are different kinds of fact and must not merge.
    evidence_type text not null default 'stated' check (evidence_type in (
        'stated',             -- someone said it in the conversation
        'inferred',           -- read out of the conversation, not said outright
        'platform_confirmed', -- an external system asserted it
        'operator'            -- a human recorded it
    )),
    confidence numeric(5,4) not null default 1.0
        check (confidence >= 0 and confidence <= 1),

    -- Provenance back to the turn that raised it. Fingerprints rather than
    -- message text, matching services/reply_provenance.py: this table must not
    -- become a second copy of the conversation.
    source_message_fingerprint text null,
    source_turn_id text null,
    evidence_text text null check (evidence_text is null or length(evidence_text) <= 500),

    -- Lifetimes. §4: "Temporary mood and durable preference need different
    -- lifetimes." A thread with no expires_at persists indefinitely, which is
    -- the right default for an unanswered question and the wrong one for a
    -- passing aside; the caller decides.
    first_seen_at timestamptz not null default now(),
    last_seen_at timestamptz not null default now(),
    expires_at timestamptz null,

    resolved_at timestamptz null,
    resolved_by text null check (resolved_by is null or resolved_by in (
        'fan_message', 'creator_reply', 'operator', 'expiry', 'supersession'
    )),
    resolution_note text null check (resolution_note is null or length(resolution_note) <= 400),

    -- §4: "A later correction should supersede an old preference, not create
    -- two simultaneously authoritative facts." Supersession is a link, not a
    -- delete, so the history of what changed survives.
    superseded_by uuid null references public.conversation_open_threads(id) on delete set null,

    -- One conversation must not accumulate the same obligation once per turn
    -- it is mentioned in. The caller derives this from (fan, kind, subject).
    dedupe_key text not null,

    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),

    -- Scoped by creator AND fan, enforced rather than trusted.
    unique (creator_id, fan_id, dedupe_key),

    -- A resolved thread must say how and when; an open one must not claim to.
    constraint conversation_open_threads_resolution_complete check (
        (status = 'open' and resolved_at is null and resolved_by is null)
        or (status <> 'open' and resolved_at is not null and resolved_by is not null)
    ),
    -- Only a superseded thread points at a successor.
    constraint conversation_open_threads_supersession check (
        superseded_by is null or status = 'superseded'
    )
);

-- The hot read: "what is this conversation still carrying?", newest first.
create index if not exists conversation_open_threads_open_idx
    on public.conversation_open_threads (creator_id, fan_id, last_seen_at desc)
    where status = 'open';

-- The expiry sweep.
create index if not exists conversation_open_threads_expiry_idx
    on public.conversation_open_threads (expires_at)
    where status = 'open' and expires_at is not null;

create index if not exists conversation_open_threads_fan_idx
    on public.conversation_open_threads (fan_id, updated_at desc);

-- ---------------------------------------------------------------------------
-- Episodes: what a completed stretch of conversation was about
-- ---------------------------------------------------------------------------

create table if not exists public.conversation_episodes (
    id uuid primary key default gen_random_uuid(),
    creator_id uuid not null references public.creators(id) on delete cascade,
    fan_id uuid not null references public.fans(id) on delete cascade,

    summary text not null check (length(summary) between 1 and 800),
    -- How it ended, in the conversation's own terms: he went quiet, he said
    -- goodbye, it was interrupted, it was resolved. Not a commercial outcome.
    ended_with text not null default 'unknown' check (ended_with in (
        'resolved', 'went_quiet', 'said_goodbye', 'interrupted', 'unknown'
    )),

    -- The source range, so an episode can be read back to the messages it
    -- describes rather than believed on its own.
    first_message_at timestamptz not null,
    last_message_at timestamptz not null,
    message_count integer not null default 0 check (message_count >= 0),

    -- Same evidence discipline as a thread. An episode summary is a model's
    -- reading of a conversation; saying so is what stops it being quoted back
    -- as though it were a record.
    evidence_type text not null default 'inferred' check (evidence_type in (
        'stated', 'inferred', 'operator'
    )),

    -- DELIBERATELY ABSENT: amount, price, purchased, order id. The review says
    -- an episode is "never proof of payment". ppv_deliveries is the authority
    -- on money and must stay the only one; a summary that could be read as a
    -- receipt is how a model comes to believe a purchase that did not happen.

    dedupe_key text not null,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),

    unique (creator_id, fan_id, dedupe_key),
    constraint conversation_episodes_range check (last_message_at >= first_message_at)
);

create index if not exists conversation_episodes_recent_idx
    on public.conversation_episodes (creator_id, fan_id, last_message_at desc);

alter table public.conversation_open_threads enable row level security;
alter table public.conversation_episodes enable row level security;

comment on table public.conversation_open_threads is
    'Unfinished business a conversation is carrying: unanswered questions, unkept promises, deferred topics, unresolved complaints, corrections. Persists until fulfilled, cancelled, superseded or expired.';
comment on table public.conversation_episodes is
    'Compact summaries of completed stretches of conversation, with source ranges. Never proof of payment: ppv_deliveries is the authority on money.';
