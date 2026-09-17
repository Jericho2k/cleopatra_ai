-- Owner-only diagnostics leave the browser-readable row.
--
-- THE FINDING
-- -----------
-- services/ai_stack_visibility.py redacts model routing out of the responses
-- main.py serves. It does that correctly, and it was never enough, because the
-- dashboard does not only read those responses:
--
--     app/simulator/page.tsx:205   supabase.from('messages').select('*')
--     app/page.tsx:554,588,728,1021        .select('*')
--
-- Those are browser Supabase reads with the operator's own JWT. They return
-- `messages.media_context` verbatim, and db/browser_least_privilege_v1.sql
-- grants `select` on the whole table, every column, to `authenticated`. So an
-- agency operator's browser already held:
--
--   * media_context.reply_provenance.writer.actual.provider / .model — which
--     model actually served the reply;
--   * .writer.attempts[] — the whole fallback ladder, including which provider
--     failed with what;
--   * .writer.requested — what was asked for before the ladder ran;
--   * media_context.ai_stack.provider / .model / .route / .prompt_version —
--     the same routing the registry redaction exists to withhold.
--
-- Realtime carries it too: a subscription on `messages` is the same SELECT
-- privilege, so a new row arrives with the same payload.
--
-- WHY THIS IS A STORAGE CHANGE AND NOT ANOTHER REDACTION
-- -----------------------------------------------------
-- Every response-shaped fix has the same hole: it only covers the responses
-- somebody remembered. The data is sitting in a column the browser is granted,
-- so the only durable boundary is for the data not to be in that column.
--
-- services/reply_provenance.py chose media_context deliberately ("It writes
-- nothing new to the database schema"). That constraint was about avoiding a
-- migration, and it is worth exactly one migration to stop shipping the supply
-- chain to every agency operator's browser. The record is unchanged; only where
-- it lives changes.
--
-- THE MECHANISM
-- -------------
-- `public.owner_only_tables` is a registry, not a convention. Both discovery
-- migrations consult it:
--
--   * tenant_isolation_v1 creates no policy for a registered table;
--   * browser_least_privilege_v1 grants it no SELECT and revokes what it has.
--
-- That matters because both of those files DISCOVER tables by looking for a
-- creator_id or fan_id column. `message_diagnostics` has both. Without the
-- registry, adding this table would have handed `authenticated` a SELECT on it
-- automatically — the migrations would have re-opened the hole this file
-- exists to close, silently, at the next run.
--
-- A future owner-only table is one INSERT away from the same treatment, which
-- is the point: the next person does not have to rediscover any of this.
--
-- Idempotent and additive. Safe to re-run. MUST precede tenant_isolation_v1.

create extension if not exists pgcrypto;

-- ---------------------------------------------------------------------------
-- 1. The registry
-- ---------------------------------------------------------------------------
create table if not exists public.owner_only_tables (
    table_name text primary key,
    -- Why it is owner-only, in one sentence, readable in psql. A registry of
    -- bare table names decays into a list nobody dares change.
    reason text not null,
    registered_at timestamptz not null default now()
);

comment on table public.owner_only_tables is
    'Tables the browser roles may never read. Consulted by tenant_isolation_v1 '
    'and browser_least_privilege_v1, which otherwise discover creator-owned '
    'tables automatically and would grant SELECT to authenticated.';

-- ---------------------------------------------------------------------------
-- 2. The diagnostics themselves
--
-- One row per delivered message part, the same grain as the provenance record
-- it carries: one turn can send several bubbles, and they differ only in
-- `part`. `record` is the document services/reply_provenance.py already built,
-- moved rather than reshaped, plus the routing half of the ai_stack marker.
-- ---------------------------------------------------------------------------
create table if not exists public.message_diagnostics (
    message_id uuid primary key
        references public.messages(id) on delete cascade,
    creator_id uuid not null references public.creators(id) on delete cascade,
    fan_id uuid not null references public.fans(id) on delete cascade,
    -- Denormalised out of `record` so the parts of one turn can be found
    -- without a jsonb scan. Nullable: a row backfilled from an older
    -- media_context may carry only an ai_stack marker and no turn.
    turn_id text null,
    part integer not null default 0,
    record jsonb not null default '{}'::jsonb,
    recorded_at timestamptz not null default now()
);

create index if not exists message_diagnostics_turn_idx
    on public.message_diagnostics (creator_id, turn_id);
create index if not exists message_diagnostics_fan_idx
    on public.message_diagnostics (creator_id, fan_id, recorded_at desc);

insert into public.owner_only_tables (table_name, reason)
values (
    'message_diagnostics',
    'Model routing, fallback attempts and internal trace detail. Owner-only by '
    'the same rule as services/ai_stack_visibility.py: an agency is told which '
    'AI stack answered, never what it routes to.'
)
on conflict (table_name) do update set reason = excluded.reason;

-- ---------------------------------------------------------------------------
-- 3. Backfill: move what is already in the browser-readable column
--
-- Rows written before this migration carry the record in media_context. Moving
-- them is the whole point — a fix that only applies to new messages leaves
-- every existing conversation readable.
--
-- `part` is read back out of the record rather than defaulted, so the bubbles
-- of one historical turn stay distinguishable.
-- ---------------------------------------------------------------------------
insert into public.message_diagnostics (
    message_id, creator_id, fan_id, turn_id, part, record, recorded_at
)
select
    m.id,
    m.creator_id,
    m.fan_id,
    m.media_context -> 'reply_provenance' ->> 'turn_id',
    coalesce(
        nullif(m.media_context -> 'reply_provenance' ->> 'part', '')::integer,
        0
    ),
    jsonb_strip_nulls(
        jsonb_build_object(
            'reply_provenance', m.media_context -> 'reply_provenance',
            'ai_stack', nullif(
                case
                    when jsonb_typeof(m.media_context -> 'ai_stack') = 'object'
                        then (m.media_context -> 'ai_stack') - 'profile'
                    else null
                end,
                '{}'::jsonb
            )
        )
    ),
    m.sent_at
from public.messages m
where jsonb_typeof(m.media_context) = 'object'
  and (
        m.media_context ? 'reply_provenance'
        or (
            jsonb_typeof(m.media_context -> 'ai_stack') = 'object'
            and ((m.media_context -> 'ai_stack') - 'profile') <> '{}'::jsonb
        )
      )
on conflict (message_id) do nothing;

-- Strip the moved keys from the row the browser can read. `ai_stack` keeps
-- `profile` and nothing else: that is the one product-level fact the marker
-- carries, and lib/aiStack.ts renders it.
update public.messages m
   set media_context = (m.media_context - 'reply_provenance')
        || case
               when jsonb_typeof(m.media_context -> 'ai_stack') = 'object'
                   then jsonb_build_object(
                       'ai_stack',
                       jsonb_strip_nulls(
                           jsonb_build_object(
                               'profile', m.media_context -> 'ai_stack' -> 'profile'
                           )
                       )
                   )
               else '{}'::jsonb
           end
 where jsonb_typeof(m.media_context) = 'object'
   and (
         m.media_context ? 'reply_provenance'
         or (
             jsonb_typeof(m.media_context -> 'ai_stack') = 'object'
             and ((m.media_context -> 'ai_stack') - 'profile') <> '{}'::jsonb
         )
       );

-- ---------------------------------------------------------------------------
-- 4. Privileges
--
-- Restated here rather than left to browser_least_privilege_v1 alone, so this
-- file is correct even if it is the only one that runs. RLS is enabled with no
-- policy for `authenticated`: with the grant withheld too, a browser client is
-- refused by both mechanisms independently.
-- ---------------------------------------------------------------------------
alter table public.message_diagnostics enable row level security;
alter table public.owner_only_tables enable row level security;

revoke all on public.message_diagnostics from anon, authenticated;
revoke all on public.owner_only_tables from anon, authenticated;

grant all on public.message_diagnostics to service_role;
grant all on public.owner_only_tables to service_role;
