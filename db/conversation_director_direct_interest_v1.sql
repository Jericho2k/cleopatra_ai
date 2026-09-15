-- Production drift repair: fan_conversation_directors.direct_interest.
--
-- models/conversation_director.py has computed `direct_interest` since the
-- director shipped, and services/conversation_director.save_conversation_director
-- upserts the WHOLE to_context() dict — so every persisted director write has
-- been sending a column production does not have. PostgREST rejects the row,
-- the service swallows the exception ("persistence failed"), and the director
-- silently degrades to a stateless recompute on every turn.
--
-- conversation_director_v1.sql is deliberately NOT edited to add the column.
-- That file has already been applied to production; editing it would fix a
-- fresh database and leave every existing deployment exactly as broken, which
-- is the drift this migration exists to end.
--
-- Additive, idempotent, and defaulted: `false` is what advance_conversation_director
-- computes for a fan who has not asked for content, so every pre-existing row
-- keeps the meaning it already had.

alter table public.fan_conversation_directors
    add column if not exists direct_interest boolean not null default false;

comment on column public.fan_conversation_directors.direct_interest is
    'True when the fan has already asked for content in as many words, so no warm-up is owed. Written by models/conversation_director.advance_conversation_director.';
