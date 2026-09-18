-- Selectable conversational runtime. NULL preserves the legacy path.
--
-- Creator overrides are persistent because Auto and scheduled work run without
-- a browser session. Fan overrides are read only for platform_fan_id LIKE
-- 'test_%' by services/conversation_core.py; the database column alone never
-- makes a real fan eligible for the replacement runtime.

alter table public.creators
    add column if not exists conversation_core text null;

alter table public.fans
    add column if not exists conversation_core text null;

comment on column public.creators.conversation_core is
    'Conversational runtime override. NULL preserves deployment/default routing.';
comment on column public.fans.conversation_core is
    'Simulation-only conversational runtime override; ignored for real fans.';

alter table public.creators
    drop constraint if exists creators_conversation_core_known;
alter table public.creators
    add constraint creators_conversation_core_known
    check (conversation_core is null or conversation_core in ('legacy', 'semantic_v1'))
    not valid;

alter table public.fans
    drop constraint if exists fans_conversation_core_known;
alter table public.fans
    add constraint fans_conversation_core_known
    check (conversation_core is null or conversation_core in ('legacy', 'semantic_v1'))
    not valid;

-- Rollback is data-preserving: set overrides back to NULL. Dropping the columns
-- is intentionally unnecessary, so rolling the backend back cannot lose the
-- selected value an owner may want to restore later.
