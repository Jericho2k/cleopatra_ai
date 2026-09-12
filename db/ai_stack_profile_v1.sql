-- AI Stack Profile selection: persistent creator override + owner-only test-fan override.
--
-- WHY THIS EXISTS
--
-- The whole conversational AI configuration — every stage's provider, model,
-- fallback, prompt version, reasoning setting and generation parameters — is
-- now a named, versioned profile (ai/stack_profiles.py). Two profiles ship:
-- ``cleo_legacy_v1`` (a frozen snapshot of the AI configuration that shipped
-- before the V2 pass) and ``cleo_v2``.
--
-- Selection has to be PERSISTENT and CREATOR-SCOPED rather than a browser or
-- session setting, because Full Auto answers asynchronously from a worker where
-- no browser session exists. A creator answered by a background AUTO_REPLY at
-- 3am must use the same brain the owner selected in the dashboard.
--
-- fans.ai_stack_profile exists for the simulator only. It is honoured by
-- services/ai_stack.py exclusively for fans whose platform_fan_id starts with
-- 'test_', which is the same boundary core/simulation.py already enforces, so
-- the owner can run "Test Fan A -> legacy" against "Test Fan B -> v2" under one
-- creator. A value written onto a real fan row has no effect: the read path
-- re-checks the prefix rather than trusting the column.
--
-- DESTRUCTIVENESS: none. Two nullable columns and two CHECK constraints. NULL
-- everywhere means "no override", which is the pre-migration behaviour.
--
-- Idempotent: safe to re-run.

alter table public.creators
    add column if not exists ai_stack_profile text null;

comment on column public.creators.ai_stack_profile is
'Persistent AI Stack Profile override for this creator (cleo_legacy_v1 | cleo_v2). NULL means use the deployment default from AI_STACK_PROFILE.';

alter table public.fans
    add column if not exists ai_stack_profile text null;

comment on column public.fans.ai_stack_profile is
'Simulator-only AI Stack Profile override. Honoured ONLY for fans whose platform_fan_id starts with test_; ignored entirely for real fans. NULL means inherit the creator override.';

-- The set of valid identifiers is owned by ai/stack_profiles.py. The database
-- constraint exists so a bad value cannot be written by any route at all, not
-- as a second registry: adding a profile means adding it in both places, which
-- is the intended friction for something that changes what every fan is
-- answered by.
do $$
begin
    if not exists (
        select 1 from pg_constraint where conname = 'creators_ai_stack_profile_known'
    ) then
        alter table public.creators
            add constraint creators_ai_stack_profile_known
            check (ai_stack_profile is null
                   or ai_stack_profile in ('cleo_legacy_v1', 'cleo_v2'))
            not valid;
    end if;

    if not exists (
        select 1 from pg_constraint where conname = 'fans_ai_stack_profile_known'
    ) then
        alter table public.fans
            add constraint fans_ai_stack_profile_known
            check (ai_stack_profile is null
                   or ai_stack_profile in ('cleo_legacy_v1', 'cleo_v2'))
            not valid;
    end if;
end
$$;

-- NOT VALID on purpose: the constraint applies to every future write without
-- taking a full scan of a large fans table at migration time. Both are no-ops
-- on a database where the column has only just been added, so validating them
-- is cheap whenever convenient:
--
--   alter table public.creators validate constraint creators_ai_stack_profile_known;
--   alter table public.fans validate constraint fans_ai_stack_profile_known;
