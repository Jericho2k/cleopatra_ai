-- Register ``cleo_v3`` as a selectable AI Stack Profile.
--
-- WHY THIS EXISTS
--
-- db/ai_stack_profile_v1.sql added creators.ai_stack_profile and
-- fans.ai_stack_profile with a CHECK listing the profiles that existed then:
-- ``cleo_legacy_v1`` and ``cleo_v2``. That constraint is deliberate friction —
-- the set of valid identifiers is owned by ai/stack_profiles.py, and the
-- database refuses anything else so a bad value cannot be written by any route
-- at all. The friction is the point, and this is it being paid: a third
-- profile exists in the registry, so the constraint has to learn about it or
-- the owner cannot select it for a creator or a test fan.
--
-- WHAT cleo_v3 IS
--
-- V2's exact model routing with a much smaller writer prompt (``writer_v3``):
-- Full Auto asks the writer for ONE reply instead of three options, the
-- deterministic bubble-count policy is bypassed, the creator no longer mirrors
-- the fan's typing or claims to be his favourite, and ordinary personal facts
-- she improvises are persisted into the existing creators.legend. Nothing about
-- the commercial engine, pricing, inventory or simulation isolation changes.
--
-- DESTRUCTIVENESS: none. It drops and recreates two CHECK constraints with a
-- strictly wider set of accepted values. No column, row or index is touched,
-- and every value that satisfied the old constraint satisfies the new one.
--
-- Idempotent: safe to re-run.

comment on column public.creators.ai_stack_profile is
'Persistent AI Stack Profile override for this creator (cleo_legacy_v1 | cleo_v2 | cleo_v3). NULL means use the deployment default from AI_STACK_PROFILE.';

comment on column public.fans.ai_stack_profile is
'Simulator-only AI Stack Profile override (cleo_legacy_v1 | cleo_v2 | cleo_v3). Honoured ONLY for fans whose platform_fan_id starts with test_; ignored entirely for real fans. NULL means inherit the creator override.';

alter table public.creators
    drop constraint if exists creators_ai_stack_profile_known;

alter table public.creators
    add constraint creators_ai_stack_profile_known
    check (ai_stack_profile is null
           or ai_stack_profile in ('cleo_legacy_v1', 'cleo_v2', 'cleo_v3'))
    not valid;

alter table public.fans
    drop constraint if exists fans_ai_stack_profile_known;

alter table public.fans
    add constraint fans_ai_stack_profile_known
    check (ai_stack_profile is null
           or ai_stack_profile in ('cleo_legacy_v1', 'cleo_v2', 'cleo_v3'))
    not valid;

-- NOT VALID for the same reason as in v1: the constraint applies to every
-- future write without taking a full scan of a large fans table at migration
-- time. Validating is cheap whenever convenient, and cannot fail here because
-- the accepted set only grew:
--
--   alter table public.creators validate constraint creators_ai_stack_profile_known;
--   alter table public.fans validate constraint fans_ai_stack_profile_known;
