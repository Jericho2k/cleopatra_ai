-- Incremental offers: one next unlock at a time, no quick/full menu.
--
-- WHAT CHANGED IN THE PRODUCT
-- ---------------------------
-- The commercial engine used to build TWO packages, show the fan both prices,
-- wait for him to choose one, confirm the choice, and only then start sending
-- the parts he had already paid for. That is a checkout, and it is what
-- produced "quick $60 or full $140?" followed by "so which one?" followed by
-- "want part 1?".
--
-- It is now one offer at a time: the next thing and its price, purchase-gated,
-- with the progression after it kept internal and never quoted. The columns
-- below existed only to run the two-branch menu, so they are dropped rather
-- than left as switches with nothing behind them.
--
-- SAFETY
-- ------
-- Idempotent and safe to re-run. Every statement is guarded, and the fan-state
-- columns are ADDED and BACKFILLED before the old ones are dropped, so an
-- in-flight conversation keeps the offer it was shown across the deploy.
-- db/commercial_queries.py additionally reads the legacy columns when they are
-- still present, so the application is correct both before and after this runs.

begin;

-- --------------------------------------------------------------------------
-- creator_commercial_policies: one content budget, not two, and no step range
-- --------------------------------------------------------------------------

alter table public.creator_commercial_policies
    add column if not exists next_offer_target_cents integer not null default 2500;

-- Carry the old opener budget over, since that is what sized the FIRST thing
-- the fan was shown and the next unlock plays the same role.
do $$
begin
    if exists (
        select 1 from information_schema.columns
        where table_schema = 'public'
          and table_name = 'creator_commercial_policies'
          and column_name = 'quick_package_target_cents'
    ) then
        update public.creator_commercial_policies
           set next_offer_target_cents = coalesce(quick_package_target_cents, 2500)
         where next_offer_target_cents = 2500;
    end if;
end $$;

-- offer_two_packages      — the menu toggle itself.
-- quick/full target cents — the two content budgets the two branches used.
-- session_min/max_steps   — how many parts one prepaid session was split into.
--                           A sold unlock is now exactly one step, decided at
--                           the moment it is offered.
alter table public.creator_commercial_policies
    drop column if exists offer_two_packages,
    drop column if exists quick_package_target_cents,
    drop column if exists full_package_target_cents,
    drop column if exists session_min_steps,
    drop column if exists session_max_steps;

-- --------------------------------------------------------------------------
-- fan_commercial_states: one pending offer, not an ordered snapshot of two
-- --------------------------------------------------------------------------

alter table public.fan_commercial_states
    add column if not exists pending_offer jsonb,
    add column if not exists accepted_offer_id text,
    add column if not exists accepted_offer_set_id text,
    add column if not exists accepted_offer_label text,
    add column if not exists accepted_offer_price_cents integer,
    add column if not exists last_session_offer_id text;

-- Backfill from the old shape so a fan who is mid-offer at deploy time is still
-- held to the exact thing he was shown. The first entry of the old ordered
-- snapshot is the offer that was on the table.
do $$
begin
    if exists (
        select 1 from information_schema.columns
        where table_schema = 'public'
          and table_name = 'fan_commercial_states'
          and column_name = 'offered_packages'
    ) then
        update public.fan_commercial_states
           set pending_offer = jsonb_build_object(
                   'offer_id', coalesce(offered_packages -> 0 ->> 'package_id',
                                        'offer:' || coalesce(offered_packages -> 0 ->> 'set_id', '')),
                   'label', coalesce(offered_packages -> 0 ->> 'label', 'private photo set'),
                   'price_cents', coalesce((offered_packages -> 0 ->> 'price_cents')::int, 0),
                   'set_id', coalesce(
                       offered_packages -> 0 -> 'set_ids' ->> 0,
                       offered_packages -> 0 ->> 'set_id'
                   ),
                   'experience', offered_packages -> 0 ->> 'experience',
                   'legal_description', offered_packages -> 0 ->> 'legal_description',
                   'media_count', coalesce((offered_packages -> 0 ->> 'media_count')::int, 0),
                   'asset_type', coalesce(offered_packages -> 0 -> 'asset_types' ->> 0, 'photo_set'),
                   'content_floor_cents', (offered_packages -> 0 ->> 'content_floor_cents')::int,
                   'content_ceiling_cents', (offered_packages -> 0 ->> 'content_ceiling_cents')::int,
                   'price_reason_codes', coalesce(offered_packages -> 0 -> 'price_reason_codes', '[]'::jsonb)
               )
         where pending_offer is null
           and jsonb_typeof(offered_packages) = 'array'
           and jsonb_array_length(offered_packages) > 0
           and coalesce(
                   offered_packages -> 0 -> 'set_ids' ->> 0,
                   offered_packages -> 0 ->> 'set_id'
               ) is not null;
    end if;
end $$;

do $$
begin
    if exists (
        select 1 from information_schema.columns
        where table_schema = 'public'
          and table_name = 'fan_commercial_states'
          and column_name = 'selected_package_id'
    ) then
        update public.fan_commercial_states
           set accepted_offer_id = coalesce(accepted_offer_id, selected_package_id),
               accepted_offer_label = coalesce(accepted_offer_label, selected_package_label),
               accepted_offer_price_cents = coalesce(
                   accepted_offer_price_cents, selected_package_price_cents
               );
    end if;
    if exists (
        select 1 from information_schema.columns
        where table_schema = 'public'
          and table_name = 'fan_commercial_states'
          and column_name = 'selected_package_set_id'
    ) then
        update public.fan_commercial_states
           set accepted_offer_set_id = coalesce(
                   accepted_offer_set_id, selected_package_set_id
               );
    end if;
    if exists (
        select 1 from information_schema.columns
        where table_schema = 'public'
          and table_name = 'fan_commercial_states'
          and column_name = 'last_session_package_id'
    ) then
        update public.fan_commercial_states
           set last_session_offer_id = coalesce(
                   last_session_offer_id, last_session_package_id
               );
    end if;
end $$;

alter table public.fan_commercial_states
    drop column if exists offered_packages,
    drop column if exists selected_package_id,
    drop column if exists selected_package_set_id,
    drop column if exists selected_package_set_ids,
    drop column if exists selected_package_label,
    drop column if exists selected_package_price_cents,
    drop column if exists last_session_package_id;

commit;
