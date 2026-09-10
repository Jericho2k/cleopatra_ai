-- Least-privilege browser access.
--
-- Audit reference: SEC-001. Idempotent. MUST run immediately after
-- db/tenant_isolation_v1.sql, which it deliberately narrows.
--
-- The problem
-- -----------
-- tenant_isolation_v1 closed the tenant boundary correctly: an operator for
-- agency A cannot see or touch agency B's rows. What it did not do is limit
-- what an operator may do INSIDE their own tenant. It creates
--
--     create policy ... for ALL to authenticated
--
-- on every creator-owned table. Tenancy is not authorization. A legitimate
-- operator holding a browser Supabase client — which is just an HTTP API and a
-- JWT they already have — could therefore INSERT, UPDATE and DELETE rows on
-- tables the product never edits from the browser: scheduled_actions,
-- ppv_deliveries, the PPV approval queue, commercial state, delivery journals.
-- That is not a hypothetical escalation; it is a bypass of every business rule
-- the backend enforces. Someone could mark a PPV purchased without a purchase,
-- cancel or forge durable work, or move a fan's commercial state directly.
--
-- The shape of the fix
-- --------------------
-- Two mechanisms, because they answer different questions:
--
--   * RLS policies decide WHICH ROWS. Split per operation, so SELECT stays open
--     (realtime subscriptions need it) while INSERT/UPDATE/DELETE exist only
--     where the product actually writes.
--   * Column GRANTs decide WHICH COLUMNS. PostgreSQL has no per-column RLS, and
--     "the operator may edit this fan's hobbies" must not also mean "the
--     operator may edit this fan's total_spent". Only GRANT can express that.
--
-- Default deny, explicit allow. Every creator-owned table gets SELECT. A table
-- gets a write policy only by appearing in the allowlist below, and that
-- allowlist was built by inventorying every .from(...).insert/update/upsert/
-- delete call in the dashboard — not by guessing what might be needed.
--
-- What is deliberately NOT writable from a browser after this:
--   scheduled_actions      durable work; forging or cancelling it is severe
--   ppv_deliveries         the PPV ledger and its single-flight invariant
--   ppv_approval_requests  the approval workflow itself
--   messages               conversation history; sends go through the backend
--   suggestions            model output
--   platform_purchase_events, and every commercial/lifecycle table
--
-- Re-running
-- ----------
-- tenant_isolation_v1 drops every policy on each table it discovers before
-- recreating its own, so re-running it ALONE silently restores FOR ALL. The two
-- files are a pair: run tenant_isolation_v1 then this one, always, in that
-- order. db/MIGRATIONS.md says so and scripts/production_preflight.py checks
-- for it.

-- ---------------------------------------------------------------------------
-- 1. Reset table-level privileges for the browser roles.
--
-- Supabase ships broad grants to anon and authenticated on public. Those are
-- what the policies above were riding on, so narrowing the policies without
-- narrowing the grants would leave column-level access wide open.
--
-- anon gets nothing at all: every dashboard read happens after login, when the
-- role is `authenticated`. Nothing in the product reads as anon.
-- ---------------------------------------------------------------------------
do $$
declare
    target_table text;
begin
    for target_table in
        select t.table_name
          from information_schema.tables t
         where t.table_schema = 'public'
           and t.table_type = 'BASE TABLE'
    loop
        execute format(
            'revoke all on public.%I from anon, authenticated',
            target_table
        );
    end loop;
end
$$;

-- ---------------------------------------------------------------------------
-- 2. SELECT everywhere a tenant policy already exists.
--
-- Read access is what realtime subscriptions and the whole dashboard depend on,
-- and it is already bounded to the operator's own creators by the tenancy
-- predicate. Narrowing reads is not what SEC-001 is about and would break the
-- product for no security gain.
-- ---------------------------------------------------------------------------
do $$
declare
    target_table text;
    policy_row record;
begin
    for target_table in
        select c.table_name
          from information_schema.columns c
          join information_schema.tables t
            on t.table_schema = c.table_schema
           and t.table_name = c.table_name
         where c.table_schema = 'public'
           and t.table_type = 'BASE TABLE'
           and c.column_name in ('creator_id', 'fan_id')
           and c.table_name <> 'chatter_creators'
         group by c.table_name
    loop
        -- Replace the FOR ALL policy tenant_isolation_v1 created with a
        -- SELECT-only one carrying the SAME predicate. The predicate is read
        -- back from the existing policy rather than rebuilt, so this file can
        -- never disagree with the tenancy rules about which rows belong to whom.
        for policy_row in
            select policyname, cmd, qual
              from pg_policies
             where schemaname = 'public'
               and tablename = target_table
               and cmd = 'ALL'
        loop
            execute format(
                'drop policy %I on public.%I',
                policy_row.policyname, target_table
            );
            execute format(
                'create policy %I on public.%I for select to authenticated using (%s)',
                policy_row.policyname || '_read',
                target_table,
                policy_row.qual
            );
        end loop;

        execute format('grant select on public.%I to authenticated', target_table);
    end loop;
end
$$;

-- creators is discovered by neither loop above (its key is `id`), so it is
-- handled explicitly and identically.
do $$
declare
    policy_row record;
begin
    for policy_row in
        select policyname, qual
          from pg_policies
         where schemaname = 'public'
           and tablename = 'creators'
           and cmd = 'ALL'
    loop
        execute format('drop policy %I on public.creators', policy_row.policyname);
        execute format(
            'create policy %I on public.creators for select to authenticated using (%s)',
            policy_row.policyname || '_read',
            policy_row.qual
        );
    end loop;
end
$$;
grant select on public.creators to authenticated;

-- chatter_creators already had a SELECT-only policy. It only needs its grant
-- back after the blanket revoke.
grant select on public.chatter_creators to authenticated;

-- Creator-owned VIEWS (fan_conversation_summaries) are security_invoker thanks
-- to tenant_isolation_v1, so they inherit the policies above. They still need
-- their own SELECT grant, which the revoke loop did not touch (it only walks
-- base tables) but which is restated here so a fresh database is correct.
do $$
declare
    target_view text;
begin
    for target_view in
        select c.table_name
          from information_schema.columns c
          join information_schema.tables t
            on t.table_schema = c.table_schema
           and t.table_name = c.table_name
         where c.table_schema = 'public'
           and t.table_type = 'VIEW'
           and c.column_name in ('creator_id', 'fan_id')
         group by c.table_name
    loop
        execute format('revoke all on public.%I from anon', target_view);
        execute format('grant select on public.%I to authenticated', target_view);
    end loop;
end
$$;

-- ---------------------------------------------------------------------------
-- 3. The allowlist: writes the dashboard actually performs.
--
-- Each block names the UI that needs it. If a block has no UI behind it any
-- more, delete the block — that is the point of naming them.
-- ---------------------------------------------------------------------------

-- creators — Settings: persona, sleep hours, spend caps, crisis policy.
--
-- No INSERT and no DELETE: creating and disconnecting a creator goes through
-- the backend, which also provisions the API Fansly binding and the session.
-- The column list is what Settings edits and nothing else: notably NOT
-- apifansly_account_id or fansly_account_id, which would let an operator
-- repoint a creator at another account, and NOT the vault/lists sync
-- bookkeeping columns, which would let them forge sync state.
do $$
begin
    if to_regclass('public.creators') is null then
        return;
    end if;
    drop policy if exists tenant_creator_settings_write on public.creators;
    create policy tenant_creator_settings_write
    on public.creators
    for update
    to authenticated
    using (public.can_access_creator(id::text))
    with check (public.can_access_creator(id::text));
end
$$;


-- fans — the operator's own notes on a fan (FanPanel "FAN DETAILS").
--
-- Deliberately NOT total_spent, spend_tier, sales_log, pending_ppv_check,
-- needs_human_review, auto_mode or active_session: those are commercial and
-- automation state the backend owns. Freezing a fan for review, toggling Auto,
-- and recording spend all have backend routes with rules attached.
do $$
begin
    if to_regclass('public.fans') is null then
        return;
    end if;
    drop policy if exists tenant_fan_notes_write on public.fans;
    create policy tenant_fan_notes_write
    on public.fans
    for update
    to authenticated
    using (public.can_access_creator(creator_id::text))
    with check (public.can_access_creator(creator_id::text));
end
$$;


-- blocked_words — Settings: add and remove a word. Whole rows, no update.
do $$
begin
    if to_regclass('public.blocked_words') is null then
        return;
    end if;
    drop policy if exists tenant_blocked_words_insert on public.blocked_words;
    drop policy if exists tenant_blocked_words_delete on public.blocked_words;
    create policy tenant_blocked_words_insert
    on public.blocked_words
    for insert
    to authenticated
    with check (public.can_access_creator(creator_id::text));
    create policy tenant_blocked_words_delete
    on public.blocked_words
    for delete
    to authenticated
    using (public.can_access_creator(creator_id::text));
    grant insert, delete on public.blocked_words to authenticated;
end
$$;

-- fan_lists — Sidebar list management, for Cleopatra's OWN lists only.
--
-- source = 'local' in every policy. A Fansly-mirrored list is not the
-- operator's to rename or delete: the sync would recreate it, and Auto Audience
-- rules reference fan_lists.id, so dropping a mirror silently changes which
-- fans those rules select (which is why fansly_lists_v1 archives instead of
-- deleting). The UI already treats mirrors as read-only — isEditableList() in
-- lib/fanLists.ts — so this makes an existing product rule actually true rather
-- than changing behaviour.
--
-- The column grant is what stops a local list being turned into a forged
-- mirror, or a mirror's identity being hijacked, by writing source or
-- external_list_id directly.
do $$
begin
    if to_regclass('public.fan_lists') is null then
        return;
    end if;
    drop policy if exists tenant_fan_lists_insert on public.fan_lists;
    drop policy if exists tenant_fan_lists_update on public.fan_lists;
    drop policy if exists tenant_fan_lists_delete on public.fan_lists;

    create policy tenant_fan_lists_insert
    on public.fan_lists
    for insert
    to authenticated
    with check (
        public.can_access_creator(creator_id::text)
        and coalesce(source, 'local') = 'local'
    );

    create policy tenant_fan_lists_update
    on public.fan_lists
    for update
    to authenticated
    using (
        public.can_access_creator(creator_id::text)
        and coalesce(source, 'local') = 'local'
    )
    with check (
        public.can_access_creator(creator_id::text)
        and coalesce(source, 'local') = 'local'
    );

    create policy tenant_fan_lists_delete
    on public.fan_lists
    for delete
    to authenticated
    using (
        public.can_access_creator(creator_id::text)
        and coalesce(source, 'local') = 'local'
    );

    grant insert, delete on public.fan_lists to authenticated;
end
$$;


-- fan_list_members — adding and removing a fan from a local list.
--
-- source = 'local' for the same reason as above: reconciliation owns the
-- membership of a mirrored list, and an operator hand-editing it would have
-- their change silently reverted on the next sync. UPDATE is granted because
-- the UI upserts (INSERT ... ON CONFLICT DO UPDATE), not because anything
-- edits an existing membership row.
do $$
begin
    if to_regclass('public.fan_list_members') is null then
        return;
    end if;
    drop policy if exists tenant_fan_list_members_insert on public.fan_list_members;
    drop policy if exists tenant_fan_list_members_update on public.fan_list_members;
    drop policy if exists tenant_fan_list_members_delete on public.fan_list_members;

    create policy tenant_fan_list_members_insert
    on public.fan_list_members
    for insert
    to authenticated
    with check (
        public.can_access_fan(fan_id::text)
        and public.can_access_fan_list(list_id::text)
        and coalesce(source, 'local') = 'local'
    );

    create policy tenant_fan_list_members_update
    on public.fan_list_members
    for update
    to authenticated
    using (
        public.can_access_fan(fan_id::text)
        and public.can_access_fan_list(list_id::text)
        and coalesce(source, 'local') = 'local'
    )
    with check (
        public.can_access_fan(fan_id::text)
        and public.can_access_fan_list(list_id::text)
        and coalesce(source, 'local') = 'local'
    );

    create policy tenant_fan_list_members_delete
    on public.fan_list_members
    for delete
    to authenticated
    using (
        public.can_access_fan(fan_id::text)
        and public.can_access_fan_list(list_id::text)
        and coalesce(source, 'local') = 'local'
    );

    grant insert, update, delete on public.fan_list_members to authenticated;
end
$$;

-- vault_sets — the Scripts page: an operator curating sellable sets by hand.
-- Genuinely operator-owned content, so whole-row writes are correct here.
do $$
begin
    if to_regclass('public.vault_sets') is null then
        return;
    end if;
    drop policy if exists tenant_vault_sets_insert on public.vault_sets;
    drop policy if exists tenant_vault_sets_update on public.vault_sets;
    drop policy if exists tenant_vault_sets_delete on public.vault_sets;

    create policy tenant_vault_sets_insert
    on public.vault_sets
    for insert
    to authenticated
    with check (public.can_access_creator(creator_id::text));

    create policy tenant_vault_sets_update
    on public.vault_sets
    for update
    to authenticated
    using (public.can_access_creator(creator_id::text))
    with check (public.can_access_creator(creator_id::text));

    create policy tenant_vault_sets_delete
    on public.vault_sets
    for delete
    to authenticated
    using (public.can_access_creator(creator_id::text));

    grant insert, update, delete on public.vault_sets to authenticated;
end
$$;

-- creator_vault_media — the Vault preview panel: correcting what the classifier
-- decided about a piece of media.
--
-- Deliberately NOT url, fansly_media_id, album_title or filename: those are the
-- vault sync's record of what exists on the platform, and editing them locally
-- would desynchronise the mirror without changing anything on Fansly.
do $$
begin
    if to_regclass('public.creator_vault_media') is null then
        return;
    end if;
    drop policy if exists tenant_vault_media_update on public.creator_vault_media;
    create policy tenant_vault_media_update
    on public.creator_vault_media
    for update
    to authenticated
    using (public.can_access_creator(creator_id::text))
    with check (public.can_access_creator(creator_id::text));
end
$$;


-- ---------------------------------------------------------------------------
-- Column-level privileges.
--
-- PostgreSQL has no per-column RLS, so this is the only mechanism that can say
-- "an operator may edit this fan's hobbies but not their total_spent". One
-- table drives all of it, so the allowlist is readable as a single matrix
-- rather than scattered through the file.
--
-- A column that does not exist is skipped rather than failing: these migrations
-- run against databases at slightly different points in their history, and a
-- missing optional column must not stop the security narrowing from applying.
-- ---------------------------------------------------------------------------
do $$
declare
    grants constant text[][] := array[
        -- table,                privilege,  column
        -- creators: what Settings edits. NOT apifansly_account_id or
        -- fansly_account_id (repointing a creator at another account), and NOT
        -- the sync bookkeeping columns (forging sync state).
        ['creators', 'update', 'persona'],
        ['creators', 'update', 'sleep_hours_start'],
        ['creators', 'update', 'sleep_hours_end'],
        ['creators', 'update', 'caps_enabled'],
        ['creators', 'update', 'max_ppv_per_fan_per_day'],
        ['creators', 'update', 'max_spend_per_fan_per_day'],
        ['creators', 'update', 'crisis_policy'],
        ['creators', 'update', 'whale_handoff_threshold'],

        -- fans: the operator's own notes. NOT total_spent, spend_tier,
        -- sales_log, pending_ppv_check, needs_human_review, auto_mode or
        -- active_session -- commercial and automation state the backend owns.
        ['fans', 'update', 'age'],
        ['fans', 'update', 'payday'],
        ['fans', 'update', 'hobbies'],
        ['fans', 'update', 'relationship_status'],
        ['fans', 'update', 'member_note'],
        ['fans', 'update', 'notes'],
        ['fans', 'update', 'preferences'],

        -- fan_lists: name and targeting flag only. Withholding source and
        -- external_list_id is what stops a local list being turned into a
        -- forged Fansly mirror, or a real mirror's identity being hijacked.
        ['fan_lists', 'update', 'name'],
        ['fan_lists', 'update', 'color'],
        ['fan_lists', 'update', 'exclude_from_auto'],
        -- The insertable set differs from the updatable one: creator_id is set
        -- once at creation and never edited afterwards.
        ['fan_lists', 'insert', 'creator_id'],
        ['fan_lists', 'insert', 'name'],
        ['fan_lists', 'insert', 'color'],
        ['fan_lists', 'insert', 'exclude_from_auto'],

        -- creator_vault_media: correcting the classifier. NOT url,
        -- fansly_media_id, album_title or filename -- the vault sync's record of
        -- what exists on the platform, which editing locally would only
        -- desynchronise.
        ['creator_vault_media', 'update', 'content_category'],
        ['creator_vault_media', 'update', 'ai_description'],
        ['creator_vault_media', 'update', 'price_min'],
        ['creator_vault_media', 'update', 'price_max'],
        ['creator_vault_media', 'update', 'scene_location'],
        ['creator_vault_media', 'update', 'scene_outfit'],
        ['creator_vault_media', 'update', 'scene_lighting'],
        ['creator_vault_media', 'update', 'scene_id'],
        ['creator_vault_media', 'update', 'classification_version'],
        ['creator_vault_media', 'update', 'classification_source'],
        ['creator_vault_media', 'update', 'classification_confidence'],
        ['creator_vault_media', 'update', 'classified_at']
    ];
    entry text[];
    target_table text;
    privilege text;
    target_column text;
begin
    for index in 1 .. array_length(grants, 1) loop
        entry := grants[index:index][1:3];
        target_table := grants[index][1];
        privilege := grants[index][2];
        target_column := grants[index][3];

        if to_regclass(current_schema() || '.' || quote_ident(target_table)) is null then
            continue;
        end if;
        if not exists (
            select 1
              from information_schema.columns c
             where c.table_schema = current_schema()
               and c.table_name = target_table
               and c.column_name = target_column
        ) then
            raise notice 'skipping %.% (column absent)', target_table, target_column;
            continue;
        end if;

        execute format(
            'grant %s (%I) on public.%I to authenticated',
            privilege, target_column, target_table
        );
    end loop;
end
$$;

-- ---------------------------------------------------------------------------
-- 4. service_role is untouched.
--
-- It carries BYPASSRLS, so every background job — the schedulers, the workers,
-- the webhook handlers, vault sync, list reconciliation — is unaffected by all
-- of the above. Restated here because "did this break the backend" is the first
-- question anyone reading this file will have.
-- ---------------------------------------------------------------------------
do $$
declare
    target_table text;
begin
    for target_table in
        select t.table_name
          from information_schema.tables t
         where t.table_schema = 'public'
           and t.table_type = 'BASE TABLE'
    loop
        execute format(
            'grant all on public.%I to service_role',
            target_table
        );
    end loop;
end
$$;
