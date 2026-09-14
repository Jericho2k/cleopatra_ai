-- Experience Director: persistent conversational scene state, and the
-- experience metadata that makes a media set a BEAT rather than an inventory row.
--
-- Two halves, one migration, because neither is useful alone: the scene state
-- is what remembers which beat a fan is on, and the set metadata is the only
-- approved source of what that beat can be ABOUT. Run after
-- conversation_director_v1.sql and before tenant_isolation_v1.sql.

create extension if not exists pgcrypto;

-- ---------------------------------------------------------------------------
-- 1. The scene.
--
-- Deliberately NOT part of fan_commercial_states or the active_session JSON.
-- A commercial session is one unlock and is over when it is paid; the scene
-- outlives it, which is the entire reason this table exists. Nothing here
-- authorizes anything: no price, no media id the fan has not already unlocked,
-- no session total.
-- ---------------------------------------------------------------------------
create table if not exists public.fan_experience_scenes (
    fan_id uuid primary key references public.fans(id) on delete cascade,
    creator_id uuid not null references public.creators(id) on delete cascade,
    beat text not null default 'SETUP',
    previous_beat text null,
    scene_key text not null default '',
    premise text not null default '',
    last_unlocked_set_id text null,
    last_unlocked_description text not null default '',
    last_fan_reaction text not null default 'NONE',
    reaction_processed boolean not null default false,
    intimacy_level smallint not null default 0,
    tension_level smallint not null default 0,
    open_hook text not null default '',
    desired_direction text not null default '',
    another_unlock_ready boolean not null default false,
    beats_in_scene integer not null default 0,
    turns_since_unlock integer not null default 0,
    unlocks_in_scene integer not null default 0,
    transition_reason text not null default 'new_scene',
    scene_version integer not null default 1,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create index if not exists fan_experience_scene_creator_idx
    on public.fan_experience_scenes (creator_id, beat, updated_at desc);

alter table public.fan_experience_scenes enable row level security;

comment on table public.fan_experience_scenes is
    'Current conversational scene. Decides WHEN something may happen; commercial policy still decides WHETHER and at what price.';
comment on column public.fan_experience_scenes.another_unlock_ready is
    'True only when the scene has produced a natural bridge. A veto that narrows offer discovery; it can never authorize one.';

-- ---------------------------------------------------------------------------
-- 2. Media as beats.
--
-- Generated once during catalog classification (db/queries.propose_sets and
-- propose_video_ppvs) and stored, rather than asked of the live writer. The
-- writer may only describe facts that are grounded in these approved columns
-- and in the media itself.
-- ---------------------------------------------------------------------------
alter table public.vault_sets
    add column if not exists paid_sellable boolean not null default true,
    add column if not exists scene_key text null,
    add column if not exists scene_premise text null,
    add column if not exists intensity_level smallint null,
    add column if not exists reveals text null,
    add column if not exists setup_line text null,
    add column if not exists continuation text null;

comment on column public.vault_sets.paid_sellable is
    'Explicit authority over automatic selling. NULL-free and true by default: false is a deliberate decision by the classifier or an operator, and outranks every inference in models/content_pricing.paid_sellable_block_reason.';
comment on column public.vault_sets.scene_key is
    'Stable identity of the scene this set belongs to, so a later turn can tell a continuation from a new premise.';
comment on column public.vault_sets.scene_premise is
    'What the scene is about, in approved words derived from the classified media.';
comment on column public.vault_sets.intensity_level is
    'Where this set sits on the escalation ladder, 0-5. Derived from the classified explicitness of its own media.';
comment on column public.vault_sets.reveals is
    'What unlocking this set actually shows or advances. The only thing the writer may say it contains.';
comment on column public.vault_sets.setup_line is
    'The approved setup or bridge this set can be led into with.';
comment on column public.vault_sets.continuation is
    'Where this set can honestly lead next. Never shown to the fan as a future price, step or total.';
