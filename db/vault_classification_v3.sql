-- Hybrid vault classification: adult classifier + vision model (classifier v3).
-- Apply after agency_operability_v1.sql.
--
-- Explicitness now comes from a purpose-built adult classifier and the
-- semantics from a vision model. These columns keep both readings plus the
-- reconciliation outcome, so an operator can see which items the two sources
-- disagreed about instead of only seeing the winner.
--
-- The classification_version / classification_source / classification_model /
-- classification_confidence / classified_at columns are what the dashboard's
-- "CLASSIFICATION QUALITY" panel already reads. They are created here with
-- "if not exists" so this migration is safe whether or not they were added
-- out of band. classification_confidence stays numeric: the dashboard renders
-- it as a percentage.

alter table public.creator_vault_media
    add column if not exists classification_version smallint not null default 0,
    add column if not exists classification_source text not null default '',
    add column if not exists classification_model text not null default '',
    add column if not exists classification_confidence numeric null,
    add column if not exists classified_at timestamptz null,
    -- high | low | vision_only | classifier_only | unavailable
    add column if not exists classification_evidence text not null default 'unavailable',
    add column if not exists classification_needs_review boolean not null default false,
    -- '' | classifier_above_category | classifier_below_category
    -- | classifier_unconfident | vision_unavailable | no_evidence | error
    add column if not exists classification_disagreement text not null default '',
    add column if not exists classifier_explicitness smallint null,
    add column if not exists vision_explicitness smallint null,
    -- Raw per-class scores, kept so a mapping change can be re-evaluated
    -- against already-classified media without re-billing the classifier.
    add column if not exists classifier_scores jsonb not null default '{}'::jsonb,
    -- 0 for a failed fetch, 1 for a photo, N for a sampled video.
    add column if not exists analyzed_frame_count smallint not null default 0;

-- The review queue and the staleness sweep are the only hot reads here.
create index if not exists creator_vault_media_needs_review_idx
    on public.creator_vault_media (creator_id)
    where classification_needs_review;

create index if not exists creator_vault_media_classification_version_idx
    on public.creator_vault_media (creator_id, classification_version);

-- Counts the media an upgrade run would touch.
--
-- "Stale" is anything below the current classifier version that already has a
-- category: those rows were written by a classifier that rated explicitness
-- with a general vision model and never looked at a video's frames, so their
-- category — and therefore their price band — cannot be trusted. Uncategorized
-- rows are excluded because the ordinary initial/new runs already own them.
--
-- vault_sets.media_ids holds fansly_media_id values, which is how an item is
-- tied back to an approved set.
create or replace function public.vault_classification_staleness(
    p_creator_id uuid,
    p_current_version integer
)
returns table (stale integer, stale_approved integer)
language sql
stable
security definer
set search_path = public
as $$
    with stale_media as (
        select fansly_media_id
          from public.creator_vault_media
         where creator_id = p_creator_id
           and content_category is not null
           and content_category <> ''
           and coalesce(classification_version, 0) < p_current_version
    ),
    approved_media as (
        select distinct jsonb_array_elements_text(coalesce(media_ids, '[]'::jsonb)) as fansly_media_id
          from public.vault_sets
         where creator_id = p_creator_id
           and status = 'approved'
    )
    select
        (select count(*)::integer from stale_media),
        (
            select count(*)::integer
              from stale_media s
             where exists (
                 select 1 from approved_media a
                  where a.fansly_media_id = s.fansly_media_id
             )
        );
$$;

revoke all on function public.vault_classification_staleness(uuid, integer)
    from public, anon, authenticated;
grant execute on function public.vault_classification_staleness(uuid, integer)
    to service_role;
