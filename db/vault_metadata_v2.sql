-- Cleopatra vault metadata v2
--
-- Apply before deploying the matching backend/dashboard commits. Existing
-- classifications become version 0 and are eligible for one explicit upgrade;
-- normal syncs still process only new uncategorized media.

alter table public.creator_vault_media
    add column if not exists classification_version integer not null default 0,
    add column if not exists classification_model text null,
    add column if not exists classification_source text null,
    add column if not exists classification_confidence numeric(4, 3) null,
    add column if not exists classification_metadata jsonb not null default '{}'::jsonb,
    add column if not exists classified_at timestamptz null;

alter table public.creator_vault_media
    drop constraint if exists creator_vault_media_classification_confidence_check;

alter table public.creator_vault_media
    add constraint creator_vault_media_classification_confidence_check
    check (
        classification_confidence is null
        or classification_confidence between 0 and 1
    );

create index if not exists creator_vault_media_classifier_version_idx
    on public.creator_vault_media (creator_id, classification_version);

alter table public.vault_sets
    add column if not exists description text null,
    add column if not exists metadata_version integer not null default 0;

comment on column public.creator_vault_media.classification_version is
'Version of Cleopatra structured vault metadata. Zero denotes legacy/unversioned data.';
comment on column public.creator_vault_media.classification_source is
'Evidence inspected by the classifier: image, video_thumbnail, video_frames, or filename_album.';
comment on column public.creator_vault_media.classification_metadata is
'Provider-neutral structured visual facts used for semantic package matching.';
comment on column public.vault_sets.description is
'Detailed approved semantic description aggregated from the exact media in this set.';

