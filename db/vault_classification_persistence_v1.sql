-- Vault classification persistence, retrieval provenance and media cost.
--
-- Three things this makes possible, none of which the existing columns could
-- express:
--
-- 1. "This classification is FINISHED" versus "this one is partial because a
--    deep video scan was skipped" versus "this one errored". Without a status,
--    a partial result is indistinguishable from a complete one, so either the
--    sync re-spends on everything or it never retries anything.
--
-- 2. "These bytes are the ones that were classified." A signed vault URL
--    rotates constantly and rotating it is not a change of content, so the
--    identity key is derived from the platform media id and MIME type. A row
--    whose key still matches is known-good and is skipped.
--
-- 3. "This classification cost N billed bytes, retrieved this way." Attribution
--    per item, so vault media spend can be traced to a creator and an asset
--    rather than inferred from a process-wide total.
--
-- Additive and idempotent. Every column is nullable or defaulted, so a backend
-- running ahead of this migration behaves exactly as it did before: unknown
-- status reads as "not unfinished", and an absent identity key disables the
-- identity check rather than forcing a re-classification.
--
-- Must precede tenant_isolation_v1.sql, which discovers creator-owned tables at
-- run time.

alter table public.creator_vault_media
    add column if not exists classification_status text null,
    add column if not exists classification_skip_reason text null,
    add column if not exists classification_media_key text null,
    add column if not exists classification_retrieval_method text null,
    add column if not exists classification_media_bytes bigint not null default 0,
    add column if not exists classification_media_credits numeric(12, 3) not null default 0,
    add column if not exists classification_frames_sampled integer not null default 0;

alter table public.creator_vault_media
    drop constraint if exists creator_vault_media_classification_status_check;

alter table public.creator_vault_media
    add constraint creator_vault_media_classification_status_check
    check (
        classification_status is null
        or classification_status in ('complete', 'partial', 'pending', 'error')
    );

alter table public.creator_vault_media
    drop constraint if exists creator_vault_media_classification_media_bytes_check;

alter table public.creator_vault_media
    add constraint creator_vault_media_classification_media_bytes_check
    check (classification_media_bytes >= 0 and classification_media_credits >= 0);

-- The sync's hot question: "which of this creator's rows still need work?"
-- Partial index because the answer is almost always a small minority of a
-- large vault, and a complete row must cost nothing to skip.
create index if not exists creator_vault_media_classification_status_idx
    on public.creator_vault_media (creator_id, classification_status)
    where classification_status is distinct from 'complete';

-- Backfill: anything already carrying a successful classification is complete
-- finished work, and must not be re-analysed merely because this migration
-- introduced a column. Rows with no classification stay NULL, which the
-- application reads as "new".
update public.creator_vault_media
   set classification_status = 'complete'
 where classification_status is null
   and classified_at is not null
   and content_category is not null
   and content_category <> '';

comment on column public.creator_vault_media.classification_status is
'complete | partial | pending | error. Anything other than complete is eligible for retry on an ordinary sync; complete is finished work and is skipped.';
comment on column public.creator_vault_media.classification_skip_reason is
'Stable code for why deeper analysis was not performed, e.g. exceeds_size_limit or unknown_size. Never a CDN or auth detail.';
comment on column public.creator_vault_media.classification_media_key is
'Fingerprint of the platform media identity that was classified. A mismatch means the row now points at different bytes.';
comment on column public.creator_vault_media.classification_retrieval_method is
'How the classified pixels were obtained: platform_thumbnail, direct_cdn, direct_video_frames, refreshed_signed_url, or apifansly_media_download.';
comment on column public.creator_vault_media.classification_media_bytes is
'Bytes transferred through the billed API Fansly media proxy for this item. Zero for every free retrieval path.';
comment on column public.creator_vault_media.classification_media_credits is
'Estimated API Fansly credits for classification_media_bytes, at 2 credits per MB.';
comment on column public.creator_vault_media.classification_frames_sampled is
'How many video keyframes were sampled. Zero for images and thumbnail-only results.';
