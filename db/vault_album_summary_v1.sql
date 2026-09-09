-- Album counts for the vault browser, without shipping the vault to the browser.
--
-- Audit reference: FE-003.
--
-- The dashboard used to page the ENTIRE creator_vault_media table into React
-- state just to know which albums exist and how many items each holds — 26
-- columns per row, ~75 MB of JSON and 50 sequential requests at 50,000 items,
-- repeated in full whenever one row changed. The counts are an aggregate; they
-- should cost one round trip and transfer no rows.
--
-- SECURITY INVOKER on purpose. This runs with the caller's own privileges, so
-- the row-level security already on creator_vault_media applies exactly as it
-- does to the direct select this replaces. It grants the browser nothing it did
-- not already have, and it deliberately does not paper over SEC-001, which is a
-- separate piece of work.
--
-- The dashboard falls back to scanning album_title when this function is
-- absent, so it may be applied before or after the frontend deploy.

create or replace function public.vault_album_summary(p_creator_id uuid)
returns table (album_title text, item_count bigint)
language sql
stable
security invoker
set search_path = public
as $$
    select
        coalesce(nullif(btrim(m.album_title), ''), 'Uncategorized') as album_title,
        count(*) as item_count
      from public.creator_vault_media as m
     where m.creator_id = p_creator_id
     group by 1
     order by 1;
$$;

grant execute on function public.vault_album_summary(uuid)
    to authenticated, service_role;

-- The grouping and the per-album page the browser then requests share this.
create index if not exists creator_vault_media_album_idx
    on public.creator_vault_media (creator_id, album_title, id);
