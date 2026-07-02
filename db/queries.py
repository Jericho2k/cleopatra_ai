"""All database reads and writes. No AI logic."""

import asyncio
from collections import Counter
from datetime import datetime

from core.supabase import get_supabase
from models.schemas import ExchangeExample, Fan, Message, Persona


def _row_to_fan(row: dict) -> Fan:
    last_active = row.get("last_active")
    if last_active is not None and isinstance(last_active, str):
        last_active = datetime.fromisoformat(last_active.replace("Z", "+00:00"))
    return Fan(
        id=str(row["id"]),
        display_name=row["display_name"],
        auto_mode=row.get("auto_mode"),
        platform_fan_id=str(row["platform_fan_id"]) if row.get("platform_fan_id") is not None else None,
        fansly_group_id=str(row["fansly_group_id"]) if row.get("fansly_group_id") is not None else None,
        total_spent=row.get("total_spent", 0),
        spend_tier=row.get("spend_tier", "cold"),
        last_active=last_active,
        preferences=row.get("preferences") or [],
        notes=row.get("notes", ""),
        member_note=row.get("member_note", ""),
        model_note=row.get("model_note", ""),
        ai_summary=row.get("ai_summary"),
        pre_session_qual=row.get("pre_session_qual"),
    )


def _row_to_message(row: dict) -> Message:
    sent_at = row.get("sent_at")
    if sent_at is not None and isinstance(sent_at, str):
        sent_at = datetime.fromisoformat(sent_at.replace("Z", "+00:00"))
    return Message(
        role=row["role"],
        content=row["content"],
        sent_at=sent_at,
        media_context=row.get("media_context"),
    )


async def get_fan(creator_id: str, platform_fan_id: str) -> Fan | None:
    def _get():
        r = get_supabase().table("fans").select("*").eq("creator_id", creator_id).eq("platform_fan_id", platform_fan_id).execute()
        if not r.data or len(r.data) == 0:
            return None
        return _row_to_fan(r.data[0])

    return await asyncio.to_thread(_get)


async def get_fan_by_id(fan_id: str) -> Fan | None:
    def _get():
        r = get_supabase().table("fans").select("*").eq("id", fan_id).execute()
        if not r.data or len(r.data) == 0:
            return None
        return _row_to_fan(r.data[0])

    return await asyncio.to_thread(_get)


async def create_fan(creator_id: str, platform_fan_id: str, display_name: str) -> Fan:
    def _create():
        db = get_supabase()
        creator = (
            db.table("creators")
            .select("auto_mode_new_fans")
            .eq("id", creator_id)
            .limit(1)
            .execute()
        )
        auto_new = bool((creator.data or [{}])[0].get("auto_mode_new_fans", False))

        row = {
            "creator_id": creator_id,
            "platform_fan_id": platform_fan_id,
            "display_name": display_name,
        }
        # Only set when the toggle is on, so existing global-fallback behavior
        # is untouched when it's off (auto_mode stays NULL → falls back to creator.auto_mode).
        if auto_new:
            row["auto_mode"] = True

        r = db.table("fans").insert(row).execute()
        return _row_to_fan(r.data[0])

    return await asyncio.to_thread(_create)


async def update_fan_spend(fan_id: str, total_spent: int, spend_tier: str) -> None:
    def _update():
        get_supabase().table("fans").update({"total_spent": total_spent, "spend_tier": spend_tier}).eq("id", fan_id).execute()

    await asyncio.to_thread(_update)


async def increment_fan_total_spent(fan_id: str, amount: int) -> None:
    def _update():
        get_supabase().rpc(
            "increment_fan_spent",
            {
                "fan_id_input": fan_id,
                "amount_input": amount,
            },
        ).execute()

    await asyncio.to_thread(_update)


async def get_sent_ppv(fan_id: str) -> list[dict]:
    """Return list of PPV media already sent to this fan with purchase status."""
    def _get():
        r = get_supabase().table("messages") \
            .select("media_context, sent_at") \
            .eq("fan_id", fan_id) \
            .eq("role", "creator") \
            .not_.is_("media_context", "null") \
            .order("sent_at", desc=False) \
            .execute()
        sent = []
        for row in (r.data or []):
            mc = row.get("media_context") or {}
            ppv = mc.get("ppv")
            if ppv and ppv.get("media_id"):
                sent.append({
                    "media_id": ppv["media_id"],
                    "price": ppv.get("price", 0),
                    "purchased": ppv.get("purchased", False),
                    "sent_at": row.get("sent_at", ""),
                })
        return sent
    return await asyncio.to_thread(_get)


async def mark_ppv_purchased(fan_id: str, media_id: str, sent_at: str | None = None) -> bool:
    def _mark() -> bool:
        db = get_supabase()
        r = db.table("messages").select("id, media_context, sent_at") \
            .eq("fan_id", fan_id).eq("role", "creator") \
            .not_.is_("media_context", "null").order("sent_at", desc=True).execute()
        for row in (r.data or []):
            mc = row.get("media_context") or {}
            ppv = mc.get("ppv")
            if not ppv or str(ppv.get("media_id")) != str(media_id):
                continue
            if sent_at and row.get("sent_at") and row["sent_at"] != sent_at:
                continue
            ppv["purchased"] = True
            mc["ppv"] = ppv
            db.table("messages").update({"media_context": mc}).eq("id", row["id"]).execute()
            return True
        return False
    return await asyncio.to_thread(_mark)


async def get_conversation_history(fan_id: str, limit: int = 40) -> list[Message]:
    def _get():
        r = get_supabase().table("messages").select("role, content, sent_at, media_context").eq("fan_id", fan_id).order("sent_at", desc=False).limit(limit).execute()
        return [_row_to_message(row) for row in (r.data or [])]

    return await asyncio.to_thread(_get)


async def save_message(
    fan_id: str,
    creator_id: str,
    role: str,
    content: str,
    was_ai_suggested: bool = False,
    fansly_message_id: str | None = None,
    media_context: dict | None = None,
) -> None:
    """media_context: e.g. {"attachments": [...]} for fan, {"ppv": {...}} for creator PPV."""

    def _save():
        row = {
            "fan_id": fan_id,
            "creator_id": creator_id,
            "role": role,
            "content": content,
            "was_ai_suggested": was_ai_suggested,
        }
        if fansly_message_id is not None:
            row["fansly_message_id"] = fansly_message_id
        if media_context is not None:
            row["media_context"] = media_context
        get_supabase().table("messages").insert(row).execute()

    await asyncio.to_thread(_save)


async def get_creator_fansly_account_id(creator_id: str) -> str | None:
    def _get():
        r = (
            get_supabase()
            .table("creators")
            .select("fansly_account_id")
            .eq("id", creator_id)
            .limit(1)
            .execute()
        )
        if not r.data:
            return None
        v = r.data[0].get("fansly_account_id")
        return str(v) if v is not None else None

    return await asyncio.to_thread(_get)


async def get_creator_persona(creator_id: str) -> Persona | None:
    def _get():
        r = get_supabase().table("creators").select("persona").eq("id", creator_id).execute()
        if not r.data or len(r.data) == 0:
            return None
        raw = r.data[0].get("persona")
        if raw is None:
            return None
        return Persona.model_validate(raw)

    return await asyncio.to_thread(_get)


async def get_ppv_offers(creator_id: str) -> list[dict]:
    def _get():
        # Custom PPV offers
        ppv = (
            get_supabase()
            .table("ppv_offers")
            .select("title, description, price")
            .eq("creator_id", creator_id)
            .execute()
        )
        # Vault items as sellable content
        vault = (
            get_supabase()
            .table("creator_vault_media")
            .select("id, fansly_media_id, ai_description, content_category, price_min, price_max, scene_outfit, scene_location, good_for, explicitness_level")
            .eq("creator_id", creator_id)
            .neq("content_category", "")
            .neq("content_category", "other")
            .neq("content_category", "teaser_clothed")
            .neq("content_category", "teaser_bundle")
            .not_.is_("content_category", "null")
            .order("explicitness_level", desc=False)
            .limit(30)
            .execute()
        )
        offers: list[dict] = []
        for row in ppv.data or []:
            offers.append({
                "title": row["title"],
                "description": row.get("description", ""),
                "price": row["price"],
            })
        for row in vault.data or []:
            media_id = row.get("fansly_media_id", "")
            if not media_id:
                continue
            description = row.get("ai_description", "")
            category = row.get("content_category", "")
            outfit = row.get("scene_outfit", "")
            location = row.get("scene_location", "")
            price_min = row.get("price_min") or 15
            price_max = row.get("price_max") or 50
            # Use lower third of range as default — session planner will set actual price
            price = round(price_min + (price_max - price_min) * 0.25)
            # Build a natural title from category + scene
            category_labels = {
                "lingerie_photo": "lingerie photo",
                "lingerie_video": "lingerie video",
                "nude_photo": "nude photo",
                "striptease_video": "striptease video",
                "closeup_photo": "closeup photo",
                "closeup_video": "closeup video",
                "solo_toy_video": "toy play video",
                "solo_toy_photo": "toy play photo",
                "bg_content": "BG content",
                "legs_feet": "feet/legs photo",
                "dictate_video": "dirty talk video",
                "task": "custom task",
            }
            label = category_labels.get(category, category)
            title = f"{label}"
            if outfit and outfit != "unknown":
                title += f" — {outfit}"
            offers.append({
                "title": title,
                "description": description[:120] if description else f"Exclusive {label}",
                "price": price,
                "media_id": media_id,
                "category": category,
                "good_for": row.get("good_for", "standalone"),
                "explicitness": row.get("explicitness_level", 3),
            })
        return offers

    return await asyncio.to_thread(_get)


async def get_vault_for_session(
    creator_id: str,
    fan_kinks: list[str] = [],
    exclude_media_ids: set = set(),
    limit: int = 200,
    min_explicitness: int = 1,
) -> list[dict]:
    """Fetch categorized vault items suitable for session planning."""
    def _get():
        r = (
            get_supabase()
            .table("creator_vault_media")
            .select("id, fansly_media_id, ai_description, content_category, price_min, price_max, explicitness_level, good_for, scene_id, scene_location, scene_outfit, scene_lighting, tags, filename, mimetype, album_title")
            .eq("creator_id", creator_id)
            .neq("content_category", "")
            .neq("content_category", "other")
            .neq("content_category", "teaser_clothed")
            .neq("content_category", "teaser_bundle")
            .not_.is_("content_category", "null")
            .gte("explicitness_level", min_explicitness)
            .gt("price_min", 0)
            .order("explicitness_level", desc=False)
            .limit(limit)
            .execute()
        )
        items = []
        for row in r.data or []:
            if row.get("fansly_media_id") in exclude_media_ids:
                continue
            items.append({
                "media_id": row.get("fansly_media_id", ""),
                "db_id": row.get("id", ""),
                "description": row.get("ai_description", ""),
                "category": row.get("content_category", ""),
                "price_min": row.get("price_min", 10),
                "price_max": row.get("price_max", 50),
                "explicitness": row.get("explicitness_level", 3),
                "good_for": row.get("good_for", "standalone"),
                "scene_id": row.get("scene_id", ""),
                "scene_location": row.get("scene_location", ""),
                "scene_outfit": row.get("scene_outfit", ""),
                "album_title": row.get("album_title", ""),
                "tags": row.get("tags") or [],
                "is_video": (row.get("mimetype") or "").startswith("video"),
            })
        return items
    return await asyncio.to_thread(_get)


def _scene_key(item: dict) -> str:
    """One shoot = one key. Priority: creator's own album folder, then the
    classifier's scene_id, then a location+outfit signature as last resort."""
    album = (item.get("album_title") or "").strip()
    if album and not album.lower().startswith("album_"):   # named folder = a real shoot
        return f"album:{album.lower()}"
    sid = (item.get("scene_id") or "").strip().lower()
    if sid and sid != "unknown":
        return f"scene:{sid}"
    loc = (item.get("scene_location") or "unknown").strip().lower()
    outfit = (item.get("scene_outfit") or "unknown").strip().lower()
    return f"sig:{loc}|{outfit}"


def _mode(values: list[str]) -> str:
    vals = [v for v in values if v and v != "unknown"]
    return Counter(vals).most_common(1)[0][0] if vals else ""


def build_scenes(vault_items: list[dict], min_items: int = 2) -> list[dict]:
    """Group flat vault items into coherent scenes (one shoot each), each
    internally ordered low -> high explicitness."""
    groups: dict[str, list[dict]] = {}
    for it in vault_items:
        groups.setdefault(_scene_key(it), []).append(it)

    scenes = []
    for key, items in groups.items():
        if len(items) < min_items:
            continue
        items.sort(key=lambda x: x.get("explicitness", 3))
        scenes.append({
            "scene_key": key,
            "location": _mode([i.get("scene_location") for i in items]),
            "outfit": _mode([i.get("scene_outfit") for i in items]),
            "categories": sorted({i.get("category", "") for i in items if i.get("category")}),
            "explicit_min": items[0].get("explicitness", 3),
            "explicit_max": items[-1].get("explicitness", 3),
            "count": len(items),
            "items": items,
        })
    scenes.sort(key=lambda s: (s["explicit_max"], s["count"]), reverse=True)
    return scenes


_JUNK_META = {"", "unclear", "unknown", "none", "n/a", "na", "null"}


def _set_key_coarse(item: dict) -> str:
    sid = (item.get("scene_id") or "").strip().lower()
    if sid and sid not in _JUNK_META:
        return f"scene:{sid}"
    album = (item.get("album_title") or "").strip()
    if album and not album.lower().startswith("album_"):
        return f"album:{album.lower()}"
    return ""   # no reliable shoot signal -> leave for manual, don't fake a set


def _collect_tags(items, limit=8):
    seen = []
    for it in items:
        vals = []
        cc = (it.get("content_category") or "").strip().lower()
        if cc: vals.append(cc)
        for t in (it.get("tags") or []):
            t = (t or "").strip().lower()
            if t: vals.append(t)
        gf = it.get("good_for")
        if isinstance(gf, list):
            vals += [(g or "").strip().lower() for g in gf]
        elif isinstance(gf, str) and gf.strip():
            vals.append(gf.strip().lower())
        for v in vals:
            if v and v not in _JUNK_META and v not in seen:
                seen.append(v)
    return seen[:limit]


def propose_sets(vault_items, max_per_set=6, min_per_set=3, min_level=2):
    groups = {}
    for it in vault_items:
        if (it.get("mimetype") or "").startswith("video"):
            continue
        key = _set_key_coarse(it)
        if not key:
            continue
        groups.setdefault(key, []).append(it)

    sets = []
    for key, items in groups.items():
        base = key.split(":", 1)[-1].replace("-", " ").replace("_", " ").strip()
        # split by explicitness AND content focus
        sub: dict[tuple, list] = {}
        for it in items:
            lvl = it.get("explicitness_level") or 0
            if lvl < min_level:
                continue
            cat = (it.get("content_category") or "").strip().lower()
            sub.setdefault((lvl, cat), []).append(it)

        for (lvl, cat), bucket in sorted(sub.items()):
            if len(bucket) < min_per_set:
                continue
            n = len(bucket)
            num = max(1, -(-n // max_per_set))
            size = -(-n // num)
            chunks = [bucket[i:i + size] for i in range(0, n, size)]
            for ci, chunk in enumerate(chunks):
                if len(chunk) < 2:
                    continue
                top = chunk[-1]
                price = round((((top.get("price_min") or 15) + (top.get("price_max") or 40)) / 2) / 5) * 5
                loc = _mode([i.get("scene_location") for i in chunk]).replace("_", " ").strip()
                outfit = _mode([i.get("scene_outfit") for i in chunk]).strip()
                if loc.lower() in _JUNK_META: loc = ""
                if outfit.lower() in _JUNK_META: outfit = "nude"
                cat_label = cat if cat and cat not in _JUNK_META else ""
                part = f" ({ci + 1})" if len(chunks) > 1 else ""
                title = f"{base}{(' · ' + cat_label) if cat_label else ''} · lvl {lvl}{part}"
                sets.append({
                    "title": title[:80],
                    "location": loc or None, "outfit": outfit or None,
                    "explicit_min": lvl, "explicit_max": lvl,
                    "media_ids": [i["fansly_media_id"] for i in chunk if i.get("fansly_media_id")],
                    "preview_media_id": chunk[0].get("fansly_media_id"),
                    "suggested_price": price,
                    "tags": _collect_tags(chunk),
                })
    sets.sort(key=lambda s: (s["explicit_max"], len(s["media_ids"])), reverse=True)
    return sets


async def get_fan_session(fan_id: str) -> dict | None:
    """Get active session plan for a fan."""
    def _get():
        r = get_supabase().table("fans").select("active_session").eq("id", fan_id).single().execute()
        return (r.data or {}).get("active_session")
    return await asyncio.to_thread(_get)


async def save_fan_session(fan_id: str, session: dict | None) -> None:
    """Save or clear active session plan for a fan."""
    def _save():
        get_supabase().table("fans").update({"active_session": session}).eq("id", fan_id).execute()
    await asyncio.to_thread(_save)


async def save_persona(creator_id: str, persona: Persona) -> None:
    def _save():
        get_supabase().table("creators").update({"persona": persona.model_dump()}).eq("id", creator_id).execute()

    await asyncio.to_thread(_save)


async def get_similar_exchanges(embedding: list[float], creator_id: str, limit: int = 5) -> list[ExchangeExample]:
    def _get():
        r = get_supabase().rpc("match_similar_exchanges", {
            "query_embedding": embedding,
            "p_creator_id": creator_id,
            "match_count": limit,
        }).execute()
        data = r.data or []
        return [
            ExchangeExample(
                fan_message=row["fan_message"],
                creator_reply=row["creator_reply"],
            )
            for row in data
        ]

    return await asyncio.to_thread(_get)


async def save_embedding(creator_id: str, fan_message: str, creator_response: str, stage: str, embedding: list[float]) -> None:
    def _save():
        get_supabase().table("message_embeddings").insert({
            "creator_id": creator_id,
            "fan_message": fan_message,
            "creator_response": creator_response,
            "conversation_stage": stage,
            "embedding": embedding,
        }).execute()

    await asyncio.to_thread(_save)


async def update_fan_memory(
    fan_id: str,
    notes: str,
    preferences: list[str],
    spend_tier: str,
    member_note: str = "",
    model_note: str = "",
) -> None:
    def _update():
        get_supabase().table("fans").update({
            "notes": notes,
            "preferences": preferences,
            "spend_tier": spend_tier,
            "member_note": member_note,
            "model_note": model_note,
        }).eq("id", fan_id).execute()

    await asyncio.to_thread(_update)


async def update_fan_ai_summary(fan_id: str, summary: dict) -> None:
    def _update():
        get_supabase().table("fans").update({
            "ai_summary": summary,
        }).eq("id", fan_id).execute()

    await asyncio.to_thread(_update)


# --- Creator autonomy caps (per-creator, agency-configurable) ---

async def get_creator_caps(creator_id: str) -> dict:
    """Per-creator autonomy limits set by the agency. All optional; null/absent = no limit.
    Keys: caps_enabled (bool), max_ppv_per_fan_per_day (int|null),
    max_spend_per_fan_per_day (int|null), max_sets_per_session (int|null)."""
    def _get():
        r = (
            get_supabase().table("creators")
            .select("caps_enabled, max_ppv_per_fan_per_day, "
                    "max_spend_per_fan_per_day, max_sets_per_session")
            .eq("id", creator_id).single().execute()
        )
        return r.data or {}
    return await asyncio.to_thread(_get)


# --- Creator self-consistency legend (canonical, per-creator) ---

# Stable identity attributes: first value established wins and is never overwritten,
# so the persona can't contradict itself across conversations.
_LEGEND_STABLE_KEYS = ("name", "origin", "age", "job", "background")


async def get_creator_legend(creator_id: str) -> dict:
    """Return the creator's canonical self-facts, e.g.
    {"name": "Eliza", "origin": "California", "age": "", "job": "", "background": "",
     "other": ["has a cat named Milo"]}."""
    def _get():
        r = (
            get_supabase().table("creators")
            .select("legend").eq("id", creator_id).single().execute()
        )
        return (r.data or {}).get("legend") or {}
    return await asyncio.to_thread(_get)


async def update_creator_legend(creator_id: str, new_facts: dict) -> dict:
    """First-established-wins merge into creators.legend.
    - Stable keys: only fill if currently empty; never overwrite a locked value.
    - 'other': accumulate distinct freeform details (deduped, capped).
    Returns the merged legend."""
    existing = await get_creator_legend(creator_id)
    merged = dict(existing)

    for key in _LEGEND_STABLE_KEYS:
        incoming = (new_facts.get(key) or "").strip()
        if incoming and not (merged.get(key) or "").strip():
            merged[key] = incoming  # lock it in

    # freeform accumulation
    other_existing = merged.get("other") or []
    if not isinstance(other_existing, list):
        other_existing = [str(other_existing)]
    incoming_other = new_facts.get("other") or []
    if isinstance(incoming_other, str):
        incoming_other = [incoming_other]
    seen = {o.strip().lower() for o in other_existing}
    for item in incoming_other:
        it = (item or "").strip()
        if it and it.lower() not in seen:
            other_existing.append(it)
            seen.add(it.lower())
    merged["other"] = other_existing[:20]  # cap to avoid unbounded growth

    if merged != existing:
        def _write():
            get_supabase().table("creators").update(
                {"legend": merged}
            ).eq("id", creator_id).execute()
        await asyncio.to_thread(_write)
    return merged