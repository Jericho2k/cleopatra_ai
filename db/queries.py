"""All database reads and writes. No AI logic."""

import asyncio
from collections import Counter
from datetime import datetime

from core.supabase import get_supabase
from models.schemas import ExchangeExample, Fan, Message, Persona
from services.shoot_fingerprint import build_shoot_clusters, shoot_fingerprint
from services.vault_metadata import VAULT_CLASSIFIER_VERSION, build_set_description


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
        needs_human_review=row.get("needs_human_review", False),
        sale_paused_at=row.get("sale_paused_at"),
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


async def get_creator_auto_mode_default(creator_id: str) -> bool:
    def _get() -> bool:
        response = (
            get_supabase().table("creators")
            .select("auto_mode")
            .eq("id", creator_id)
            .single()
            .execute()
        )
        return bool((response.data or {}).get("auto_mode", False))

    return await asyncio.to_thread(_get)


async def get_creator_sleep_hours(creator_id: str) -> tuple[int, int]:
    def _get() -> tuple[int, int]:
        response = (
            get_supabase().table("creators")
            .select("sleep_hours_start, sleep_hours_end")
            .eq("id", creator_id)
            .single()
            .execute()
        )
        row = response.data or {}
        return (
            int(row.get("sleep_hours_start") if row.get("sleep_hours_start") is not None else 0),
            int(row.get("sleep_hours_end") if row.get("sleep_hours_end") is not None else 7),
        )

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
        approved_sets = (
            db.table("vault_sets")
            .select("id", count="exact", head=True)
            .eq("creator_id", creator_id)
            .eq("status", "approved")
            .execute()
        )
        auto_available = int(approved_sets.count or 0) > 0

        row = {
            "creator_id": creator_id,
            "platform_fan_id": platform_fan_id,
            "display_name": display_name,
        }
        # Only set when the toggle is on, so existing global-fallback behavior
        # is untouched when it's off (auto_mode stays NULL → falls back to creator.auto_mode).
        if auto_new and auto_available:
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
                media_ids = [
                    str(value)
                    for value in (ppv.get("media_ids") or [ppv["media_id"]])
                    if value
                ]
                sent.append({
                    "media_id": str(ppv["media_id"]),
                    "media_ids": media_ids,
                    "set_id": str(ppv["set_id"]) if ppv.get("set_id") else None,
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
    """Return the newest ``limit`` messages in chronological prompt order.

    Supabase applies LIMIT after ORDER BY. Querying ascending returned the oldest
    40 messages in long conversations, so auto mode could answer stale context.
    """
    def _get():
        r = (
            get_supabase().table("messages")
            .select("role, content, sent_at, media_context")
            .eq("fan_id", fan_id)
            .order("sent_at", desc=True)
            .limit(limit)
            .execute()
        )
        rows = list(reversed(r.data or []))
        return [_row_to_message(row) for row in rows]

    return await asyncio.to_thread(_get)


async def save_message(
    fan_id: str,
    creator_id: str,
    role: str,
    content: str,
    was_ai_suggested: bool = False,
    fansly_message_id: str | None = None,
    media_context: dict | None = None,
) -> str | None:
    """Persist a message and return its internal ID when Supabase returns it.

    Existing callers may ignore the return value. The passive intelligence layer uses
    it as evidence provenance for manually submitted fan messages.
    """

    def _save() -> str | None:
        if fansly_message_id is not None:
            try:
                existing = (
                    get_supabase().table("messages")
                    .select("id")
                    .eq("fan_id", fan_id)
                    .eq("creator_id", creator_id)
                    .eq("fansly_message_id", fansly_message_id)
                    .limit(1)
                    .execute()
                )
                if existing.data and existing.data[0].get("id") is not None:
                    return str(existing.data[0]["id"])
            except Exception as exc:
                # Delivery already happened. A failed defensive read must not
                # turn the response into an apparent send failure.
                print(
                    "[MESSAGE DEDUPE READ ERROR] "
                    f"fan={fan_id} platform_message={fansly_message_id}: {exc}"
                )
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
        response = get_supabase().table("messages").insert(row).execute()
        data = response.data or []
        if data and data[0].get("id") is not None:
            return str(data[0]["id"])
        return None

    return await asyncio.to_thread(_save)


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
            .select("id, fansly_media_id, ai_description, content_category, price_min, price_max, explicitness_level, good_for, scene_id, scene_location, scene_outfit, scene_lighting, tags, filename, mimetype, album_title, classification_metadata")
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
                "classification_metadata": row.get("classification_metadata") or {},
            })
        return items
    return await asyncio.to_thread(_get)


def _mode(values: list[str]) -> str:
    vals = [v for v in values if v and v != "unknown"]
    return Counter(vals).most_common(1)[0][0] if vals else ""


def build_scenes(vault_items: list[dict], min_items: int = 2) -> list[dict]:
    """Group flat vault items into coherent scenes (one shoot each), each
    internally ordered low -> high explicitness."""
    scenes = []
    for shoot in build_shoot_clusters(vault_items):
        items = shoot["items"]
        if len(items) < min_items:
            continue
        items.sort(key=lambda x: x.get("explicitness", 3))
        scenes.append({
            "scene_key": shoot["shoot_id"],
            "grouping_method": shoot["method"],
            "grouping_confidence": shoot["confidence"],
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
    sets = []
    photo_items = [
        item
        for item in vault_items
        if not str(item.get("mimetype") or "").lower().startswith("video")
    ]
    for shoot in build_shoot_clusters(photo_items):
        bucket = [
            item
            for item in shoot["items"]
            if int(item.get("explicitness_level") or 0) >= min_level
        ]
        if len(bucket) < min_per_set:
            continue
        bucket.sort(
            key=lambda item: (
                int(item.get("explicitness_level") or 0),
                str(item.get("good_for") or ""),
                str(item.get("fansly_media_id") or ""),
            )
        )
        chunk_count = max(1, -(-len(bucket) // max_per_set))
        base_size, remainder = divmod(len(bucket), chunk_count)
        chunks = []
        start = 0
        for chunk_index in range(chunk_count):
            size = base_size + (1 if chunk_index < remainder else 0)
            chunks.append(bucket[start:start + size])
            start += size

        for chunk_index, chunk in enumerate(chunks):
            if len(chunk) < min_per_set:
                continue
            top = chunk[-1]
            price = round(
                (
                    (
                        (top.get("price_min") or 15)
                        + (top.get("price_max") or 40)
                    )
                    / 2
                )
                / 5
            ) * 5
            loc = _mode(
                [item.get("scene_location") for item in chunk]
            ).replace("_", " ").strip()
            outfit = _mode(
                [item.get("scene_outfit") for item in chunk]
            ).strip()
            if loc.lower() in _JUNK_META:
                loc = ""
            if outfit.lower() in _JUNK_META:
                outfit = ""
            category = _mode(
                [item.get("content_category") for item in chunk]
            ).replace("_", " ").strip()
            palette = []
            for item in chunk:
                local = shoot_fingerprint(item).get("local") or {}
                for colour in local.get("palette_names") or []:
                    if colour not in palette:
                        palette.append(colour)
            title_parts = [
                value
                for value in (
                    loc.title() if loc else "",
                    palette[0].title() if palette else "",
                    outfit,
                    category,
                )
                if value
            ]
            base = " · ".join(title_parts[:3]) or str(shoot["shoot_id"])
            levels = [
                int(item.get("explicitness_level") or 0) for item in chunk
            ]
            level_label = (
                f"lvl {min(levels)}–{max(levels)}"
                if min(levels) != max(levels)
                else f"lvl {levels[0]}"
            )
            part = (
                f" ({chunk_index + 1})"
                if len(chunks) > 1
                else ""
            )
            title = f"{base} · {level_label}{part}"
            tags = _collect_tags(chunk)
            for colour in palette[:4]:
                if colour not in tags:
                    tags.append(colour)
            sets.append({
                "title": title[:80],
                "description": build_set_description(chunk),
                "location": loc or None,
                "outfit": outfit or None,
                "explicit_min": min(levels),
                "explicit_max": max(levels),
                "media_ids": [
                    item["fansly_media_id"]
                    for item in chunk
                    if item.get("fansly_media_id")
                ],
                "preview_media_id": chunk[0].get("fansly_media_id"),
                "suggested_price": price,
                "tags": tags[:12],
                "metadata_version": VAULT_CLASSIFIER_VERSION,
                "shoot_id": shoot["shoot_id"],
                "shoot_method": shoot["method"],
                "shoot_confidence": shoot["confidence"],
            })
    sets.sort(key=lambda s: (s["explicit_max"], len(s["media_ids"])), reverse=True)
    return sets


def propose_video_ppvs(vault_items, min_level=2):
    """Turn each frame-classified video into its own approvable PPV asset.

    Videos are deliberately never bundled here. An approved row still uses the
    existing ``vault_sets`` contract, but contains exactly one media id so the
    commercial planner and delivery ledger can treat every clip as an
    individual locked message.
    """
    proposals = []
    for item in vault_items:
        if not str(item.get("mimetype") or "").lower().startswith("video"):
            continue
        if str(item.get("classification_source") or "") != "video_frames":
            continue
        media_id = str(item.get("fansly_media_id") or "").strip()
        category = str(item.get("content_category") or "").strip().lower()
        level = int(item.get("explicitness_level") or 0)
        if (
            not media_id
            or level < min_level
            or category in _JUNK_META
            or category in {"other", "teaser_clothed", "teaser_bundle"}
        ):
            continue

        minimum = max(1, round(float(item.get("price_min") or 15)))
        maximum = max(minimum, round(float(item.get("price_max") or minimum)))
        suggested = int(round(((minimum + maximum) / 2) / 5) * 5)
        suggested = max(minimum, min(maximum, suggested))
        location = str(item.get("scene_location") or "").replace("_", " ").strip()
        outfit = str(item.get("scene_outfit") or "").strip()
        if location.lower() in _JUNK_META:
            location = ""
        if outfit.lower() in _JUNK_META:
            outfit = ""
        label = category.replace("_video", "").replace("_", " ").strip()
        title_parts = [
            value
            for value in (
                location.title() if location else "Private",
                outfit,
                label,
                "video",
            )
            if value
        ]
        tags = _collect_tags([item], limit=10)
        for tag in ("video", "individual_video"):
            if tag not in tags:
                tags.append(tag)
        proposals.append({
            "title": " · ".join(title_parts[:4])[:80],
            "description": build_set_description([item]),
            "location": location or None,
            "outfit": outfit or None,
            "explicit_min": level,
            "explicit_max": level,
            "media_ids": [media_id],
            "preview_media_id": media_id,
            "suggested_price": suggested,
            "base_price_cents": suggested * 100,
            "min_price_cents": minimum * 100,
            "max_price_cents": maximum * 100,
            "tags": tags[:12],
            "metadata_version": VAULT_CLASSIFIER_VERSION,
        })
    proposals.sort(
        key=lambda row: (row["explicit_max"], row["suggested_price"], row["media_ids"][0]),
        reverse=True,
    )
    return proposals


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

async def freeze_fan_for_review(fan_id: str, reason: str) -> None:
    """Mark a fan's conversation as frozen and needing human intervention.
    Auto-mode will skip frozen fans until a human clears the flag in the dashboard."""
    from datetime import datetime, timezone
    def _update():
        get_supabase().table("fans").update({
            "needs_human_review": True,
            "review_reason": reason,
            "frozen_at": datetime.now(timezone.utc).isoformat(),
        }).eq("id", fan_id).execute()
    await asyncio.to_thread(_update)


async def clear_fan_review(fan_id: str) -> None:
    """Clear a review hold only after its backend resolution has completed."""
    def _update():
        get_supabase().table("fans").update({
            "needs_human_review": False,
            "review_reason": None,
            "frozen_at": None,
        }).eq("id", fan_id).execute()

    await asyncio.to_thread(_update)


async def set_fan_decline_lock(fan_id: str, price: float | None) -> None:
    """Mark that the fan declined on affordability. While locked, auto-mode does NOT
    sell to this fan (no PPV, no cheaper-item resurfacing). Cleared when he signals
    money is available / payday arrives."""
    from datetime import datetime, timezone
    def _update():
        get_supabase().table("fans").update({
            "sale_paused_at": datetime.now(timezone.utc).isoformat(),
            "sale_paused_price": price,
        }).eq("id", fan_id).execute()
    await asyncio.to_thread(_update)


async def clear_fan_decline_lock(fan_id: str) -> None:
    """Lift the decline lock (fan signaled money is available)."""
    def _update():
        get_supabase().table("fans").update({
            "sale_paused_at": None,
            "sale_paused_price": None,
        }).eq("id", fan_id).execute()
    await asyncio.to_thread(_update)


async def get_creator_caps(creator_id: str) -> dict:
    """Per-creator autonomy limits set by the agency. All optional; null/absent = no limit.
    Keys: caps_enabled (bool), max_ppv_per_fan_per_day (int|null),
    max_spend_per_fan_per_day (int|null), max_sets_per_session (int|null),
    crisis_policy ('continue'|'freeze'), whale_handoff_threshold (int|null;
    null/0 = disabled, else hand the fan to a human once total_spent crosses it)."""
    def _get():
        r = (
            get_supabase().table("creators")
            .select("caps_enabled, max_ppv_per_fan_per_day, "
                    "max_spend_per_fan_per_day, max_sets_per_session, crisis_policy, "
                    "whale_handoff_threshold")
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
