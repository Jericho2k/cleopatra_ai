"""FastAPI entrypoint for Cleopatra.

Routes are thin and delegate all logic to services.
"""

import asyncio
import os
import random
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

from fastapi import BackgroundTasks, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ai.generator import generate_replies
from ai.prompt_builder import build_prompt
from ai.situation_analyzer import analyze_situation
from ai.rag import find_similar_exchanges
from ai.stage_classifier import classify_stage
from core.supabase import get_supabase
from db.queries import (
    create_fan,
    get_conversation_history,
    get_creator_persona,
    get_fan,
    get_fan_by_id,
    get_ppv_offers,
    save_message,
)
from models.schemas import (
    ConversationContext,
    Fan,
    Persona,
    SuggestionRequest,
    SuggestionResponse,
)
from services.fansly_poller import FanslyPoller
from services.fansly_session_store import SessionStore
from services.suggestions import (
    _should_update_memory,
    _update_fan_ai_summary,
    _update_fan_memory,
    get_suggestions,
    schedule_auto_reply,
)


_processed_messages: set = set()


async def send_fansly_message(account_id: str, group_id: str, text: str) -> bool:
    import httpx

    api_key = os.environ.get("APIFANSLY_API_KEY")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"https://v1.apifansly.com/api/fansly/{account_id}/chats/{group_id}/messages",
                headers={
                    "x-api-key": api_key,
                    "Content-Type": "application/json",
                },
                json={"content": text},
                timeout=10,
            )
            print(f"[SEND] status={response.status_code} body={response.text[:200]}")
            return response.status_code == 201
    except Exception as e:
        print(f"[SEND ERROR] {e}")
        return False


async def process_incoming_fan_message(
    fan_id: str,
    creator_id: str,
    message_content: str,
    auto_mode: bool,
    message_id: str | None,
) -> None:
    """Shared pipeline: history already includes the new fan message."""
    conversation_history = await get_conversation_history(fan_id)
    fan_profile = await get_fan_by_id(fan_id)
    if fan_profile is None:
        fan_profile = Fan(id=fan_id, display_name=fan_id)

    fan_auto = fan_profile.auto_mode
    if fan_auto is None:
        effective_auto = auto_mode
    else:
        effective_auto = fan_auto

    print(
        f"[AUTO MODE] creator={creator_id} creator_auto={auto_mode} "
        f"fan_auto={fan_auto} effective_auto={effective_auto} fan={fan_id}"
    )

    excluded = await asyncio.to_thread(
        lambda: get_supabase()
        .from_("fan_list_members")
        .select("fan_lists(exclude_from_auto)")
        .eq("fan_id", fan_id)
        .execute()
    )
    is_excluded = any(
        row.get("fan_lists", {}).get("exclude_from_auto", False)
        for row in (excluded.data or [])
    )

    if effective_auto and not is_excluded:
        schedule_auto_reply(fan_id, creator_id)
        fan_msg_count = len([m for m in conversation_history if m.role == "fan"])
        print(
            f"[MEMORY CHECK] fan={fan_id} fan_messages={fan_msg_count} "
            f"should_update={_should_update_memory(conversation_history)}"
        )
        if _should_update_memory(conversation_history):
            asyncio.create_task(_update_fan_memory(fan_id, creator_id, conversation_history, fan_profile.total_spent))
            asyncio.create_task(_update_fan_ai_summary(fan_id, conversation_history))
        return

    creator_persona = await get_creator_persona(creator_id)
    if creator_persona is None:
        creator_persona = Persona()
    ppv_offers = await get_ppv_offers(creator_id)

    conversation_stage = classify_stage(conversation_history, fan_profile)
    similar_exchanges = await find_similar_exchanges(
        message_content, creator_id, enabled=False
    )

    ctx_without_situation = ConversationContext(
        fan_message=message_content,
        conversation_history=conversation_history,
        fan_profile=fan_profile,
        creator_persona=creator_persona,
        similar_exchanges=similar_exchanges,
        conversation_stage=conversation_stage,
        creator_name="a creator",
        ppv_offers=ppv_offers,
    )

    situation = await analyze_situation(ctx_without_situation)

    ctx = ConversationContext(
        fan_message=message_content,
        conversation_history=conversation_history,
        fan_profile=fan_profile,
        creator_persona=creator_persona,
        similar_exchanges=similar_exchanges,
        conversation_stage=conversation_stage,
        creator_name="a creator",
        situation=situation,
        ppv_offers=ppv_offers,
    )

    prompt = build_prompt(ctx)
    replies = await generate_replies(prompt, creator_persona)

    db = get_supabase()
    if message_id:
        await asyncio.to_thread(
            lambda: db.table("suggestions").insert({
                "fan_id": fan_id,
                "creator_id": creator_id,
                "message_id": message_id,
                "suggestions": replies,
                "stage": conversation_stage.value,
            }).execute()
        )

    fan_msg_count = len([m for m in conversation_history if m.role == "fan"])
    print(
        f"[MEMORY CHECK] fan={fan_id} fan_messages={fan_msg_count} "
        f"should_update={_should_update_memory(conversation_history)}"
    )
    if _should_update_memory(conversation_history):
        asyncio.create_task(_update_fan_memory(fan_id, creator_id, conversation_history, fan_profile.total_spent))
        asyncio.create_task(_update_fan_ai_summary(fan_id, conversation_history))


class ReplyRequest(BaseModel):
    fan_id: str
    creator_id: str
    content: str
    was_ai_suggested: bool = False


class WebhookPayload(BaseModel):
    type: str
    record: dict


class ConnectCreatorRequest(BaseModel):
    name: str
    email: str
    password: str
    user_id: str
    countryCode: str = "US"


class Connect2FARequest(BaseModel):
    twofa_token: str
    code: str
    name: str
    email: str
    password: str
    countryCode: str = "US"
    user_id: str = ""


async def handle_new_fan_message(account_id: str, group_id: str, message: dict):
    """
    Fires when a fan sends a message to a model account we're polling.

    account_id  = Fansly ID of the model (e.g. "707604041756061697")
    group_id    = Fansly conversation ID (e.g. "813798181052637184")
    message     = raw Fansly message dict with keys:
                  id, senderId, content, createdAt, attachments, etc.
    """
    fan_id = message["senderId"]
    content = message.get("content", "")
    print(f"[NEW MESSAGE] model={account_id} fan={fan_id}: {content[:80]}")


session_store: SessionStore = None
fansly_poller: FanslyPoller = None
reengagement_task: asyncio.Task | None = None


async def reengagement_scheduler():
    """Runs re-engagement check every 6 hours."""
    while True:
        await asyncio.sleep(6 * 60 * 60)
        try:
            print("[CRON] Running re-engagement check...")
            result = await run_reengagement()
            print(f"[CRON] Re-engagement done: {result}")
        except Exception as e:
            print(f"[CRON ERROR] {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global session_store, fansly_poller, reengagement_task

    supabase = get_supabase()
    session_store = SessionStore(
        supabase=supabase,
        encryption_key=os.environ["FANSLY_SESSION_KEY"],
    )
    await session_store.load_all()

    fansly_poller = FanslyPoller(
        session_store=session_store,
        on_new_message=handle_new_fan_message,
    )
    await fansly_poller.start_all()
    reengagement_task = asyncio.create_task(reengagement_scheduler())

    yield

    if fansly_poller:
        await fansly_poller.stop_all()
    if reengagement_task:
        reengagement_task.cancel()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/suggestions", response_model=SuggestionResponse)
async def suggestions(req: SuggestionRequest) -> SuggestionResponse:
    return await get_suggestions(
        fan_id=req.fan_id,
        creator_id=req.creator_id,
        fan_message=req.message,
        creator_name="a creator",
    )


@app.post("/regenerate-suggestions", response_model=SuggestionResponse)
async def regenerate_suggestions(req: SuggestionRequest) -> SuggestionResponse:
    result = await get_suggestions(
        fan_id=req.fan_id,
        creator_id=req.creator_id,
        fan_message=req.message,
        creator_name="a creator",
        save_fan_message=False,
    )

    if result.suggestions:
        db = get_supabase()
        await asyncio.to_thread(
            lambda: db.table("suggestions").insert({
                "fan_id": req.fan_id,
                "creator_id": req.creator_id,
                "suggestions": result.suggestions,
                "stage": result.stage.value,
            }).execute()
        )

    return result


@app.post("/reply")
async def save_reply(req: ReplyRequest) -> dict:
    await save_message(
        req.fan_id,
        req.creator_id,
        "creator",
        req.content,
        req.was_ai_suggested
    )

    db = get_supabase()
    fan_row = await asyncio.to_thread(
        lambda: db.table("fans")
        .select("fansly_group_id")
        .eq("id", req.fan_id).single().execute()
    )
    creator_row = await asyncio.to_thread(
        lambda: db.table("creators")
        .select("apifansly_account_id")
        .eq("id", req.creator_id).single().execute()
    )

    group_id = (fan_row.data or {}).get("fansly_group_id")
    apifansly_id = (creator_row.data or {}).get("apifansly_account_id")

    if group_id and apifansly_id:
        await send_fansly_message(apifansly_id, group_id, req.content)

    return {"status": "ok"}


@app.post("/reengagement")
async def run_reengagement() -> dict:
    db = get_supabase()
    settings_rows = await asyncio.to_thread(
        lambda: db.table("reengagement_settings")
        .select("*, creators(id, apifansly_account_id)")
        .eq("enabled", True)
        .execute()
    )

    total = 0
    for setting in (settings_rows.data or []):
        creator_id = setting["creator_id"]
        hours = setting.get("hours_threshold", 48)
        use_ai = setting.get("use_ai", True)
        templates = setting.get("templates", [])
        ai_instructions = setting.get("ai_instructions", "")
        _ = ai_instructions  # Placeholder for future prompt customization.
        exclude_list_id = setting.get("exclude_list_id")

        excluded_ids = set()
        if exclude_list_id:
            excl = await asyncio.to_thread(
                lambda lid=exclude_list_id: db.table("fan_list_members")
                .select("fan_id")
                .eq("list_id", lid)
                .execute()
            )
            excluded_ids = {m["fan_id"] for m in (excl.data or [])}

        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        jitter_hours = random.uniform(-2, 2)
        jitter_cutoff = (
            datetime.now(timezone.utc) - timedelta(hours=hours + jitter_hours)
        ).isoformat()

        fans = await asyncio.to_thread(
            lambda cid=creator_id: db.table("fans")
            .select("id, display_name, fansly_group_id, auto_mode")
            .eq("creator_id", cid)
            .execute()
        )

        for fan in (fans.data or []):
            fan_id = fan["id"]

            if fan_id in excluded_ids:
                continue
            if fan.get("auto_mode") is False:
                continue

            last_msg = await asyncio.to_thread(
                lambda fid=fan_id, cid=creator_id: db.table("messages")
                .select("role, sent_at")
                .eq("fan_id", fid)
                .eq("creator_id", cid)
                .order("sent_at", desc=True)
                .limit(1)
                .maybe_single()
                .execute()
            )

            if not last_msg.data:
                continue
            if last_msg.data["role"] != "creator":
                continue
            if last_msg.data["sent_at"] > jitter_cutoff:
                continue

            last_log = await asyncio.to_thread(
                lambda fid=fan_id, cid=creator_id: db.table("reengagement_log")
                .select("template_index, sent_at")
                .eq("fan_id", fid)
                .eq("creator_id", cid)
                .order("sent_at", desc=True)
                .limit(1)
                .maybe_single()
                .execute()
            )

            if last_log.data:
                last_sent = last_log.data["sent_at"]
                if last_sent > cutoff:
                    continue

            if use_ai:
                schedule_auto_reply(fan_id, creator_id)
                total += 1
            else:
                if not templates:
                    continue

                last_index = last_log.data["template_index"] if last_log.data else -1
                next_index = last_index + 1
                if next_index >= len(templates):
                    continue

                template = templates[next_index]
                msg = template.replace("{name}", fan.get("display_name") or "")
                apifansly_id = (setting.get("creators") or {}).get("apifansly_account_id")
                group_id = fan.get("fansly_group_id")

                if msg and apifansly_id and group_id:
                    await send_fansly_message(apifansly_id, group_id, msg)
                    await save_message(fan_id, creator_id, "creator", msg, False)
                    await asyncio.to_thread(
                        lambda fid=fan_id, cid=creator_id, idx=next_index: db.table("reengagement_log").insert({
                            "fan_id": fid,
                            "creator_id": cid,
                            "template_index": idx,
                        }).execute()
                    )
                    total += 1

    return {"status": "ok", "reengaged": total}


@app.post("/connect-creator")
async def connect_creator(req: ConnectCreatorRequest) -> dict:
    print(f"[CONNECT] req={req.dict()}")
    import httpx

    api_key = os.environ.get("APIFANSLY_API_KEY")

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://v1.apifansly.com/api/fansly/connect",
            headers={"x-api-key": api_key, "Content-Type": "application/json"},
            json={
                "username": req.email,
                "password": req.password,
                "name": req.name,
                "countryCode": req.countryCode,
            },
            timeout=30,
        )
        data = response.json()
        print(f"[CONNECT] response={data}")

        if data.get("data", {}).get("requires_2fa"):
            return {
                "requires_2fa": True,
                "twofa_token": data["data"].get("twofa_token", ""),
                "masked_email": data["data"].get("masked_email", ""),
                "message": data["data"].get("message", ""),
            }

        apifansly_account_id = data.get("data", {}).get("account_id")
        fansly_account_id = data.get("data", {}).get("data", {}).get("response", {}).get("accountId")

        if not apifansly_account_id:
            return {"success": False, "error": "Failed to connect account"}

        db = get_supabase()
        creator_row = await asyncio.to_thread(
            lambda: db.table("creators").insert({
                "platform_username": req.name,
                "platform": "fansly",
                "fansly_account_id": str(fansly_account_id),
                "apifansly_account_id": apifansly_account_id,
                "auto_mode": False,
            }).execute()
        )

        creator = creator_row.data[0] if creator_row.data else None
        if not creator:
            return {"success": False, "error": "Failed to create creator"}

        await asyncio.to_thread(
            lambda: db.table("chatter_creators").insert({
                "chatter_id": req.user_id,
                "creator_id": creator["id"],
            }).execute()
        )

        asyncio.create_task(sync_chats_background(creator["id"]))

        return {"success": True, "creator": creator}


@app.post("/connect-creator-2fa")
async def connect_creator_2fa(req: Connect2FARequest) -> dict:
    import httpx

    api_key = os.environ.get("APIFANSLY_API_KEY")

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://v1.apifansly.com/api/fansly/verify-2fa",
            headers={"x-api-key": api_key, "Content-Type": "application/json"},
            json={
                "username": req.email,
                "password": req.password,
                "name": req.name,
                "twoFactorToken": req.twofa_token,
                "twoFactorCode": req.code,
                "countryCode": req.countryCode,
            },
            timeout=30,
        )
        data = response.json()
        print(f"[2FA] response={data}")

        apifansly_account_id = data.get("data", {}).get("account_id")
        fansly_account_id = data.get("data", {}).get("data", {}).get("response", {}).get("accountId")

        if not apifansly_account_id:
            return {"success": False, "error": "2FA verification failed"}

        db = get_supabase()
        creator_row = await asyncio.to_thread(
            lambda: db.table("creators").insert({
                "platform_username": req.name,
                "platform": "fansly",
                "fansly_account_id": str(fansly_account_id),
                "apifansly_account_id": apifansly_account_id,
                "auto_mode": False,
            }).execute()
        )

        creator = creator_row.data[0] if creator_row.data else None
        if not creator:
            return {"success": False, "error": "Failed to create creator"}

        await asyncio.to_thread(
            lambda: db.table("chatter_creators").insert({
                "chatter_id": req.user_id,
                "creator_id": creator["id"],
            }).execute()
        )

        asyncio.create_task(sync_chats_background(creator["id"]))

        return {"success": True, "creator": creator}


async def sync_chats_background(creator_id: str) -> None:
    try:
        await sync_chats(creator_id)
        print(f"[SYNC] Chats synced for creator={creator_id}")
    except Exception as e:
        print(f"[SYNC ERROR] {e}")


@app.post("/sync-chats/{creator_id}")
async def sync_chats(creator_id: str) -> dict:
    import httpx

    db = get_supabase()
    creator_row = await asyncio.to_thread(
        lambda cid=creator_id: db.table("creators")
        .select("apifansly_account_id, fansly_account_id")
        .eq("id", cid)
        .single()
        .execute()
    )

    apifansly_id = (creator_row.data or {}).get("apifansly_account_id")
    if not apifansly_id:
        return {"status": "error", "message": "no apifansly account"}

    api_key = os.environ.get("APIFANSLY_API_KEY")

    async with httpx.AsyncClient() as client:
        all_chats = []
        cursor = None

        while True:
            params = {}
            if cursor:
                params["cursor"] = cursor

            response = await client.get(
                f"https://v1.apifansly.com/api/fansly/{apifansly_id}/chats",
                headers={"x-api-key": api_key},
                params=params,
                timeout=30,
            )
            data = response.json()
            response_data = data.get("data", {}).get("data", {}).get("response", {})
            chats = response_data.get("data", [])
            cursor = response_data.get("nextCursor") or response_data.get("Nextcursor")
            if not all_chats:
                print(f"[SYNC CHATS RAW] {str(response_data)[:500]}")

            print(
                f"[SYNC CHATS] batch={len(chats)} total={len(all_chats) + len(chats)} "
                f"cursor={cursor}"
            )

            if not chats:
                break

            all_chats.extend(chats)

            if not cursor:
                break

        synced = 0
        for chat in all_chats:
            platform_fan_id = str(chat.get("partnerAccountId", ""))
            fan_name = chat.get("partnerUsername", f"Fan_{platform_fan_id[-6:]}")
            group_id = str(chat.get("groupId", ""))
            fan_avatar = chat.get("partnerAvatar") or chat.get("partnerImage") or None
            _ = fan_avatar

            if not platform_fan_id or not group_id:
                continue

            fan = await get_fan(creator_id, platform_fan_id)
            if not fan:
                fan = await create_fan(creator_id, platform_fan_id, fan_name)

            await asyncio.to_thread(
                lambda fid=fan.id, gid=group_id, name=fan_name: db.table("fans")
                .update({
                    "fansly_group_id": gid,
                    "display_name": name,
                })
                .eq("id", fid)
                .execute()
            )
            synced += 1

        print(f"[SYNC CHATS] total_chats={len(all_chats)} synced={synced}")
        return {"status": "ok", "synced": synced}


@app.post("/load-history/{creator_id}/{fan_id}")
async def load_fan_history(creator_id: str, fan_id: str) -> dict:
    import httpx

    db = get_supabase()

    fan_row = await asyncio.to_thread(
        lambda: db.table("fans")
        .select("fansly_group_id, platform_fan_id")
        .eq("id", fan_id)
        .single()
        .execute()
    )
    creator_row = await asyncio.to_thread(
        lambda: db.table("creators")
        .select("apifansly_account_id, fansly_account_id")
        .eq("id", creator_id)
        .single()
        .execute()
    )

    group_id = (fan_row.data or {}).get("fansly_group_id")
    apifansly_id = (creator_row.data or {}).get("apifansly_account_id")
    fansly_account_id = str((creator_row.data or {}).get("fansly_account_id", ""))

    if not group_id or not apifansly_id:
        return {"status": "error", "message": "missing fan or creator info"}

    api_key = os.environ.get("APIFANSLY_API_KEY")

    existing = await asyncio.to_thread(
        lambda: db.table("messages")
        .select("fansly_message_id")
        .eq("fan_id", fan_id)
        .execute()
    )
    existing_ids = {r["fansly_message_id"] for r in (existing.data or []) if r.get("fansly_message_id")}

    all_messages = []
    all_media = {}
    cursor = None

    async with httpx.AsyncClient() as client:
        while True:
            params = {}
            if cursor:
                params["cursor"] = cursor

            response = await client.get(
                f"https://v1.apifansly.com/api/fansly/{apifansly_id}/chats/{group_id}/messages",
                headers={"x-api-key": api_key},
                params=params,
                timeout=30,
            )
            data = response.json()
            response_data = data.get("data", {}).get("data", {}).get("response", {})
            messages = response_data.get("messages", [])
            account_media = response_data.get("accountMedia", [])
            cursor = response_data.get("cursor")

            for am in account_media:
                content_id = am.get("id") or am.get("mediaId")
                media = am.get("media", {})
                locations = media.get("locations", [])
                variants = media.get("variants", [])
                url = None
                if locations:
                    url = locations[0].get("location")
                elif variants and variants[0].get("locations"):
                    url = variants[0]["locations"][0].get("location")
                if content_id and url:
                    all_media[str(content_id)] = url

            all_messages.extend(messages)
            print(f"[LOAD HISTORY] batch={len(messages)} total={len(all_messages)} cursor={cursor}")

            if not cursor or not messages:
                break

    imported = 0
    for msg in reversed(all_messages):
        msg_id = str(msg.get("id", ""))
        if not msg_id or msg_id in existing_ids:
            continue

        content = msg.get("content", "")
        sender_id = str(msg.get("senderId", ""))
        role = "fan" if sender_id != fansly_account_id else "creator"

        created_at = msg.get("createdAt")
        if created_at and created_at > 0:
            ts = float(created_at)
            if ts > 1e12:
                ts /= 1000.0
            sent_at = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        else:
            sent_at = datetime.now(timezone.utc).isoformat()

        attachments = msg.get("attachments", [])
        media_context = None
        if attachments:
            resolved = []
            for att in attachments:
                content_id = str(att.get("contentId", ""))
                url = all_media.get(content_id)
                resolved.append({
                    "contentId": content_id,
                    "url": url,
                    "type": att.get("contentType", 1),
                })
            media_context = {"attachments": resolved}

        if not content and not attachments:
            continue

        row = {
            "fan_id": fan_id,
            "creator_id": creator_id,
            "role": role,
            "content": content,
            "fansly_message_id": msg_id,
            "sent_at": sent_at,
            "media_context": media_context,
        }

        await asyncio.to_thread(
            lambda r=row: db.table("messages").insert(r).execute()
        )
        existing_ids.add(msg_id)
        imported += 1

    if imported > 0:
        conversation_history = await get_conversation_history(fan_id)
        fan_profile = await get_fan_by_id(fan_id)

        if fan_profile and len(conversation_history) >= 10:
            asyncio.create_task(_update_fan_ai_summary(fan_id, conversation_history))
            asyncio.create_task(
                _update_fan_memory(fan_id, creator_id, conversation_history, fan_profile.total_spent)
            )

    return {"status": "ok", "imported": imported}


@app.post("/mark-all-read/{creator_id}")
async def mark_all_read(creator_id: str) -> dict:
    import httpx

    db = get_supabase()
    creator_row = await asyncio.to_thread(
        lambda cid=creator_id: db.table("creators")
        .select("apifansly_account_id")
        .eq("id", cid)
        .single()
        .execute()
    )
    apifansly_id = (creator_row.data or {}).get("apifansly_account_id")
    if not apifansly_id:
        return {"status": "error"}

    api_key = os.environ.get("APIFANSLY_API_KEY")
    async with httpx.AsyncClient() as client:
        await client.post(
            f"https://v1.apifansly.com/api/fansly/{apifansly_id}/chats/mark-as-read",
            headers={"x-api-key": api_key},
            timeout=10,
        )
    return {"status": "ok"}


@app.post("/sync-vault/{creator_id}")
async def sync_vault(creator_id: str) -> dict:
    import httpx

    creator_row = await asyncio.to_thread(
        lambda: get_supabase()
        .table("creators")
        .select("apifansly_account_id")
        .eq("id", creator_id)
        .single()
        .execute()
    )

    apifansly_id = (creator_row.data or {}).get("apifansly_account_id")
    if not apifansly_id:
        return {"status": "error", "message": "no apifansly account id"}

    api_key = os.environ.get("APIFANSLY_API_KEY")

    synced = 0
    async with httpx.AsyncClient() as client:
        albums_resp = await client.get(
            f"https://v1.apifansly.com/api/fansly/{apifansly_id}/vault/albums",
            headers={"x-api-key": api_key},
            timeout=30,
        )
        try:
            albums_data = albums_resp.json()
        except Exception:
            albums_data = {}
        albums = (
            albums_data.get("data", {})
            .get("data", {})
            .get("response", {})
            .get("albums", [])
        )

        for album in albums:
            album_id = album.get("id")
            album_title = album.get("title", "")
            if album_id is None:
                continue

            media_resp = await client.get(
                f"https://v1.apifansly.com/api/fansly/{apifansly_id}/vault/albums/{album_id}/media",
                headers={"x-api-key": api_key},
                timeout=30,
            )
            print(f"[VAULT] album={album_id} media_resp={media_resp.text[:500]}")
            try:
                media_data = media_resp.json()
            except Exception:
                media_data = {}
            items = (
                media_data.get("data", {})
                .get("data", {})
                .get("response", {})
                .get("accountMedia", [])
            )

            for item in items:
                media_id = item.get("mediaId") or item.get("id")
                account_media_id = item.get("id")
                media = item.get("media", {}) or {}

                thumbnail_url = None
                variants = media.get("variants", [])
                locations = media.get("locations", [])
                if locations:
                    thumbnail_url = locations[0].get("location")
                elif variants:
                    variant_locs = variants[0].get("locations", [])
                    if variant_locs:
                        thumbnail_url = variant_locs[0].get("location")

                media_type = media.get("type", 1)

                if not media_id:
                    continue

                row_payload = {
                    "creator_id": creator_id,
                    "fansly_media_id": str(media_id),
                    "account_media_id": str(account_media_id) if account_media_id else None,
                    "title": f"{album_title} content",
                    "description": "",
                    "price": 0,
                    "media_type": media_type,
                    "thumbnail_url": thumbnail_url,
                    "is_active": True,
                }

                await asyncio.to_thread(
                    lambda p=row_payload: get_supabase()
                    .table("creator_vault_media")
                    .upsert(p, on_conflict="creator_id,fansly_media_id")
                    .execute()
                )
                synced += 1

        print(f"[VAULT SYNC] albums={len(albums)} synced_rows={synced}")
        return {"status": "ok", "synced": synced}


@app.get("/media/{account_id}/{content_id}")
async def get_media_url(account_id: str, content_id: str) -> dict:
    import httpx

    api_key = os.environ.get("APIFANSLY_API_KEY")
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"https://v1.apifansly.com/api/fansly/{account_id}/media/{content_id}",
            headers={"x-api-key": api_key},
            timeout=10,
        )
        print(f"[MEDIA] status={response.status_code} body={response.text[:300]}")
        return response.json()


@app.post("/generate-suggestions")
async def generate_suggestions_webhook(
    payload: WebhookPayload,
    background_tasks: BackgroundTasks,
) -> dict:
    if payload.type != "INSERT":
        return {"status": "skipped"}
    record = payload.record
    message_id = record.get("id")
    message_content = record.get("content")
    print(f"[WEBHOOK] message_id={message_id} role={record.get('role')} content={message_content[:30]}")
    if record.get("role") != "fan":
        return {"status": "skipped"}
    fansly_msg_id = record.get("fansly_message_id")
    if fansly_msg_id:
        return {"status": "skipped - handled by fansly webhook"}
    if message_id in _processed_messages:
        return {"status": "duplicate"}
    _processed_messages.add(message_id)
    if len(_processed_messages) > 1000:
        _processed_messages.clear()

    fan_id = record.get("fan_id")
    creator_id = record.get("creator_id")
    if not all([fan_id, creator_id, message_content, message_id]):
        return {"status": "skipped"}

    db = get_supabase()
    creator_row = await asyncio.to_thread(
        lambda: db.table("creators").select("auto_mode").eq("id", creator_id).single().execute()
    )
    auto_mode = (creator_row.data or {}).get("auto_mode", False)

    await process_incoming_fan_message(
        str(fan_id), str(creator_id), str(message_content), auto_mode, str(message_id),
    )
    return {"status": "ok"}


@app.post("/webhook/fansly")
async def fansly_webhook(payload: dict) -> dict:
    print(f"[FANSLY WEBHOOK RAW] {payload}")

    event = payload.get("event")
    data = payload.get("data") or {}

    if event == "tips.received":
        sender_id = str(data.get("fromUser", {}).get("id", ""))
        amount = data.get("netAmount", 0) or data.get("grossAmount", 0)

        print(f"[TIP] sender={sender_id} amount={amount}")

        if sender_id and amount:
            try:
                amt = float(amount)
            except (TypeError, ValueError):
                amt = 0.0
            if amt:
                db = get_supabase()
                fan_row = await asyncio.to_thread(
                    lambda: db.table("fans")
                    .select("id, total_spent")
                    .eq("platform_fan_id", sender_id)
                    .limit(1)
                    .execute()
                )
                rows = fan_row.data or []
                if rows:
                    r = rows[0]
                    new_total = int((r.get("total_spent") or 0) + round(amt))
                    fid = str(r["id"])
                    await asyncio.to_thread(
                        lambda: db.table("fans")
                        .update({"total_spent": new_total})
                        .eq("id", fid)
                        .execute()
                    )
                    print(f"[TIP] Updated total_spent={new_total} for fan={fid}")
        return {"status": "ok"}

    if event != "messages.received":
        return {"status": "skipped"}

    platform_fan_id = str(data.get("senderId", ""))
    message_content = (data.get("content") or "").strip()
    group_id = data.get("groupId", "")
    message_id = data.get("id", "")

    attachments_raw = data.get("attachments")
    if attachments_raw is None:
        attachments_raw = []
    elif not isinstance(attachments_raw, list):
        attachments_raw = [attachments_raw]
    has_attachments = len(attachments_raw) > 0

    if not platform_fan_id:
        return {"status": "skipped"}
    if not message_content and not has_attachments:
        return {"status": "skipped"}

    interactions = data.get("interactions") or []
    creator_platform_id = None
    if interactions:
        creator_platform_id = str(interactions[0].get("userId", "") or "")

    if not creator_platform_id:
        return {"status": "skipped"}

    if platform_fan_id == creator_platform_id:
        return {"status": "skipped"}

    print(
        f"[WEBHOOK] message_id={message_id} fan={platform_fan_id} "
        f"creator_platform={creator_platform_id} content={message_content[:50]}"
    )

    db = get_supabase()
    print(f"[DEBUG] looking up creator with fansly_account_id='{creator_platform_id}' len={len(creator_platform_id)}")
    creator_row = await asyncio.to_thread(
        lambda: db.table("creators")
        .select("id, auto_mode")
        .eq("fansly_account_id", creator_platform_id)
        .limit(1)
        .execute()
    )
    if not creator_row.data:
        print(f"[WEBHOOK] creator not found for platform_id={creator_platform_id}")
        return {"status": "creator_not_found"}

    creator_id = creator_row.data[0]["id"]
    auto_mode = creator_row.data[0].get("auto_mode", False)

    fan = await get_fan(creator_id, platform_fan_id)
    if not fan:
        fan = await create_fan(creator_id, platform_fan_id, f"Fan_{platform_fan_id[-6:]}")

    if message_id:
        mid = str(message_id)
        existing = await asyncio.to_thread(
            lambda: db.table("messages")
            .select("id")
            .eq("fansly_message_id", mid)
            .limit(1)
            .execute()
        )
        if existing.data:
            return {"status": "duplicate"}

    if group_id:
        await asyncio.to_thread(
            lambda: db.table("fans")
            .update({"fansly_group_id": str(group_id)})
            .eq("id", fan.id)
            .execute()
        )

    mid = str(message_id) if message_id else ""

    media_context = (
        {"attachments": [{"contentId": a.get("contentId")} for a in attachments_raw]}
        if attachments_raw
        else None
    )

    if mid:
        await save_message(
            fan.id,
            creator_id,
            "fan",
            message_content,
            fansly_message_id=mid,
            media_context=media_context,
        )
    else:
        await save_message(
            fan.id,
            creator_id,
            "fan",
            message_content,
            media_context=media_context,
        )

    await process_incoming_fan_message(
        fan.id,
        creator_id,
        message_content,
        auto_mode,
        mid if mid else None,
    )

    return {"status": "ok"}


@app.delete("/creators/{creator_id}")
async def delete_creator(creator_id: str) -> dict:
    db = get_supabase()

    await asyncio.to_thread(
        lambda cid=creator_id: db.table("reengagement_log").delete().eq("creator_id", cid).execute()
    )
    await asyncio.to_thread(
        lambda cid=creator_id: db.table("reengagement_settings").delete().eq("creator_id", cid).execute()
    )
    await asyncio.to_thread(
        lambda cid=creator_id: db.table("creator_vault_media").delete().eq("creator_id", cid).execute()
    )
    await asyncio.to_thread(
        lambda cid=creator_id: db.table("fan_lists").delete().eq("creator_id", cid).execute()
    )
    await asyncio.to_thread(
        lambda cid=creator_id: db.table("blocked_words").delete().eq("creator_id", cid).execute()
    )
    await asyncio.to_thread(
        lambda cid=creator_id: db.table("scripts").delete().eq("creator_id", cid).execute()
    )
    await asyncio.to_thread(
        lambda cid=creator_id: db.table("ppv_offers").delete().eq("creator_id", cid).execute()
    )

    fans = await asyncio.to_thread(
        lambda cid=creator_id: db.table("fans").select("id").eq("creator_id", cid).execute()
    )
    fan_ids = [f["id"] for f in (fans.data or [])]

    for fan_id in fan_ids:
        await asyncio.to_thread(
            lambda fid=fan_id: db.table("suggestions").delete().eq("fan_id", fid).execute()
        )
        await asyncio.to_thread(
            lambda fid=fan_id: db.table("messages").delete().eq("fan_id", fid).execute()
        )
        await asyncio.to_thread(
            lambda fid=fan_id: db.table("fan_list_members").delete().eq("fan_id", fid).execute()
        )

    await asyncio.to_thread(
        lambda cid=creator_id: db.table("fans").delete().eq("creator_id", cid).execute()
    )
    await asyncio.to_thread(
        lambda cid=creator_id: db.table("chatter_creators").delete().eq("creator_id", cid).execute()
    )
    await asyncio.to_thread(
        lambda cid=creator_id: db.table("creators").delete().eq("id", cid).execute()
    )

    return {"status": "ok"}


@app.get("/my-creators")
async def get_my_creators(user_id: str) -> dict:
    db = get_supabase()
    links = await asyncio.to_thread(
        lambda: db.table("chatter_creators")
        .select("creator_id")
        .eq("chatter_id", user_id)
        .execute()
    )
    creator_ids = [r["creator_id"] for r in (links.data or [])]
    if not creator_ids:
        return {"creators": []}

    creators = await asyncio.to_thread(
        lambda: db.table("creators")
        .select("id, platform_username, fansly_account_id, apifansly_account_id, persona, auto_mode")
        .in_("id", creator_ids)
        .execute()
    )
    return {"creators": creators.data or []}


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


from routes.fansly import fansly_router

app.include_router(fansly_router)

