"""FastAPI entrypoint for Cleopatra.

Routes are thin and delegate all logic to services.
"""

import asyncio
import json
import os
import random
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

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
_vault_sync_state: dict = {}


async def get_or_fetch_group_id(apifansly_id: str, platform_fan_id: str, fan_id: str) -> str | None:
    """Find the group_id for a fan by scanning recent chats."""
    import httpx

    api_key = os.environ.get("APIFANSLY_API_KEY")
    try:
        async with httpx.AsyncClient() as client:
            cursor = None
            for _ in range(5):  # check up to 5 pages
                params = {}
                if cursor:
                    params["cursor"] = cursor
                resp = await client.get(
                    f"https://v1.apifansly.com/api/fansly/{apifansly_id}/chats",
                    headers={"x-api-key": api_key},
                    params=params,
                    timeout=15,
                )
                data = resp.json()
                data_inner = data.get("data", {}).get("data", {})
                response_data = data_inner.get("response", {})
                cursor = data_inner.get("nextCursor")
                chats = response_data.get("data", [])
                accounts = response_data.get("aggregationData", {}).get("accounts", [])
                account_lookup = {str(a.get("id", "")): a for a in accounts}

                for chat in chats:
                    if str(chat.get("partnerAccountId", "")) == str(platform_fan_id):
                        group_id = str(chat.get("groupId", ""))
                        if group_id:
                            db = get_supabase()
                            update = {"fansly_group_id": group_id}
                            # Also grab display name and avatar
                            account = account_lookup.get(str(platform_fan_id), {})
                            display_name = account.get("displayName") or account.get("username")
                            if display_name:
                                update["display_name"] = display_name
                            avatar = account.get("avatar", {})
                            if avatar and avatar.get("locations"):
                                update["avatar_url"] = avatar["locations"][0].get("location")
                            await asyncio.to_thread(
                                lambda u=update: db.table("fans")
                                .update(u)
                                .eq("id", fan_id)
                                .execute()
                            )
                            print(f"[GROUP_ID] Found group_id={group_id} name={display_name} for fan={fan_id}")
                            return group_id
                if not cursor or not chats:
                    break
    except Exception as e:
        print(f"[GROUP_ID ERROR] {e}")
    return None


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
                "fansly_message_id": message_id,
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
        account_lookup: dict[str, dict] = {}
        cursor = None

        while True:
            params = {}
            if cursor is not None:
                params["cursor"] = cursor

            response = await client.get(
                f"https://v1.apifansly.com/api/fansly/{apifansly_id}/chats",
                headers={"x-api-key": api_key},
                params=params,
                timeout=30,
            )
            data = response.json()
            data_inner = data.get("data", {}).get("data", {})
            response_data = data_inner.get("response", {})
            cursor = data_inner.get("nextCursor")
            chats = response_data.get("data", [])
            accounts = response_data.get("aggregationData", {}).get("accounts", [])
            for a in accounts:
                aid = str(a.get("id", ""))
                if aid:
                    account_lookup[aid] = a

            if not all_chats:
                print(f"[SYNC RAW] {str(response_data)[:500]}")

            print(f"[SYNC CHATS] batch={len(chats)} total={len(all_chats)+len(chats)} nextCursor={cursor}")

            if not chats:
                break

            all_chats.extend(chats)

            if not cursor:
                break

        synced = 0
        for chat in all_chats:
            platform_fan_id = str(chat.get("partnerAccountId", ""))
            account_data = account_lookup.get(platform_fan_id, {})
            fan_name = (
                account_data.get("displayName")
                or account_data.get("username")
                or chat.get("partnerUsername", f"Fan_{platform_fan_id[-6:]}")
            )
            avatar_url = None
            avatar = account_data.get("avatar", {})
            if avatar and avatar.get("locations"):
                avatar_url = avatar["locations"][0].get("location")
            group_id = str(chat.get("groupId", ""))

            if not platform_fan_id or not group_id:
                continue

            fan = await get_fan(creator_id, platform_fan_id)
            if not fan:
                fan = await create_fan(creator_id, platform_fan_id, fan_name)

            update_payload = {
                "fansly_group_id": group_id,
                "display_name": fan_name,
            }
            if avatar_url:
                update_payload["avatar_url"] = avatar_url

            await asyncio.to_thread(
                lambda fid=fan.id, p=update_payload: db.table("fans").update(p).eq("id", fid).execute()
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

    print(f"[LOAD HISTORY URL] apifansly_id={apifansly_id} group_id={group_id}")

    async with httpx.AsyncClient() as client:
        while True:
            params = {"limit": 10}
            if cursor:
                params["cursor"] = cursor

            response = await client.get(
                f"https://v1.apifansly.com/api/fansly/{apifansly_id}/chats/{group_id}/messages",
                headers={"x-api-key": api_key},
                params=params,
                timeout=30,
            )
            data = response.json()
            data_inner = data.get("data", {}).get("data", {})
            response_data = data_inner.get("response", {})
            cursor = data_inner.get("nextCursor")
            messages = response_data.get("messages", [])
            account_media_batch = response_data.get("accountMedia", [])
            print(f"[CURSOR CHECK] keys={list(response_data.keys())} cursor={cursor}")
            if not all_messages:
                print(f"[HISTORY END] {str(response_data)[-500:]}")

            print(f"[LOAD HISTORY] batch={len(messages)} total={len(all_messages)+len(messages)} nextCursor={cursor}")

            for am in account_media_batch:
                content_id_1 = str(am.get("id", ""))
                content_id_2 = str(am.get("mediaId", ""))
                media = am.get("media", {})
                locations = media.get("locations", [])
                variants = media.get("variants", [])
                url = None
                if locations:
                    url = locations[0].get("location")
                elif variants and variants[0].get("locations"):
                    url = variants[0]["locations"][0].get("location")
                price = int(am.get("price") or 0)
                purchased = bool(am.get("purchased", am.get("isPurchased", False)))
                access = am.get("access")
                media_info = {
                    "url": url,
                    "price": price / 100 if price > 100 else price,
                    "is_ppv": price > 0,
                    "purchased": purchased,
                    "access": access,
                }
                if content_id_1:
                    all_media[content_id_1] = media_info
                if content_id_2 and content_id_2 != content_id_1:
                    all_media[content_id_2] = media_info

            all_messages.extend(messages)

            if not cursor or not messages:
                break

    print(f"[MEDIA LOOKUP] keys={list(all_media.keys())[:5]}")

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
                print(f"[ATT RESOLVE] contentId={content_id} found={content_id in all_media}")
                info = all_media.get(content_id)
                if isinstance(info, dict):
                    resolved.append({
                        "contentId": content_id,
                        "url": info.get("url"),
                        "type": att.get("contentType", 1),
                        "price": info.get("price"),
                        "is_ppv": info.get("is_ppv"),
                        "purchased": info.get("purchased"),
                        "access": info.get("access"),
                    })
                else:
                    resolved.append({
                        "contentId": content_id,
                        "url": info,
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

    db = get_supabase()
    creator_row = await asyncio.to_thread(
        lambda: db.table("creators")
        .select("apifansly_account_id")
        .eq("id", creator_id)
        .single()
        .execute()
    )

    apifansly_id = (creator_row.data or {}).get("apifansly_account_id")
    api_key = os.environ.get("APIFANSLY_API_KEY")

    async with httpx.AsyncClient() as client:
        # Step 1: Get all albums
        resp = await client.get(
            f"https://v1.apifansly.com/api/fansly/{apifansly_id}/vault/albums",
            headers={"x-api-key": api_key},
            timeout=30,
        )
        albums_data = resp.json()
        albums = albums_data.get("data", {}).get("data", {}).get("response", {}).get("albums", [])
        print(f"[VAULT] found {len(albums)} albums")

        total_synced = 0

        # Step 2: For each album, fetch media
        for album in albums:
            album_id = album.get("id")
            album_title = album.get("title") or f"Album_{album_id}"
            item_count = album.get("itemCount", 0)
            print(f"[VAULT] album={album_title} items={item_count}")

            cursor = None
            while True:
                params = {"limit": 50}
                if cursor:
                    params["cursor"] = cursor

                media_resp = await client.get(
                    f"https://v1.apifansly.com/api/fansly/{apifansly_id}/vault/albums/{album_id}/media",
                    headers={"x-api-key": api_key},
                    params=params,
                    timeout=30,
                )
                media_data = media_resp.json()
                print(f"[VAULT ALBUM RAW] {str(media_data)[:500]}")
                data_l1 = media_data.get("data", {})
                cursor = data_l1.get("nextCursor")
                outer = data_l1.get("data", {})
                response_data = outer.get("response", {})
                items = response_data if isinstance(response_data, list) else (
                    response_data.get("data") or response_data.get("media") or []
                )

                print(f"[VAULT] album={album_title} batch={len(items)} cursor={cursor}")

                if not items:
                    # Log raw to debug structure
                    print(f"[VAULT RAW] {str(media_data)[:300]}")
                    break

                for item in items:
                    media = item.get("media", {})
                    media_id = str(media.get("id", ""))
                    mimetype = media.get("mimetype", "")
                    locations = media.get("locations", [])
                    variants = media.get("variants", [])
                    price = item.get("price", 0)

                    url = None
                    if locations:
                        url = locations[0].get("location")
                    elif variants and variants[0].get("locations"):
                        url = variants[0]["locations"][0].get("location")

                    if not media_id or not url:
                        continue

                    await asyncio.to_thread(
                        lambda cid=creator_id, mid=media_id, u=url, mt=mimetype, fn=media.get("filename", ""), aid=album_id, at=album_title, pr=price: db.table("creator_vault_media").upsert({
                            "creator_id": cid,
                            "media_id": mid,
                            "fansly_media_id": mid,
                            "url": u,
                            "mimetype": mt,
                            "filename": fn,
                            "album_id": aid,
                            "album_title": at,
                            "price": pr,
                        }, on_conflict="creator_id,media_id").execute()
                    )
                    total_synced += 1

                if not cursor:
                    break

        return {"status": "ok", "synced": total_synced}


@app.post("/sync-vault-start/{creator_id}")
async def sync_vault_start(creator_id: str) -> dict:
    if creator_id in _vault_sync_state and _vault_sync_state[creator_id].get("status") == "running":
        return {"status": "already_running"}
    _vault_sync_state[creator_id] = {"status": "running", "synced": 0, "total": 0, "album": ""}
    asyncio.create_task(_run_vault_sync(creator_id))
    return {"status": "started"}


@app.get("/sync-vault-status/{creator_id}")
async def sync_vault_status(creator_id: str) -> dict:
    state = _vault_sync_state.get(creator_id, {"status": "idle", "synced": 0, "total": 0, "album": ""})
    return state


async def _run_vault_sync(creator_id: str) -> None:
    import httpx

    db = get_supabase()
    try:
        creator_row = await asyncio.to_thread(
            lambda: db.table("creators")
            .select("apifansly_account_id")
            .eq("id", creator_id)
            .single()
            .execute()
        )
        apifansly_id = (creator_row.data or {}).get("apifansly_account_id")
        api_key = os.environ.get("APIFANSLY_API_KEY")

        existing_rows = await asyncio.to_thread(
            lambda: db.table("creator_vault_media")
            .select("media_id")
            .eq("creator_id", creator_id)
            .execute()
        )
        existing_ids = {r["media_id"] for r in (existing_rows.data or [])}

        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"https://v1.apifansly.com/api/fansly/{apifansly_id}/vault/albums",
                headers={"x-api-key": api_key},
                timeout=30,
            )
            albums_data = resp.json()
            albums = albums_data.get("data", {}).get("data", {}).get("response", {}).get("albums", [])

            total = sum(a.get("itemCount", 0) for a in albums)
            already = len(existing_ids)
            new_total = max(total - already, 0)
            synced = 0

            _vault_sync_state[creator_id] = {"status": "running", "synced": 0, "total": 0, "album": "Starting..."}

            for album in albums:
                album_id = album.get("id")
                album_title = album.get("title") or f"Album_{album_id}"
                cursor = None
                consecutive_dupe_batches = 0

                while True:
                    params = {"limit": 50}
                    if cursor:
                        params["cursor"] = cursor

                    media_resp = await client.get(
                        f"https://v1.apifansly.com/api/fansly/{apifansly_id}/vault/albums/{album_id}/media",
                        headers={"x-api-key": api_key},
                        params=params,
                        timeout=30,
                    )
                    media_data = media_resp.json()
                    data_l1 = media_data.get("data", {})
                    cursor = data_l1.get("nextCursor")
                    outer = data_l1.get("data", {})
                    response_data = outer.get("response", {})
                    items = response_data if isinstance(response_data, list) else []

                    if not items:
                        break

                    batch = []
                    all_dupes = True
                    for item in items:
                        media = item.get("media", {})
                        media_id = str(media.get("id", ""))
                        if not media_id or media_id in existing_ids:
                            continue
                        all_dupes = False

                        mimetype = media.get("mimetype", "")
                        locations = media.get("locations", [])
                        variants = media.get("variants", [])
                        price = item.get("price", 0)

                        url = None
                        if locations:
                            url = locations[0].get("location")
                        elif variants and variants[0].get("locations"):
                            url = variants[0]["locations"][0].get("location")

                        if not url:
                            continue

                        batch.append({
                            "creator_id": creator_id,
                            "media_id": media_id,
                            "fansly_media_id": media_id,
                            "url": url,
                            "mimetype": mimetype,
                            "filename": media.get("filename", ""),
                            "album_id": album_id,
                            "album_title": album_title,
                            "price": price,
                        })
                        existing_ids.add(media_id)
                    consecutive_dupe_batches = consecutive_dupe_batches + 1 if all_dupes else 0

                    if batch:
                        await asyncio.to_thread(
                            lambda b=batch: db.table("creator_vault_media")
                            .upsert(b, on_conflict="creator_id,media_id")
                            .execute()
                        )
                        synced += len(batch)

                    _vault_sync_state[creator_id] = {"status": "running", "synced": synced, "total": new_total, "album": album_title}
                    print(f"[VAULT SYNC] album={album_title} synced={synced}/{new_total} cursor={cursor}")

                    if not cursor or consecutive_dupe_batches >= 3:
                        break

        _vault_sync_state[creator_id] = {"status": "done", "synced": synced, "total": new_total, "album": ""}
        print(f"[VAULT SYNC] done synced={synced}")

    except Exception as e:
        import traceback
        print(f"[VAULT SYNC ERROR] {e}")
        print(traceback.format_exc())
        _vault_sync_state[creator_id] = {"status": "error", "synced": 0, "total": 0, "album": str(e)}


@app.post("/upload-vault-media/{creator_id}")
async def upload_vault_media(creator_id: str, request: Request) -> dict:
    import httpx

    db = get_supabase()
    creator_row = await asyncio.to_thread(
        lambda: db.table("creators")
        .select("apifansly_account_id")
        .eq("id", creator_id)
        .single()
        .execute()
    )
    apifansly_id = (creator_row.data or {}).get("apifansly_account_id")
    api_key = os.environ.get("APIFANSLY_API_KEY")

    form = await request.form()
    file = form.get("file")
    album_title = str(form.get("album_title") or "Uncategorized")
    album_id = str(form.get("album_id") or "")

    if not file:
        return {"status": "error", "message": "no file"}

    file_bytes = await file.read()
    filename = file.filename
    mimetype = file.content_type

    async with httpx.AsyncClient() as client:
        upload_resp = await client.post(
            f"https://v1.apifansly.com/api/fansly/{apifansly_id}/media/upload",
            headers={"x-api-key": api_key},
            files={"file": (filename, file_bytes, mimetype)},
            timeout=60,
        )
        upload_data = upload_resp.json()
        print(f"[UPLOAD] initiate response: {upload_data}")

        job_id = upload_data.get("data", {}).get("jobId")
        if not job_id:
            return {"status": "error", "message": "no jobId returned", "raw": str(upload_data)}

        media_id = None
        url = None
        for attempt in range(30):
            await asyncio.sleep(2)
            status_resp = await client.get(
                f"https://v1.apifansly.com/api/fansly/media/upload/{job_id}/status",
                headers={"x-api-key": api_key},
                timeout=15,
            )
            status_data = status_resp.json()
            state = status_data.get("data", {}).get("state")
            print(f"[UPLOAD] job={job_id} attempt={attempt+1} state={state}")

            if state == "completed":
                result = status_data.get("data", {}).get("result", {})
                media_id = str(result.get("mediaId", ""))
                account_media = result.get("accountMedia", [])
                if account_media:
                    media_obj = account_media[0].get("media", {})
                    locations = media_obj.get("locations", [])
                    variants = media_obj.get("variants", [])
                    if locations:
                        url = locations[0].get("location")
                    elif variants and variants[0].get("locations"):
                        url = variants[0]["locations"][0].get("location")
                break
            elif state == "failed":
                return {"status": "error", "message": "upload job failed"}

        if not media_id or not url:
            return {"status": "error", "message": "upload timed out or no media URL"}

        ai_description = str(form.get("ai_description") or "")
        row = {
            "creator_id": creator_id,
            "media_id": media_id,
            "fansly_media_id": media_id,
            "url": url,
            "mimetype": mimetype,
            "filename": filename,
            "album_id": album_id,
            "album_title": album_title,
            "price": 0,
        }
        if ai_description:
            row["ai_description"] = ai_description
        db_result = await asyncio.to_thread(
            lambda: db.table("creator_vault_media")
            .upsert(row, on_conflict="creator_id,media_id")
            .select()
            .single()
            .execute()
        )
        saved = db_result.data or row
        print(f"[UPLOAD] saved to DB media_id={media_id}")
        # Auto-categorize in background
        if saved.get("id"):
            asyncio.create_task(_categorize_single_item_and_save(saved))
        return {"status": "ok", "item": saved}


async def _categorize_single_item_and_save(item: dict) -> None:
    try:
        result = await _categorize_single_item(item)
        db = get_supabase()
        await asyncio.to_thread(
            lambda: db.table("creator_vault_media").update({
                "content_category": result["content_category"],
                "ai_description": result["ai_description"],
                "price_min": result["price_min"],
                "price_max": result["price_max"],
                "explicitness_level": result.get("explicitness", 3),
                "good_for": result.get("good_for", "standalone"),
                "tags": result.get("tags", []),
                "scene_id": result.get("scene_id", ""),
                "scene_location": result.get("scene_location", ""),
                "scene_outfit": result.get("scene_outfit", ""),
                "scene_lighting": result.get("scene_lighting", ""),
            }).eq("id", item["id"]).execute()
        )
        print(f"[UPLOAD CATEGORIZE] item={item['id']} category={result['content_category']}")
    except Exception as e:
        print(f"[UPLOAD CATEGORIZE ERROR] {e}")


VAULT_CATEGORIES = {
    "teaser_clothed":   {"min": 0,   "max": 0,   "label": "Clothed teaser (free)"},
    "teaser_bundle":    {"min": 0,   "max": 0,   "label": "Teaser bundle no nudity (free)"},
    "legs_feet":        {"min": 15,  "max": 70,  "label": "Legs / feet / armpits"},
    "lingerie_photo":   {"min": 10,  "max": 80,  "label": "Lingerie photo"},
    "lingerie_video":   {"min": 15,  "max": 90,  "label": "Lingerie video"},
    "nude_photo":       {"min": 15,  "max": 80,  "label": "Nude photo"},
    "striptease_video": {"min": 15,  "max": 100, "label": "Striptease video"},
    "closeup_photo":    {"min": 25,  "max": 130, "label": "Closeup photo"},
    "closeup_video":    {"min": 25,  "max": 130, "label": "Closeup video"},
    "dictate_video":    {"min": 15,  "max": 50,  "label": "Dictate / dirty talk video"},
    "solo_toy_video":   {"min": 30,  "max": 150, "label": "Solo / toy / orgasm video"},
    "solo_toy_photo":   {"min": 20,  "max": 80,  "label": "Solo / toy photo"},
    "bg_content":       {"min": 50,  "max": 300, "label": "BG (boy-girl) content"},
    "task":             {"min": 10,  "max": 50,  "label": "Task / custom request"},
    "other":            {"min": 0,   "max": 0,   "label": "Other / unclear"},
}

CATEGORY_LIST = "\n".join([
    f"- {k}: {v['label']} (price range ${v['min']}-${v['max']})"
    for k, v in VAULT_CATEGORIES.items()
])


async def _categorize_single_item(item: dict) -> dict:
    """Run Claude Vision on one vault item and return category + description."""
    from anthropic import AsyncAnthropic
    import httpx

    client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    url = item.get("url", "")
    mimetype = item.get("mimetype", "")
    item_id = item.get("id", "")
    is_video = mimetype.startswith("video") if mimetype else False

    try:
        if is_video:
            # For videos we can't send frames easily — use filename + album as context
            filename = item.get("filename", "")
            album = item.get("album_title", "")
            prompt = (
                f"This is a video file. Filename: '{filename}'. Album: '{album}'.\n"
                f"Based on the filename and album name only, classify this into one of these categories:\n{CATEGORY_LIST}\n\n"
                "Return ONLY valid JSON:\n"
                '{"category": "category_key", "description": "one sentence description of likely content", "mood": "playful|intimate|explicit|teasing"}'
            )
            response = await client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=150,
                messages=[{"role": "user", "content": prompt}],
            )
        else:
            # Fetch image and send to Claude Vision
            img_b64 = None
            media_type = mimetype if mimetype in ["image/jpeg", "image/png", "image/webp", "image/gif"] else "image/jpeg"
            try:
                async with httpx.AsyncClient() as hc:
                    img_resp = await hc.get(url, timeout=15)
                    if img_resp.status_code == 200 and len(img_resp.content) > 1000:
                        img_data = img_resp.content
                        # Resize if over 4MB to stay under Claude's 5MB limit
                        if len(img_data) > 4 * 1024 * 1024:
                            try:
                                from PIL import Image
                                import io
                                img_obj = Image.open(io.BytesIO(img_data))
                                img_obj.thumbnail((1024, 1024), Image.LANCZOS)
                                buf = io.BytesIO()
                                img_obj.save(buf, format="JPEG", quality=85)
                                img_data = buf.getvalue()
                                media_type = "image/jpeg"
                                print(f"[CATEGORIZE] resized large image item={item_id} to {len(img_data)} bytes")
                            except Exception as resize_err:
                                print(f"[CATEGORIZE] resize failed item={item_id}: {resize_err}")
                        img_b64 = __import__("base64").b64encode(img_data).decode()
            except Exception as fetch_err:
                print(f"[CATEGORIZE] image fetch failed for item={item_id}: {fetch_err}")

            if img_b64:
                response = await client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=400,
                    messages=[{
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": img_b64,
                                },
                            },
                            {
                                "type": "text",
                                "text": (
                                    f"Classify this OnlyFans creator image. Pick one category:\n{CATEGORY_LIST}\n\n"
                                    "Return ONLY a single line of JSON, no newlines, no formatting:\n"
                                    '{"category":"category_key","description":"1-2 sentences","mood":"playful","explicitness":3,"good_for":"opener","tags":["tag1"],"scene_location":"bedroom","scene_outfit":"red lingerie","scene_lighting":"dim","scene_id":"bedroom-red-lingerie"}'
                                ),
                            },
                        ],
                    }],
                )
            else:
                # CDN URL expired or fetch failed — fall back to filename+album
                filename = item.get("filename", "")
                album = item.get("album_title", "")
                print(f"[CATEGORIZE] falling back to filename analysis for item={item_id}")
                response = await client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=300,
                    messages=[{"role": "user", "content": (
                        f"This is an OnlyFans image. Filename: '{filename}'. Album: '{album}'.\n"
                        f"Based on filename and album only, classify into one of:\n{CATEGORY_LIST}\n\n"
                        "Return ONLY valid JSON:\n"
                        "{\n"
                        '  "category": "category_key",\n'
                        '  "description": "best guess description based on filename/album",\n'
                        '  "mood": "playful|intimate|explicit|teasing",\n'
                        '  "explicitness": 3,\n'
                        '  "good_for": "opener|mid_session|closer|standalone",\n'
                        '  "tags": [],\n'
                        '  "scene": {"location": "unknown", "outfit": "unknown", "hair": "unknown", "lighting": "unknown", "scene_id": "unknown"}\n'
                        "}"
                    )}],
                )

        content = response.content[0].text.strip()
        content = content.replace("```json", "").replace("```", "").strip()
        print(f"[CATEGORIZE RAW] item={item_id} response={content[:300]}")
        data = json.loads(content)
        category = data.get("category", "other")
        if category not in VAULT_CATEGORIES:
            category = "other"
        description = data.get("description", "")
        mood = data.get("mood", "")
        price_info = VAULT_CATEGORIES[category]

        explicitness = min(max(int(data.get("explicitness", 3)), 1), 5)
        good_for = data.get("good_for", "standalone")
        if good_for not in ["opener", "mid_session", "closer", "standalone"]:
            good_for = "standalone"
        tags = data.get("tags", [])
        if not isinstance(tags, list):
            tags = []
        scene_id = data.get("scene_id", "")
        location = data.get("scene_location", "")
        outfit = data.get("scene_outfit", "")
        lighting = data.get("scene_lighting", "")

        full_description = f"[{mood}|{good_for}|explicit:{explicitness}] {description}"
        if outfit:
            full_description += f" Outfit: {outfit}."
        if location:
            full_description += f" Location: {location}."
        if tags:
            full_description += f" Tags: {', '.join(tags)}."

        return {
            "id": item_id,
            "content_category": category,
            "ai_description": full_description,
            "price_min": price_info["min"],
            "price_max": price_info["max"],
            "explicitness": explicitness,
            "good_for": good_for,
            "tags": tags,
            "scene_id": scene_id,
            "scene_location": location,
            "scene_outfit": outfit,
            "scene_lighting": lighting,
        }

    except Exception as e:
        print(f"[CATEGORIZE] item={item_id} error={e}")
        return {
            "id": item_id,
            "content_category": "other",
            "ai_description": "",
            "price_min": 0,
            "price_max": 0,
        }


_categorize_state: dict = {}


@app.post("/categorize-vault/{creator_id}")
async def categorize_vault(creator_id: str) -> dict:
    if _categorize_state.get(creator_id, {}).get("status") == "running":
        return {"status": "already_running", "state": _categorize_state[creator_id]}
    _categorize_state[creator_id] = {"status": "running", "done": 0, "total": 0, "errors": 0}
    asyncio.create_task(_run_vault_categorization(creator_id))
    return {"status": "started"}


@app.get("/categorize-vault-status/{creator_id}")
async def categorize_vault_status(creator_id: str) -> dict:
    return _categorize_state.get(creator_id, {"status": "idle", "done": 0, "total": 0})


async def _run_vault_categorization(creator_id: str) -> None:
    db = get_supabase()
    try:
        # Paginate to get ALL uncategorized items past 1000 limit
        all_items = []
        page_size = 1000
        from_idx = 0
        while True:
            rows = await asyncio.to_thread(
                lambda f=from_idx: db.table("creator_vault_media")
                .select("id, url, mimetype, filename, album_title")
                .eq("creator_id", creator_id)
                .or_("content_category.is.null,content_category.eq.")
                .range(f, f + page_size - 1)
                .execute()
            )
            batch = rows.data or []
            all_items.extend(batch)
            if len(batch) < page_size:
                break
            from_idx += page_size

        total = len(all_items)
        _categorize_state[creator_id]["total"] = total
        print(f"[CATEGORIZE] creator={creator_id} uncategorized={total}")

        done = 0
        errors = 0
        # 2 concurrent max to respect 30k token/min rate limit
        batch_size = 2
        for i in range(0, total, batch_size):
            batch = all_items[i:i + batch_size]
            results = await asyncio.gather(
                *[_categorize_single_item_with_retry(item) for item in batch],
                return_exceptions=True,
            )
            for result in results:
                if isinstance(result, Exception):
                    errors += 1
                    continue
                await asyncio.to_thread(
                    lambda r=result: db.table("creator_vault_media").update({
                        "content_category": r["content_category"],
                        "ai_description": r["ai_description"],
                        "price_min": r["price_min"],
                        "price_max": r["price_max"],
                        "explicitness_level": r.get("explicitness", 3),
                        "good_for": r.get("good_for", "standalone"),
                        "tags": r.get("tags", []),
                        "scene_id": r.get("scene_id", ""),
                        "scene_location": r.get("scene_location", ""),
                        "scene_outfit": r.get("scene_outfit", ""),
                        "scene_lighting": r.get("scene_lighting", ""),
                    }).eq("id", r["id"]).execute()
                )
                done += 1
            _categorize_state[creator_id].update({"done": done, "errors": errors})
            print(f"[CATEGORIZE] done={done}/{total} errors={errors}")
            await asyncio.sleep(1.5)  # respect rate limit between batches

        _categorize_state[creator_id]["status"] = "done"
        print(f"[CATEGORIZE] complete done={done} errors={errors}")

    except Exception as e:
        import traceback
        print(f"[CATEGORIZE ERROR] {e}")
        traceback.print_exc()
        _categorize_state[creator_id]["status"] = "error"


async def _categorize_single_item_with_retry(item: dict, max_retries: int = 3) -> dict:
    """Wrap _categorize_single_item with exponential backoff on 429."""
    for attempt in range(max_retries):
        try:
            return await _categorize_single_item(item)
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                wait = 10 * (attempt + 1)
                print(f"[CATEGORIZE] rate limited, waiting {wait}s before retry")
                await asyncio.sleep(wait)
            else:
                raise
    return await _categorize_single_item(item)


@app.post("/recategorize-item/{item_id}")
async def recategorize_item(item_id: str) -> dict:
    db = get_supabase()
    row = await asyncio.to_thread(
        lambda: db.table("creator_vault_media")
        .select("id, url, mimetype, filename, album_title")
        .eq("id", item_id)
        .single()
        .execute()
    )
    item = row.data
    if not item:
        return {"status": "error", "message": "item not found"}

    result = await _categorize_single_item(item)
    await asyncio.to_thread(
        lambda: db.table("creator_vault_media").update({
            "content_category": result["content_category"],
            "ai_description": result["ai_description"],
            "price_min": result["price_min"],
            "price_max": result["price_max"],
            "explicitness_level": result.get("explicitness", 3),
            "good_for": result.get("good_for", "standalone"),
            "tags": result.get("tags", []),
            "scene_id": result.get("scene_id", ""),
            "scene_location": result.get("scene_location", ""),
            "scene_outfit": result.get("scene_outfit", ""),
            "scene_lighting": result.get("scene_lighting", ""),
        }).eq("id", item_id).execute()
    )
    updated = await asyncio.to_thread(
        lambda: db.table("creator_vault_media")
        .select("*")
        .eq("id", item_id)
        .single()
        .execute()
    )
    return {"status": "ok", "item": updated.data}


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


async def _enrich_fan_profile(fan_id: str, creator_id: str, platform_fan_id: str) -> None:
    """Fetch real username, avatar and group_id by scanning chats list."""
    try:
        db = get_supabase()
        creator_row = await asyncio.to_thread(
            lambda: db.table("creators")
            .select("apifansly_account_id")
            .eq("id", creator_id)
            .single()
            .execute()
        )
        apifansly_id = (creator_row.data or {}).get("apifansly_account_id")
        if apifansly_id:
            await get_or_fetch_group_id(apifansly_id, platform_fan_id, fan_id)
    except Exception as e:
        print(f"[FAN ENRICH ERROR] {e}")


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

    # Detect outgoing message (creator sending to fan)
    # senderId = creator, interactions[0].userId = fan
    db_check = get_supabase()
    creator_row_check = await asyncio.to_thread(
        lambda: db_check.table("creators")
        .select("id, auto_mode, fansly_account_id, auto_mode_new_fans")
        .eq("fansly_account_id", platform_fan_id)
        .limit(1)
        .execute()
    )
    if creator_row_check.data:
        # Check if recipient is also a creator — if so, skip outgoing capture
        recipient_creator_check = await asyncio.to_thread(
            lambda: db_check.table("creators")
            .select("id")
            .eq("fansly_account_id", creator_platform_id)
            .limit(1)
            .execute()
        )
        if not recipient_creator_check.data:
            # platform_fan_id is a creator, recipient is a real fan — outgoing message
            outgoing_creator_id = creator_row_check.data[0]["id"]
            fan_recipient_id = creator_platform_id
            if fan_recipient_id and group_id:
                existing_fan = await get_fan(outgoing_creator_id, fan_recipient_id)
                if not existing_fan:
                    existing_fan = await create_fan(
                        outgoing_creator_id, fan_recipient_id, f"Fan_{fan_recipient_id[-6:]}"
                    )
                    asyncio.create_task(
                        _enrich_fan_profile(existing_fan.id, outgoing_creator_id, fan_recipient_id)
                    )
                    auto_new = creator_row_check.data[0].get("auto_mode_new_fans", False)
                    if auto_new:
                        await asyncio.to_thread(
                            lambda fid=existing_fan.id: get_supabase()
                            .table("fans").update({"auto_mode": True}).eq("id", fid).execute()
                        )
                await asyncio.to_thread(
                    lambda fid=existing_fan.id, gid=str(group_id): get_supabase()
                    .table("fans")
                    .update({"fansly_group_id": gid})
                    .eq("id", fid)
                    .execute()
                )
                if message_content and message_id:
                    msg_id = str(message_id)
                    existing_msg = await asyncio.to_thread(
                        lambda: get_supabase().table("messages")
                        .select("id")
                        .eq("fansly_message_id", msg_id)
                        .limit(1)
                        .execute()
                    )
                    if not existing_msg.data:
                        await save_message(
                            existing_fan.id,
                            outgoing_creator_id,
                            "creator",
                            message_content,
                            fansly_message_id=msg_id,
                        )
                print(f"[OUTGOING] Captured creator→fan msg, fan={existing_fan.id} group={group_id}")
            return {"status": "outgoing_captured"}
        # Both are creators (test scenario) — sender is fan, recipient is creator
        # The normal webhook flow below will handle this correctly
        # since creator_platform_id = interactions[0].userId = the real creator
        pass

    print(
        f"[WEBHOOK] message_id={message_id} fan={platform_fan_id} "
        f"creator_platform={creator_platform_id} content={message_content[:50]}"
    )

    db = get_supabase()
    print(f"[DEBUG] looking up creator with fansly_account_id='{creator_platform_id}' len={len(creator_platform_id)}")
    creator_row = await asyncio.to_thread(
        lambda: db.table("creators")
        .select("id, auto_mode, auto_mode_new_fans")
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
        asyncio.create_task(_enrich_fan_profile(fan.id, creator_id, platform_fan_id))

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


@app.post("/plan-session/{creator_id}/{fan_id}")
async def plan_session(creator_id: str, fan_id: str) -> dict:
    """
    Called when a sexting session is starting.
    Picks 5-8 vault items ordered by escalating explicitness,
    grouped by scene continuity where possible.
    """
    from db.queries import get_fan_by_id, get_sent_ppv, get_vault_for_session, save_fan_session
    import json as _json

    fan = await get_fan_by_id(fan_id)
    sent_ppv = await get_sent_ppv(fan_id)
    purchased_ids = {s["media_id"] for s in sent_ppv if s.get("purchased")}
    sent_ids = {s["media_id"] for s in sent_ppv}
    exclude = purchased_ids

    kinks = []
    if fan and fan.ai_summary:
        kinks = fan.ai_summary.get("kinks", [])

    vault = await get_vault_for_session(creator_id, fan_kinks=kinks, exclude_media_ids=exclude)
    if not vault:
        return {"status": "no_content", "session": None}

    from anthropic import AsyncAnthropic

    client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    vault_summary = "\n".join(
        [
            f"[{i['media_id']}] cat={i['category']} explicit={i['explicitness']} scene={i['scene_id']} good_for={i['good_for']} price=${i['price_min']}-${i['price_max']} desc={i['description'][:80]}"
            for i in vault[:80]
        ]
    )

    fan_kinks_str = ", ".join(kinks) if kinks else "unknown"
    fan_spent = fan.total_spent if fan else 0
    fan_tier = fan.spend_tier if fan else "cold"
    prompt = ""
    prompt += "You are planning a sexting content session for an OnlyFans fan.\n\n"
    prompt += "Fan profile:\n"
    prompt += f"- Total spent: ${fan_spent} ({fan_tier} tier)\n"
    prompt += f"- Known kinks/interests: {fan_kinks_str}\n"
    prompt += f"- Already sent (not purchased): {len(sent_ids - purchased_ids)} items\n"
    prompt += f"- Already purchased: {len(purchased_ids)} items\n\n"
    prompt += "Available vault content (media_id | details):\n"
    prompt += f"{vault_summary}\n\n"
    prompt += "Select 6-8 items that would make the best escalating sexting session for this fan.\n"
    prompt += "Rules:\n"
    prompt += "1. Start with lower explicitness (1-3), escalate to higher (4-5)\n"
    prompt += "2. Prefer items matching fan's known kinks\n"
    prompt += "3. Group by scene_id where possible for continuity - avoid jumping between different scenes too quickly\n"
    prompt += "4. Mix good_for values: start with opener, mid_session items, end with closer\n"
    prompt += "5. Never pick the same scene_id more than 3 times in a row\n"
    prompt += f"6. Pick realistic prices based on fan's spend tier ({fan_tier})\n\n"
    prompt += "Return ONLY a JSON array of objects:\n"
    prompt += '[{"media_id":"id","price":25,"reason":"why this fits here","transition":"natural line to say before sending this e.g. \'I actually took this last night...\'"}]'

    response = await client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    )

    content = response.content[0].text.strip().replace("```json", "").replace("```", "").strip()

    try:
        plan = _json.loads(content)
        vault_by_id = {i["media_id"]: i for i in vault}
        enriched = []
        for item in plan:
            mid = item.get("media_id", "")
            vault_item = vault_by_id.get(mid, {})
            enriched.append(
                {
                    "media_id": mid,
                    "price": item.get("price", vault_item.get("price_min", 15)),
                    "reason": item.get("reason", ""),
                    "transition": item.get("transition", ""),
                    "category": vault_item.get("category", ""),
                    "description": vault_item.get("description", ""),
                    "scene_id": vault_item.get("scene_id", ""),
                    "is_video": vault_item.get("is_video", False),
                    "sent": False,
                    "purchased": False,
                }
            )

        session = {
            "plan": enriched,
            "current_index": 0,
            "started_at": __import__("datetime").datetime.utcnow().isoformat(),
            "fan_kinks": kinks,
        }
        await save_fan_session(fan_id, session)
        return {"status": "ok", "session": session}

    except Exception as e:
        print(f"[SESSION PLAN ERROR] {e}")
        return {"status": "error", "message": str(e)}


@app.get("/session/{fan_id}")
async def get_session(fan_id: str) -> dict:
    from db.queries import get_fan_session

    session = await get_fan_session(fan_id)
    return {"session": session}


@app.post("/session/{fan_id}/advance")
async def advance_session(fan_id: str) -> dict:
    """Mark current session item as sent and move to next."""
    from db.queries import get_fan_session, save_fan_session

    session = await get_fan_session(fan_id)
    if not session:
        return {"status": "no_session"}
    plan = session.get("plan", [])
    idx = session.get("current_index", 0)
    if idx < len(plan):
        plan[idx]["sent"] = True
        session["current_index"] = idx + 1
        await save_fan_session(fan_id, session)
    return {
        "status": "ok",
        "next_index": session["current_index"],
        "remaining": len(plan) - session["current_index"],
    }


@app.post("/session/{fan_id}/purchased/{media_id}")
async def mark_session_purchased(fan_id: str, media_id: str) -> dict:
    """Mark a session item as purchased."""
    from db.queries import get_fan_session, save_fan_session

    session = await get_fan_session(fan_id)
    if not session:
        return {"status": "no_session"}
    for item in session.get("plan", []):
        if item["media_id"] == media_id:
            item["purchased"] = True
            break
    await save_fan_session(fan_id, session)
    return {"status": "ok"}


@app.delete("/session/{fan_id}")
async def clear_session(fan_id: str) -> dict:
    """End and clear the active session."""
    from db.queries import save_fan_session

    await save_fan_session(fan_id, None)
    return {"status": "ok"}


@app.post("/enrich-fan/{fan_id}")
async def enrich_fan_endpoint(fan_id: str) -> dict:
    db = get_supabase()
    fan_row = await asyncio.to_thread(
        lambda: db.table("fans")
        .select("platform_fan_id, creator_id")
        .eq("id", fan_id)
        .single()
        .execute()
    )
    data = fan_row.data or {}
    platform_fan_id = data.get("platform_fan_id")
    creator_id = data.get("creator_id")
    if not platform_fan_id or not creator_id:
        return {"status": "error", "message": "fan not found"}
    await _enrich_fan_profile(fan_id, creator_id, platform_fan_id)
    return {"status": "ok"}


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


from routes.fansly import fansly_router

app.include_router(fansly_router)

