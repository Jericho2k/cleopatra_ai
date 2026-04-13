"""FastAPI entrypoint for Cleopatra.

Routes are thin and delegate all logic to services.
"""

import asyncio
import os
from contextlib import asynccontextmanager

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

    print(f"[AUTO MODE] creator={creator_id} auto_mode={auto_mode} fan={fan_id}")

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

    if auto_mode and not is_excluded:
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
    countryCode: str = "US"


class Connect2FARequest(BaseModel):
    twofa_token: str
    code: str
    name: str
    email: str
    password: str


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    global session_store, fansly_poller

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

    yield

    if fansly_poller:
        await fansly_poller.stop_all()


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
    return {"status": "ok"}


@app.post("/connect-creator")
async def connect_creator(req: ConnectCreatorRequest) -> dict:
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
                "twofa_token": data["data"]["twofa_token"],
                "masked_email": data["data"]["masked_email"],
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

        return {"success": True, "creator": creator}


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


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


from routes.fansly import fansly_router

app.include_router(fansly_router)

