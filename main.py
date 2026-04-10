"""FastAPI entrypoint for Cleopatra.

Routes are thin and delegate all logic to services.
"""

import asyncio
import os
from contextlib import asynccontextmanager

import httpx
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
    increment_fan_total_spent,
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
    _send_auto_reply,
    _should_update_memory,
    _update_fan_ai_summary,
    _update_fan_memory,
    get_suggestions,
)


_processed_messages: set = set()


async def send_fansly_message(account_id: str, fan_platform_id: str, text: str) -> bool:
    api_key = os.environ.get("APIFANSLY_API_KEY")
    base_url = os.environ.get("APIFANSLY_BASE_URL", "https://app.apifansly.com/api")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{base_url.rstrip('/')}/{account_id}/chats/{fan_platform_id}/messages",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"text": text},
                timeout=10,
            )
            return response.status_code == 200
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

    creator_persona = await get_creator_persona(creator_id)
    if creator_persona is None:
        creator_persona = Persona()
    ppv_offers = await get_ppv_offers(creator_id)

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
    if auto_mode and not is_excluded:
        if replies:
            asyncio.create_task(_send_auto_reply(fan_id, creator_id, replies[0]))
    elif message_id:
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


async def handle_fan_message(
    fan_id: str,
    creator_id: str,
    message_content: str,
    auto_mode: bool,
    message_id: str | None = None,
) -> None:
    await save_message(fan_id, creator_id, "fan", message_content)
    await process_incoming_fan_message(
        fan_id, creator_id, message_content, auto_mode, message_id,
    )


class ReplyRequest(BaseModel):
    fan_id: str
    creator_id: str
    content: str
    was_ai_suggested: bool = False


class WebhookPayload(BaseModel):
    type: str
    record: dict


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
    return await get_suggestions(
        fan_id=req.fan_id,
        creator_id=req.creator_id,
        fan_message=req.message,
        creator_name="a creator",
        save_fan_message=False,
    )


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
    account_id = payload.get("account_id")
    data = payload.get("payload") or {}

    if event == "messages.received":
        from_user = data.get("fromUser") or {}

        if from_user.get("isPerformer"):
            return {"status": "skipped"}

        platform_fan_id = str(from_user.get("id"))
        fan_name = from_user.get("name", "Fan")
        message_content = data.get("text", "")
        message_id = data.get("id")

        if not message_content or not account_id:
            return {"status": "skipped"}

        if message_id is not None:
            mid = str(message_id)
            if mid in _processed_messages:
                return {"status": "duplicate"}
            _processed_messages.add(mid)
            if len(_processed_messages) > 1000:
                _processed_messages.clear()

        db = get_supabase()
        creator_row = await asyncio.to_thread(
            lambda: db.table("creators")
            .select("id, auto_mode")
            .eq("fansly_account_id", str(account_id))
            .limit(1)
            .execute()
        )
        if not creator_row.data:
            return {"status": "creator_not_found"}

        creator_id = creator_row.data[0]["id"]
        auto_mode = creator_row.data[0].get("auto_mode", False)

        fan = await get_fan(creator_id, platform_fan_id)
        if not fan:
            fan = await create_fan(creator_id, platform_fan_id, fan_name)

        await handle_fan_message(
            fan.id,
            creator_id,
            message_content,
            auto_mode,
            str(message_id) if message_id is not None else None,
        )

        return {"status": "ok"}

    if event == "tips.received":
        from_user = data.get("fromUser") or {}
        fan_id_platform = str(from_user.get("id")) if from_user.get("id") is not None else ""
        raw_amount = data.get("netAmount", 0)
        try:
            tip_amount = int(round(float(raw_amount)))
        except (TypeError, ValueError):
            tip_amount = 0
        print(f"[TIP] account={account_id} fan={fan_id_platform} amount={raw_amount}")
        if account_id and fan_id_platform and tip_amount:
            db = get_supabase()
            creator_row = await asyncio.to_thread(
                lambda: db.table("creators")
                .select("id")
                .eq("fansly_account_id", str(account_id))
                .limit(1)
                .execute()
            )
            if creator_row.data:
                creator_id = creator_row.data[0]["id"]
                fan = await get_fan(creator_id, fan_id_platform)
                if fan:
                    await increment_fan_total_spent(fan.id, tip_amount)
        return {"status": "ok"}

    if event == "subscriptions.new":
        from_user = data.get("fromUser") or {}
        fan_id_platform = str(from_user.get("id")) if from_user.get("id") is not None else ""
        fan_name = from_user.get("name", "Fan")
        print(f"[NEW SUB] account={account_id} fan={fan_id_platform} name={fan_name}")
        # TODO: send welcome message
        return {"status": "ok"}

    if event == "users.typing":
        pass

    return {"status": "ok"}


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


from routes.fansly import fansly_router

app.include_router(fansly_router)

