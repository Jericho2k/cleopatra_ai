"""FastAPI entrypoint for Cleopatra.

Routes are thin and delegate all logic to services.
"""

import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ai.generator import generate_replies
from ai.prompt_builder import build_prompt
from ai.rag import find_similar_exchanges
from ai.stage_classifier import classify_stage
from core.supabase import get_supabase
from db.queries import (
    get_conversation_history,
    get_creator_persona,
    get_fan_by_id,
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
from services.suggestions import _send_auto_reply, get_suggestions


_debounce_tasks: dict[str, asyncio.Task] = {}


async def _delayed_auto_reply(fan_id: str, creator_id: str, delay: int = 90) -> None:
    """Wait for delay seconds, then generate and send auto reply."""
    try:
        await asyncio.sleep(delay)

        # Fetch latest conversation history (includes all messages sent during delay)
        conversation_history = await get_conversation_history(fan_id)

        # Get the latest fan message
        fan_messages = [m for m in conversation_history if m.role == "fan"]
        if not fan_messages:
            return

        latest_fan_message = fan_messages[-1].content

        # Now run the full AI pipeline with complete context
        fan_profile = await get_fan_by_id(fan_id)
        if fan_profile is None:
            fan_profile = Fan(id=fan_id, display_name=fan_id)

        creator_persona = await get_creator_persona(creator_id)
        if creator_persona is None:
            creator_persona = Persona()

        conversation_stage = classify_stage(conversation_history, fan_profile)
        similar_exchanges = await find_similar_exchanges(latest_fan_message, creator_id)

        ctx = ConversationContext(
            fan_message=latest_fan_message,
            conversation_history=conversation_history,
            fan_profile=fan_profile,
            creator_persona=creator_persona,
            similar_exchanges=similar_exchanges,
            conversation_stage=conversation_stage,
            creator_name="a creator",
        )

        prompt = build_prompt(ctx)
        replies = await generate_replies(prompt, creator_persona)

        if replies:
            best_reply = replies[0]
            asyncio.create_task(
                _send_auto_reply(fan_id, creator_id, best_reply)
            )

    except asyncio.CancelledError:
        raise
    finally:
        current = asyncio.current_task()
        if current is not None and _debounce_tasks.get(fan_id) is current:
            _debounce_tasks.pop(fan_id, None)


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
async def generate_suggestions_webhook(payload: WebhookPayload) -> dict:
    if payload.type != "INSERT":
        return {"status": "skipped"}
    record = payload.record
    if record.get("role") != "fan":
        return {"status": "skipped"}
    fan_id = record.get("fan_id")
    creator_id = record.get("creator_id")
    message_content = record.get("content")
    message_id = record.get("id")
    if not all([fan_id, creator_id, message_content, message_id]):
        return {"status": "skipped"}
    conversation_history = await get_conversation_history(fan_id)
    fan_profile = await get_fan_by_id(fan_id)
    if fan_profile is None:
        fan_profile = Fan(id=fan_id, display_name=fan_id)

    # Check auto mode EARLY — right after getting fan profile
    db = get_supabase()
    creator_row = await asyncio.to_thread(
        lambda: db.table("creators").select("auto_mode").eq("id", creator_id).single().execute()
    )
    auto_mode = (creator_row.data or {}).get("auto_mode", False)

    # If auto mode, still need to generate a reply but skip saving to suggestions
    # If NOT auto mode, run full suggestions pipeline

    creator_persona = await get_creator_persona(creator_id)
    if creator_persona is None:
        creator_persona = Persona()

    # Check if fan is in any auto-excluded list
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

    from services.suggestions import (
        _should_update_memory,
        _update_fan_ai_summary,
        _update_fan_memory,
    )

    if auto_mode and not is_excluded:
        # Cancel existing debounce for this fan if any
        existing = _debounce_tasks.get(fan_id)
        if existing and not existing.done():
            existing.cancel()

        # Schedule new debounced reply
        task = asyncio.create_task(_delayed_auto_reply(fan_id, creator_id, delay=60))
        _debounce_tasks[fan_id] = task

        if _should_update_memory(conversation_history):
            asyncio.create_task(_update_fan_memory(fan_id, creator_id, conversation_history))
            asyncio.create_task(_update_fan_ai_summary(fan_id, conversation_history))

        return {"status": "queued"}

    conversation_stage = classify_stage(conversation_history, fan_profile)
    similar_exchanges = await find_similar_exchanges(message_content, creator_id)
    ctx = ConversationContext(
        fan_message=message_content,
        conversation_history=conversation_history,
        fan_profile=fan_profile,
        creator_persona=creator_persona,
        similar_exchanges=similar_exchanges,
        conversation_stage=conversation_stage,
        creator_name="a creator",
    )
    prompt = build_prompt(ctx)
    replies = await generate_replies(prompt, creator_persona)

    if _should_update_memory(conversation_history):
        asyncio.create_task(_update_fan_memory(fan_id, creator_id, conversation_history))
        asyncio.create_task(_update_fan_ai_summary(fan_id, conversation_history))

    await asyncio.to_thread(
        lambda: db.table("suggestions").insert({
            "fan_id": fan_id,
            "creator_id": creator_id,
            "message_id": message_id,
            "suggestions": replies,
            "stage": conversation_stage.value,
        }).execute()
    )
    return {"status": "ok"}


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


from routes.fansly import fansly_router

app.include_router(fansly_router)

