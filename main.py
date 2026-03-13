"""FastAPI entrypoint for Cleopatra.

Routes are thin and delegate all logic to services.
"""

import asyncio

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
    get_fan,
    save_message,
)
from models.schemas import (
    ConversationContext,
    Fan,
    Persona,
    SuggestionRequest,
    SuggestionResponse,
)
from services.suggestions import get_suggestions


class ReplyRequest(BaseModel):
    fan_id: str
    creator_id: str
    content: str
    was_ai_suggested: bool = False


class WebhookPayload(BaseModel):
    type: str
    record: dict


app = FastAPI()

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
    fan_profile = await get_fan(creator_id, fan_id)
    if fan_profile is None:
        fan_profile = Fan(id=fan_id, display_name=fan_id)
    creator_persona = await get_creator_persona(creator_id)
    if creator_persona is None:
        creator_persona = Persona()
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
    from services.suggestions import _should_update_memory, _update_fan_memory
    if _should_update_memory(conversation_history):
        asyncio.create_task(_update_fan_memory(fan_id, creator_id, conversation_history))
    db = get_supabase()
    await asyncio.to_thread(
        lambda: db.table("suggestions").insert({
            "fan_id": fan_id,
            "creator_id": creator_id,
            "message_id": message_id,
            "suggestions": replies,
        }).execute()
    )
    return {"status": "ok"}


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}

