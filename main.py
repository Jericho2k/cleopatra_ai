"""FastAPI entrypoint for Cleopatra.

Routes are thin and delegate all logic to services.
"""

import asyncio

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core.supabase import get_supabase
from db.queries import save_message
from models.schemas import SuggestionRequest, SuggestionResponse
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
    result = await get_suggestions(
        fan_id=fan_id,
        creator_id=creator_id,
        fan_message=message_content,
        creator_name="a creator",
    )
    db = get_supabase()
    await asyncio.to_thread(
        lambda: db.table("suggestions").insert({
            "fan_id": fan_id,
            "creator_id": creator_id,
            "message_id": message_id,
            "suggestions": result.suggestions,
        }).execute()
    )
    return {"status": "ok"}


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}

