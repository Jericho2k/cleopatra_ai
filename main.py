"""FastAPI entrypoint for Cleopatra.

Routes are thin and delegate all logic to services.
"""

from fastapi import FastAPI
from pydantic import BaseModel

from db.queries import save_message
from models.schemas import SuggestionRequest, SuggestionResponse
from services.suggestions import get_suggestions


class ReplyRequest(BaseModel):
    fan_id: str
    creator_id: str
    content: str
    was_ai_suggested: bool = False


app = FastAPI()


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


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}

