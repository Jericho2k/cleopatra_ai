"""FastAPI entrypoint for Cleopatra.

Routes are thin and delegate all logic to services.
"""

from fastapi import FastAPI

from models.schemas import SuggestionRequest, SuggestionResponse
from services.suggestions import get_suggestions


app = FastAPI()


@app.post("/suggestions", response_model=SuggestionResponse)
async def suggestions(req: SuggestionRequest) -> SuggestionResponse:
    return await get_suggestions(
        fan_id=req.fan_id,
        creator_id=req.creator_id,
        fan_message=req.message,
        creator_name="a creator",
    )


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}

