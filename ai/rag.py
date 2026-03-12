"""RAG helpers: embeddings + similar exchange lookup.

Uses OpenAI embeddings and Supabase pgvector via db.queries.
"""

from openai import AsyncOpenAI

from core.config import get_settings
from db.queries import get_similar_exchanges
from models.schemas import ExchangeExample


# Module level — create once, used for embeddings only
embeddings_client = AsyncOpenAI(
    api_key=get_settings().OPENAI_API_KEY,
)


async def get_embedding(text: str) -> list[float]:
    """Get an embedding vector for the given text using OpenAI embeddings."""
    response = await embeddings_client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return list(response.data[0].embedding)


async def find_similar_exchanges(
    fan_message: str,
    creator_id: str,
    limit: int = 5,
) -> list[ExchangeExample]:
    embedding = await get_embedding(fan_message)
    return await get_similar_exchanges(embedding, creator_id, limit)

