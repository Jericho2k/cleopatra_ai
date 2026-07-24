"""Explicit, operator-approved creator voice calibration.

Only human-selected creator messages influence the writer. Candidate discovery
never turns a sent message into style evidence by itself, which avoids an
AI-output feedback loop.
"""
from __future__ import annotations

import asyncio
from typing import Any, Iterable

from ai.voice_calibration import normalize_sample
from core.supabase import get_supabase
from db.queries import get_creator_persona, save_persona
from models.schemas import Persona


MAX_APPROVED_SAMPLES = 30


def normalize_message_ids(values: Iterable[object]) -> list[str]:
    return list(
        dict.fromkeys(
            str(value).strip()
            for value in values
            if str(value or "").strip()
        )
    )[:MAX_APPROVED_SAMPLES]


async def list_voice_calibration_candidates(
    creator_id: str,
    *,
    limit: int = 120,
) -> list[dict[str, Any]]:
    """Return possible samples without approving any of them."""

    def _load() -> list[dict[str, Any]]:
        result = (
            get_supabase()
            .table("messages")
            .select("id, content, sent_at, was_ai_suggested")
            .eq("creator_id", creator_id)
            .eq("role", "creator")
            .order("sent_at", desc=True)
            .limit(max(1, min(int(limit), 300)))
            .execute()
        )
        return result.data or []

    rows = await asyncio.to_thread(_load)
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        # A known AI suggestion is never eligible. Imported platform history can
        # be unknown/null, but still requires explicit operator approval below.
        if row.get("was_ai_suggested") is True:
            continue
        content = normalize_sample(row.get("content"))
        key = content.casefold()
        if not content or len(content) < 2 or key in seen:
            continue
        seen.add(key)
        candidates.append(
            {
                "id": str(row.get("id") or ""),
                "content": content,
                "sent_at": row.get("sent_at"),
            }
        )
    return [candidate for candidate in candidates if candidate["id"]]


async def save_voice_calibration(
    creator_id: str,
    *,
    enabled: bool,
    approved_message_ids: Iterable[object],
) -> Persona:
    message_ids = normalize_message_ids(approved_message_ids)

    def _load_selected() -> list[dict[str, Any]]:
        if not message_ids:
            return []
        result = (
            get_supabase()
            .table("messages")
            .select("id, content, was_ai_suggested")
            .eq("creator_id", creator_id)
            .eq("role", "creator")
            .in_("id", message_ids)
            .execute()
        )
        return result.data or []

    selected = await asyncio.to_thread(_load_selected)
    by_id = {
        str(row.get("id")): normalize_sample(row.get("content"))
        for row in selected
        if row.get("was_ai_suggested") is not True and normalize_sample(row.get("content"))
    }
    approved_ids: list[str] = []
    samples: list[str] = []
    seen_samples: set[str] = set()
    for message_id in message_ids:
        sample = by_id.get(message_id)
        key = str(sample or "").casefold()
        if not sample or key in seen_samples:
            continue
        approved_ids.append(message_id)
        samples.append(sample)
        seen_samples.add(key)

    persona = await get_creator_persona(creator_id) or Persona()
    persona.voice_calibration_enabled = bool(enabled and samples)
    persona.voice_calibration_message_ids = approved_ids
    persona.voice_calibration_samples = samples
    await save_persona(creator_id, persona)
    return persona
