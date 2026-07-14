"""Persistence helpers for passive fan intelligence."""

from __future__ import annotations

import asyncio
import os
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from core.supabase import get_supabase
from models.fan_intelligence import FactStatus, ValidatedObservation


async def insert_observation(payload: dict[str, Any]) -> bool:
    """Insert immutable evidence. False means the same observation already exists."""

    def _insert() -> bool:
        db = get_supabase()
        existing = (
            db.table("fan_fact_observations")
            .select("id")
            .eq("dedupe_key", payload["dedupe_key"])
            .limit(1)
            .execute()
        )
        if existing.data:
            return False
        try:
            db.table("fan_fact_observations").insert(payload).execute()
            return True
        except Exception:
            # The unique constraint closes the race between duplicate webhook/poller tasks.
            existing = (
                db.table("fan_fact_observations")
                .select("id")
                .eq("dedupe_key", payload["dedupe_key"])
                .limit(1)
                .execute()
            )
            if existing.data:
                return False
            raise

    return await asyncio.to_thread(_insert)


async def get_facts_for_key(fan_id: str, fact_key: str) -> list[dict[str, Any]]:
    def _get() -> list[dict[str, Any]]:
        response = (
            get_supabase()
            .table("fan_facts")
            .select("*")
            .eq("fan_id", fan_id)
            .eq("fact_key", fact_key)
            .order("updated_at", desc=True)
            .execute()
        )
        return list(response.data or [])

    return await asyncio.to_thread(_get)


async def insert_fact(
    *,
    creator_id: str,
    fan_id: str,
    observation: ValidatedObservation,
    source_message_id: str | None,
    status: FactStatus,
    is_active: bool,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    payload = {
        "creator_id": creator_id,
        "fan_id": fan_id,
        "category": observation.category.value,
        "fact_key": observation.fact_key,
        "value_json": observation.value_json,
        "normalized_value": observation.normalized_value,
        "confidence": observation.confidence,
        "status": status.value,
        "source_type": observation.source_type,
        "first_observed_at": now,
        "last_observed_at": now,
        "confirmation_count": 1,
        "first_evidence_message_id": source_message_id,
        "last_evidence_message_id": source_message_id,
        "first_evidence_text": observation.evidence_text,
        "last_evidence_text": observation.evidence_text,
        "is_active": is_active,
    }

    def _insert() -> None:
        get_supabase().table("fan_facts").upsert(
            payload,
            on_conflict="fan_id,fact_key,normalized_value",
        ).execute()

    await asyncio.to_thread(_insert)


async def update_fact(fact_id: str, patch: dict[str, Any]) -> None:
    payload = {
        **patch,
        "last_observed_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    await asyncio.to_thread(
        lambda: get_supabase().table("fan_facts").update(payload).eq("id", fact_id).execute()
    )


async def mark_facts_contradicted(fact_ids: list[str], *, deactivate: bool) -> None:
    if not fact_ids:
        return
    now = datetime.now(timezone.utc).isoformat()

    def _update() -> None:
        db = get_supabase()
        for fact_id in fact_ids:
            db.table("fan_facts").update(
                {
                    "status": FactStatus.CONTRADICTED.value,
                    "is_active": not deactivate,
                    "contradicted_at": now,
                    "updated_at": now,
                }
            ).eq("id", fact_id).execute()

    await asyncio.to_thread(_update)


async def get_fan_intelligence_context(fan_id: str) -> dict[str, Any]:
    """Return compact evidence-backed context for writers and commercial logic."""

    enabled = os.getenv("FAN_INTELLIGENCE_ENABLED", "false").strip().lower() in {
        "1", "true", "yes", "on"
    }
    if not enabled:
        return {}

    try:
        def _get() -> list[dict[str, Any]]:
            response = (
                get_supabase()
                .table("fan_facts")
                .select(
                    "id, category, fact_key, value_json, confidence, status, "
                    "source_type, is_active, confirmation_count, last_observed_at"
                )
                .eq("fan_id", fan_id)
                .order("updated_at", desc=True)
                .limit(100)
                .execute()
            )
            return list(response.data or [])

        rows = await asyncio.to_thread(_get)
    except Exception as exc:
        print(f"[FAN INTELLIGENCE] context read failed fan={fan_id}: {exc}")
        return {}

    active: list[dict[str, Any]] = []
    contradicted: dict[str, list[Any]] = defaultdict(list)
    for row in rows:
        if row.get("status") == FactStatus.CONTRADICTED.value:
            value = row.get("value_json")
            if value not in contradicted[row.get("fact_key") or "unknown"]:
                contradicted[row.get("fact_key") or "unknown"].append(value)
            continue
        if not row.get("is_active", True):
            continue
        if float(row.get("confidence") or 0) < 0.65:
            continue
        active.append(
            {
                "category": row.get("category"),
                "fact_key": row.get("fact_key"),
                "value": row.get("value_json"),
                "confidence": float(row.get("confidence") or 0),
                "status": row.get("status"),
                "source_type": row.get("source_type"),
                "confirmation_count": int(row.get("confirmation_count") or 1),
            }
        )

    hard_limits = [
        fact["value"]
        for fact in active
        if fact["fact_key"] == "hard_limit"
        and fact["status"] in {FactStatus.EXPLICIT.value, FactStatus.CONFIRMED.value}
    ]
    commercial: dict[str, Any] = {}
    for fact in active:
        if fact["category"] == "commercial" and fact["status"] in {
            FactStatus.EXPLICIT.value,
            FactStatus.CONFIRMED.value,
        }:
            commercial.setdefault(fact["fact_key"], fact["value"])

    return {
        "facts": active[:40],
        "hard_limits": hard_limits[:20],
        "commercial": commercial,
        "conflicts": [
            {"fact_key": key, "values": values[:5]}
            for key, values in contradicted.items()
            if len(values) >= 2
        ][:10],
    }
