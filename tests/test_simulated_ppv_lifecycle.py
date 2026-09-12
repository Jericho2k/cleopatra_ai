"""A simulated PPV is locked, then unlocked — and the AI-stack marker survives.

The Simulator's whole value is judging how an offer lands, which means it has to
show what the fan sees: a priced, locked card before a purchase, and the actual
media after one. That distinction is not a simulator invention — it is the
``media_context.ppv.purchased`` flag the production purchase transition writes,
which is what the dashboard renders from.

These tests pin the transition at the level the UI reads it, and pin that the
"which brain wrote this" marker added alongside the PPV payload is not lost when
a purchase rewrites that same jsonb column.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from db import queries
from services.suggestions import _with_ai_stack, message_ai_stack_metadata


def _ppv_message(message_id: str, media_id: str, *, with_stack: bool = True) -> dict:
    context = {
        "ppv": {
            "media_id": media_id,
            "media_ids": [media_id],
            "price": 25,
            "price_cents": 2500,
            "access_type": "ppv",
            "source": "auto",
        }
    }
    if with_stack:
        context = _with_ai_stack(
            context,
            message_ai_stack_metadata(
                SimpleNamespace(
                    route=SimpleNamespace(value="commercial_complex"),
                    prompt_version="writer_v2",
                    primary_target=SimpleNamespace(
                        provider="openrouter", model="moonshotai/kimi-k2.6"
                    ),
                ),
                profile_id="cleo_v2",
            ),
        )
    return {
        "id": message_id,
        "fan_id": "fan-test",
        "role": "creator",
        "content": "here it is",
        "sent_at": "2026-09-12T10:00:00Z",
        "media_context": context,
    }


class _Messages:
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows
        self.filters: list[tuple[str, str]] = []
        self.payload: dict | None = None

    def table(self, _name):
        return _Messages(self.rows)

    def select(self, *_a, **_k):
        return self

    def update(self, payload):
        self.payload = payload
        return self

    def eq(self, column, value):
        self.filters.append((column, str(value)))
        return self

    @property
    def not_(self):
        return self

    def is_(self, *_a, **_k):
        return self

    def order(self, *_a, **_k):
        return self

    def range(self, *_a, **_k):
        return self

    def limit(self, *_a, **_k):
        return self

    def execute(self):
        matched = [
            row
            for row in self.rows
            if all(str(row.get(col)) == val for col, val in self.filters)
        ]
        if self.payload is not None:
            for row in matched:
                row.update(self.payload)
            return SimpleNamespace(data=[dict(row) for row in matched])
        return SimpleNamespace(data=[dict(row) for row in matched])


@pytest.fixture
def store(monkeypatch):
    rows = [_ppv_message("msg-ppv", "sim:abc123:fansly-media-1")]
    root = _Messages(rows)
    monkeypatch.setattr(queries, "get_supabase", lambda: root)
    return rows


def test_a_simulated_ppv_is_locked_until_a_purchase_is_confirmed(store):
    ppv = store[0]["media_context"]["ppv"]

    # No purchased flag at all: the UI reads that as locked, which is what the
    # fan is actually looking at.
    assert ppv.get("purchased") is not True


def test_confirming_the_purchase_unlocks_exactly_that_message(store):
    unlocked = asyncio.run(
        queries.mark_ppv_purchased("fan-test", "sim:abc123:fansly-media-1")
    )

    assert unlocked is True
    assert store[0]["media_context"]["ppv"]["purchased"] is True


def test_unlocking_keeps_the_media_ids_and_price_intact(store):
    asyncio.run(queries.mark_ppv_purchased("fan-test", "sim:abc123:fansly-media-1"))

    ppv = store[0]["media_context"]["ppv"]
    assert ppv["media_ids"] == ["sim:abc123:fansly-media-1"]
    assert ppv["price_cents"] == 2500


def test_the_ai_stack_marker_survives_the_purchase_rewrite(store):
    """A purchase rewrites media_context. The marker lives in the same column,
    so losing it here would make the audit trail silently incomplete."""
    asyncio.run(queries.mark_ppv_purchased("fan-test", "sim:abc123:fansly-media-1"))

    assert store[0]["media_context"]["ai_stack"]["profile"] == "cleo_v2"
    assert store[0]["media_context"]["ai_stack"]["route"] == "commercial_complex"


def test_an_unmatched_media_id_unlocks_nothing(store):
    unlocked = asyncio.run(queries.mark_ppv_purchased("fan-test", "some-other-id"))

    assert unlocked is False
    assert store[0]["media_context"]["ppv"].get("purchased") is not True
