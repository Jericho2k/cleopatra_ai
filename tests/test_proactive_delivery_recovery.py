from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from models.schemas import Fan, Persona
from services import proactive


def run(coro):
    return asyncio.run(coro)


def test_ambiguous_platform_acceptance_is_reconciled_by_exact_message(monkeypatch):
    started = datetime.now(timezone.utc) - timedelta(minutes=2)
    created_ms = int((started + timedelta(seconds=4)).timestamp() * 1000)

    async def messages(*_args, **_kwargs):
        return ([{
            "id": "platform-123",
            "senderId": "creator-platform",
            "content": "thought of u",
            "createdAt": created_ms,
        }], [], None)

    monkeypatch.setattr(proactive, "list_chat_messages", messages)

    message_id = run(proactive._reconcile_ambiguous_delivery(
        account_id="api-account",
        group_id="chat-1",
        creator_platform_id="creator-platform",
        text="thought  of u",
        started_at=started,
    ))

    assert message_id == "platform-123"


def test_confirmed_scheduled_delivery_is_persisted_without_resending(monkeypatch):
    fan = Fan(
        id="fan-1",
        display_name="Fan",
        platform_fan_id="platform-fan",
        fansly_group_id="chat-1",
    )
    saved = []
    journals = []

    class Query:
        def table(self, _name):
            return self

        def select(self, _columns):
            return self

        def eq(self, _column, _value):
            return self

        def single(self):
            return self

        def execute(self):
            return SimpleNamespace(data={
                "apifansly_account_id": "api-account",
                "fansly_account_id": "creator-platform",
            })

    async def value(result):
        return result

    async def save(*args, **kwargs):
        saved.append((args, kwargs))
        return "local-message"

    async def journal(action_id, payload):
        journals.append((action_id, payload))

    import db.queries as queries

    monkeypatch.setattr(proactive, "get_fan_by_id", lambda _fan_id: value(fan))
    monkeypatch.setattr(proactive, "get_creator_persona", lambda _creator_id: value(Persona()))
    monkeypatch.setattr(proactive, "get_conversation_history", lambda *_args, **_kwargs: value([]))
    monkeypatch.setattr(proactive, "get_sent_ppv", lambda _fan_id: value([]))
    monkeypatch.setattr(proactive, "get_fan_session", lambda _fan_id: value(None))
    monkeypatch.setattr(queries, "get_creator_legend", lambda _creator_id: value({}))
    monkeypatch.setattr(proactive, "get_supabase", lambda: Query())
    monkeypatch.setattr(proactive, "save_message", save)
    monkeypatch.setattr(proactive, "update_action_payload", journal)

    result = run(proactive.send_proactive_message(
        "creator-1",
        "fan-1",
        "reopen naturally",
        action_id="action-1",
        action_payload={
            "_delivery": {
                "text": "thought of u",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "platform_message_id": "platform-123",
            }
        },
    ))

    assert result is True
    assert saved[0][1]["fansly_message_id"] == "platform-123"
    assert journals[0][1]["_delivery"]["platform_message_id"] == "platform-123"

