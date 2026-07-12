import sys
import types
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import asyncio
from datetime import datetime, timezone

# The delivered patch is tested in a minimal repository snapshot as well as the
# full application. Provide tiny import stubs only when the application's normal
# modules are not present; production/test installs use the real modules.
try:
    import core.supabase  # type: ignore  # noqa: F401
except ModuleNotFoundError:
    core_module = types.ModuleType("core")
    supabase_module = types.ModuleType("core.supabase")
    supabase_module.get_supabase = lambda: None
    sys.modules.setdefault("core", core_module)
    sys.modules["core.supabase"] = supabase_module

try:
    import models.schemas  # type: ignore  # noqa: F401
except ModuleNotFoundError:
    schemas_module = types.ModuleType("models.schemas")

    @dataclass
    class Message:
        role: str
        content: str
        sent_at: datetime
        media_context: dict | None = None

    class Placeholder:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    schemas_module.Message = Message
    schemas_module.Fan = Placeholder
    schemas_module.Persona = Placeholder
    schemas_module.ExchangeExample = Placeholder
    sys.modules["models.schemas"] = schemas_module

import db.queries as queries


class Result:
    def __init__(self, data):
        self.data = data


class FakeQuery:
    def __init__(self, rows):
        self.rows = rows
        self.desc = None
        self.limit_value = None

    def select(self, *_args, **_kwargs): return self
    def eq(self, *_args, **_kwargs): return self
    def order(self, _field, desc=False):
        self.desc = desc
        return self
    def limit(self, value):
        self.limit_value = value
        return self
    def execute(self):
        ordered = list(reversed(self.rows)) if self.desc else list(self.rows)
        return Result(ordered[: self.limit_value])


class FakeSupabase:
    def __init__(self, rows):
        self.query = FakeQuery(rows)
    def table(self, _name): return self.query


def test_fetches_newest_rows_but_returns_them_chronologically(monkeypatch):
    rows = [
        {"role": "fan", "content": f"m{i}", "sent_at": datetime(2026, 1, i + 1, tzinfo=timezone.utc).isoformat(), "media_context": None}
        for i in range(5)
    ]
    fake = FakeSupabase(rows)
    monkeypatch.setattr(queries, "get_supabase", lambda: fake)
    history = asyncio.run(queries.get_conversation_history("fan", limit=3))
    assert fake.query.desc is True
    assert [message.content for message in history] == ["m2", "m3", "m4"]
