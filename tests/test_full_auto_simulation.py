"""The simulator runs the REAL Full Auto pipeline, locally, with zero remote I/O.

These tests deliberately do NOT stub the pipeline. ``run_simulated_inbound``
drives the same ``_debounced_auto_reply`` the durable AUTO_REPLY worker drives;
what is faked here is the world around it — the database, the model calls and
the clock — exactly as the existing Full Auto tests fake them. Every stage in
between is the production code.

The zero-remote-call claim is checked at the transport, by counting requests
that reach ``httpx``. That is stronger than asserting a particular branch was
not taken: a path nobody thought of still fails the count.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import httpx
import pytest

from models.schemas import Fan, Persona
from services import suggestions


NOW = datetime(2026, 7, 18, 12, 0, tzinfo=timezone.utc)


class SpyTransport(httpx.AsyncBaseTransport):
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(200, json={"success": True, "response": {}})


class FakeTable:
    """One in-memory ``messages``/``fans``/``creators`` table stand-in."""

    def __init__(self, store: "FakeDB", name: str) -> None:
        self.store = store
        self.name = name
        self.filters: list[tuple[str, object]] = []
        self.payload: dict | None = None
        self.mode = "select"

    def select(self, *_a, **_k):
        self.mode = "select"
        return self

    def insert(self, payload):
        self.mode = "insert"
        self.payload = payload
        return self

    def update(self, payload):
        self.mode = "update"
        self.payload = payload
        return self

    def upsert(self, payload, **_k):
        self.mode = "insert"
        self.payload = payload
        return self

    def eq(self, column, value):
        self.filters.append((column, value))
        return self

    def in_(self, *_a, **_k):
        return self

    def like(self, *_a, **_k):
        return self

    def is_(self, *_a, **_k):
        return self

    def not_(self, *_a, **_k):
        return self

    def order(self, column, desc=False):
        self._order = (column, desc)
        return self

    def limit(self, value):
        self._limit = value
        return self

    def single(self):
        self._single = True
        return self

    def maybe_single(self):
        self._single = True
        return self

    def _rows(self) -> list[dict]:
        rows = list(self.store.tables.get(self.name, []))
        for column, value in self.filters:
            rows = [r for r in rows if str(r.get(column)) == str(value)]
        order = getattr(self, "_order", None)
        if order:
            rows.sort(key=lambda r: str(r.get(order[0]) or ""), reverse=order[1])
        limit = getattr(self, "_limit", None)
        if limit:
            rows = rows[:limit]
        if order and order[1]:
            pass
        return rows

    def execute(self):
        if self.mode == "insert":
            payload = self.payload
            rows = payload if isinstance(payload, list) else [payload]
            created = []
            for row in rows:
                row = dict(row)
                row.setdefault("id", f"{self.name}-{len(self.store.tables.setdefault(self.name, [])) + 1}")
                row.setdefault("sent_at", self.store.next_timestamp())
                self.store.tables.setdefault(self.name, []).append(row)
                created.append(row)
            return SimpleNamespace(data=created, count=len(created))
        if self.mode == "update":
            rows = self._rows()
            for row in rows:
                row.update(self.payload or {})
            return SimpleNamespace(data=rows, count=len(rows))
        rows = self._rows()
        if getattr(self, "_single", False):
            return SimpleNamespace(data=rows[0] if rows else None, count=len(rows))
        return SimpleNamespace(data=rows, count=len(rows))


class FakeDB:
    def __init__(self, tables: dict[str, list[dict]]) -> None:
        self.tables = tables
        self._clock = 0

    def next_timestamp(self) -> str:
        self._clock += 1
        return (NOW + timedelta(seconds=self._clock)).isoformat()

    def table(self, name):
        return FakeTable(self, name)

    def from_(self, name):
        return FakeTable(self, name)

    def rpc(self, _name, _params):
        return SimpleNamespace(execute=lambda: SimpleNamespace(data="attached"))


@pytest.fixture
def spy():
    transport = SpyTransport()
    client = httpx.AsyncClient(transport=transport)
    from services import apifansly

    apifansly.set_shared_client(client)
    yield transport
    apifansly.set_shared_client(None)


@pytest.fixture
def world(monkeypatch):
    """A creator, a test fan, and two fan messages already in SQL history."""
    db = FakeDB(
        {
            "messages": [
                {
                    "id": "msg-1",
                    "fan_id": "fan-test",
                    "creator_id": "creator-1",
                    "role": "fan",
                    "content": "hiii",
                    "sent_at": "2026-07-18T11:58:00+00:00",
                },
                {
                    "id": "msg-2",
                    "fan_id": "fan-test",
                    "creator_id": "creator-1",
                    "role": "fan",
                    "content": "how r u doing?)",
                    "sent_at": "2026-07-18T11:59:00+00:00",
                },
            ],
            "fans": [
                {
                    "id": "fan-test",
                    "creator_id": "creator-1",
                    "platform_fan_id": "test_jostar",
                    "display_name": "Jostar",
                    "fansly_group_id": "stale-group-99",
                    "pending_tip": None,
                    "pending_ppv_check": None,
                }
            ],
            "creators": [
                {
                    "id": "creator-1",
                    "name": "Sophia",
                    "apifansly_account_id": "acct-1",
                    "fansly_account_id": "plat-1",
                    "auto_mode": True,
                }
            ],
        }
    )

    calls: dict[str, list] = {
        "analyzer": [],
        "writer": [],
        "director": [],
        "orchestrator": [],
        "planner": [],
        "route": [],
        "sleeps": [],
    }

    async def fake_analyze(ctx, telemetry_context=None):
        calls["analyzer"].append(ctx)
        return {
            "purchase_signal": "none",
            "crisis_signal": "none",
            "resend_requested": "false",
            "strategic_move": "build_rapport",
        }

    async def fake_generate(prompt, persona, **kwargs):
        calls["writer"].append({"prompt": prompt, **kwargs})
        return ["hey | what are you doing?"]

    def fake_route(ctx):
        calls["route"].append(ctx)
        return SimpleNamespace(
            route=SimpleNamespace(value="primary"),
            reason="test",
            primary_target=SimpleNamespace(model="test-writer"),
            fallback_target=None,
            telemetry_metadata=lambda: {},
        )

    async def fake_direct(**kwargs):
        calls["director"].append(kwargs)
        return {"phase": "rapport", "action": "chat", "transition_reason": "test"}

    async def fake_plan_next(**kwargs):
        calls["planner"].append(kwargs)
        return {"goal": "rapport", "next_action": "chat"}

    async def noop_async(*_a, **_k):
        return None

    async def empty_dict(*_a, **_k):
        return {}

    async def fake_sleep(seconds):
        calls["sleeps"].append(seconds)

    monkeypatch.setattr(suggestions, "get_supabase", lambda: db)
    monkeypatch.setattr("db.queries.get_supabase", lambda: db)
    monkeypatch.setattr(suggestions, "analyze_situation", fake_analyze)
    monkeypatch.setattr(suggestions, "generate_replies", fake_generate)
    monkeypatch.setattr(suggestions, "select_writer_route", fake_route)
    monkeypatch.setattr(suggestions, "direct_conversation", fake_direct)
    monkeypatch.setattr(suggestions, "plan_next_action", fake_plan_next)
    monkeypatch.setattr(suggestions, "get_creator_persona", lambda _c: _value(Persona()))
    monkeypatch.setattr(suggestions, "get_ppv_offers", lambda _c: _value([]))
    monkeypatch.setattr(suggestions, "get_sent_ppv", lambda _f: _value([]))
    monkeypatch.setattr(suggestions, "get_fan_session", lambda _f: _value(None))
    monkeypatch.setattr(suggestions, "find_similar_exchanges", lambda *_a, **_k: _value([]))
    monkeypatch.setattr(suggestions, "get_fan_intelligence_context", lambda _f: _value({}))
    monkeypatch.setattr(suggestions, "get_fan_lifecycle_context", lambda _f: _value({"stage": "new"}))
    monkeypatch.setattr(suggestions, "get_affordability_context", lambda _f: _value({}))
    monkeypatch.setattr(suggestions, "get_price_learning_context", lambda _f: _value({"mode": "learning"}))
    monkeypatch.setattr(suggestions, "refresh_affordability_from_situation", lambda **_k: _value({}))
    monkeypatch.setattr(suggestions, "refresh_fan_lifecycle", lambda **_k: _value({"stage": "new"}))
    monkeypatch.setattr(suggestions, "refresh_price_learning", lambda **_k: _value({"mode": "learning"}))
    monkeypatch.setattr(suggestions, "learn_from_fan_message", lambda **_k: _value(None))
    monkeypatch.setattr(suggestions, "_crisis_freezes_chat", lambda *_a, **_k: _value(False))
    monkeypatch.setattr(
        suggestions,
        "get_fan_by_id",
        lambda _f: _value(
            Fan(
                id="fan-test",
                display_name="Jostar",
                platform_fan_id="test_jostar",
                fansly_group_id="stale-group-99",
            )
        ),
    )
    monkeypatch.setattr(
        "services.inactivity_reengagement.schedule_inactivity_reengagement", noop_async
    )
    monkeypatch.setattr(suggestions.asyncio, "sleep", fake_sleep)
    suggestions._pending_auto_replies.clear()
    return db, calls


async def _value(value):
    return value


def _run(coro):
    return asyncio.run(coro)


def _fan_rows(db) -> list[dict]:
    return [r for r in db.tables["messages"] if r["role"] == "fan"]


def _creator_rows(db) -> list[dict]:
    return [r for r in db.tables["messages"] if r["role"] == "creator"]


# --- 19, 20: exactly one fan row, full history visible ----------------------


def test_fan_message_persisted_exactly_once(world, spy):
    db, _ = world
    result = _run(
        suggestions.run_simulated_inbound(
            fan_id="fan-test", creator_id="creator-1", message="what are you up to?", fast=True
        )
    )
    new_rows = [r for r in _fan_rows(db) if r["content"] == "what are you up to?"]
    assert len(new_rows) == 1
    assert result["fan_message_id"] == new_rows[0]["id"]


def test_previous_sql_history_is_included_in_the_turn(world, spy):
    """Rows inserted by hand before the simulator existed are ordinary history."""
    db, calls = world
    _run(
        suggestions.run_simulated_inbound(
            fan_id="fan-test", creator_id="creator-1", message="still there?", fast=True
        )
    )
    ctx = calls["analyzer"][0]
    contents = [m.content for m in ctx.conversation_history]
    assert "hiii" in contents
    assert "how r u doing?)" in contents
    assert "still there?" in contents
    # The latest fan message is the CURRENT TURN, not the only context.
    assert ctx.fan_message == "still there?"
    assert len(ctx.conversation_history) >= 3


def test_history_reaches_every_downstream_stage(world, spy):
    db, calls = world
    _run(
        suggestions.run_simulated_inbound(
            fan_id="fan-test", creator_id="creator-1", message="hey you", fast=True
        )
    )
    director_kwargs = calls["director"][0]
    assert [m.content for m in director_kwargs["conversation_history"]][:2] == [
        "hiii",
        "how r u doing?)",
    ]
    assert director_kwargs["latest_fan_message"] == "hey you"
    writer_ctx = calls["route"][0]
    assert len(writer_ctx.conversation_history) >= 3


# --- 21, 22, 23: the REAL brain ---------------------------------------------


def test_real_analyzer_commercial_and_writer_are_invoked(world, spy, monkeypatch):
    monkeypatch.setenv("COMMERCIAL_LAYER_ENABLED", "true")
    db, calls = world
    orchestrated: list[dict] = []

    async def fake_orchestrate(**kwargs):
        orchestrated.append(kwargs)
        from models.commercial import ActionType

        return SimpleNamespace(
            action=ActionType.CONTINUE_NORMAL_CHAT,
            selected_package_set_ids=None,
            session_budget_cents=None,
            model_dump=lambda mode=None: {"action": "CONTINUE_NORMAL_CHAT"},
        )

    monkeypatch.setattr(suggestions, "orchestrate", fake_orchestrate)
    monkeypatch.setattr(suggestions, "_within_daily_caps", lambda *_a, **_k: _value((True, "")))

    _run(
        suggestions.run_simulated_inbound(
            fan_id="fan-test", creator_id="creator-1", message="hi there", fast=True
        )
    )

    assert len(calls["analyzer"]) == 1, "the real situation analyzer must run"
    assert len(orchestrated) == 1, "the commercial orchestrator must run"
    assert orchestrated[0]["creator_id"] == "creator-1"
    assert len(calls["director"]) == 1, "the conversation director must run"
    assert len(calls["planner"]) == 1, "session planning must run"
    assert len(calls["route"]) == 1, "writer routing must run"
    assert len(calls["writer"]) == 1, "the writer must run"
    assert calls["writer"][0]["telemetry_context"]["feature"] == "auto_reply"


def test_simulator_reuses_the_production_entry_point(world):
    """No second engine: the simulator awaits the same function the durable
    AUTO_REPLY worker awaits, with the same flags plus a timing skip."""
    import inspect

    source = inspect.getsource(suggestions.run_simulated_inbound)
    assert "_debounced_auto_reply(" in source
    assert "skip_debounce=True" in source
    assert "skip_availability=True" in source
    assert "simulation_scope()" in source
    # Assisted is a different pipeline and must not be substituted for Auto.
    assert "get_suggestions" not in source
    assert "regenerate" not in source


def test_analyzer_fail_closed_behaviour_is_preserved(world, spy, monkeypatch):
    """A degraded analysis sends nothing — the real Full Auto safety rule."""
    db, calls = world

    async def degraded(_ctx, telemetry_context=None):
        return {"analysis_degraded": True, "degraded_reason": "provider_down"}

    monkeypatch.setattr(suggestions, "analyze_situation", degraded)
    monkeypatch.setattr(suggestions, "analysis_is_degraded", lambda _s: True)
    monkeypatch.setattr(suggestions, "degraded_reason", lambda _s: "provider_down")

    result = _run(
        suggestions.run_simulated_inbound(
            fan_id="fan-test", creator_id="creator-1", message="hi", fast=True
        )
    )
    assert result["analysis_degraded"] is True
    assert result["creator_messages"] == []
    assert calls["writer"] == [], "nothing may be generated on a degraded analysis"
    assert spy.requests == []


# --- 24, 25: local persistence and multipart order --------------------------


def test_generated_creator_message_is_persisted_locally(world, spy):
    db, _ = world
    result = _run(
        suggestions.run_simulated_inbound(
            fan_id="fan-test", creator_id="creator-1", message="hi", fast=True
        )
    )
    assert len(result["creator_messages"]) == 2
    assert [r["content"] for r in _creator_rows(db)] == ["hey", "what are you doing?"]


def test_multipart_order_is_preserved(world, spy):
    db, _ = world
    result = _run(
        suggestions.run_simulated_inbound(
            fan_id="fan-test", creator_id="creator-1", message="hi", fast=True
        )
    )
    assert [m["content"] for m in result["creator_messages"]] == [
        "hey",
        "what are you doing?",
    ]


# --- 26: fast mode skips intentional timing only ---------------------------


def test_fast_mode_skips_intentional_timing(world, spy):
    db, calls = world
    _run(
        suggestions.run_simulated_inbound(
            fan_id="fan-test", creator_id="creator-1", message="hi", fast=True
        )
    )
    assert all(s == 0 for s in calls["sleeps"]), calls["sleeps"]


def test_slow_mode_still_waits(world, spy):
    db, calls = world
    _run(
        suggestions.run_simulated_inbound(
            fan_id="fan-test", creator_id="creator-1", message="hi", fast=False
        )
    )
    assert any(s > 0 for s in calls["sleeps"]), "fast=False must keep human timing"


# --- 27: zero API Fansly requests for a plain simulated reply --------------


@pytest.mark.parametrize("connector", ["true", "false"])
def test_zero_remote_calls_for_a_plain_simulated_reply(world, spy, monkeypatch, connector):
    """Counted at the transport, and asserted with the connector BOTH on and
    off: simulation must not depend on the kill switch to stay local."""
    monkeypatch.setenv("APIFANSLY_ENABLED", connector)
    db, _ = world
    result = _run(
        suggestions.run_simulated_inbound(
            fan_id="fan-test", creator_id="creator-1", message="hi", fast=True
        )
    )
    assert result["status"] == "ok"
    assert len(spy.requests) == 0


def test_stale_group_binding_does_not_trigger_a_typing_call(world, spy, monkeypatch):
    """The fan carries fansly_group_id='stale-group-99'. Before this change the
    typing indicator fired on the presence of a binding, not on the test flag."""
    monkeypatch.setenv("APIFANSLY_ENABLED", "true")
    db, _ = world
    _run(
        suggestions.run_simulated_inbound(
            fan_id="fan-test", creator_id="creator-1", message="hi", fast=True
        )
    )
    assert [str(r.url) for r in spy.requests] == []


def test_missing_group_binding_does_not_trigger_a_chat_listing(world, spy, monkeypatch):
    """A test fan with no group id must not cause get_or_fetch_group_id to list
    chats against the provider."""
    monkeypatch.setenv("APIFANSLY_ENABLED", "true")
    db, _ = world
    for row in db.tables["fans"]:
        row["fansly_group_id"] = None

    async def should_not_run(*_a, **_k):
        raise AssertionError("a test fan must not resolve a group id remotely")

    monkeypatch.setattr("main.get_or_fetch_group_id", should_not_run)
    _run(
        suggestions.run_simulated_inbound(
            fan_id="fan-test", creator_id="creator-1", message="hi", fast=True
        )
    )
    assert spy.requests == []


def test_resend_request_does_not_send_a_real_ppv(world, spy, monkeypatch):
    """The resend branch returns before the delivery block, so without its own
    guard a test fan could send a REAL PPV through the platform."""
    monkeypatch.setenv("APIFANSLY_ENABLED", "true")
    db, calls = world
    for row in db.tables["fans"]:
        row["pending_ppv_check"] = {
            "media_id": "m1",
            "media_ids": ["m1"],
            "price": 25.0,
            "reference": "ref-1",
        }

    async def resend_requested(_ctx, telemetry_context=None):
        return {
            "purchase_signal": "none",
            "crisis_signal": "none",
            "resend_requested": "true",
            "strategic_move": "build_rapport",
        }

    async def should_not_send(*_a, **_k):
        raise AssertionError("a simulated turn must not send a real PPV")

    monkeypatch.setattr(suggestions, "analyze_situation", resend_requested)
    monkeypatch.setattr(suggestions, "send_apifansly_message", should_not_send)

    _run(
        suggestions.run_simulated_inbound(
            fan_id="fan-test", creator_id="creator-1", message="i cant see it", fast=True
        )
    )
    assert spy.requests == []
    # Normal generation proceeded instead of a fake "resent" claim.
    assert [r["content"] for r in _creator_rows(db)] == ["hey", "what are you doing?"]


# --- 28, 29: the PPV path stays local and schedules nothing remote ---------


@pytest.fixture
def ppv_world(world, monkeypatch):
    db, calls = world
    monkeypatch.setenv("COMMERCIAL_LAYER_ENABLED", "true")

    async def ppv_reply(prompt, persona, **kwargs):
        calls["writer"].append({"prompt": prompt, **kwargs})
        return ["here it is [PPV:media-77:35.00]"]

    async def fake_orchestrate(**kwargs):
        from models.commercial import ActionType

        return SimpleNamespace(
            action=ActionType.SEND_NEXT_PPV_STEP,
            selected_package_set_ids=None,
            session_budget_cents=None,
            model_dump=lambda mode=None: {"action": "SEND_NEXT_PPV_STEP"},
        )

    from models.commercial import CreatorPolicy

    monkeypatch.setattr(suggestions, "generate_replies", ppv_reply)
    monkeypatch.setattr(suggestions, "orchestrate", fake_orchestrate)
    monkeypatch.setattr(suggestions, "_within_daily_caps", lambda *_a, **_k: _value((True, "")))
    monkeypatch.setattr(
        suggestions, "get_creator_policy", lambda _c: _value(CreatorPolicy())
    )
    monkeypatch.setattr(
        "db.commercial_queries.get_creator_policy", lambda _c: _value(CreatorPolicy())
    )
    return db, calls


def test_simulated_ppv_makes_zero_remote_calls_and_preserves_intent(
    ppv_world, spy, monkeypatch
):
    monkeypatch.setenv("APIFANSLY_ENABLED", "true")
    db, _ = ppv_world
    receipts: list[dict] = []
    reconciliations: list[dict] = []

    async def capture_receipt(**kwargs):
        receipts.append(kwargs)
        return "ppv-message-1"

    async def capture_reconciliation(**kwargs):
        reconciliations.append(kwargs)
        return kwargs.get("session"), NOW + timedelta(hours=1), True

    monkeypatch.setattr(suggestions, "save_ppv_message_receipt", capture_receipt)
    monkeypatch.setattr(suggestions, "persist_ppv_reconciliation", capture_reconciliation)

    _run(
        suggestions.run_simulated_inbound(
            fan_id="fan-test", creator_id="creator-1", message="show me", fast=True
        )
    )

    assert spy.requests == [], "a simulated PPV must not reach the platform"
    assert len(receipts) == 1, "a clearly simulated local PPV must be persisted"
    ppv = receipts[0]["media_context"]["ppv"]
    # Intent is preserved for inspection: what Cleopatra WOULD have sent.
    assert ppv["media_id"] == "media-77"
    assert ppv["price"] == 35.0
    assert ppv["price_cents"] == 3500
    assert receipts[0]["platform_message_id"].startswith("local-test:")
    assert len(reconciliations) == 1, "local commercial/session state is preserved"


def test_no_remote_reconciliation_is_triggered_after_a_simulated_ppv(
    ppv_world, spy, monkeypatch
):
    """29 — the eager remote verification spawned on a 'bought' signal used to
    run before the local-test branch was even evaluated."""
    monkeypatch.setenv("APIFANSLY_ENABLED", "true")
    db, _ = ppv_world
    verified: list[tuple] = []

    async def bought(_ctx, telemetry_context=None):
        return {
            "purchase_signal": "bought",
            "crisis_signal": "none",
            "resend_requested": "false",
            "strategic_move": "close",
        }

    async def capture_verify(*args, **kwargs):
        verified.append(args)

    for row in db.tables["fans"]:
        row["pending_ppv_check"] = {
            "media_id": "m1",
            "media_ids": ["m1"],
            "price": 25.0,
            "reference": "ref-1",
            "sent_at": NOW.isoformat(),
        }

    monkeypatch.setattr(suggestions, "analyze_situation", bought)
    monkeypatch.setattr(suggestions, "_verify_ppv_purchase", capture_verify)
    monkeypatch.setattr(suggestions, "save_ppv_message_receipt", lambda **_k: _value("m"))
    monkeypatch.setattr(
        suggestions,
        "persist_ppv_reconciliation",
        lambda **k: _value((k.get("session"), NOW + timedelta(hours=1), True)),
    )

    _run(
        suggestions.run_simulated_inbound(
            fan_id="fan-test", creator_id="creator-1", message="just bought it", fast=True
        )
    )
    assert verified == [], "no remote purchase verification for a simulated PPV"
    assert spy.requests == []


def test_scheduled_reconcile_for_a_test_fan_never_calls_the_provider(spy, monkeypatch):
    """The durable PPV_RECONCILE action a simulated PPV leaves behind resolves
    entirely locally, so it can never become a future API Fansly job."""
    from models.commercial import CreatorPolicy
    from services import ppv_reconciliation as reconciliation
    from services.ppv_reconciliation import PPVReconcileDisposition

    monkeypatch.setenv("APIFANSLY_ENABLED", "true")
    pending = {
        "reference": "ref-1",
        "media_id": "m1",
        "media_ids": ["m1"],
        "price": 25.0,
        "sent_at": "2026-07-18T11:00:00+00:00",
        "expires_at": "2026-07-18T11:30:00+00:00",
    }
    finalized: list[dict] = []

    monkeypatch.setattr(
        reconciliation, "get_creator_policy", lambda _c: _value(CreatorPolicy())
    )
    monkeypatch.setattr(
        reconciliation,
        "_load_platform_context",
        lambda _c, _f: _value(
            (
                {"apifansly_account_id": "acct-1"},
                {"platform_fan_id": "test_jostar", "pending_ppv_check": pending},
            )
        ),
    )
    monkeypatch.setattr(
        reconciliation,
        "_finalize_abandonment",
        lambda **kwargs: _append(finalized, kwargs),
    )

    result = _run(
        reconciliation.reconcile_pending_ppv(
            creator_id="creator-1", fan_id="fan-test", now=NOW
        )
    )
    assert result.disposition == PPVReconcileDisposition.ABANDONED
    assert len(finalized) == 1
    assert spy.requests == []


async def _append(bucket, value):
    bucket.append(value)
    # _finalize_abandonment returns truthy when it actually finalized; a falsy
    # return means "the pending PPV changed underneath us" and yields STALE.
    return True


# --- 31: real-fan Auto behaviour is unchanged when the connector is on ------


def test_real_fan_auto_still_delivers_through_the_platform(world, spy, monkeypatch):
    """The simulation changes must not turn a real fan's Auto reply local."""
    monkeypatch.setenv("APIFANSLY_ENABLED", "true")
    db, calls = world
    sent: list[tuple] = []

    for row in db.tables["fans"]:
        row["platform_fan_id"] = "884422113355"
        row["fansly_group_id"] = "group-1"

    monkeypatch.setattr(
        suggestions,
        "get_fan_by_id",
        lambda _f: _value(
            Fan(
                id="fan-test",
                display_name="Real Fan",
                platform_fan_id="884422113355",
                fansly_group_id="group-1",
            )
        ),
    )

    async def fake_send(account_id, group_id, text):
        sent.append((account_id, group_id, text))
        return f"platform-{len(sent)}"

    monkeypatch.setattr("main.send_fansly_message", fake_send)

    task_holder: dict = {}

    async def scenario():
        task = asyncio.create_task(
            suggestions._debounced_auto_reply(
                "fan-test",
                "creator-1",
                skip_debounce=True,
                skip_availability=True,
                skip_human_delays=True,
            )
        )
        suggestions._pending_auto_replies["fan-test"] = task
        task_holder["task"] = task
        await task

    _run(scenario())

    assert [s[2] for s in sent] == ["hey", "what are you doing?"]
    assert [r["content"] for r in _creator_rows(db)] == ["hey", "what are you doing?"]


def test_real_fan_auto_is_blocked_when_the_connector_is_disabled(world, spy, monkeypatch):
    """Never persist a creator message as delivered when nothing was sent."""
    monkeypatch.setenv("APIFANSLY_ENABLED", "false")
    db, _ = world
    for row in db.tables["fans"]:
        row["platform_fan_id"] = "884422113355"
        row["fansly_group_id"] = "group-1"

    monkeypatch.setattr(
        suggestions,
        "get_fan_by_id",
        lambda _f: _value(
            Fan(
                id="fan-test",
                display_name="Real Fan",
                platform_fan_id="884422113355",
                fansly_group_id="group-1",
            )
        ),
    )

    async def scenario():
        task = asyncio.create_task(
            suggestions._debounced_auto_reply(
                "fan-test",
                "creator-1",
                skip_debounce=True,
                skip_availability=True,
                skip_human_delays=True,
            )
        )
        suggestions._pending_auto_replies["fan-test"] = task
        await task

    _run(scenario())
    assert _creator_rows(db) == []
    assert spy.requests == []


# --- the simulated turn is processed once, by the simulator -----------------


def test_simulated_fan_message_is_marked_as_an_owner_simulation_event(world, spy):
    """The marker the database webhook keys off, written where the row is.

    Without it the INSERT also reached POST /generate-suggestions and the
    ordinary inbound pipeline ran a second time for one simulated turn.
    """
    from core.simulation import is_simulation_message

    db, _ = world
    _run(
        suggestions.run_simulated_inbound(
            fan_id="fan-test", creator_id="creator-1", message="hii", fast=True
        )
    )

    rows = [r for r in _fan_rows(db) if r["content"] == "hii"]
    assert len(rows) == 1
    assert is_simulation_message(rows[0]["media_context"]) is True


def test_one_simulated_inbound_runs_full_auto_exactly_once(world, spy, monkeypatch):
    """Every analysis and planning stage runs once, not twice.

    This is the duplicate-processing bug stated as a count. The double pass used
    to run situation analysis, the commercial orchestrator, price learning and
    the conversation director twice for a single simulated fan turn.
    """
    monkeypatch.setenv("COMMERCIAL_LAYER_ENABLED", "true")
    db, calls = world
    orchestrated: list[dict] = []
    priced: list[dict] = []

    async def fake_orchestrate(**kwargs):
        orchestrated.append(kwargs)
        from models.commercial import ActionType

        return SimpleNamespace(
            action=ActionType.CONTINUE_NORMAL_CHAT,
            selected_package_set_ids=None,
            session_budget_cents=None,
            model_dump=lambda mode=None: {"action": "CONTINUE_NORMAL_CHAT"},
        )

    async def fake_refresh_price_learning(**kwargs):
        priced.append(kwargs)
        return {"mode": "learning"}

    monkeypatch.setattr(suggestions, "orchestrate", fake_orchestrate)
    monkeypatch.setattr(suggestions, "refresh_price_learning", fake_refresh_price_learning)
    monkeypatch.setattr(suggestions, "_within_daily_caps", lambda *_a, **_k: _value((True, "")))

    result = _run(
        suggestions.run_simulated_inbound(
            fan_id="fan-test", creator_id="creator-1", message="hii", fast=True
        )
    )

    assert len(_fan_rows(db)) == 3, "exactly one new fan message"
    assert len(calls["analyzer"]) == 1, "situation analysis must run once"
    assert len(orchestrated) == 1, "no duplicate commercial mutation"
    # Price learning is refreshed twice WITHIN one turn by design — once before
    # the analysis and once after it writes commercial state. Two is therefore
    # the single-pass count here; the doubled turn this test exists to catch
    # showed four.
    assert len(priced) == 2, "price learning must run once per pass, not twice"
    assert len(calls["director"]) == 1, "the conversation director must run once"
    assert len(calls["route"]) == 1
    assert len(calls["writer"]) == 1, "one Full Auto generation, not two"
    assert result["outcome"] == "replied"
    assert spy.requests == [], "zero API Fansly requests"


def test_simulator_never_runs_assisted_generation(world, spy, monkeypatch):
    """Auto and Assisted are different pipelines. Only Auto may run here."""
    db, calls = world
    assisted: list[tuple] = []

    async def forbidden(*args, **kwargs):
        assisted.append((args, kwargs))
        raise AssertionError("the simulator must never run Assisted generation")

    monkeypatch.setattr(suggestions, "get_suggestions", forbidden, raising=False)
    monkeypatch.setattr(
        suggestions, "process_incoming_fan_message", forbidden, raising=False
    )

    _run(
        suggestions.run_simulated_inbound(
            fan_id="fan-test", creator_id="creator-1", message="hii", fast=True
        )
    )

    assert assisted == []
    assert len(calls["writer"]) == 1
    assert all(
        call["telemetry_context"]["feature"] == "auto_reply"
        for call in calls["writer"]
    )


# --- a writer failure is not a decision ------------------------------------


def test_writer_failure_reports_outcome_writer_failed(world, spy, monkeypatch):
    """generate_replies fails closed with an empty list. That is a broken
    deployment, and the simulator must never present it as Full Auto choosing
    to stay quiet."""
    db, calls = world

    async def writer_fails(prompt, persona, **kwargs):
        calls["writer"].append({"prompt": prompt, **kwargs})
        return []

    monkeypatch.setattr(suggestions, "generate_replies", writer_fails)

    result = _run(
        suggestions.run_simulated_inbound(
            fan_id="fan-test", creator_id="creator-1", message="hii", fast=True
        )
    )

    assert result["outcome"] == "writer_failed"
    assert result["creator_messages"] == []
    assert result["analysis_degraded"] is False
    assert len(calls["writer"]) == 1, "the writer was reached and failed there"
    assert spy.requests == []


def test_intentional_no_send_is_distinct_from_writer_failure(world, spy, monkeypatch):
    """A turn that stops before the writer is a real Full Auto decision."""
    db, calls = world

    monkeypatch.setattr(
        suggestions,
        "get_fan_by_id",
        lambda _f: _value(
            Fan(
                id="fan-test",
                display_name="Jostar",
                platform_fan_id="test_jostar",
                fansly_group_id="stale-group-99",
                needs_human_review=True,
            )
        ),
    )
    monkeypatch.setattr(suggestions, "_crisis_freezes_chat", lambda *_a, **_k: _value(True))

    result = _run(
        suggestions.run_simulated_inbound(
            fan_id="fan-test", creator_id="creator-1", message="hii", fast=True
        )
    )

    assert result["outcome"] == "no_send"
    assert result["creator_messages"] == []
    assert calls["writer"] == [], "a no-send decision never reaches the writer"


def test_degraded_analysis_outranks_every_other_outcome(world, spy, monkeypatch):
    """The fail-closed analyzer keeps its own name rather than collapsing into
    the generic no-send."""
    db, calls = world

    async def degraded(_ctx, telemetry_context=None):
        return {"analysis_degraded": True, "degraded_reason": "provider_down"}

    monkeypatch.setattr(suggestions, "analyze_situation", degraded)
    monkeypatch.setattr(suggestions, "analysis_is_degraded", lambda _s: True)
    monkeypatch.setattr(suggestions, "degraded_reason", lambda _s: "provider_down")

    result = _run(
        suggestions.run_simulated_inbound(
            fan_id="fan-test", creator_id="creator-1", message="hi", fast=True
        )
    )

    assert result["outcome"] == "analyzer_degraded"
    assert result["analysis_degraded"] is True


def test_a_successful_turn_reports_replied(world, spy):
    result = _run(
        suggestions.run_simulated_inbound(
            fan_id="fan-test", creator_id="creator-1", message="hi", fast=True
        )
    )

    assert result["outcome"] == "replied"
    assert len(result["creator_messages"]) == 2


def test_live_inbound_path_is_unaffected_by_the_outcome_sink(world, spy, monkeypatch):
    """Every non-simulator caller passes no sink and behaves exactly as before."""
    db, calls = world

    async def writer_fails(prompt, persona, **kwargs):
        calls["writer"].append({"prompt": prompt, **kwargs})
        return []

    monkeypatch.setattr(suggestions, "generate_replies", writer_fails)

    async def scenario():
        task = asyncio.create_task(
            suggestions._debounced_auto_reply(
                "fan-test",
                "creator-1",
                skip_debounce=True,
                skip_availability=True,
                skip_human_delays=True,
            )
        )
        suggestions._pending_auto_replies["fan-test"] = task
        await task

    _run(scenario())

    assert _creator_rows(db) == []
    assert spy.requests == []
