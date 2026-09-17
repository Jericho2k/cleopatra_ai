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

    def upsert(self, payload, **kwargs):
        # on_conflict matters: fan_experience_scenes and fan_commercial_states
        # are one row per fan, and a fake that appends instead of replacing
        # would hand every turn the FIRST scene ever written — i.e. it would
        # make multi-turn choreography untestable while looking like it worked.
        self.mode = "upsert"
        self.payload = payload
        self.conflict = kwargs.get("on_conflict")
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
        if self.mode == "upsert":
            rows = self.payload if isinstance(self.payload, list) else [self.payload]
            table = self.store.tables.setdefault(self.name, [])
            keys = [key.strip() for key in str(self.conflict or "").split(",") if key.strip()]
            written = []
            for row in rows:
                row = dict(row)
                existing = None
                if keys:
                    existing = next(
                        (
                            candidate
                            for candidate in table
                            if all(
                                str(candidate.get(key)) == str(row.get(key))
                                for key in keys
                            )
                        ),
                        None,
                    )
                if existing is not None:
                    existing.update(row)
                    written.append(existing)
                else:
                    row.setdefault("id", f"{self.name}-{len(table) + 1}")
                    table.append(row)
                    written.append(row)
            return SimpleNamespace(data=written, count=len(written))
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
def two_bubble_turn(monkeypatch):
    """Pin this turn's bubble count so multipart delivery can be asserted.

    Full Auto now chooses the bubble count deterministically per turn
    (services/message_shape.py), and most turns are one bubble by design. These
    tests are about how a multipart reply is ordered, persisted and delivered,
    not about how often one occurs, so the shape is fixed here and the
    distribution is covered by tests/test_commercial_realism.py.
    """
    from services.message_shape import MessageShape

    monkeypatch.setattr(
        suggestions,
        "choose_message_shape",
        lambda **kwargs: MessageShape(target_bubbles=2, reason="pinned_by_test"),
    )


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

    async def fake_analyze(ctx, telemetry_context=None, **_kwargs):
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

    def fake_route(ctx, **kwargs):
        calls["route"].append(ctx)
        return SimpleNamespace(
            route=SimpleNamespace(value="primary"),
            reason="test",
            primary_target=SimpleNamespace(model="test-writer", provider="test"),
            fallback_target=None,
            prompt_version="writer_v1",
            ai_stack_profile=str(kwargs.get("profile_id") or "cleo_legacy_v1"),
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
    # The Experience Director and the commercial policy/state reads behind the
    # text-intimacy decision run for real against this fake, rather than being
    # stubbed out: the scene is now part of what a Full Auto turn IS, and a
    # world that fakes it away would let it silently stop working.
    monkeypatch.setattr("db.experience_director_queries.get_supabase", lambda: db)
    monkeypatch.setattr("db.commercial_queries.get_supabase", lambda: db)
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
            accepted_offer_set_id=None,
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

    async def degraded(_ctx, telemetry_context=None, **_kwargs):
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


def test_generated_creator_message_is_persisted_locally(world, spy, two_bubble_turn):
    db, _ = world
    result = _run(
        suggestions.run_simulated_inbound(
            fan_id="fan-test", creator_id="creator-1", message="hi", fast=True
        )
    )
    assert len(result["creator_messages"]) == 2
    assert [r["content"] for r in _creator_rows(db)] == ["hey", "what are you doing?"]


def test_multipart_order_is_preserved(world, spy, two_bubble_turn):
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
    """An access complaint is a support handoff, including in simulation."""
    monkeypatch.setenv("APIFANSLY_ENABLED", "true")
    db, calls = world
    for row in db.tables["fans"]:
        row["pending_ppv_check"] = {
            "media_id": "m1",
            "media_ids": ["m1"],
            "price": 25.0,
            "reference": "ref-1",
        }

    async def resend_requested(_ctx, telemetry_context=None, **_kwargs):
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

    result = _run(
        suggestions.run_simulated_inbound(
            fan_id="fan-test", creator_id="creator-1", message="i cant see it", fast=True
        )
    )
    assert spy.requests == []
    assert _creator_rows(db) == []
    assert calls["writer"] == []
    assert result["outcome"] == "human_review"
    assert db.tables["fans"][0]["needs_human_review"] is True
    assert db.tables["fans"][0]["review_reason"] == "content_access_issue"


@pytest.mark.parametrize("commercial_enabled", ["true", "false"])
@pytest.mark.parametrize("platform_fan_id", ["test_jostar", "real-fan-99"])
@pytest.mark.parametrize("pending", [None, {"media_id": "m1", "price": 25, "reference": "ref-1"}])
@pytest.mark.parametrize("resend_signal", ["true", True])
def test_access_issue_preempts_selling_and_preserves_payment_state(
    world, spy, monkeypatch, commercial_enabled, platform_fan_id, pending, resend_signal
):
    """Even a mixed request to buy more cannot turn a delivery complaint into a sale.

    Exercise both live and simulated routing through the actual Auto pipeline.
    An already-purchased item has no pending payment, but still needs support.
    """
    db, calls = world
    fan = db.tables["fans"][0]
    fan.update(platform_fan_id=platform_fan_id, pending_ppv_check=pending)
    monkeypatch.setenv("APIFANSLY_ENABLED", "true")
    monkeypatch.setenv("COMMERCIAL_LAYER_ENABLED", commercial_enabled)
    monkeypatch.setattr(
        suggestions, "get_fan_by_id",
        lambda _f: _value(Fan(
            id="fan-test", display_name="Jostar", platform_fan_id=platform_fan_id,
            fansly_group_id="stale-group-99",
        )),
    )
    monkeypatch.setattr(suggestions, "analyze_situation", lambda *_a, **_k: _value({
        "resend_requested": resend_signal, "purchase_signal": "ready_to_buy",
        "offer_response": "accepted", "crisis_signal": "none",
    }))
    downstream_calls = []

    async def unexpected(*_a, **_k):
        downstream_calls.append(True)
        raise AssertionError("access complaint reached commercial mutation or delivery")

    for name in (
        "refresh_affordability_from_situation", "refresh_price_learning", "orchestrate",
        "plan_session_for_fan", "send_apifansly_message", "generate_replies",
    ):
        monkeypatch.setattr(suggestions, name, unexpected)
    outcome = {}
    _run(suggestions._debounced_auto_reply(
        "fan-test", "creator-1", skip_debounce=True, skip_availability=True,
        skip_human_delays=True, outcome_sink=outcome,
    ))
    assert downstream_calls == []
    assert spy.requests == []
    assert _creator_rows(db) == []
    assert fan["pending_ppv_check"] == pending
    assert fan["needs_human_review"] is True
    assert fan["review_reason"] == "content_access_issue"
    assert outcome == {"outcome": "human_review"}


def test_access_issue_hold_failure_is_visible_and_never_falls_through(world, spy, monkeypatch):
    db, calls = world
    monkeypatch.setattr(suggestions, "analyze_situation", lambda *_a, **_k: _value({
        "resend_requested": "true", "crisis_signal": "none",
    }))

    async def unavailable(*_a, **_k):
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(suggestions, "freeze_fan_for_review", unavailable)
    with pytest.raises(suggestions.HumanReviewHandoffError):
        _run(suggestions.run_simulated_inbound(
            fan_id="fan-test", creator_id="creator-1", message="I paid but cannot open it", fast=True,
        ))
    assert calls["writer"] == []
    assert _creator_rows(db) == []
    assert spy.requests == []


def test_crisis_hold_takes_priority_over_access_issue(world, spy, monkeypatch):
    db, calls = world
    monkeypatch.setattr(suggestions, "analyze_situation", lambda *_a, **_k: _value({
        "resend_requested": "true", "crisis_signal": "self_harm",
    }))

    async def crisis_hold(_creator, fan_id, _situation):
        await suggestions.freeze_fan_for_review(fan_id, "crisis:self_harm")
        return True

    monkeypatch.setattr(suggestions, "_crisis_freezes_chat", crisis_hold)
    _run(suggestions.run_simulated_inbound(
        fan_id="fan-test", creator_id="creator-1", message="help with my account", fast=True,
    ))
    assert db.tables["fans"][0]["review_reason"] == "crisis:self_harm"
    assert calls["writer"] == []
    assert spy.requests == []


def test_full_auto_loads_creator_facts_on_each_turn(world, spy, monkeypatch):
    """Saved creator identity must reach the Auto writer, not just Assisted mode.

    A later correction must replace stale facts on the next turn; it must not
    depend on whether those facts happen to fit in the recent transcript.
    """
    db, calls = world
    legend = {"name": "Maya", "origin": "Lisbon", "other": ["plays the cello"]}
    reads = []

    async def load_legend(creator_id):
        reads.append(creator_id)
        return dict(legend)

    monkeypatch.setattr(suggestions, "get_creator_legend", load_legend)
    for origin in ("Lisbon", "Porto"):
        legend["origin"] = origin
        _run(suggestions.run_simulated_inbound(
            fan_id="fan-test", creator_id="creator-1", message="where are you from again?", fast=True,
        ))
        assert calls["analyzer"][-1].creator_legend == legend
        assert calls["route"][-1].creator_legend == legend
        prompt = str(calls["writer"][-1]["prompt"])
        assert "Maya" in prompt and origin in prompt and "plays the cello" in prompt
    assert reads == ["creator-1", "creator-1"]
    assert spy.requests == []


# --- 28, 29: the PPV path stays local and schedules nothing remote ---------


@pytest.fixture
def ppv_world(world, monkeypatch):
    db, calls = world
    monkeypatch.setenv("COMMERCIAL_LAYER_ENABLED", "true")

    # The writer writes ORDINARY TEXT. What is attached, at what price, comes
    # from the persisted plan below — the writer cannot cause, reprice or
    # mis-address a delivery, and no tag appears anywhere in this fixture.
    async def ppv_reply(prompt, persona, **kwargs):
        calls["writer"].append({"prompt": prompt, **kwargs})
        return ["here it is"]

    session = {
        "status": "active",
        "current_index": 0,
        "awaiting_purchase_index": None,
        "plan": [
            {
                "step_number": 1,
                "step_count": 1,
                "media_ids": ["media-77"],
                "media_id": "media-77",
                "price": 35.0,
                "price_cents": 3500,
                "set_id": "set-77",
                "asset_type": "photo_set",
                "description": "bedroom bundle (4 pcs)",
                "sent": False,
                "purchased": False,
            }
        ],
    }

    async def fake_orchestrate(**kwargs):
        from models.commercial import ActionType

        return SimpleNamespace(
            action=ActionType.SEND_NEXT_PPV_STEP,
            accepted_offer_set_id=None,
            session_budget_cents=None,
            model_dump=lambda mode=None: {"action": "SEND_NEXT_PPV_STEP"},
        )

    from models.commercial import CreatorPolicy

    monkeypatch.setattr(suggestions, "get_fan_session", lambda _f: _value(session))
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

    async def bought(_ctx, telemetry_context=None, **_kwargs):
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


def test_real_fan_auto_still_delivers_through_the_platform(world, spy, monkeypatch, two_bubble_turn):
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


def test_real_fan_auto_does_not_depend_on_the_simulator_turn_record(
    world, spy, monkeypatch, two_bubble_turn
):
    """Backend-driven, and it stays backend-driven.

    The durable/pollable turn record exists for operator VISIBILITY in the
    Simulator. A real fan's reply must not acquire a dependency on it — nothing
    is watching, nobody polls, and a delivery that waited on a dashboard
    connection would be a far worse outage than the one this pass fixes.
    """
    monkeypatch.setenv("APIFANSLY_ENABLED", "true")
    db, _calls = world
    sent: list[tuple] = []
    turn_writes: list[str] = []

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

    from services import simulation_turns

    async def forbidden(*_args, **_kwargs):
        turn_writes.append("start_turn")
        raise AssertionError("real Full Auto must not create a simulation turn")

    monkeypatch.setattr(simulation_turns, "start_turn", forbidden)
    monkeypatch.setattr(simulation_turns, "execute_turn", forbidden)

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

    assert [s[2] for s in sent] == ["hey", "what are you doing?"]
    assert turn_writes == []
    assert db.tables.get("simulation_turns", []) == []


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
            accepted_offer_set_id=None,
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


def test_existing_review_hold_is_distinct_from_writer_failure(world, spy, monkeypatch):
    """The simulator identifies an existing hold instead of claiming silence."""
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

    assert result["outcome"] == "human_review"
    assert result["creator_messages"] == []
    assert calls["writer"] == [], "a no-send decision never reaches the writer"


def test_degraded_analysis_outranks_every_other_outcome(world, spy, monkeypatch):
    """The fail-closed analyzer keeps its own name rather than collapsing into
    the generic no-send."""
    db, calls = world

    async def degraded(_ctx, telemetry_context=None, **_kwargs):
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


def test_a_successful_turn_reports_replied(world, spy, two_bubble_turn):
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


# --- fast mode really is fast ----------------------------------------------
#
# Railway showed ``fast=True`` next to
# ``[AUTO TIMING] mode=live away=5.39s compose=6.03s``. The pauses were in fact
# already skipped — the line reported the schedule the planner COMPUTED, not the
# one the turn awaited — but an operator cannot tell those apart from a log, and
# "is the simulator actually waiting six seconds?" is not a question that should
# need a code read. So the behaviour is pinned and the line now says which it is.


def test_a_fast_simulated_turn_awaits_no_human_delay(world, spy):
    db, calls = world

    _run(
        suggestions.run_simulated_inbound(
            fan_id="fan-test", creator_id="creator-1", message="hii", fast=True
        )
    )

    assert all(seconds == 0 for seconds in calls["sleeps"]), (
        f"a fast simulated turn must not wait: {calls['sleeps']}"
    )
    assert _creator_rows(db), "and it still produces the reply"


def test_fast_mode_says_so_in_the_timing_line(world, spy, capsys):
    _run(
        suggestions.run_simulated_inbound(
            fan_id="fan-test", creator_id="creator-1", message="hii", fast=True
        )
    )

    timing = [
        line
        for line in capsys.readouterr().out.splitlines()
        if "[AUTO TIMING]" in line and "parts=" in line
    ]
    assert timing, "the turn must report its timing"
    line = timing[-1]
    assert "mode=simulation_fast" in line
    assert "delays_skipped=true" in line
    assert "away=0.00s" in line
    assert "compose=0.00s" in line
    # The computed schedule is still reported, so the two can be compared.
    assert "planned_mode=" in line
    assert "planned_away=" in line


def test_a_fast_turn_still_runs_the_real_pipeline(world, spy, monkeypatch):
    """Fast removes the waiting and nothing else. Analysis, planning, routing
    and generation all run exactly as they do live."""
    db, calls = world

    _run(
        suggestions.run_simulated_inbound(
            fan_id="fan-test", creator_id="creator-1", message="hii", fast=True
        )
    )

    assert len(calls["analyzer"]) == 1
    assert len(calls["director"]) == 1
    assert len(calls["route"]) == 1
    assert len(calls["writer"]) == 1


def test_live_timing_is_untouched(world, spy, capsys):
    """The same turn with fast=False keeps the real schedule and says so."""
    db, calls = world

    _run(
        suggestions.run_simulated_inbound(
            fan_id="fan-test", creator_id="creator-1", message="hii", fast=False
        )
    )

    timing = [
        line
        for line in capsys.readouterr().out.splitlines()
        if "[AUTO TIMING]" in line and "parts=" in line
    ]
    assert timing
    assert "mode=simulation_fast" not in timing[-1]
    # Availability is skipped for every simulator turn (there is nobody to be
    # away from); the composition pause is real and is genuinely awaited.
    assert any(seconds > 0 for seconds in calls["sleeps"]), (
        "fast=False must still exercise the human-delay path"
    )


# --- inventory authority, end to end ---------------------------------------
#
# The unit-level contract lives in tests/test_inventory_authority.py. What is
# pinned here is that the real Full Auto turn carries it: the writer is TOLD
# what exists, and a writer that promises a clip anyway does not get to send it.


def _photo_only_decision(action=None):
    from models.commercial import ActionType, CommercialDecision, Offer

    return CommercialDecision(
        action=action or ActionType.OFFER_NEXT_UNLOCK,
        goal="offer him the one next thing",
        next_offer=Offer(
            offer_id="offer:a",
            label="private photo set",
            price_cents=3000,
            set_id="a",
            experience="bedroom, black lingerie",
            legal_description="bedroom, black lingerie",
            media_count=5,
            asset_type="photo_set",
        ),
        authorized_asset_types=["photo_set"],
        vault_asset_types=["photo_set"],
        unavailable_asset_type_requested="video",
    )


def test_the_writer_is_told_there_is_no_video(world, spy, monkeypatch):
    monkeypatch.setenv("COMMERCIAL_LAYER_ENABLED", "true")
    db, calls = world

    async def fake_orchestrate(**_kwargs):
        return _photo_only_decision()

    monkeypatch.setattr(suggestions, "orchestrate", fake_orchestrate)
    monkeypatch.setattr(
        suggestions, "_within_daily_caps", lambda *_a, **_k: _value((True, ""))
    )

    _run(
        suggestions.run_simulated_inbound(
            fan_id="fan-test",
            creator_id="creator-1",
            message="do you have any videos?",
            fast=True,
        )
    )

    prompt = calls["writer"][-1]["prompt"]
    text = "\n".join(
        str(message["content"])
        for message in prompt
        if isinstance(message, dict)
    )
    assert "CONTENT INVENTORY" in text
    assert "NO video" in text
    assert "photo sets" in text
    # And the pivot instruction, because he asked for one in as many words.
    assert "he just asked for video and there is none" in text


def test_an_explicit_video_request_with_no_videos_still_gets_a_real_reply(
    world, spy, monkeypatch
):
    """Not silence, not a fabricated clip, not an inventory explanation."""
    monkeypatch.setenv("COMMERCIAL_LAYER_ENABLED", "true")
    db, calls = world

    async def fake_orchestrate(**_kwargs):
        return _photo_only_decision()

    async def promises_video(prompt, persona, **kwargs):
        calls["writer"].append({"prompt": prompt, **kwargs})
        return [
            "omg yes i have a video for you 😏",
            "wait till you see the clip",
            "i'll send you a vid",
        ]

    monkeypatch.setattr(suggestions, "orchestrate", fake_orchestrate)
    monkeypatch.setattr(suggestions, "generate_replies", promises_video)
    monkeypatch.setattr(
        suggestions, "_within_daily_caps", lambda *_a, **_k: _value((True, ""))
    )

    result = _run(
        suggestions.run_simulated_inbound(
            fan_id="fan-test",
            creator_id="creator-1",
            message="got any videos?",
            fast=True,
        )
    )

    assert result["outcome"] == "replied", "an unavailable format is not a no-send"
    sent = " ".join(row["content"] for row in result["creator_messages"])
    assert sent.strip(), "the fan gets an actual message"
    for word in ("video", "clip", "vid"):
        assert word not in sent.lower(), f"promised {word} with none in inventory"
    # And no internal language leaks in its place.
    for leak in ("inventory", "vault", "approved", "system", "unavailable"):
        assert leak not in sent.lower()


def test_an_authorised_video_is_still_allowed_through(world, spy, monkeypatch):
    monkeypatch.setenv("COMMERCIAL_LAYER_ENABLED", "true")
    db, calls = world

    async def fake_orchestrate(**_kwargs):
        decision = _photo_only_decision()
        decision.authorized_asset_types = ["photo_set", "video"]
        decision.vault_asset_types = ["photo_set", "video"]
        decision.unavailable_asset_type_requested = None
        return decision

    async def promises_video(prompt, persona, **kwargs):
        calls["writer"].append({"prompt": prompt, **kwargs})
        return ["wait till you see the video 😈"]

    monkeypatch.setattr(suggestions, "orchestrate", fake_orchestrate)
    monkeypatch.setattr(suggestions, "generate_replies", promises_video)
    monkeypatch.setattr(
        suggestions, "_within_daily_caps", lambda *_a, **_k: _value((True, ""))
    )

    result = _run(
        suggestions.run_simulated_inbound(
            fan_id="fan-test",
            creator_id="creator-1",
            message="got any videos?",
            fast=True,
        )
    )

    sent = " ".join(row["content"] for row in result["creator_messages"])
    assert "video" in sent.lower(), "an approved video may be offered as planned"


# --- a recoverable plan failure is not a no-send ----------------------------


def test_a_stale_offer_recovers_instead_of_sending_nothing(world, spy, monkeypatch):
    """The exact production trace: SEND_NEXT_PPV_STEP, selected_set_unavailable,
    outcome=no_send — with the same message succeeding on the next attempt."""
    from models.commercial import ActionType, Offer
    from services import session_plan_recovery

    monkeypatch.setenv("COMMERCIAL_LAYER_ENABLED", "true")
    db, calls = world

    async def fake_orchestrate(**_kwargs):
        decision = _photo_only_decision(action=ActionType.SEND_NEXT_PPV_STEP)
        decision.accepted_offer_set_id = "gone-1"
        decision.session_budget_cents = 3000
        return decision

    async def stale_plan(*_args, **_kwargs):
        return {"status": "selected_set_unavailable", "session": None, "missing_set_ids": ["gone-1"]}

    replacement = Offer(
        offer_id="offer:fresh-1",
        label="private photo set",
        price_cents=3000,
        set_id="fresh-1",
        asset_type="photo_set",
    )

    async def fake_recover(**kwargs):
        assert kwargs["status"] == "selected_set_unavailable"
        assert kwargs["had_accepted_contract"] is True
        return session_plan_recovery.PlanRecovery(
            status="selected_set_unavailable",
            failure_class=session_plan_recovery.PlanFailureClass.RECOVERABLE,
            replacement_offer=replacement,
            present_replacement=True,
            accepted_contract_lost=True,
            reason="accepted_contract_unavailable",
        )

    monkeypatch.setattr(suggestions, "orchestrate", fake_orchestrate)
    monkeypatch.setattr(suggestions, "plan_session_for_fan", stale_plan)
    monkeypatch.setattr(session_plan_recovery, "recover_session_plan", fake_recover)
    monkeypatch.setattr(
        suggestions, "_within_daily_caps", lambda *_a, **_k: _value((True, ""))
    )

    result = _run(
        suggestions.run_simulated_inbound(
            fan_id="fan-test", creator_id="creator-1", message="yes the $30 one", fast=True
        )
    )

    assert result["outcome"] == "replied", (
        "a recoverable planning failure must never read as a decision to send nothing"
    )
    assert result["creator_messages"], "the fan gets an answer"
    # The replacement is presented, not charged for: no PPV was delivered.
    assert all(
        not (row.get("media_context") or {}).get("ppv")
        for row in result["creator_messages"]
    )


def test_an_unrecoverable_plan_reports_itself_rather_than_a_no_send(
    world, spy, monkeypatch
):
    from models.commercial import ActionType
    from services import session_plan_recovery

    monkeypatch.setenv("COMMERCIAL_LAYER_ENABLED", "true")
    db, calls = world

    async def fake_orchestrate(**_kwargs):
        decision = _photo_only_decision(action=ActionType.SEND_NEXT_PPV_STEP)
        decision.accepted_offer_set_id = "gone-1"
        decision.session_budget_cents = 3000
        return decision

    async def stale_plan(*_args, **_kwargs):
        return {"status": "selected_set_unavailable", "session": None}

    async def exploding_recovery(**_kwargs):
        raise RuntimeError("PostgREST connection terminated")

    monkeypatch.setattr(suggestions, "orchestrate", fake_orchestrate)
    monkeypatch.setattr(suggestions, "plan_session_for_fan", stale_plan)
    monkeypatch.setattr(
        session_plan_recovery, "recover_session_plan", exploding_recovery
    )
    monkeypatch.setattr(
        suggestions, "_within_daily_caps", lambda *_a, **_k: _value((True, ""))
    )

    result = _run(
        suggestions.run_simulated_inbound(
            fan_id="fan-test", creator_id="creator-1", message="yes the $30 one", fast=True
        )
    )

    assert result["outcome"] == "plan_unrecoverable", (
        "a broken sale must be reported as broken, not as the product working"
    )
    assert result["outcome"] != "no_send"


# --- a fresh fan's very first turn -----------------------------------------
#
# The director opens every new conversation with
# phase=OPENING action=RESPOND_AND_OPEN, and derive_session_strategy mapped that
# to NextBestAction.RESPOND_AND_OPEN — an enum member that has never existed.
# Because derive_session_strategy runs OUTSIDE the persistence try/except in
# plan_next_action, the AttributeError propagated out of the whole Auto turn:
# no strategy, no writer call, no reply. The unit contract lives in
# tests/test_session_strategy_director.py; what is pinned here is that the real
# pipeline survives the turn end to end.


def test_a_fresh_fans_opening_turn_runs_the_writer_and_replies(
    world, spy, monkeypatch
):
    from services.adaptive_session_planner import plan_next_action

    monkeypatch.setenv("ADAPTIVE_SESSION_PLANNER_ENABLED", "true")
    db, calls = world

    async def opening_director(**_kwargs):
        # Exactly what advance_conversation_director emits for a new fan.
        return {
            "phase": "OPENING",
            "action": "RESPOND_AND_OPEN",
            "transition_reason": "first_contact",
            "turns_in_phase": 1,
            "engagement_score": 0,
        }

    monkeypatch.setattr(suggestions, "direct_conversation", opening_director)
    # The real planner, not the fixture's stub: derive_session_strategy is the
    # code under test.
    monkeypatch.setattr(suggestions, "plan_next_action", plan_next_action)

    result = _run(
        suggestions.run_simulated_inbound(
            fan_id="fan-test", creator_id="creator-1", message="hey there", fast=True
        )
    )

    assert result["outcome"] == "replied", (
        "an opening turn must not fail closed; it is the product's first message"
    )
    assert result["creator_messages"], "the fan gets an actual reply"
    assert len(calls["writer"]) == 1, "the writer ran"

    strategy = calls["writer"][-1]
    assert strategy, "the writer received a prompt"


def test_the_opening_strategy_reaches_the_writer_prompt(world, spy, monkeypatch):
    """Not just 'no exception' — the strategy has to arrive intact."""
    from services.adaptive_session_planner import plan_next_action

    monkeypatch.setenv("ADAPTIVE_SESSION_PLANNER_ENABLED", "true")
    db, calls = world

    async def opening_director(**_kwargs):
        return {
            "phase": "OPENING",
            "action": "RESPOND_AND_OPEN",
            "transition_reason": "first_contact",
        }

    monkeypatch.setattr(suggestions, "direct_conversation", opening_director)
    monkeypatch.setattr(suggestions, "plan_next_action", plan_next_action)

    _run(
        suggestions.run_simulated_inbound(
            fan_id="fan-test", creator_id="creator-1", message="hey there", fast=True
        )
    )

    prompt = calls["writer"][-1]["prompt"]
    text = "\n".join(
        str(message["content"]) for message in prompt if isinstance(message, dict)
    )
    assert "ADAPTIVE SESSION STRATEGY" in text
    assert "next action: CONTINUE_CHAT" in text
    assert "respond specifically to what he said" in text


# --- which AI stack wrote this message --------------------------------------
#
# Every creator message the pipeline writes carries the profile that produced
# it, inside the existing media_context jsonb. Without it, "which brain wrote
# this?" is unanswerable from the row months later, which is the whole point of
# running two profiles side by side.


def test_a_generated_creator_message_records_the_effective_profile(
    world, spy, two_bubble_turn, monkeypatch
):
    monkeypatch.setenv("AI_STACK_PROFILE", "cleo_v2")
    monkeypatch.setenv("AI_STACK_CACHE_SECONDS", "0")
    from services.ai_stack import clear_ai_stack_cache

    clear_ai_stack_cache()
    db, _ = world

    _run(
        suggestions.run_simulated_inbound(
            fan_id="fan-test", creator_id="creator-1", message="hi", fast=True
        )
    )

    creator_rows = _creator_rows(db)
    assert creator_rows
    for row in creator_rows:
        marker = (row.get("media_context") or {}).get("ai_stack")
        assert marker is not None, "every creator message names the stack that wrote it"
        assert marker["profile"] == "cleo_v2"
        # Enough to debug a bad reply without a telemetry join.
        assert marker["route"]
        assert marker["model"]


def test_the_marker_travels_alongside_a_ppv_rather_than_replacing_it():
    """The PPV payload is what delivery and purchase reconciliation read. The
    stack marker is additive metadata and must never displace it."""
    from services.suggestions import _with_ai_stack, message_ai_stack_metadata

    ppv_context = {"ppv": {"media_ids": ["111"], "price": 25, "price_cents": 2500}}
    marker = message_ai_stack_metadata(
        SimpleNamespace(
            route=SimpleNamespace(value="commercial_complex"),
            prompt_version="writer_v2",
            primary_target=SimpleNamespace(provider="openrouter", model="kimi"),
        ),
        profile_id="cleo_v2",
    )

    merged = _with_ai_stack(ppv_context, marker)

    assert merged["ppv"] == ppv_context["ppv"]
    assert merged["ai_stack"]["profile"] == "cleo_v2"
    assert merged["ai_stack"]["route"] == "commercial_complex"


def test_a_plain_message_with_no_other_metadata_still_records_the_stack():
    from services.suggestions import _with_ai_stack, message_ai_stack_metadata

    merged = _with_ai_stack(None, message_ai_stack_metadata(None, profile_id="cleo_legacy_v1"))

    assert merged == {"ai_stack": {"profile": "cleo_legacy_v1"}}


def test_the_simulator_marker_and_the_stack_marker_coexist():
    """A simulated fan message carries the simulator's ownership marker; a
    creator message carries the stack marker. Neither may shadow the other."""
    from core.simulation import is_simulation_message, simulation_message_marker
    from services.suggestions import _with_ai_stack, message_ai_stack_metadata

    merged = _with_ai_stack(
        simulation_message_marker(),
        message_ai_stack_metadata(None, profile_id="cleo_v2"),
    )

    assert is_simulation_message(merged) is True
    assert merged["ai_stack"]["profile"] == "cleo_v2"


# --- Sprint 0: ground truth travels with the message it explains ------------


@pytest.fixture
def traced_writer(monkeypatch):
    """A writer stub that fills the trace the way the real ladder does.

    The default ``world`` stub ignores the trace, which is correct for tests
    that do not care. These ones assert that the model which ANSWERED reaches
    the persisted row, so the stub has to answer as a specific model — and it
    deliberately answers as the FALLBACK, which is the case finding H says the
    old marker got wrong.
    """
    from ai.writer_recovery import ROLE_FALLBACK
    from models.model_runtime import ModelTarget

    async def fake_generate(prompt, persona, **kwargs):
        trace = kwargs.get("trace")
        if trace is not None:
            trace.record_request(
                primary_target=ModelTarget(
                    name="kimi", provider="openrouter", model="moonshotai/kimi-k2.6"
                ),
                fallback_target=None,
                profile="cleo_legacy_v1",
                policy="legacy",
                deadline_seconds=30.0,
            )
            trace.record_success(
                target=ModelTarget(
                    name="qwen", provider="together", model="Qwen/Qwen3.7-Plus"
                ),
                role=ROLE_FALLBACK,
                attempt_index=3,
                upstream_provider="together",
                outcome="qwen_emergency_fallback",
                attempts=3,
                pinned_attempts=2,
                alternate_attempts=0,
                elapsed_ms=900,
            )
        return ["hey"]

    monkeypatch.setattr(suggestions, "generate_replies", fake_generate)


def _provenance_rows(db) -> list[dict]:
    from services.reply_provenance import provenance_of

    return [
        provenance_of(row.get("media_context"))
        for row in _creator_rows(db)
        if provenance_of(row.get("media_context"))
    ]


def test_a_sent_reply_records_the_event_that_caused_it(world, spy, traced_writer):
    db, _ = world
    _run(
        suggestions.run_simulated_inbound(
            fan_id="fan-test", creator_id="creator-1", message="are you there?", fast=True
        )
    )

    from services.reply_provenance import fingerprint

    records = _provenance_rows(db)
    assert records, "every reply this pipeline sends carries its provenance"
    assert records[0]["trigger"]["kind"] == "fan_message"
    assert records[0]["trigger"]["text_fingerprint"] == fingerprint("are you there?")
    assert records[0]["mode"] == "auto"


def test_a_sent_reply_records_the_model_that_actually_answered(
    world, spy, traced_writer
):
    """Finding H, end to end: the row names Qwen because Qwen answered."""
    db, _ = world
    _run(
        suggestions.run_simulated_inbound(
            fan_id="fan-test", creator_id="creator-1", message="hi", fast=True
        )
    )

    record = _provenance_rows(db)[0]
    assert record["writer"]["requested"]["model"] == "moonshotai/kimi-k2.6"
    assert record["writer"]["actual"]["model"] == "Qwen/Qwen3.7-Plus"
    assert record["writer"]["served_by_requested_model"] is False

    # And the compact stack marker on the same row agrees with it, rather than
    # still naming the model the router asked for.
    stack = _creator_rows(db)[0]["media_context"]["ai_stack"]
    assert stack["model"] == "Qwen/Qwen3.7-Plus"
    assert stack["requested_model"] == "moonshotai/kimi-k2.6"


def test_a_sent_reply_records_the_context_it_was_allowed_to_see(
    world, spy, traced_writer
):
    db, _ = world
    _run(
        suggestions.run_simulated_inbound(
            fan_id="fan-test", creator_id="creator-1", message="hi", fast=True
        )
    )

    context = _provenance_rows(db)[0]["context"]
    assert context["history_messages"] >= 1
    # Finding D: the analyzer never sees more than the writer.
    assert context["analyzer_window"] <= context["writer_window"]
    assert context["stack_profile"]


def test_a_sent_reply_records_the_commit_and_flag_digest_that_produced_it(
    world, spy, traced_writer
):
    db, _ = world
    _run(
        suggestions.run_simulated_inbound(
            fan_id="fan-test", creator_id="creator-1", message="hi", fast=True
        )
    )

    build = _provenance_rows(db)[0]["build"]
    assert build["sha"]
    assert build["flags_digest"]
    assert "flags" not in build, "the mapping lives on /build, not on every row"


def test_the_bubbles_of_one_reply_share_one_turn(
    world, spy, two_bubble_turn, monkeypatch
):
    async def fake_generate(prompt, persona, **kwargs):
        return ["hey | what are you doing?"]

    monkeypatch.setattr(suggestions, "generate_replies", fake_generate)
    db, _ = world
    _run(
        suggestions.run_simulated_inbound(
            fan_id="fan-test", creator_id="creator-1", message="hi", fast=True
        )
    )

    records = _provenance_rows(db)
    assert len(records) == 2
    assert records[0]["turn_id"] == records[1]["turn_id"]
    assert [r["part"] for r in records] == [0, 1]
    assert records[0]["parts"] == 2


def test_a_reply_the_code_rewrote_is_not_attributed_to_the_model(
    world, spy, monkeypatch
):
    """A writer-emitted delivery tag is stripped; the record has to say so."""

    async def tagged_reply(prompt, persona, **kwargs):
        return ["here you go [PPV:set-1]"]

    monkeypatch.setattr(suggestions, "generate_replies", tagged_reply)
    db, _ = world
    _run(
        suggestions.run_simulated_inbound(
            fan_id="fan-test", creator_id="creator-1", message="hi", fast=True
        )
    )

    from services.reply_provenance import TRANSFORM_PPV_TAG_STRIPPED

    record = _provenance_rows(db)[0]
    assert TRANSFORM_PPV_TAG_STRIPPED in record["transforms"]


def test_a_local_test_delivery_is_not_claimed_as_a_platform_delivery(
    world, spy, traced_writer
):
    """The simulator's text path gets no platform receipt, so it claims none."""
    db, _ = world
    _run(
        suggestions.run_simulated_inbound(
            fan_id="fan-test", creator_id="creator-1", message="hi", fast=True
        )
    )

    delivery = _provenance_rows(db)[0]["delivery"]
    assert delivery["kind"] == "text"
    assert delivery["accepted_by_platform"] is False
    assert delivery["platform_message_id"] is None


# --- Sprint 2: what the conversation is carrying reaches the turn -----------


@pytest.fixture
def continuity_store(world, monkeypatch):
    """Point the continuity layer at the simulator's in-memory database."""
    from services import conversation_continuity

    db, _ = world
    db.tables.setdefault("conversation_open_threads", [])
    db.tables.setdefault("conversation_episodes", [])
    monkeypatch.setattr(conversation_continuity, "get_supabase", lambda: db)
    return conversation_continuity


def test_an_unanswered_question_reaches_the_writer(world, spy, continuity_store):
    """A question from thirty turns ago is in no transcript window at all."""
    from models.conversation_continuity import OpenThread, ThreadKind, ThreadParty

    db, calls = world
    _run(
        continuity_store.record_open_thread(
            OpenThread(
                creator_id="creator-1",
                fan_id="fan-test",
                kind=ThreadKind.QUESTION,
                raised_by=ThreadParty.FAN,
                summary="whether you ever visit Chicago",
                resolution_condition="you answer it",
            )
        )
    )

    _run(
        suggestions.run_simulated_inbound(
            fan_id="fan-test", creator_id="creator-1", message="hey", fast=True
        )
    )

    prompt = str(calls["writer"][0]["prompt"])
    assert "whether you ever visit Chicago" in prompt


def test_an_access_complaint_becomes_something_the_conversation_carries(
    world, spy, continuity_store, monkeypatch
):
    """Otherwise the next turn has no idea anything was ever wrong."""
    db, _ = world

    async def complains(ctx, **kwargs):
        return {
            "purchase_signal": "none",
            "strategic_move": "build_rapport",
            "resend_requested": "true",
        }

    monkeypatch.setattr(suggestions, "analyze_situation", complains)

    _run(
        suggestions.run_simulated_inbound(
            fan_id="fan-test",
            creator_id="creator-1",
            message="i paid but it will not open",
            fast=True,
        )
    )

    threads = db.tables["conversation_open_threads"]
    assert len(threads) == 1
    assert threads[0]["kind"] == "complaint"
    assert threads[0]["status"] == "open"
    assert _creator_rows(db) == [], "the complaint is answered by a human, not a sale"


def test_the_same_complaint_twice_is_one_obligation(
    world, spy, continuity_store, monkeypatch
):
    db, _ = world

    async def complains(ctx, **kwargs):
        return {
            "purchase_signal": "none",
            "strategic_move": "build_rapport",
            "resend_requested": "true",
        }

    monkeypatch.setattr(suggestions, "analyze_situation", complains)
    monkeypatch.setattr(
        suggestions, "freeze_fan_for_review", lambda *_a, **_k: _value(None)
    )

    for _ in range(2):
        _run(
            suggestions.run_simulated_inbound(
                fan_id="fan-test",
                creator_id="creator-1",
                message="i paid but it will not open",
                fast=True,
            )
        )

    assert len(db.tables["conversation_open_threads"]) == 1


def test_a_reply_records_what_the_packet_actually_assembled(
    world, spy, continuity_store, traced_writer
):
    db, _ = world
    _run(
        suggestions.run_simulated_inbound(
            fan_id="fan-test", creator_id="creator-1", message="hi", fast=True
        )
    )

    packet = _provenance_rows(db)[0]["context"]["packet"]
    assert packet["turns"] >= 1
    assert packet["turn_budget"] >= 1
    assert "dropped_turns" in packet


def test_a_continuity_store_that_is_unavailable_never_stops_a_reply(
    world, spy, traced_writer
):
    """The `world` fixture has no continuity tables at all, which is the test."""
    db, _ = world
    _run(
        suggestions.run_simulated_inbound(
            fan_id="fan-test", creator_id="creator-1", message="hi", fast=True
        )
    )

    assert _creator_rows(db), "losing continuity costs a later turn, never this one"
