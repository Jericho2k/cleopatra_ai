"""A small world for driving the REAL Core v1/v2 turn path, many turns long.

Only the edges are faked: the data sources ``load_evidence`` reads, the GLM
and Kimi transports, and the delivery adapters. Everything between them —
``load_evidence`` itself, state loading and persistence (against a
PostgREST-shaped fake), reconciliation, delta validation, the shared
deterministic validator, the Kimi prompt builder, the writer contract and
``execute_auto_turn`` — is production code.

Scenarios are deliberately neutral: "content A / B / C", an interactive date
scenario, a shared roleplay scene. No explicit copy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, Callable

from models.commercial import CreatorPolicy, FanCommercialState, FanStatus, Offer
from models.model_runtime import ModelTarget
from models.schemas import Fan, Message, Persona
from services import conversation_core, conversational_core, conversational_session
from services import conversational_v2
from services import live_orchestration as lo
from tests.fake_supabase import FakeSupabase

CREATOR_ID = "11111111-1111-1111-1111-111111111111"
FAN_ID = "22222222-2222-2222-2222-222222222222"
PLATFORM_FAN_ID = "test_session_fan"

OWNER_TARGET = ModelTarget(
    name="glm-owner",
    provider="test",
    model="glm-4.6",
    input_per_million=1.0,
    output_per_million=1.0,
)
WRITER_TARGET = ModelTarget(
    name="kimi-writer",
    provider="test",
    model="kimi-k2",
    input_per_million=1.0,
    output_per_million=1.0,
)


async def _value(result: Any) -> Any:
    return result


@dataclass
class Item:
    set_id: str
    price_cents: int
    description: str
    asset_type: str = "photo_set"

    def offer(self) -> Offer:
        return Offer(
            offer_id=f"offer:{self.set_id}",
            label="private video" if self.asset_type == "video" else "private photo set",
            price_cents=self.price_cents,
            set_id=self.set_id,
            legal_description=self.description,
            experience=self.description,
            asset_type=self.asset_type,
            media_count=1 if self.asset_type == "video" else 3,
        )


OwnerStep = dict[str, Any] | Callable[[dict[str, Any]], dict[str, Any]]


@dataclass
class V2World:
    core: str = conversation_core.CORE_CONVERSATIONAL_V2
    catalog: dict[str, Item] = field(default_factory=dict)
    messages: list[Message] = field(default_factory=list)
    sent_ppv: list[dict[str, Any]] = field(default_factory=list)
    pending_offer: Offer | None = None
    last_offer_at: datetime | None = None
    pending_payment: dict[str, Any] | None = None
    affordability: dict[str, Any] = field(default_factory=dict)
    owner_script: list[OwnerStep] = field(default_factory=list)
    writer_script: list[list[str]] = field(default_factory=list)
    owner_payloads: list[dict[str, Any]] = field(default_factory=list)
    owner_systems: list[str] = field(default_factory=list)
    writer_prompts: list[list[dict[str, str]]] = field(default_factory=list)
    results: list[dict[str, Any]] = field(default_factory=list)
    transcript: list[dict[str, Any]] = field(default_factory=list)
    frozen: list[str] = field(default_factory=list)
    clock: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc) - timedelta(hours=2)
    )
    db: FakeSupabase = field(
        default_factory=lambda: FakeSupabase(
            {
                "creators": [{"id": CREATOR_ID, "conversation_core": None}],
                "fans": [
                    {
                        "id": FAN_ID,
                        "creator_id": CREATOR_ID,
                        "platform_fan_id": PLATFORM_FAN_ID,
                        "conversation_core": conversation_core.CORE_CONVERSATIONAL_V2,
                    }
                ],
                conversational_session.SESSION_TABLE: [],
                conversational_core.STATE_TABLE: [],
            }
        )
    )
    _ids: int = 0

    # --- world events ---------------------------------------------------

    def _next_id(self, prefix: str) -> str:
        self._ids += 1
        return f"{prefix}-{self._ids}"

    def _tick(self) -> datetime:
        self.clock += timedelta(minutes=1)
        return self.clock

    def fan_says(self, text: str) -> str:
        message_id = self._next_id("fan-msg")
        self.messages.append(
            Message(id=message_id, role="fan", content=text, sent_at=self._tick())
        )
        self.transcript.append({"speaker": "fan", "text": text})
        return message_id

    def purchase(self, set_id: str) -> None:
        """The platform confirms payment for the locked message (the webhook)."""
        for row in self.sent_ppv:
            if row["set_id"] == set_id:
                row["purchased"] = True
                row["purchased_at"] = self._tick().isoformat()
                row["reference"] = f"purchase-{set_id}"
        self.pending_payment = None
        self.transcript.append({"speaker": "ledger", "event": f"purchased {set_id}"})

    # --- state views ----------------------------------------------------

    def session_row(self) -> dict[str, Any] | None:
        rows = self.db.tables[conversational_session.SESSION_TABLE]
        return rows[0] if rows else None

    def session_state(self):
        row = self.session_row()
        if row is None:
            return conversational_session.empty_session_state()
        payload = dict(row["state"])
        payload["revision"] = row["revision"]
        from models.conversational_session import ConversationalSessionState

        return ConversationalSessionState.model_validate(payload)

    def eligible(self, ceiling: int | None = None) -> list[Offer]:
        sent = {row["set_id"] for row in self.sent_ppv}
        return [
            item.offer()
            for item in self.catalog.values()
            if item.set_id not in sent and (ceiling is None or item.price_cents <= ceiling)
        ]

    def handle_for(self, payload: dict[str, Any], set_id: str) -> str:
        description = self.catalog[set_id].description
        for row in payload["interaction_session"]["content_candidates"]:
            if row["description"] == description:
                return row["candidate_handle"]
        raise AssertionError(f"{set_id} is not an exposed candidate")

    def dialogue_without_media(self) -> list[str]:
        return [
            row["text"]
            for row in self.transcript
            if row.get("speaker") in {"fan", "creator"} and not row.get("media")
        ]

    # --- running turns --------------------------------------------------

    def queue(self, owner: OwnerStep, writer: list[str] | None = None) -> None:
        self.owner_script.append(owner)
        if writer is not None:
            self.writer_script.append(writer)

    async def turn(
        self, text: str, owner: OwnerStep, writer: list[str] | None = None
    ) -> dict[str, Any]:
        message_id = self.fan_says(text)
        self.queue(owner, writer)
        result = await lo.run_auto_turn(
            creator_id=CREATOR_ID,
            fan_id=FAN_ID,
            latest_message=text,
            trigger_identity=message_id,
            conversation_core=self.core,
        )
        self.results.append(result)
        return result


def owner_json(
    *,
    goal: str = "continue the shared moment",
    move: str | dict[str, Any] = "converse",
    operation: dict[str, Any] | None = None,
    session_delta: dict[str, Any] | None = None,
    state_delta: dict[str, Any] | None = None,
    disposition: str = "reply",
    **extra: Any,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "disposition": disposition,
        "response_goal": goal,
        "operation_proposal": operation or {"kind": "none"},
        "confidence": 0.9,
        "next_experience_move": move if isinstance(move, dict) else {"kind": move, "intent": goal},
    }
    if session_delta is not None:
        body["session_delta"] = session_delta
    if state_delta is not None:
        body["state_delta"] = state_delta
    body.update(extra)
    return body


def install(monkeypatch, world: V2World) -> V2World:
    conversation_core.clear_conversation_core_cache()

    def fan() -> Fan:
        return Fan(
            id=FAN_ID,
            display_name="Session Fan",
            creator_id=CREATOR_ID,
            platform_fan_id=PLATFORM_FAN_ID,
            auto_mode=True,
        )

    def commercial_state() -> FanCommercialState:
        return FanCommercialState(
            pending_offer=world.pending_offer,
            last_offer_at=world.last_offer_at,
            status=FanStatus.OFFER_PENDING if world.pending_offer else FanStatus.IDLE,
        )

    def stage(name: str):
        target = WRITER_TARGET if "writer" in name else OWNER_TARGET
        return SimpleNamespace(
            primary_target=lambda: target,
            fallback_target=lambda: None,
            prompt_version=f"{name}_test",
            max_tokens=2048,
        )

    stack = SimpleNamespace(
        profile_id="cleo_v3", profile=SimpleNamespace(stage=stage)
    )

    async def next_offer_with_inventory(*_a, hard_ceiling_cents=None, **_k):
        offers = world.eligible(hard_ceiling_cents)
        return (offers[0] if offers else None), tuple(o.asset_type for o in offers)

    async def next_offer_with_candidates(
        *_a, hard_ceiling_cents=None, desired_experience=None, **_k
    ):
        offers = world.eligible(hard_ceiling_cents)
        if desired_experience and "video" in desired_experience.lower():
            offers.sort(key=lambda offer: offer.asset_type != "video")
        return (
            (offers[0] if offers else None),
            tuple(o.asset_type for o in offers),
            offers[:4],
        )

    async def complete(target, *, system, messages, **_kwargs):
        payload = json.loads(messages[0]["content"].split("\nFORMAT REPAIR")[0])
        world.owner_payloads.append(payload)
        world.owner_systems.append(system)
        step = world.owner_script.pop(0)
        body = step(payload) if callable(step) else step
        return SimpleNamespace(
            text=json.dumps(body),
            target=target,
            upstream_provider="test",
            latency_ms=5,
            usage=None,
            reported_cost_usd=None,
        )

    async def generate_replies(prompt, _persona, **_kwargs):
        world.writer_prompts.append([dict(message) for message in prompt])
        return list(world.writer_script.pop(0)) if world.writer_script else ["mm"]

    async def current_revision(prepared):
        return prepared.loaded.snapshot.state_revision

    async def deliver_reply(prepared, *, expected_revision):
        text = prepared.replies[0].replace("|", " ").strip()
        message_id = world._next_id("creator-msg")
        world.messages.append(
            Message(id=message_id, role="creator", content=text, sent_at=world._tick())
        )
        world.transcript.append(
            {
                "speaker": "creator",
                "text": text,
                "operation": prepared.execution.operation,
            }
        )
        return {"outcome": lo.OUTCOME_REPLIED, "message_ids": [message_id]}

    async def commit_presented_offer(prepared):
        world.pending_offer = prepared.loaded.next_offer
        world.last_offer_at = world.clock
        world.transcript.append(
            {"speaker": "system", "media": True, "event": f"offer card {world.pending_offer.set_id}"}
        )

    async def plan_session_for_fan(_creator, _fan, *, accepted_set_id, accepted_price_cents, **_k):
        return {
            "status": "ok",
            "session": {
                "status": "active",
                "plan": [
                    {
                        "media_ids": [f"media-{accepted_set_id}"],
                        "set_id": accepted_set_id,
                        "price_cents": accepted_price_cents,
                        "asset_type": world.catalog[accepted_set_id].asset_type,
                    }
                ],
                "current_index": 0,
            },
        }

    async def commit_locked_plan(prepared):
        return prepared.loaded.snapshot.state_revision

    async def send_locked_ppv(**kwargs):
        set_id = kwargs["set_id"]
        message_id = world._next_id("ppv-msg")
        world.sent_ppv.append(
            {
                "set_id": set_id,
                "price_cents": kwargs["price_cents"],
                "sent_at": world._tick().isoformat(),
                "platform_message_id": f"local-test:{message_id}",
                "purchased": False,
            }
        )
        world.pending_offer = None
        world.pending_payment = {
            "reference": f"pay-{set_id}",
            "set_id": set_id,
            "price_cents": kwargs["price_cents"],
        }
        world.messages.append(
            Message(
                id=message_id,
                role="creator",
                content=kwargs["message_content"],
                sent_at=world.clock,
            )
        )
        world.transcript.append(
            {"speaker": "creator", "text": kwargs["message_content"], "operation": "send_locked_paid_message"}
        )
        world.transcript.append({"speaker": "system", "media": True, "event": f"locked media {set_id}"})
        return {"message_id": message_id}

    async def freeze(fan_id, reason):
        world.frozen.append(reason)

    patches = {
        "get_conversation_history": lambda *_a, **_k: _value(list(world.messages)),
        "get_fan_by_id": lambda *_a, **_k: _value(fan()),
        "get_creator_persona": lambda *_a: _value(Persona()),
        "get_creator_legend": lambda *_a: _value({}),
        "get_fan_intelligence_context": lambda *_a: _value({}),
        "get_fan_lifecycle_context": lambda *_a: _value({}),
        "get_affordability_context": lambda *_a: _value(dict(world.affordability)),
        "get_price_learning_context": lambda *_a: _value({}),
        "get_sent_ppv": lambda *_a: _value([dict(row) for row in world.sent_ppv]),
        "get_fan_session": lambda *_a: _value(None),
        "get_fan_state": lambda *_a: _value(commercial_state()),
        "get_creator_policy": lambda *_a: _value(CreatorPolicy()),
        "open_threads_for": lambda *_a: _value([]),
        "recent_episodes_for": lambda *_a: _value([]),
        "_fan_pending_payment": lambda *_a: _value(
            dict(world.pending_payment) if world.pending_payment else None
        ),
        "current_generation": lambda *_a: _value(0),
        "get_creator_caps": lambda *_a: _value({"caps_enabled": False}),
        "get_next_offer_with_inventory": next_offer_with_inventory,
        "get_next_offer_with_candidates": next_offer_with_candidates,
        "resolve_ai_stack": lambda **_k: _value(stack),
        "complete": complete,
        "generate_replies": generate_replies,
        "_current_revision": current_revision,
        "deliver_reply": deliver_reply,
        "_commit_presented_offer": commit_presented_offer,
        "plan_session_for_fan": plan_session_for_fan,
        "_commit_locked_plan": commit_locked_plan,
        "send_locked_ppv": send_locked_ppv,
        "freeze_fan_for_review": freeze,
        "retrieve_examples": lambda **_k: [],
        "retrieval_enabled": lambda *_a: False,
    }
    for name, replacement in patches.items():
        monkeypatch.setattr(lo, name, replacement)
    monkeypatch.setattr(conversational_v2, "retrieve_examples", lambda **_k: [])
    monkeypatch.setattr(conversational_v2, "retrieval_enabled", lambda *_a: False)
    monkeypatch.setattr(conversational_session, "get_supabase", lambda: world.db)
    monkeypatch.setattr(conversational_core, "get_supabase", lambda: world.db)
    return world


def writer_payload(prompt: list[dict[str, str]]) -> dict[str, Any]:
    return json.loads(prompt[1]["content"])
