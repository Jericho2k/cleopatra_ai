"""REL-006: the Fansly webhook acknowledges only what it durably accepted.

Acknowledgement must mean "this event survives a restart", which is why the
tests assert on what was written before the 2xx rather than on how fast it
returned. Speed is measured too, but as a consequence of the shape, not as the
guarantee.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import statistics
import time
from types import SimpleNamespace

import pytest

import main
from db.queries import MessageWriteResult

SECRET = "test-webhook-secret"


def signed_request(payload: dict):
    body = json.dumps(payload).encode()
    signature = hmac.new(SECRET.encode(), body, hashlib.sha256).hexdigest()

    class Request:
        headers: dict = {"signature": signature}  # noqa: RUF012

        async def body(self):
            return body

    return Request()


def message_event(message_id="msg-1", content="hey there"):
    return {
        "event": "messages.received",
        "accountId": "api-account-1",
        "data": {
            "id": message_id,
            "senderId": "fan-platform-1",
            "groupId": "group-1",
            "content": content,
            "interactions": [{"userId": "creator-platform-1"}],
        },
    }


class Recorder:
    """Captures everything the acceptance path writes.

    Mirrors the REL-002 upsert semantics of the real ``save_message_result``:
    a first write for a given platform-message key inserts, and every
    subsequent write for the same key resolves to the existing row with
    ``inserted=False`` rather than creating a second one.
    """

    def __init__(self):
        self.messages: dict[str, dict] = {}
        self.actions: dict[str, dict] = {}
        self.pipeline_runs: list[tuple] = []
        self.enrich_calls = 0
        self.fail_persistence = False

    async def save_message_result(
        self, fan_id, creator_id, role, content, **kwargs
    ) -> MessageWriteResult:
        if self.fail_persistence:
            raise RuntimeError("supabase unavailable")
        fansly_message_id = kwargs.get("fansly_message_id")
        if fansly_message_id is None:
            row = {
                "id": f"row-{len(self.messages)}",
                "fan_id": fan_id,
                "content": content,
            }
            self.messages[f"local:{len(self.messages)}"] = row
            return MessageWriteResult(message_id=row["id"], inserted=True)
        key = (creator_id, fansly_message_id)
        if key in self.messages:
            return MessageWriteResult(
                message_id=self.messages[key]["id"], inserted=False
            )
        row = {"id": f"row-{len(self.messages)}", "fan_id": fan_id, "content": content}
        self.messages[key] = row
        return MessageWriteResult(message_id=row["id"], inserted=True)

    async def schedule_action(self, **kwargs):
        key = kwargs["dedupe_key"]
        if kwargs.get("replace_existing", True) is False and key in self.actions:
            return
        self.actions[key] = kwargs

    async def pipeline(self, fan_id, creator_id, content, auto_mode, message_id):
        self.pipeline_runs.append((fan_id, creator_id, content, auto_mode, message_id))


@pytest.fixture
def wired(monkeypatch):
    monkeypatch.setenv("APIFANSLY_WEBHOOK_SECRET", SECRET)
    recorder = Recorder()

    class DB:
        def table(self, name):
            self._name = name
            return self

        def select(self, *_a, **_k):
            return self

        def eq(self, *_a):
            return self

        def limit(self, *_a):
            return self

        def update(self, *_a):
            return self

        def execute(self):
            if self._name == "creators":
                return SimpleNamespace(data=[{
                    "id": "creator-1",
                    "auto_mode": True,
                    "auto_mode_new_fans": True,
                    "fansly_account_id": "creator-platform-1",
                }])
            return SimpleNamespace(data=[])

    monkeypatch.setattr(main, "get_supabase", lambda: DB())
    monkeypatch.setattr(
        main, "get_fan",
        lambda *_a: _value(SimpleNamespace(id="fan-1", fansly_group_id="group-1")),
    )
    monkeypatch.setattr(main, "save_message_result", recorder.save_message_result)
    monkeypatch.setattr(main, "schedule_action", recorder.schedule_action)
    monkeypatch.setattr(main, "process_incoming_fan_message", recorder.pipeline)
    monkeypatch.setattr(main, "notify_scheduled_worker", lambda: None)
    return recorder


async def _value(v):
    return v


def test_webhook_acks_only_after_persisting_and_enqueuing(wired):
    response = asyncio.run(main.fansly_webhook(signed_request(message_event())))

    assert response == {"status": "accepted"}
    # The message is durable.
    assert ("creator-1", "msg-1") in wired.messages
    # And exactly one processing obligation exists for it.
    assert list(wired.actions) == ["inbound-message:fan-1:msg-1"]
    action = wired.actions["inbound-message:fan-1:msg-1"]
    assert action["action_type"] == "PROCESS_INBOUND_MESSAGE"
    assert action["replace_existing"] is False


def test_no_model_work_happens_inside_the_request(wired):
    asyncio.run(main.fansly_webhook(signed_request(message_event())))
    # The analyzer/writer pipeline is exactly what was moved out of the request.
    assert wired.pipeline_runs == []


def test_failed_persistence_returns_an_error_so_the_platform_retries(wired):
    wired.fail_persistence = True

    with pytest.raises(main.HTTPException) as excinfo:
        asyncio.run(main.fansly_webhook(signed_request(message_event())))

    assert excinfo.value.status_code == 503
    # Nothing was acknowledged, and no obligation was invented.
    assert wired.actions == {}


def test_redelivery_produces_one_message_and_one_obligation(wired):
    request = message_event()
    statuses = [
        asyncio.run(main.fansly_webhook(signed_request(request)))["status"]
        for _ in range(4)
    ]

    # First delivery inserts; every redelivery resolves to the same row and
    # reports it rather than re-running the pipeline.
    assert statuses == ["accepted", "duplicate", "duplicate", "duplicate"]
    assert len(wired.messages) == 1
    assert len(wired.actions) == 1


def test_redelivery_after_processing_does_not_reprocess(wired):
    asyncio.run(main.fansly_webhook(signed_request(message_event())))
    key = "inbound-message:fan-1:msg-1"
    # Simulate the worker having finished the obligation.
    wired.actions[key] = {**wired.actions[key], "status": "COMPLETED"}

    asyncio.run(main.fansly_webhook(signed_request(message_event())))

    # replace_existing=False means the completed action is untouched, so the
    # message is not answered a second time.
    assert wired.actions[key]["status"] == "COMPLETED"


def test_dedupe_key_is_stable_without_a_platform_message_id():
    first = main.inbound_message_dedupe_key("fan-1", "", "hello", "group-1")
    second = main.inbound_message_dedupe_key("fan-1", "", "hello", "group-1")
    other = main.inbound_message_dedupe_key("fan-1", "", "different", "group-1")

    assert first == second
    assert first != other
    assert first.startswith("inbound-message:fan-1:")


def test_durable_handler_runs_the_pipeline_that_left_the_request(monkeypatch, wired):
    action = {
        "id": "action-1",
        "fan_id": "fan-1",
        "creator_id": "creator-1",
        "payload": {
            "platform_message_id": "msg-1",
            "message_row_id": "row-0",
            "message_content": "hey there",
            "group_id": "group-1",
            "api_account_id": "api-account-1",
            "creator_platform_id": "creator-platform-1",
            "auto_mode": True,
            "attachments": [],
        },
    }
    result = asyncio.run(main.run_durable_inbound_message(action))

    assert wired.pipeline_runs == [
        ("fan-1", "creator-1", "hey there", True, "msg-1")
    ]
    assert result.sent_message is False


def test_media_enrichment_moved_out_of_the_request(monkeypatch, wired):
    """The live API Fansly call now happens in the worker, not the webhook."""
    calls = []

    async def fake_list(account_id, group_id, limit=10):
        calls.append((account_id, group_id))
        return ([{"id": "msg-1"}], {}, None)

    monkeypatch.setattr(main, "apifansly_list_chat_messages", fake_list)
    monkeypatch.setattr(
        main, "_apifansly_message_row",
        lambda *a, **k: {"media_context": {"attachments": [{"url": "signed"}]}},
    )
    monkeypatch.setattr(main, "_apifansly_account_media_lookup", lambda _m: {})
    patched = []
    monkeypatch.setattr(
        main, "update_message_media_context",
        lambda mid, ctx: _record(patched, (mid, ctx)),
    )

    event = message_event()
    event["data"]["attachments"] = [{"contentId": "media-1", "contentType": 1}]
    asyncio.run(main.fansly_webhook(signed_request(event)))
    assert calls == [], "no live platform call may happen inside the webhook"

    action = {
        "id": "a1",
        "fan_id": "fan-1",
        "creator_id": "creator-1",
        "payload": {
            "platform_message_id": "msg-1",
            "message_row_id": "row-0",
            "message_content": "hey there",
            "group_id": "group-1",
            "api_account_id": "api-account-1",
            "creator_platform_id": "creator-platform-1",
            "auto_mode": True,
            "attachments": [{"contentId": "media-1"}],
        },
    }
    asyncio.run(main.run_durable_inbound_message(action))
    assert calls == [("api-account-1", "group-1")]
    assert patched == [("row-0", {"attachments": [{"url": "signed"}]})]


async def _record(target, value):
    target.append(value)


def test_acknowledgement_latency_is_dominated_by_signature_and_writes(monkeypatch, wired):
    """Synthetic, mocked dependencies. Not a production SLA claim.

    Every DB write is given a deliberate 5 ms cost and the model layer is not
    reachable from this path at all, so the numbers below describe the SHAPE of
    the request: bounded writes, no inference.
    """
    original_save = wired.save_message_result
    original_schedule = wired.schedule_action

    async def slow_save(*a, **k):
        await asyncio.sleep(0.005)
        return await original_save(*a, **k)

    async def slow_schedule(**k):
        await asyncio.sleep(0.005)
        return await original_schedule(**k)

    monkeypatch.setattr(main, "save_message_result", slow_save)
    monkeypatch.setattr(main, "schedule_action", slow_schedule)

    samples = []

    async def drive():
        for i in range(40):
            request = signed_request(message_event(message_id=f"msg-{i}"))
            started = time.perf_counter()
            await main.fansly_webhook(request)
            samples.append((time.perf_counter() - started) * 1000)

    asyncio.run(drive())
    samples.sort()
    p50 = statistics.median(samples)
    p95 = samples[int(len(samples) * 0.95) - 1]
    print(f"[WEBHOOK ACK BENCH] p50={p50:.1f}ms p95={p95:.1f}ms n={len(samples)}")

    # Two 5 ms writes plus signature work. A pipeline still inside the request
    # would be an order of magnitude above this.
    assert p50 < 60
    assert p95 < 120
