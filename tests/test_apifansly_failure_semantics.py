"""Part 15 — a passing 503 must not be handled like a revoked API key.

Every API Fansly failure used to arrive as a bare httpx error, so the call sites
that protect exactly-once delivery had no way to tell "the platform told us it
did not process this" from "the platform may have processed it and we lost the
answer" from "this key will fail identically forever". They therefore treated
all three the same, and a rate limit that cleared on its own took a fan out of
automation until an operator noticed.

The distinction is only ever allowed to *narrow* the freeze. An ambiguous
outcome still freezes, because the send may be live in the fan's chat.
"""

from __future__ import annotations

import asyncio
import uuid

import httpx
import pytest

import services.ppv_delivery as delivery
import services.ppv_reconciliation as reconciliation
from services.apifansly import ApiFanslyTransientError
from services.ppv_delivery import PPVDeliveryError


def run(coro):
    return asyncio.run(coro)


async def async_value(value):
    return value


async def async_noop(*_args, **_kwargs):
    return None


class _Row:
    def __init__(self, data):
        self.data = data


class _FakeTable:
    def __init__(self, rows):
        self._rows = rows

    def select(self, *_args, **_kwargs):
        return self

    def eq(self, *_args, **_kwargs):
        return self

    def single(self):
        return self

    def execute(self):
        return _Row(self._rows)


class _FakeDb:
    def __init__(self, fan, creator):
        self._fan = fan
        self._creator = creator

    def table(self, name):
        if name == "fans":
            return _FakeTable(self._fan)
        if name == "creators":
            return _FakeTable(self._creator)
        raise AssertionError(f"unexpected table {name}")


@pytest.fixture
def delivery_env(monkeypatch):
    """Everything send_locked_ppv needs, up to the platform send itself."""

    transitions: list[tuple] = []
    frozen: list[tuple[str, str]] = []

    monkeypatch.setattr(
        delivery,
        "get_supabase",
        lambda: _FakeDb(
            {
                "creator_id": "creator-1",
                "fansly_group_id": "group-1",
                "platform_fan_id": "p-1",
                "pending_ppv_check": None,
                "needs_human_review": False,
                "review_reason": None,
            },
            {"apifansly_account_id": "account-1"},
        ),
    )
    monkeypatch.setattr(delivery, "get_fan_session", lambda _fan_id: async_value(None))
    monkeypatch.setattr(delivery, "claim_delivery", async_noop)
    monkeypatch.setattr(delivery, "uuid", uuid)

    async def transition(reference, status, **kwargs):
        transitions.append((reference, status, kwargs.get("error")))

    async def freeze(fan_id, reason):
        frozen.append((fan_id, reason))

    monkeypatch.setattr(delivery, "transition_delivery", transition)
    monkeypatch.setattr(delivery, "freeze_fan_for_review", freeze)

    return {"transitions": transitions, "frozen": frozen}


def _send(**overrides):
    kwargs = {
        "creator_id": "creator-1",
        "fan_id": "fan-1",
        "media_ids": ["media-1"],
        "price_cents": 2500,
        "message_content": "just for you",
        "source": "auto",
        "was_ai_suggested": True,
    }
    kwargs.update(overrides)
    return delivery.send_locked_ppv(**kwargs)


def test_a_transient_platform_refusal_does_not_freeze_the_fan(
    monkeypatch, delivery_env
):
    """503 means the platform confirmed it did nothing. Nothing to recover."""

    async def refuse(*_args, **_kwargs):
        raise ApiFanslyTransientError(
            "temporarily unavailable", status_code=503
        )

    monkeypatch.setattr(delivery, "send_apifansly_message", refuse)

    with pytest.raises(PPVDeliveryError, match="temporarily unavailable"):
        run(_send())

    # The claim is released so the scheduled action can retry on its own
    # backoff...
    assert [status for _ref, status, _err in delivery_env["transitions"]] == ["failed"]
    # ...and the conversation stays in automation.
    assert delivery_env["frozen"] == []


def test_an_ambiguous_send_timeout_still_freezes_the_fan(monkeypatch, delivery_env):
    """The platform may have accepted it. That has to reach a human."""

    async def time_out(*_args, **_kwargs):
        raise httpx.ReadTimeout("no answer")

    monkeypatch.setattr(delivery, "send_apifansly_message", time_out)

    with pytest.raises(PPVDeliveryError):
        run(_send())

    assert [status for _ref, status, _err in delivery_env["transitions"]] == ["failed"]
    assert delivery_env["frozen"] == [("fan-1", "ppv_send_failed")]


def test_a_permanent_rejection_still_freezes_the_fan(monkeypatch, delivery_env):
    async def reject(*_args, **_kwargs):
        request = httpx.Request("POST", "https://example.test/send")
        raise httpx.HTTPStatusError(
            "bad request",
            request=request,
            response=httpx.Response(400, request=request),
        )

    monkeypatch.setattr(delivery, "send_apifansly_message", reject)

    with pytest.raises(PPVDeliveryError):
        run(_send())

    assert delivery_env["frozen"] == [("fan-1", "ppv_send_failed")]


# --- purchase verification --------------------------------------------------


def test_transient_verification_failure_keeps_its_status_code(monkeypatch):
    """The reconciler's transient/permanent verdict must survive the new class.

    It reads the status code to decide whether to keep rechecking. Before, that
    came off an HTTPStatusError; a transient API Fansly refusal now arrives as
    ApiFanslyTransientError and has to be read the same way, or the operator
    loses the code from the recorded reason.
    """

    persisted: list[dict] = []
    frozen: list[tuple[str, str]] = []

    async def persist(_fan_id, value):
        persisted.append(dict(value))

    async def freeze(fan_id, reason):
        frozen.append((fan_id, reason))

    import db.queries as queries

    monkeypatch.setattr(reconciliation, "_persist_pending_check", persist)
    monkeypatch.setattr(queries, "freeze_fan_for_review", freeze)

    from datetime import datetime, timedelta, timezone

    now = datetime(2026, 7, 19, 12, 0, tzinfo=timezone.utc)
    result = run(
        reconciliation._verification_unavailable(
            fan_id="fan-1",
            pending={"reference": "ref-1"},
            now=now,
            expires_at=now + timedelta(hours=2),
            recheck_minutes=5,
            error=ApiFanslyTransientError("busy", status_code=503),
        )
    )

    assert persisted[0]["last_verification_error"] == "purchase verification HTTP 503"
    # 5xx is not permanent, and this is the first failure, so nothing freezes.
    assert frozen == []
    assert result.disposition.value == "PENDING"
