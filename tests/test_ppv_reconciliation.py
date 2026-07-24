from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import httpx

from models.commercial import CreatorPolicy
from services import ppv_reconciliation as reconciliation
from services.ppv_reconciliation import PPVReconcileDisposition


NOW = datetime(2026, 7, 18, 12, 0, tzinfo=timezone.utc)


def pending(**overrides):
    value = {
        "reference": "ref-1",
        "media_id": "media-1",
        "price": 45,
        "sent_at": "2026-07-18T11:00:00+00:00",
        "expires_at": "2026-07-19T11:00:00+00:00",
        "verification_attempts": 0,
    }
    value.update(overrides)
    return value


def run(coro):
    return asyncio.run(coro)


def test_matching_purchase_filters_transaction_type_and_price():
    transactions = [
        {"type": 999, "totalGross": 4500},
        {"type": 2110, "totalGross": 8000},
        {"type": 2110, "totalGross": 4400},
    ]
    assert reconciliation._matching_purchase(transactions, 45) == 44
    assert reconciliation._matching_purchase(transactions, 20) is None


def test_abandoned_log_preserves_complete_ppv_bundle():
    entry = reconciliation._abandoned_log_entry(
        pending(media_ids=["media-1", "media-2"], source="operator"),
        now=NOW,
    )
    assert entry["media_id"] == "media-1"
    assert entry["media_ids"] == ["media-1", "media-2"]
    assert entry["chatter"] == "Operator"
    assert entry["payment_reference"] == "ref-1"


def test_unpurchased_ppv_stays_pending_until_real_expiry(monkeypatch):
    current = pending()
    persisted = []
    monkeypatch.setattr(
        reconciliation,
        "get_creator_policy",
        lambda _creator_id: async_value(CreatorPolicy(ppv_recheck_minutes=20)),
    )
    monkeypatch.setattr(
        reconciliation,
        "_load_platform_context",
        lambda _creator_id, _fan_id: async_value(
            ({"apifansly_account_id": "account"}, {"platform_fan_id": "fan", "pending_ppv_check": current})
        ),
    )
    monkeypatch.setattr(
        reconciliation,
        "_fetch_purchase_amount",
        lambda **_kwargs: async_value(None),
    )
    monkeypatch.setattr(
        reconciliation,
        "_persist_pending_check",
        lambda _fan_id, value: async_append(persisted, value.copy()),
    )

    result = run(reconciliation.reconcile_pending_ppv(
        creator_id="creator",
        fan_id="fan",
        expected_reference="ref-1",
        now=NOW,
    ))
    assert result.disposition == PPVReconcileDisposition.PENDING
    assert result.retry_at == datetime(2026, 7, 18, 12, 20, tzinfo=timezone.utc)
    assert persisted[0]["verification_attempts"] == 1


def test_expired_ppv_finalizes_once(monkeypatch):
    current = pending(expires_at="2026-07-18T11:30:00+00:00")
    finalized = []
    monkeypatch.setattr(
        reconciliation,
        "get_creator_policy",
        lambda _creator_id: async_value(CreatorPolicy()),
    )
    monkeypatch.setattr(
        reconciliation,
        "_load_platform_context",
        lambda _creator_id, _fan_id: async_value(
            ({"apifansly_account_id": "account"}, {"platform_fan_id": "fan", "pending_ppv_check": current})
        ),
    )
    monkeypatch.setattr(
        reconciliation,
        "_fetch_purchase_amount",
        lambda **_kwargs: async_value(None),
    )
    monkeypatch.setattr(
        reconciliation,
        "_finalize_abandonment",
        lambda **kwargs: async_append(finalized, kwargs) or async_value(True),
    )

    result = run(reconciliation.reconcile_pending_ppv(
        creator_id="creator",
        fan_id="fan",
        expected_reference="ref-1",
        now=NOW,
    ))
    assert result.disposition == PPVReconcileDisposition.ABANDONED
    assert len(finalized) == 1


def test_superseded_reconciliation_is_a_noop(monkeypatch):
    monkeypatch.setattr(
        reconciliation,
        "get_creator_policy",
        lambda _creator_id: async_value(CreatorPolicy()),
    )
    monkeypatch.setattr(
        reconciliation,
        "_load_platform_context",
        lambda _creator_id, _fan_id: async_value(
            ({}, {"pending_ppv_check": pending(reference="new-ref")})
        ),
    )
    result = run(reconciliation.reconcile_pending_ppv(
        creator_id="creator",
        fan_id="fan",
        expected_reference="old-ref",
        now=NOW,
    ))
    assert result.disposition == PPVReconcileDisposition.STALE


def test_local_test_ppv_never_calls_live_earnings_and_expires(monkeypatch):
    current = pending(expires_at="2026-07-18T11:30:00+00:00")
    finalized = []
    monkeypatch.setattr(
        reconciliation,
        "get_creator_policy",
        lambda _creator_id: async_value(CreatorPolicy()),
    )
    monkeypatch.setattr(
        reconciliation,
        "_load_platform_context",
        lambda _creator_id, _fan_id: async_value((
            {"apifansly_account_id": "account"},
            {"platform_fan_id": "test_123", "pending_ppv_check": current},
        )),
    )
    monkeypatch.setattr(
        reconciliation,
        "_fetch_purchase_amount",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("live API must not be called")),
    )
    monkeypatch.setattr(
        reconciliation,
        "_finalize_abandonment",
        lambda **kwargs: async_append(finalized, kwargs),
    )

    result = run(reconciliation.reconcile_pending_ppv(
        creator_id="creator",
        fan_id="fan",
        expected_reference="ref-1",
        now=NOW,
    ))
    assert result.disposition == PPVReconcileDisposition.ABANDONED
    assert len(finalized) == 1


def test_permanent_verification_error_freezes_instead_of_retrying_forever(monkeypatch):
    current = pending(expires_at="2026-07-19T11:30:00+00:00")
    persisted = []
    frozen = []
    request = httpx.Request("GET", "https://example.test/earnings")
    response = httpx.Response(402, request=request)
    error = httpx.HTTPStatusError("payment required", request=request, response=response)

    monkeypatch.setattr(
        reconciliation,
        "get_creator_policy",
        lambda _creator_id: async_value(CreatorPolicy()),
    )
    monkeypatch.setattr(
        reconciliation,
        "_load_platform_context",
        lambda _creator_id, _fan_id: async_value((
            {"apifansly_account_id": "account"},
            {"platform_fan_id": "live-fan", "pending_ppv_check": current},
        )),
    )
    monkeypatch.setattr(
        reconciliation,
        "_fetch_purchase_amount",
        lambda **_kwargs: async_raise(error),
    )
    monkeypatch.setattr(
        reconciliation,
        "_persist_pending_check",
        lambda _fan_id, value: async_append(persisted, value.copy()),
    )
    import db.queries as queries
    monkeypatch.setattr(
        queries,
        "freeze_fan_for_review",
        lambda fan_id, reason: async_append(frozen, (fan_id, reason)),
    )

    result = run(reconciliation.reconcile_pending_ppv(
        creator_id="creator",
        fan_id="fan",
        expected_reference="ref-1",
        now=NOW,
    ))
    assert result.disposition == PPVReconcileDisposition.REVIEW_REQUIRED
    assert persisted[0]["verification_failures"] == 1
    assert frozen == [("fan", "ppv_purchase_verification_unavailable")]


async def async_value(value):
    return value


async def async_append(target, value):
    target.append(value)
    return True


async def async_raise(error):
    raise error
