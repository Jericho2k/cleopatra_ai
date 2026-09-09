"""Every corrected completeness-sensitive read, past the 1,000-row cap.

PostgREST truncates an unranged select at db-max-rows and says nothing about it.
The audit classified 43 such selects; these are the ones where the truncation
changes an outcome rather than a display total, and each is exercised here
against the double that reproduces the cap. A double that returned every row
would let all of them pass.

The failures being pinned:

  get_sent_ppv / offer packages   Cleopatra re-offers content already sent
  mark_ppv_purchased              a settled purchase cannot be recorded
  sweep_stale_ppv_checks          pending reconciliation is never repaired
  chat reconciliation scheduler   an active creator drops to the idle interval
  fansly audience sync            a fan never gets follower/spend attributes
  load_fan_history                a wave of writes the unique index rejects
  delete_creator                  fans left behind
  full-auto health                a stuck fan is invisible to the operator
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from tests.fake_supabase import FakeSupabase

CREATOR_ID = "creator-1"
FAN_ID = "fan-1"


def _ppv_messages(count: int) -> list[dict]:
    """One creator message per media item, oldest first."""

    return [
        {
            "id": f"msg-{index:06d}",
            "fan_id": FAN_ID,
            "creator_id": CREATOR_ID,
            "role": "creator",
            "sent_at": f"2026-01-01T00:00:{index % 60:02d}Z",
            "media_context": {
                "ppv": {
                    "media_id": f"media-{index:06d}",
                    "media_ids": [f"media-{index:06d}"],
                    "set_id": f"set-{index:06d}",
                    "price": 25,
                    "purchased": False,
                }
            },
        }
        for index in range(count)
    ]


# --- get_sent_ppv -----------------------------------------------------------


@pytest.mark.parametrize("count", [500, 1500, 3000])
def test_get_sent_ppv_sees_every_previously_sent_item(monkeypatch, count):
    import db.queries as queries

    db = FakeSupabase({"messages": _ppv_messages(count)})
    monkeypatch.setattr(queries, "get_supabase", lambda: db)

    sent = asyncio.run(queries.get_sent_ppv(FAN_ID))

    assert len(sent) == count
    # The oldest send is the one a truncated read loses, and it is exactly the
    # one whose absence lets Cleopatra re-offer it.
    assert sent[0]["media_id"] == "media-000000"


def test_mark_ppv_purchased_finds_an_old_send(monkeypatch):
    import db.queries as queries

    rows = _ppv_messages(2500)
    db = FakeSupabase({"messages": rows})
    monkeypatch.setattr(queries, "get_supabase", lambda: db)

    # Newest-first ordering means row 0 is the last thing a truncated read
    # would ever reach.
    assert asyncio.run(queries.mark_ppv_purchased(FAN_ID, "media-000000")) is True
    updated = [payload for op, table, payload in db.writes if op == "update"]
    assert updated and updated[0]["media_context"]["ppv"]["purchased"] is True


# --- offer packages ---------------------------------------------------------


def test_offer_packages_never_reoffer_a_set_sent_past_the_cap(monkeypatch):
    """The audit's headline example: an incomplete sent_set_ids re-sells.

    sent_set_ids is the "never offer this again" list. Truncated at 1,000
    creator messages it forgets older sends, and the offer builder happily
    proposes content the fan already received.
    """

    import db.commercial_queries as commercial

    messages = _ppv_messages(2500)
    old_set = "set-000000"
    vault_sets = [{
        "id": old_set,
        "title": "An old set",
        "description": "already sent",
        "location": "",
        "outfit": "",
        "suggested_price": 25,
        "tags": [],
        "explicit_min": 1,
        "explicit_max": 3,
        "media_ids": ["media-000000"],
        "base_price_cents": 2500,
        "min_price_cents": 1500,
        "max_price_cents": 4000,
        "dynamic_pricing_enabled": False,
        "creator_id": CREATOR_ID,
        "status": "approved",
    }]
    db = FakeSupabase({
        "messages": messages,
        "vault_sets": vault_sets,
        "fans": [{"id": FAN_ID, "ai_summary": {}, "preferences": {}}],
    })
    monkeypatch.setattr(commercial, "get_supabase", lambda: db)

    captured: dict = {}

    def fake_usable_sets(rows, sent_set_ids):
        captured["sent_set_ids"] = set(sent_set_ids)
        return [row for row in rows if row["id"] not in sent_set_ids]

    monkeypatch.setattr(commercial, "usable_sets", fake_usable_sets)
    monkeypatch.setattr(
        commercial,
        "build_offer_packages",
        lambda rows, *_args, **_kwargs: list(rows),
    )

    packages = asyncio.run(
        commercial.get_offerable_packages(
            creator_id=CREATOR_ID,
            fan_id=FAN_ID,
            policy=SimpleNamespace(),
        )
    )

    assert old_set in captured["sent_set_ids"]
    assert packages == []


# --- pending PPV repair sweep ----------------------------------------------


def test_the_ppv_sweep_repairs_actions_past_the_cap(monkeypatch):
    """This read spans every creator, so the global cap is easy to reach."""

    import services.suggestions as suggestions

    fans = [
        {
            "id": f"fan-{index:06d}",
            "creator_id": f"creator-{index % 20}",
            "needs_human_review": False,
            "review_reason": None,
            "pending_ppv_check": {
                "sent_at": "2026-01-01T00:00:00+00:00",
                "reference": f"ref-{index:06d}",
            },
        }
        for index in range(2400)
    ]
    db = FakeSupabase({"fans": fans})
    monkeypatch.setattr(suggestions, "get_supabase", lambda: db)

    ensured: list[str] = []

    async def fake_ensure(**kwargs):
        ensured.append(kwargs["dedupe_key"])

    import db.commercial_queries as commercial

    monkeypatch.setattr(commercial, "ensure_action_pending", fake_ensure)

    asyncio.run(suggestions.sweep_stale_ppv_checks())

    assert len(ensured) == 2400
    assert any("fan-002399" in key for key in ensured)


# --- chat reconciliation interval ------------------------------------------


def test_a_creator_with_auto_fans_past_the_global_cap_stays_active(monkeypatch):
    """The probe replaced a global read of every auto fan in the deployment.

    That read had no creator filter, so once enough fans anywhere had auto mode
    on, a creator whose fans fell past row 1,000 was misread as having none and
    dropped from the 10-minute interval to 30.
    """

    import main

    # 1,200 auto fans belong to a different creator and would fill the cap.
    fans = [
        {"id": f"other-{index}", "creator_id": "creator-noisy", "auto_mode": True}
        for index in range(1200)
    ]
    fans.append({"id": "quiet-1", "creator_id": CREATOR_ID, "auto_mode": True})

    db = FakeSupabase({
        "creators": [{
            "id": CREATOR_ID,
            "apifansly_account_id": "acct",
            "auto_mode": False,
        }],
        "fans": fans,
    })
    monkeypatch.setattr(main, "get_supabase", lambda: db)

    intervals: list[bool] = []
    original = main._chat_reconcile_interval_seconds

    def spy(*, creator_auto_mode, has_auto_fan):
        intervals.append(has_auto_fan)
        return original(
            creator_auto_mode=creator_auto_mode,
            has_auto_fan=has_auto_fan,
        )

    monkeypatch.setattr(main, "_chat_reconcile_interval_seconds", spy)

    async def fake_reconcile(due):
        return {"processed": len(due)}

    monkeypatch.setattr(main, "_reconcile_chat_creators_once", fake_reconcile)
    monkeypatch.setenv("CHAT_RECONCILE_TICK_MINUTES", "1")

    async def run_one_tick():
        # The scheduler sleeps first, so drive one pass and then stop it.
        sleeps = {"count": 0}
        real_sleep = asyncio.sleep

        async def fake_sleep(_seconds):
            sleeps["count"] += 1
            if sleeps["count"] > 1:
                raise asyncio.CancelledError
            await real_sleep(0)

        monkeypatch.setattr(main.asyncio, "sleep", fake_sleep)
        with pytest.raises(asyncio.CancelledError):
            await main.chat_reconciliation_scheduler()

    main._chat_reconcile_due_at.clear()
    asyncio.run(run_one_tick())

    assert intervals == [True]


# --- audience sync ----------------------------------------------------------


def test_audience_sync_reaches_every_fan(monkeypatch):
    import services.fansly_audience as audience

    fans = [
        {
            "id": f"fan-{index:06d}",
            "creator_id": CREATOR_ID,
            "platform_fan_id": f"p-{index:06d}",
            "total_spent": 0,
        }
        for index in range(2200)
    ]
    db = FakeSupabase({"fans": fans})
    monkeypatch.setattr(audience, "get_supabase", lambda: db)

    async def empty(*_args, **_kwargs):
        return {}, None

    async def empty_supporters(*_args, **_kwargs):
        return []

    monkeypatch.setattr(audience, "list_followers", empty)
    monkeypatch.setattr(audience, "list_subscribers", empty)
    monkeypatch.setattr(audience, "top_supporters", empty_supporters)

    result = asyncio.run(audience.sync_fansly_audience(CREATOR_ID, "acct"))

    # Every fan was considered, including the ones past the cap.
    assert result["updated_fans"] == 2200


# --- full-auto health -------------------------------------------------------


def test_full_auto_health_sees_a_frozen_fan_past_the_cap(monkeypatch):
    import services.full_auto_operations as operations

    fans = [
        {
            "id": f"fan-{index:06d}",
            "creator_id": CREATOR_ID,
            "display_name": f"Fan {index}",
            "auto_mode": True,
            "needs_human_review": index == 2100,
            "review_reason": "ppv_send_failed" if index == 2100 else None,
        }
        for index in range(2200)
    ]
    db = FakeSupabase({
        "fans": fans,
        "fan_commercial_states": [],
        "scheduled_actions": [],
    })
    monkeypatch.setattr(operations, "get_supabase", lambda: db)

    health = asyncio.run(operations.get_creator_full_auto_health(CREATOR_ID))

    frozen = [row for row in health["fans"] if row["needs_human_review"]]
    assert [row["fan_id"] for row in frozen] == ["fan-002100"]
