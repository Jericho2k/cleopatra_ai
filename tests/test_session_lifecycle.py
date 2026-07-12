import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.session_lifecycle import (
    decrement_cooldown,
    has_pending_purchase,
    mark_step_purchased,
    mark_step_sent,
)


def session():
    return {
        "status": "active",
        "current_index": 0,
        "awaiting_purchase_index": None,
        "plan": [
            {"media_id": "m1", "media_ids": ["m1", "m2"], "set_id": "s1", "price": 20, "sent": False, "purchased": False},
            {"media_id": "m3", "media_ids": ["m3"], "set_id": "s2", "price": 40, "sent": False, "purchased": False},
        ],
    }


def test_send_does_not_advance_before_purchase():
    updated = mark_step_sent(session())
    assert updated["current_index"] == 0
    assert updated["awaiting_purchase_index"] == 0
    assert updated["plan"][0]["sent"] is True
    assert has_pending_purchase(updated) is True


def test_purchase_advances_and_starts_cooldown():
    sent = mark_step_sent(session())
    updated, completed = mark_step_purchased(sent, media_id="m1", amount_cents=2000, cooldown_messages=2)
    assert completed is False
    assert updated["current_index"] == 1
    assert updated["awaiting_purchase_index"] is None
    assert updated["post_ppv_cooldown"] is True
    assert updated["cooldown_messages_remaining"] == 2


def test_final_purchase_completes_session_and_is_idempotent():
    first = mark_step_sent(session())
    first, _ = mark_step_purchased(first, media_id="m1", amount_cents=2000, cooldown_messages=0)
    second = mark_step_sent(first)
    done, completed = mark_step_purchased(second, media_id="m3", amount_cents=4000)
    assert completed is True
    assert done["status"] == "completed"
    assert done["revenue_cents"] == 6000
    duplicate, duplicate_completed = mark_step_purchased(done, media_id="m3", amount_cents=4000)
    assert duplicate_completed is True
    assert duplicate["revenue_cents"] == 6000


def test_cooldown_decrements_on_fan_messages():
    sent = mark_step_sent(session())
    updated, _ = mark_step_purchased(sent, media_id="m1", cooldown_messages=2)
    updated = decrement_cooldown(updated)
    assert updated["cooldown_messages_remaining"] == 1
    updated = decrement_cooldown(updated)
    assert updated["post_ppv_cooldown"] is False
