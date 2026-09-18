"""The live turn applies analyzer lifecycle proposals to durable memory.

These tests call the function used by Assisted and Full Auto, not the lifecycle
helpers in isolation.  They pin the gap from the merged-main review: extraction
already parsed resolutions, but the live path ignored them.
"""

from __future__ import annotations

import asyncio

from services import conversation_continuity as continuity
from services import suggestions
from tests.fake_supabase import FakeSupabase


def run(coro):
    return asyncio.run(coro)


def situation(**overrides):
    value = {
        "open_questions_raised": [],
        "commitments_made": [],
        "topics_deferred": [],
        "corrections_stated": [],
        "threads_resolved": [],
    }
    value.update(overrides)
    return value


def test_answered_question_leaves_the_next_packet_while_another_stays_open(
    monkeypatch,
):
    db = FakeSupabase(
        {"conversation_open_threads": [], "conversation_episodes": []}
    )
    monkeypatch.setattr(continuity, "get_supabase", lambda: db)

    run(
        suggestions._record_conversation_threads(
            situation(
                open_questions_raised=[
                    "whether she visits Chicago",
                    "whether she likes hiking",
                ]
            ),
            creator_id="creator-1",
            fan_id="fan-1",
            latest_message="two questions",
            turn_id="turn-raise",
        )
    )
    opened = run(continuity.open_threads_for("creator-1", "fan-1"))
    chicago = next(thread for thread in opened if "Chicago" in thread.summary)

    changed = run(
        suggestions._record_conversation_threads(
            situation(
                threads_resolved=[
                    {
                        "thread_id": chicago.id,
                        "evidence": "the creator answered the Chicago question",
                    }
                ]
            ),
            creator_id="creator-1",
            fan_id="fan-1",
            latest_message="thanks, that answers Chicago",
            turn_id="turn-answer",
        )
    )

    assert changed is True
    packet = continuity.summarize_threads(
        run(continuity.open_threads_for("creator-1", "fan-1"))
    )
    assert "Chicago" not in " ".join(packet)
    assert "hiking" in " ".join(packet)


def test_live_correction_supersedes_only_the_named_record(monkeypatch):
    db = FakeSupabase(
        {"conversation_open_threads": [], "conversation_episodes": []}
    )
    monkeypatch.setattr(continuity, "get_supabase", lambda: db)

    run(
        suggestions._record_conversation_threads(
            situation(corrections_stated=["he prefers outdoor shoots"]),
            creator_id="creator-1",
            fan_id="fan-1",
            latest_message="I prefer outdoor shoots",
            turn_id="turn-old",
        )
    )
    old = run(continuity.open_threads_for("creator-1", "fan-1"))[0]

    run(
        suggestions._record_conversation_threads(
            situation(
                corrections_stated=[
                    {
                        "summary": "he prefers indoor shoots, not outdoor",
                        "supersedes_thread_id": old.id,
                        "evidence": "he explicitly corrected the earlier preference",
                    }
                ]
            ),
            creator_id="creator-1",
            fan_id="fan-1",
            latest_message="Actually indoors, not outdoors",
            turn_id="turn-new",
        )
    )

    packet = " ".join(
        continuity.summarize_threads(
            run(continuity.open_threads_for("creator-1", "fan-1"))
        )
    )
    assert "indoor" in packet
    assert "prefers outdoor shoots" not in packet
    old_row = next(row for row in db.tables[continuity.THREADS_TABLE] if row["id"] == old.id)
    assert old_row["status"] == "superseded"
    assert old_row["superseded_by"]


def test_ambiguous_and_cross_tenant_references_change_nothing(monkeypatch):
    db = FakeSupabase(
        {"conversation_open_threads": [], "conversation_episodes": []}
    )
    monkeypatch.setattr(continuity, "get_supabase", lambda: db)
    run(
        suggestions._record_conversation_threads(
            situation(open_questions_raised=["whether she visits Chicago"]),
            creator_id="creator-1",
            fan_id="fan-1",
            latest_message="Do you visit Chicago?",
            turn_id="turn-raise",
        )
    )
    thread = run(continuity.open_threads_for("creator-1", "fan-1"))[0]

    # Free text has no identity and is never fuzzy-matched.
    run(
        suggestions._record_conversation_threads(
            situation(threads_resolved=["the Chicago one"]),
            creator_id="creator-1",
            fan_id="fan-1",
            latest_message="that answers it",
            turn_id="turn-ambiguous",
        )
    )
    # A real id from another tenant is still outside this conversation.
    run(
        suggestions._record_conversation_threads(
            situation(
                threads_resolved=[
                    {"thread_id": thread.id, "evidence": "it sounds answered"}
                ]
            ),
            creator_id="creator-2",
            fan_id="fan-1",
            latest_message="thanks",
            turn_id="turn-cross-tenant",
        )
    )

    assert [
        item.id
        for item in run(continuity.open_threads_for("creator-1", "fan-1"))
    ] == [thread.id]

