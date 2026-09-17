"""An operator can see what the conversation remembers, and fix it.

The continuity tables were written and read entirely by the machine. An
operator could see a frozen conversation and a transcript, and had no way to
see that the system was carrying an obligation, where it came from, or that it
was wrong.

That gap has a specific cost, and it is not cosmetic. These records are fed to
a model on every turn, so a wrong one — a question recorded from a message that
did not ask it, a correction extracted the wrong way round — keeps being fed to
it, and surfaces as the model behaving strangely for reasons nobody can trace.
Before this, the only fix available was to stop using the conversation.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from models.conversation_continuity import (
    ConversationEpisode,
    EpisodeEnding,
    EvidenceType,
    OpenThread,
    ThreadKind,
    ThreadParty,
    ThreadStatus,
)
from services import conversation_continuity as continuity
from services.conversation_memory_view import (
    correct_as_operator,
    episode_view,
    fact_view,
    load_memory,
    resolve_as_operator,
    thread_view,
)
from tests.fake_supabase import FakeSupabase

NOW = datetime(2026, 9, 17, 12, 0, tzinfo=timezone.utc)


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def db(monkeypatch):
    store = FakeSupabase(
        {"conversation_open_threads": [], "conversation_episodes": []}
    )
    monkeypatch.setattr(continuity, "get_supabase", lambda: store)
    return store


def _thread(**overrides) -> OpenThread:
    values = {
        "creator_id": "creator-1",
        "fan_id": "fan-1",
        "kind": ThreadKind.QUESTION,
        "raised_by": ThreadParty.FAN,
        "summary": "whether she ever gets to chicago",
        "evidence_type": EvidenceType.STATED,
        "source_turn_id": "turn-7",
        "source_message_fingerprint": "fp-7",
    }
    values.update(overrides)
    return OpenThread(**values)


def _stored(db, **overrides) -> OpenThread:
    run(continuity.record_open_thread(_thread(**overrides)))
    return run(continuity.open_threads_for("creator-1", "fan-1"))[0]


# ===========================================================================
# What an operator sees
# ===========================================================================


def test_the_view_says_where_a_record_came_from(db):
    """The half an operator actually judges on.

    "He asked about Chicago" read out of an ambiguous message and the same line
    stated outright are different claims, and only one is worth acting on.
    """
    view = thread_view(_stored(db))

    assert view["summary"] == "whether she ever gets to chicago"
    assert view["evidence_type"] == EvidenceType.STATED.value
    assert view["source_turn_id"] == "turn-7"
    assert view["source_message_fingerprint"] == "fp-7"
    assert view["confidence"] == 1.0


def test_the_view_renders_when_a_record_was_first_and_last_seen():
    """Separate from the stored-thread test on purpose.

    first_seen_at and last_seen_at come from the column defaults in
    db/conversation_continuity_v1.sql — `not null default now()` — so the real
    database populates them and FakeSupabase, which just keeps the dict it was
    handed, does not. Asserting them off a stored double would be asserting the
    double's behaviour rather than the view's, so this builds a thread that
    carries them and checks the view renders them.
    """
    seen = _thread().model_copy(update={"first_seen_at": NOW, "last_seen_at": NOW})

    view = thread_view(seen)

    assert view["first_seen_at"] == NOW.isoformat()
    assert view["last_seen_at"] == NOW.isoformat()
    # And a record with no timestamps says so rather than inventing one.
    assert thread_view(_thread())["first_seen_at"] is None


def test_the_view_says_outright_whether_it_was_read_rather_than_said(db):
    """The question is "should I trust this", and making an operator derive it
    from an enum is how they stop reading it."""
    said = thread_view(_stored(db, evidence_type=EvidenceType.STATED))
    assert said["was_read_rather_than_said"] is False

    db.tables["conversation_open_threads"].clear()
    read = thread_view(_stored(db, evidence_type=EvidenceType.INFERRED))
    assert read["was_read_rather_than_said"] is True


def test_the_view_says_whose_move_it_is(db):
    """A question he asked and one she asked are different obligations, and
    answering the wrong one is a named failure."""
    his = thread_view(_stored(db, raised_by=ThreadParty.FAN))
    assert his["waiting_on_us"] is True


def test_the_view_never_carries_the_message_itself(db):
    """A fingerprint, not the text — the same discipline as reply_provenance."""
    view = thread_view(_stored(db, evidence_text=""))

    assert "fp-7" == view["source_message_fingerprint"]
    assert "content" not in view
    assert "message" not in view


def test_loading_memory_returns_what_a_turn_would_read(db):
    """A separate query would eventually disagree with the one that matters."""
    _stored(db)
    run(
        continuity.record_episode(
            ConversationEpisode(
                creator_id="creator-1",
                fan_id="fan-1",
                summary="an evening about his sister's wedding",
                ended_with=EpisodeEnding.WENT_QUIET,
                first_message_at=NOW,
                last_message_at=NOW,
                message_count=6,
            )
        )
    )

    memory = run(load_memory("creator-1", "fan-1"))

    assert len(memory["open_threads"]) == 1
    assert len(memory["episodes"]) == 1
    assert memory["carrying_anything"] is True


def test_an_empty_conversation_says_it_is_carrying_nothing(db):
    """Stated rather than left to an empty list: "nothing is carried" and "we
    could not read it" lead an operator to different actions."""
    memory = run(load_memory("creator-1", "fan-1"))

    assert memory["open_threads"] == []
    assert memory["carrying_anything"] is False


def test_another_conversations_memory_is_not_returned(db):
    _stored(db)
    run(continuity.record_open_thread(_thread(fan_id="fan-2", summary="about boston")))

    memory = run(load_memory("creator-1", "fan-1"))

    assert [t["summary"] for t in memory["open_threads"]] == [
        "whether she ever gets to chicago"
    ]


def test_an_episode_is_always_shown_as_read_rather_than_said():
    view = episode_view(
        ConversationEpisode(
            creator_id="c",
            fan_id="f",
            summary="an evening about the wedding",
            ended_with=EpisodeEnding.SAID_GOODBYE,
            first_message_at=NOW,
            last_message_at=NOW,
            message_count=4,
        )
    )

    assert view["was_read_rather_than_said"] is True
    assert view["ended_with"] == "said_goodbye"


def test_a_fact_view_carries_its_source():
    view = fact_view(
        {
            "id": "fact-1",
            "fact_key": "preferred_name",
            "fact_value": "Dan",
            "status": "confirmed",
            "confidence": 0.9,
            "source_message_id": "msg-3",
            "updated_at": "2026-09-17T12:00:00+00:00",
        }
    )

    assert view["key"] == "preferred_name"
    assert view["source_message_id"] == "msg-3"


def test_a_fact_view_tolerates_a_shape_it_did_not_expect():
    assert fact_view(None) == {}
    assert fact_view("not a row") == {}


# ===========================================================================
# What an operator may do about it
# ===========================================================================


def test_an_operator_can_close_an_obligation(db):
    thread = _stored(db)

    assert run(resolve_as_operator(thread.id, note="answered on the phone")) is True
    assert run(continuity.open_threads_for("creator-1", "fan-1")) == []


def test_closing_one_twice_reports_that_it_did_not(db):
    thread = _stored(db)
    run(resolve_as_operator(thread.id))

    assert run(resolve_as_operator(thread.id)) is False


def test_a_bad_extraction_is_cancelled_rather_than_reported_as_handled(db):
    """The distinction that keeps the history honest.

    An operator correcting an invention is saying "this was never real".
    Recording it as fulfilled would teach anyone reading the history that the
    system dealt with something it in fact made up.
    """
    thread = _stored(db)

    run(resolve_as_operator(thread.id, cancelled=True, note="he never asked this"))

    stored = db.tables["conversation_open_threads"][0]
    assert stored["status"] == ThreadStatus.CANCELLED.value


def test_an_operator_correction_supersedes_rather_than_overwrites(db):
    """So an operator's correction behaves exactly like the customer's."""
    thread = _stored(db)

    corrected = run(
        correct_as_operator(thread, summary="whether she gets to boston", note="misread")
    )

    assert corrected is not None
    carried = run(continuity.open_threads_for("creator-1", "fan-1"))
    assert [t.summary for t in carried] == ["whether she gets to boston"]
    # The obsolete version is kept, which is the point of supersession.
    assert len(db.tables["conversation_open_threads"]) == 2


def test_a_corrected_record_says_a_human_stated_it(db):
    """A stronger claim than anything read out of a message, and a later reader
    deciding how much to rely on it needs to know which it was."""
    thread = _stored(db, evidence_type=EvidenceType.INFERRED, confidence=0.4)

    corrected = run(correct_as_operator(thread, summary="whether she gets to boston"))

    assert corrected.evidence_type is EvidenceType.OPERATOR
    assert corrected.confidence == 1.0
    assert thread_view(corrected)["was_read_rather_than_said"] is False


def test_a_corrected_record_keeps_its_original_provenance(db):
    """Which turn produced the mistake is exactly what somebody debugging it
    needs, and a correction must not erase it."""
    thread = _stored(db)

    corrected = run(correct_as_operator(thread, summary="whether she gets to boston"))

    assert corrected.source_turn_id == "turn-7"
    assert corrected.source_message_fingerprint == "fp-7"


def test_there_is_no_way_to_delete_a_record():
    """A deleted record leaves the next reader wondering whether it was ever
    there. Resolved and corrected both say what happened."""
    import services.conversation_memory_view as view_module

    exported = [name for name in dir(view_module) if not name.startswith("_")]

    assert not any("delete" in name.lower() for name in exported)
    assert not any("remove" in name.lower() for name in exported)
