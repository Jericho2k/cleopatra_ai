"""Sprint 2 — what a conversation is still carrying, across the window.

``docs/autonomy_architecture_review.md`` §3E and §4. The existing memory is
real and the review says so; what it does not hold is the state of the
interaction — which question is unanswered, which promise unkept, whether a
misunderstanding was repaired, whether an invitation to continue is current.

Each test here is one row of the review's §5 trajectory table, reduced to the
mechanism it depends on:

* *A question deferred across 30+ turns* needs an obligation that outlives the
  recent-message window.
* *Preference correction, old preference appears in retrieved history* needs a
  correction that supersedes rather than sitting beside what it corrects.
* *Multiple open threads, customer returns to the earlier of two subjects* needs
  both kept, and the older ranked ahead of the newer.
* *Explicit decline or goodbye* needs a thread that something he said can cancel.
* *Cross-account workload* needs retrieval scoped by creator AND customer.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from models.conversation_continuity import (
    ConversationEpisode,
    EpisodeEnding,
    EvidenceType,
    OpenThread,
    ResolvedBy,
    ThreadKind,
    ThreadParty,
    ThreadStatus,
)
from services import conversation_continuity as continuity
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
        "summary": "whether you ever visit Chicago",
        "resolution_condition": "you answer it",
    }
    values.update(overrides)
    return OpenThread(**values)


# --- an obligation outlives the window it was raised in ---------------------


def test_a_question_is_kept_until_something_closes_it(db):
    kept = run(continuity.record_open_thread(_thread()))

    assert kept is not None
    assert kept.status == ThreadStatus.OPEN
    assert run(continuity.open_threads_for("creator-1", "fan-1"))[0].summary == (
        "whether you ever visit Chicago"
    )


def test_the_same_question_mentioned_four_times_is_one_obligation(db):
    for _ in range(4):
        run(continuity.record_open_thread(_thread()))

    threads = run(continuity.open_threads_for("creator-1", "fan-1"))
    assert len(threads) == 1, (
        "four mentions of one unanswered question is one thing to answer"
    )


def test_rewording_the_same_question_still_collapses_to_one(db):
    run(continuity.record_open_thread(_thread(summary="whether you ever visit Chicago")))
    run(continuity.record_open_thread(_thread(summary="Whether you ever visit Chicago!")))

    assert len(run(continuity.open_threads_for("creator-1", "fan-1"))) == 1


def test_a_question_the_creator_asked_is_a_different_obligation(db):
    """Answering the wrong one is the "wrong referent" failure in §1."""
    run(continuity.record_open_thread(_thread(raised_by=ThreadParty.FAN)))
    run(continuity.record_open_thread(_thread(raised_by=ThreadParty.CREATOR)))

    threads = run(continuity.open_threads_for("creator-1", "fan-1"))
    assert len(threads) == 2
    waiting_on_us = [t for t in threads if t.is_obligation_on_us]
    assert len(waiting_on_us) == 1
    assert waiting_on_us[0].raised_by == ThreadParty.FAN


def test_a_fulfilled_obligation_stops_being_carried(db):
    kept = run(continuity.record_open_thread(_thread()))

    assert run(
        continuity.resolve_thread(
            kept.id,
            status=ThreadStatus.FULFILLED,
            resolved_by=ResolvedBy.CREATOR_REPLY,
            note="answered it",
        )
    ) is True
    assert run(continuity.open_threads_for("creator-1", "fan-1")) == []


def test_closing_an_already_closed_thread_reports_that_it_did_not(db):
    """Two workers noticing the same answer must not both claim to have closed it."""
    kept = run(continuity.record_open_thread(_thread()))
    run(
        continuity.resolve_thread(
            kept.id, status=ThreadStatus.FULFILLED, resolved_by=ResolvedBy.CREATOR_REPLY
        )
    )

    assert run(
        continuity.resolve_thread(
            kept.id, status=ThreadStatus.CANCELLED, resolved_by=ResolvedBy.FAN_MESSAGE
        )
    ) is False


def test_mentioning_a_resolved_obligation_does_not_reopen_it(db):
    """Otherwise a kept promise comes back as an outstanding one and is kept twice."""
    kept = run(continuity.record_open_thread(_thread()))
    run(
        continuity.resolve_thread(
            kept.id, status=ThreadStatus.FULFILLED, resolved_by=ResolvedBy.CREATOR_REPLY
        )
    )

    run(continuity.record_open_thread(_thread()))

    assert run(continuity.open_threads_for("creator-1", "fan-1")) == []


def test_a_thread_cannot_be_moved_back_to_open(db):
    kept = run(continuity.record_open_thread(_thread()))
    with pytest.raises(ValueError, match="back to open"):
        run(
            continuity.resolve_thread(
                kept.id,
                status=ThreadStatus.OPEN,
                resolved_by=ResolvedBy.CREATOR_REPLY,
            )
        )


# --- a goodbye cancels a queued obligation ---------------------------------


def test_something_he_said_can_cancel_an_obligation(db):
    """§1: a previously queued follow-up that becomes due after he says goodbye."""
    promise = run(
        continuity.record_open_thread(
            _thread(
                kind=ThreadKind.PROMISE,
                raised_by=ThreadParty.CREATOR,
                summary="send him the beach set tomorrow",
            )
        )
    )

    assert run(
        continuity.resolve_thread(
            promise.id,
            status=ThreadStatus.CANCELLED,
            resolved_by=ResolvedBy.FAN_MESSAGE,
            note="he asked for no more messages",
        )
    ) is True
    assert run(continuity.open_threads_for("creator-1", "fan-1")) == []


# --- a correction supersedes rather than coexisting -------------------------


def test_a_correction_supersedes_the_thing_it_corrects(db):
    """§4: not two simultaneously authoritative facts."""
    original = run(
        continuity.record_open_thread(
            _thread(
                kind=ThreadKind.DEFERRED_TOPIC,
                summary="he wants the outdoor set next",
            )
        )
    )

    replacement = run(
        continuity.supersede_thread(
            original.id,
            _thread(
                kind=ThreadKind.CORRECTION,
                summary="he actually wants the hotel set, not the outdoor one",
            ),
        )
    )

    open_now = run(continuity.open_threads_for("creator-1", "fan-1"))
    assert [t.id for t in open_now] == [replacement.id]

    stored = {row["id"]: row for row in db.tables["conversation_open_threads"]}
    assert stored[original.id]["status"] == ThreadStatus.SUPERSEDED.value
    assert stored[original.id]["superseded_by"] == replacement.id, (
        "the link is what makes 'he changed his mind' answerable later"
    )
    assert stored[original.id]["resolved_by"] == ResolvedBy.SUPERSESSION.value


def test_nothing_is_deleted_when_it_is_superseded(db):
    original = run(continuity.record_open_thread(_thread()))
    run(continuity.supersede_thread(original.id, _thread(summary="a different thing")))

    assert len(db.tables["conversation_open_threads"]) == 2


# --- multiple open threads, ranked the way an operator would work them ------


def test_a_complaint_outranks_everything_else(db):
    """§1: a request to fix access outranks a new commercial suggestion."""
    threads = [
        _thread(
            kind=ThreadKind.DEFERRED_TOPIC,
            summary="he wanted to hear about the shoot",
            first_seen_at=NOW - timedelta(days=3),
        ),
        _thread(
            kind=ThreadKind.COMPLAINT,
            summary="he cannot open the set he bought",
            first_seen_at=NOW,
        ),
    ]

    assert continuity.rank_threads(threads)[0].kind == ThreadKind.COMPLAINT


def test_the_older_of_two_obligations_comes_first(db):
    """§1: he returns to the earlier of two subjects, not the newest."""
    older = _thread(summary="the trip he asked about", first_seen_at=NOW - timedelta(days=9))
    newer = _thread(summary="the film he asked about", first_seen_at=NOW)

    ranked = continuity.rank_threads([newer, older])

    assert ranked[0].summary == "the trip he asked about"


def test_what_he_is_waiting_on_comes_before_ordinary_deferred_chat(db):
    waiting = _thread(kind=ThreadKind.QUESTION, raised_by=ThreadParty.FAN, first_seen_at=NOW)
    chat = _thread(
        kind=ThreadKind.DEFERRED_TOPIC,
        raised_by=ThreadParty.FAN,
        summary="the band he mentioned",
        first_seen_at=NOW - timedelta(days=5),
    )

    assert continuity.rank_threads([chat, waiting])[0].kind == ThreadKind.QUESTION


def test_the_prompt_lines_carry_no_internal_vocabulary(db):
    lines = continuity.summarize_threads(
        [
            _thread(summary="whether you ever visit Chicago"),
            _thread(
                kind=ThreadKind.PROMISE,
                raised_by=ThreadParty.CREATOR,
                summary="send him the beach set",
                resolution_condition="it is sent",
            ),
        ]
    )

    joined = " ".join(lines)
    assert "he asked: whether you ever visit Chicago" in joined
    assert "you said: send him the beach set" in joined
    for leak in ("thread", "dedupe", "uuid", "status=", "fan_id"):
        assert leak not in joined


def test_the_reservation_for_obligations_is_bounded(db):
    many = [_thread(summary=f"topic number {i}") for i in range(20)]

    assert len(continuity.summarize_threads(many)) == continuity.DEFAULT_THREAD_LIMIT, (
        "a reply that services twenty obligations at once is an interrogation"
    )


# --- expiry -----------------------------------------------------------------


def test_an_expired_obligation_stops_being_carried(db):
    run(
        continuity.record_open_thread(
            _thread(expires_at=NOW - timedelta(days=1), summary="a passing aside")
        )
    )

    assert run(continuity.open_threads_for("creator-1", "fan-1")) == []
    stored = db.tables["conversation_open_threads"][0]
    assert stored["status"] == ThreadStatus.EXPIRED.value
    assert stored["resolved_by"] == ResolvedBy.EXPIRY.value


def test_an_unanswered_question_survives_a_week(db):
    """§1 tests a return after one day and one week."""
    run(continuity.record_open_thread(_thread()))

    later = NOW + timedelta(days=7)
    assert run(continuity.expire_due_threads("creator-1", "fan-1", now=later)) == 0
    assert len(run(continuity.open_threads_for("creator-1", "fan-1"))) == 1


# --- scoping ----------------------------------------------------------------


def test_two_creators_talking_to_one_customer_do_not_share_obligations(db):
    """§4: retrieval must be scoped by creator and customer."""
    run(continuity.record_open_thread(_thread(creator_id="creator-1")))
    run(continuity.record_open_thread(_thread(creator_id="creator-2", summary="something else")))

    first = run(continuity.open_threads_for("creator-1", "fan-1"))
    second = run(continuity.open_threads_for("creator-2", "fan-1"))

    assert [t.summary for t in first] == ["whether you ever visit Chicago"]
    assert [t.summary for t in second] == ["something else"]


def test_two_customers_of_one_creator_do_not_share_obligations(db):
    run(continuity.record_open_thread(_thread(fan_id="fan-1")))
    run(continuity.record_open_thread(_thread(fan_id="fan-2", summary="his own question")))

    assert [t.summary for t in run(continuity.open_threads_for("creator-1", "fan-2"))] == [
        "his own question"
    ]


def test_an_unscoped_read_returns_nothing_rather_than_everything(db):
    run(continuity.record_open_thread(_thread()))

    assert run(continuity.open_threads_for("", "fan-1")) == []
    assert run(continuity.open_threads_for("creator-1", "")) == []


# --- episodes ---------------------------------------------------------------


def _episode(**overrides) -> ConversationEpisode:
    values = {
        "creator_id": "creator-1",
        "fan_id": "fan-1",
        "summary": "talked about his trip and the shoot she was planning",
        "ended_with": EpisodeEnding.WENT_QUIET,
        "first_message_at": NOW - timedelta(days=8),
        "last_message_at": NOW - timedelta(days=8, minutes=-40),
        "message_count": 22,
    }
    values.update(overrides)
    return ConversationEpisode(**values)


def test_an_episode_records_what_it_was_about_and_how_it_ended(db):
    stored = run(continuity.record_episode(_episode()))

    assert stored is not None
    assert "trip" in stored.summary
    assert stored.ended_with == EpisodeEnding.WENT_QUIET


def test_summarising_the_same_stretch_twice_keeps_one_episode(db):
    run(continuity.record_episode(_episode()))
    run(continuity.record_episode(_episode()))

    assert len(run(continuity.recent_episodes_for("creator-1", "fan-1"))) == 1


def test_an_episode_can_never_stand_in_for_a_receipt(db):
    """The review's words: an episode is "never proof of payment"."""
    row = _episode().to_row(dedupe_key="k")

    for money in ("amount", "price", "price_cents", "purchased", "order_id"):
        assert money not in row, (
            "ppv_deliveries is the authority on money and must stay the only one"
        )


def test_an_episode_keeps_the_range_it_describes(db):
    """So a summary can be read back to the messages it came from."""
    stored = run(continuity.record_episode(_episode()))

    assert stored.first_message_at <= stored.last_message_at
    assert stored.message_count == 22


def test_an_episode_line_says_when_rather_than_how_much(db):
    line = _episode().render()

    assert "went quiet" in line
    assert "$" not in line


def test_episodes_are_scoped_to_the_conversation_they_came_from(db):
    run(continuity.record_episode(_episode(creator_id="creator-1")))
    run(continuity.record_episode(_episode(creator_id="creator-2", summary="a different one")))

    assert [e.summary for e in run(continuity.recent_episodes_for("creator-2", "fan-1"))] == [
        "a different one"
    ]


# --- evidence discipline ----------------------------------------------------


def test_what_someone_said_is_not_the_same_record_as_what_a_system_confirmed(db):
    """§4: those two must not merge into one authoritative fact."""
    said = _thread(
        kind=ThreadKind.COMPLAINT,
        summary="he says his payment went through",
        evidence_type=EvidenceType.STATED,
    )
    confirmed = _thread(
        kind=ThreadKind.COMPLAINT,
        summary="the platform confirmed order 12345",
        evidence_type=EvidenceType.PLATFORM_CONFIRMED,
    )

    run(continuity.record_open_thread(said))
    run(continuity.record_open_thread(confirmed))

    kept = run(continuity.open_threads_for("creator-1", "fan-1"))
    assert len(kept) == 2
    assert {t.evidence_type for t in kept} == {
        EvidenceType.STATED,
        EvidenceType.PLATFORM_CONFIRMED,
    }


def test_a_thread_carries_its_provenance_without_copying_the_message(db):
    stored = run(
        continuity.record_open_thread(
            _thread(
                source_message_fingerprint="abc123def456",
                source_turn_id="turn-1",
                evidence_text="do you ever come to chicago",
            )
        )
    )

    assert stored.source_message_fingerprint == "abc123def456"
    assert stored.source_turn_id == "turn-1"


# --- nothing here may stop a reply ------------------------------------------


def test_a_database_failure_costs_continuity_and_never_the_turn(monkeypatch):
    class Broken:
        def table(self, _name):
            raise RuntimeError("database is down")

    monkeypatch.setattr(continuity, "get_supabase", Broken)

    assert run(continuity.record_open_thread(_thread())) is None
    assert run(continuity.open_threads_for("creator-1", "fan-1")) == []
    assert run(continuity.recent_episodes_for("creator-1", "fan-1")) == []
    assert run(continuity.expire_due_threads("creator-1", "fan-1")) == 0
    assert run(
        continuity.resolve_thread(
            "t-1", status=ThreadStatus.FULFILLED, resolved_by=ResolvedBy.CREATOR_REPLY
        )
    ) is False


def test_a_row_written_by_a_newer_build_is_read_without_raising():
    thread = OpenThread.from_row(
        {
            "id": "t-1",
            "creator_id": "creator-1",
            "fan_id": "fan-1",
            "kind": "some_future_kind",
            "raised_by": "nobody",
            "summary": "a thing",
            "status": "parked",
            "evidence_type": "telepathy",
        }
    )

    assert thread.kind == ThreadKind.DEFERRED_TOPIC
    assert thread.status == ThreadStatus.OPEN
    assert thread.evidence_type == EvidenceType.INFERRED


def test_an_empty_thread_is_refused_rather_than_stored(db):
    assert run(continuity.record_open_thread(_thread(creator_id=""))) is None
    assert db.tables["conversation_open_threads"] == []
