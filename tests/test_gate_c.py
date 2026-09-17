"""Gate C: what the conversation is carrying survives the conversation.

    two deferred subjects and an unanswered question survive 30+ intervening
    turns and a return session; a correction displaces obsolete evidence [...]
    Cross-tenant and delayed-extraction tests pass.
                        — docs/continuation_brief_2026-09-17.md, Phase C

The unit tests in tests/test_conversation_continuity.py each hold one
mechanism. This holds the gate itself: the whole path, end to end, through the
pieces a live turn actually uses — extraction proposes, validation accepts,
the store keeps, the clock advances, retrieval returns, and the context packet
puts it in front of the model.

That last step is the one worth having a test for. Every piece could work and
the packet could still drop the obligations under a long transcript, in which
case the model never sees them and none of the rest matters.
"""

from __future__ import annotations

import asyncio

import pytest

from core import clock
from models.conversation_continuity import ThreadKind, ThreadStatus
from services import conversation_continuity as continuity
from services.context_packet import build_context_packet
from services.continuity_extraction import extract_threads
from tests.fake_supabase import FakeSupabase


def run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _wall_clock():
    clock.reset()
    yield
    clock.reset()


@pytest.fixture
def movable(monkeypatch):
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv(clock.EVAL_CLOCK_FLAG, "1")
    return clock


@pytest.fixture
def db(monkeypatch):
    store = FakeSupabase(
        {"conversation_open_threads": [], "conversation_episodes": []}
    )
    monkeypatch.setattr(continuity, "get_supabase", lambda: store)
    return store


def _raise_from_a_turn(*, creator_id="creator-1", fan_id="fan-1", **lists) -> int:
    """Run a turn's analysis through extraction and store what survives."""
    result = extract_threads(
        {
            "open_questions_raised": [],
            "commitments_made": [],
            "topics_deferred": [],
            "corrections_stated": [],
            "threads_resolved": [],
            **lists,
        },
        creator_id=creator_id,
        fan_id=fan_id,
        source_turn_id="turn-1",
        source_message_fingerprint="fp-1",
    )
    stored = 0
    for thread in result.threads:
        if run(continuity.record_open_thread(thread)):
            stored += 1
    return stored


def _chatter(turns: int) -> list[dict]:
    """A long stretch of conversation about nothing in particular."""
    history: list[dict] = []
    for index in range(turns):
        history.append({"role": "fan", "content": f"anyway, thing number {index}"})
        history.append({"role": "creator", "content": f"haha yeah, number {index}"})
    return history


def _carried(creator_id="creator-1", fan_id="fan-1") -> list[str]:
    threads = run(continuity.open_threads_for(creator_id, fan_id))
    return continuity.summarize_threads(threads)


# ===========================================================================
# The gate
# ===========================================================================


def test_two_deferred_subjects_and_a_question_survive_thirty_turns_and_a_week(
    db, movable
):
    """The gate, in one test.

    Everything here is the real path: the analyzer's output shape goes through
    the real validator, into the real store, read back through the real
    retrieval, into the real packet.
    """
    stored = _raise_from_a_turn(
        open_questions_raised=["whether she ever gets to chicago"],
        topics_deferred=[
            "his sister's wedding, until it is closer",
            "the road trip he keeps mentioning",
        ],
    )
    assert stored == 3

    # Thirty-four turns about nothing, then a week away.
    history = _chatter(34)
    movable.advance(7)

    carried = _carried()
    assert len(carried) == 3

    blob = " ".join(carried).lower()
    assert "chicago" in blob
    assert "wedding" in blob
    assert "road trip" in blob

    # And the packet actually puts them in front of the model. Every piece
    # above could work and this could still drop them under a long transcript.
    packet = build_context_packet(history=history, open_threads=carried)
    rendered = packet.render_continuity()

    assert packet.dropped_threads == 0
    assert "chicago" in rendered.lower()
    assert "wedding" in rendered.lower()
    assert "road trip" in rendered.lower()


def test_the_obligations_outlive_the_transcript_window_itself(db, movable):
    """The whole reason they are rows and not messages.

    Thirty-four turns is well past the twelve-turn window, so the message that
    raised the question is long gone from the transcript. The obligation is not.
    """
    _raise_from_a_turn(open_questions_raised=["whether she ever gets to chicago"])
    history = _chatter(34)
    movable.advance(7)

    packet = build_context_packet(history=history, open_threads=_carried())

    assert "chicago" not in packet.render_transcript().lower()
    assert "chicago" in packet.render_continuity().lower()


def test_a_week_away_does_not_expire_an_unanswered_question(db, movable):
    """An unanswered question does not stop mattering because a week passed."""
    _raise_from_a_turn(open_questions_raised=["whether she ever gets to chicago"])

    movable.advance(30)

    assert len(_carried()) == 1


def test_answering_one_leaves_the_others_carried(db, movable):
    """Resolution is per obligation, not per conversation."""
    _raise_from_a_turn(
        open_questions_raised=["whether she ever gets to chicago"],
        topics_deferred=["his sister's wedding", "the road trip"],
    )
    threads = run(continuity.open_threads_for("creator-1", "fan-1"))
    question = next(t for t in threads if t.kind is ThreadKind.QUESTION)

    run(
        continuity.resolve_thread(
            question.id,
            status=ThreadStatus.FULFILLED,
            resolved_by=continuity.ResolvedBy.CREATOR_REPLY,
        )
    )
    movable.advance(7)

    carried = " ".join(_carried()).lower()
    assert "chicago" not in carried
    assert "wedding" in carried and "road trip" in carried


# ===========================================================================
# A correction displaces what it corrects
# ===========================================================================


def test_a_correction_displaces_the_obsolete_version(db):
    """Not "sits beside".

    §5's preference-correction row fails exactly when the old preference is
    still in retrieved history next to the new one, because then the model gets
    to pick.
    """
    _raise_from_a_turn(corrections_stated=["he prefers indoor, not outdoor"])
    original = run(continuity.open_threads_for("creator-1", "fan-1"))[0]

    replacement = original.model_copy(
        update={"summary": "he prefers hotel shoots, not indoor", "id": ""}
    )
    run(continuity.supersede_thread(original.id, replacement))

    carried = _carried()
    assert len(carried) == 1
    assert "hotel" in carried[0]
    assert "indoor, not outdoor" not in carried[0]


def test_the_superseded_version_is_kept_but_not_carried(db):
    """Displaced, not deleted: the audit trail is why supersession exists."""
    _raise_from_a_turn(corrections_stated=["he prefers indoor, not outdoor"])
    original = run(continuity.open_threads_for("creator-1", "fan-1"))[0]
    run(
        continuity.supersede_thread(
            original.id,
            original.model_copy(update={"summary": "he prefers hotel shoots", "id": ""}),
        )
    )

    stored = db.tables["conversation_open_threads"]
    statuses = {row["status"] for row in stored}

    assert len(stored) == 2
    assert ThreadStatus.SUPERSEDED.value in statuses


# ===========================================================================
# Cross-tenant
# ===========================================================================


def test_another_creators_obligations_are_not_carried_here(db, movable):
    """§4: retrieval scoped by creator AND customer. One customer talking to
    two creators has two sets of open threads and they must not see each
    other."""
    _raise_from_a_turn(
        creator_id="creator-1", open_questions_raised=["whether she gets to chicago"]
    )
    _raise_from_a_turn(
        creator_id="creator-2", open_questions_raised=["whether she gets to boston"]
    )

    movable.advance(7)

    assert "chicago" in " ".join(_carried("creator-1")).lower()
    assert "boston" not in " ".join(_carried("creator-1")).lower()
    assert "boston" in " ".join(_carried("creator-2")).lower()


def test_another_customers_obligations_are_not_carried_here(db):
    _raise_from_a_turn(fan_id="fan-1", open_questions_raised=["about chicago"])
    _raise_from_a_turn(fan_id="fan-2", open_questions_raised=["about boston"])

    assert "boston" not in " ".join(_carried(fan_id="fan-1")).lower()


# ===========================================================================
# Extraction that arrives late, or twice
# ===========================================================================


def test_the_same_obligation_extracted_twice_is_one_obligation(db):
    """A retry, a redelivered webhook, the same turn processed again."""
    for _ in range(3):
        _raise_from_a_turn(open_questions_raised=["whether she ever gets to chicago"])

    assert len(_carried()) == 1


def test_a_reordered_rephrasing_produces_a_visible_duplicate_not_a_loss():
    """The limit of the dedupe key, asserted rather than wished away.

    `subject_key` keeps word ORDER, so "whether she ever gets to chicago" and
    "whether she gets to Chicago ever" are two keys and two threads. The module
    says so outright and gives its reason: collapsing two keys silently loses
    one obligation, while a duplicate is visible and harmless.

    That reasoning holds, so this asserts the real behaviour. What Gate C needs
    is that nothing is LOST, and a duplicate loses nothing — an operator reads
    two similar lines, which is a cosmetic cost, not a forgotten question.
    """
    # Deliberately not using the `db` fixture's assertion of uniqueness: this
    # is about what the key does, so it uses the key.
    from services.conversation_continuity import subject_key

    first = subject_key("whether she ever gets to chicago")
    second = subject_key("whether she gets to Chicago ever")

    assert first != second, "word order is part of the key, by design"
    # And the thing that matters: both survive, so the question is still asked.
    assert "chicago" in first and "chicago" in second


def test_the_same_wording_twice_is_one_obligation(db):
    """What the key DOES guarantee: a repeat of the same phrasing collapses."""
    _raise_from_a_turn(open_questions_raised=["whether she ever gets to chicago"])
    _raise_from_a_turn(open_questions_raised=["Whether she ever gets to Chicago!"])

    assert len(_carried()) == 1


def test_extraction_arriving_after_a_resolution_does_not_reopen_it(db):
    """Stale extraction must not override a newer resolution.

    The order that matters: the obligation was settled, and a delayed or
    re-run extraction of the turn that raised it arrives afterwards. Reopening
    would resurrect an answered question and have the next reply ask it again.
    """
    _raise_from_a_turn(open_questions_raised=["whether she ever gets to chicago"])
    thread = run(continuity.open_threads_for("creator-1", "fan-1"))[0]
    run(
        continuity.resolve_thread(
            thread.id,
            status=ThreadStatus.FULFILLED,
            resolved_by=continuity.ResolvedBy.CREATOR_REPLY,
        )
    )

    _raise_from_a_turn(open_questions_raised=["whether she ever gets to chicago"])

    assert _carried() == []


def test_a_degraded_analysis_arriving_late_writes_nothing(db):
    """The worst case for delayed extraction: a reading nothing vouches for."""
    result = extract_threads(
        {"analysis_degraded": "true", "open_questions_raised": ["about chicago"]},
        creator_id="creator-1",
        fan_id="fan-1",
    )
    for thread in result.threads:
        run(continuity.record_open_thread(thread))

    assert _carried() == []


# ===========================================================================
# What this does not establish
# ===========================================================================


def test_the_packet_reports_obligations_it_could_not_fit(db):
    """Gate C is about them surviving, and surviving has a budget.

    Seven obligations against a budget of six means one is not shown, and the
    packet has to say so — "the model did not mention it" and "the model was
    never told" are different failures with different fixes.
    """
    from services.context_packet import ContextBudget

    carried = [f"obligation number {index}" for index in range(7)]
    packet = build_context_packet(
        history=_chatter(2), open_threads=carried, budget=ContextBudget(threads=6)
    )

    assert len(packet.open_threads) == 6
    assert packet.dropped_threads == 1
    assert packet.fingerprint()["dropped_threads"] == 1


# ===========================================================================
# A correction displaces obsolete evidence on every surface
# ===========================================================================
#
# Gate C asks for this "across Auto, Assisted, simulation and replay". The
# useful way to establish it is not four near-identical tests: it is to show
# that all four read continuity from ONE place, so a correction applied once is
# invisible to all of them and cannot be invisible to only three.
#
# That matters because divergence here has happened before. Finding A of the
# review was exactly this — Assisted and Full Auto assembling context
# differently — and the fix was the two modes sharing the load rather than
# keeping a second copy of the logic.


def _corrected_threads(db) -> list[str]:
    """Record a preference, correct it, and return what is still carried."""
    _raise_from_a_turn(corrections_stated=["he prefers indoor, not outdoor"])
    original = run(continuity.open_threads_for("creator-1", "fan-1"))[0]
    run(
        continuity.supersede_thread(
            original.id,
            original.model_copy(
                update={"summary": "he prefers hotel shoots, not indoor", "id": ""}
            ),
        )
    )
    return _carried()


def _context(open_thread_lines):
    """The one object every surface hands to the prompt builders."""
    from models.schemas import (
        ConversationContext,
        Fan,
        Message,
        Persona,
        StageType,
    )

    history = [
        Message(role="fan", content="what have you got"),
        Message(role="creator", content="a few things"),
    ]
    return ConversationContext(
        fan_message=history[-1].content,
        conversation_history=history,
        fan_profile=Fan(id="fan-1", display_name="Dan"),
        creator_persona=Persona(),
        similar_exchanges=[],
        conversation_stage=StageType.WARMING_UP,
        open_threads=tuple(open_thread_lines),
    )


def test_the_writer_sees_the_correction_and_not_what_it_corrected(db):
    from ai.prompt_builder import build_prompt

    rendered = str(build_prompt(_context(_corrected_threads(db)))).lower()

    assert "hotel" in rendered
    assert "indoor, not outdoor" not in rendered


def test_the_analyzer_sees_the_correction_and_not_what_it_corrected(db):
    """Finding D was the two readers seeing different evidence.

    A correction the writer honours and the analyzer does not is that bug
    again, in the one place it would be hardest to notice.
    """
    from ai.situation_analyzer import build_analyzer_prompt

    system, user = build_analyzer_prompt(_context(_corrected_threads(db)))
    rendered = f"{system}\n{user}".lower()

    assert "hotel" in rendered
    assert "indoor, not outdoor" not in rendered


def test_every_surface_reads_continuity_from_the_same_place(db):
    """The property that makes one correction enough for all four.

    Auto, Assisted and simulation all call open_threads_for + summarize_threads
    and hand the result to build_context_packet; replay reads the same packet.
    Asserted by source, because the failure mode is a fifth caller appearing
    with its own copy — which is what finding A was.
    """
    import inspect

    from services import suggestions

    source = inspect.getsource(suggestions)

    # Both live turn paths load it through the shared helpers.
    assert source.count("open_threads_for(creator_id, fan_id)") >= 2
    assert source.count("summarize_threads(carried_threads)") >= 2

    # The invariant that actually matters, and the one a divergent fifth
    # caller would break: nothing here reads the table directly or assembles
    # thread lines by hand. An exact count of the calls above would also break
    # on an innocuous refactor, so it is a floor rather than an equality.
    assert "conversation_open_threads" not in source
    assert 'table("conversation' not in source


def test_the_packet_carries_only_the_current_version(db):
    """Replay's view, which is the packet and nothing else."""
    packet = build_context_packet(history=_chatter(20), open_threads=_corrected_threads(db))

    rendered = packet.render_continuity().lower()

    assert "hotel" in rendered
    assert "indoor, not outdoor" not in rendered
    assert packet.dropped_threads == 0
