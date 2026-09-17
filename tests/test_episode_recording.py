"""Closing a stretch of conversation, without inventing what it was about.

conversation_episodes had no producer at all. record_episode existed, was
tested, and was called by nothing — so "we talked about this before" was a
table shape rather than something the system could ever say.

The design decision worth testing is that the summary is NOT written by a
model. Asked to summarise a stretch of conversation, a model produces fluent
prose about what it thinks happened, and this record is read back to it turns
later as though it were evidence — a belief the system formed about itself,
laundered into a fact. That is the shape of the failure the review objected to
in the baseline. So the summary is assembled from obligations that stretch
actually raised, which are rows with source turn ids behind them, and when
there were none the episode says what it factually was and claims nothing
about content.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from models.conversation_continuity import EpisodeEnding, EvidenceType
from services import conversation_continuity as continuity
from services.episode_recording import (
    EPISODE_GAP,
    build_episode,
    close_finished_episode,
    find_closed_stretch,
)
from tests.fake_supabase import FakeSupabase

START = datetime(2026, 9, 10, 9, 0, tzinfo=timezone.utc)


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def db(monkeypatch):
    store = FakeSupabase(
        {"conversation_open_threads": [], "conversation_episodes": []}
    )
    monkeypatch.setattr(continuity, "get_supabase", lambda: store)
    return store


def _exchange(count: int, *, start: datetime = START, minutes: int = 5) -> list[dict]:
    """`count` messages, alternating, a few minutes apart."""
    return [
        {
            "role": "fan" if index % 2 == 0 else "creator",
            "content": f"message {index}",
            "sent_at": (start + timedelta(minutes=minutes * index)).isoformat(),
        }
        for index in range(count)
    ]


def _after_a_gap(history: list[dict], *, hours: float = 30) -> list[dict]:
    last = datetime.fromisoformat(history[-1]["sent_at"])
    return history + [
        {
            "role": "fan",
            "content": "hey, im back",
            "sent_at": (last + timedelta(hours=hours)).isoformat(),
        }
    ]


# ===========================================================================
# When a stretch is over
# ===========================================================================


def test_a_return_after_a_long_silence_closes_the_stretch_before_it():
    stretch = find_closed_stretch(_after_a_gap(_exchange(6)))

    assert stretch is not None
    assert len(stretch) == 6


def test_a_live_conversation_closes_nothing():
    """Summarising an exchange still in progress produces a summary that is
    immediately wrong."""
    assert find_closed_stretch(_exchange(20)) is None


def test_an_ordinary_pause_does_not_chop_one_exchange_into_episodes():
    """A meal, a shift, a night's sleep in another timezone."""
    history = _exchange(6)
    assert find_closed_stretch(_after_a_gap(history, hours=1)) is None


def test_a_gap_at_exactly_the_threshold_closes_it():
    hours = EPISODE_GAP.total_seconds() / 3600
    assert find_closed_stretch(_after_a_gap(_exchange(6), hours=hours)) is not None


def test_an_abandoned_hello_is_not_worth_a_record():
    """An episode per abandoned hello would bury the ones that mean something."""
    assert find_closed_stretch(_after_a_gap(_exchange(2))) is None


def test_only_the_stretch_this_silence_closed_is_taken():
    """An older stretch was already closed by its own gap.

    Re-summarising it would produce a second record of the same conversation
    with a different shape, and both would be read back as evidence.
    """
    first = _exchange(6, start=START)
    second = _exchange(
        8, start=START + timedelta(days=2)
    )
    history = _after_a_gap(first + second, hours=30)

    stretch = find_closed_stretch(history)

    assert len(stretch) == 8, "the most recent closed stretch, not everything"


def test_messages_without_timestamps_are_ignored_rather_than_guessed_at():
    history = [{"role": "fan", "content": "hey"} for _ in range(8)]

    assert find_closed_stretch(history) is None


# ===========================================================================
# What it says it was about
# ===========================================================================


def test_the_summary_comes_from_obligations_that_stretch_raised():
    """Rows with source turn ids behind them, not invented prose."""
    stretch = find_closed_stretch(_after_a_gap(_exchange(6)))

    episode = build_episode(
        stretch,
        creator_id="creator-1",
        fan_id="fan-1",
        subjects=["he asked whether she gets to chicago", "his sister's wedding"],
    )

    assert "chicago" in episode.summary
    assert "wedding" in episode.summary


def test_a_stretch_that_raised_nothing_describes_itself_rather_than_guessing():
    """A dull true record beats an interesting invented one, and this one is
    read back to a model."""
    stretch = find_closed_stretch(_after_a_gap(_exchange(6)))

    episode = build_episode(stretch, creator_id="creator-1", fan_id="fan-1")

    assert "6 messages" in episode.summary
    assert "nothing left open" in episode.summary


def test_the_summary_does_not_become_a_second_transcript():
    stretch = find_closed_stretch(_after_a_gap(_exchange(6)))

    episode = build_episode(
        stretch,
        creator_id="creator-1",
        fan_id="fan-1",
        subjects=[f"subject number {index}" for index in range(10)],
    )

    assert episode.summary.count(";") <= 2


def test_an_episode_is_always_marked_as_read_rather_than_said():
    """A later reader deciding how much to rely on it needs that distinction."""
    stretch = find_closed_stretch(_after_a_gap(_exchange(6)))

    episode = build_episode(stretch, creator_id="creator-1", fan_id="fan-1")

    assert episode.evidence_type is EvidenceType.INFERRED


def test_an_episode_carries_no_money():
    """The model forbids it and the subjects cannot supply it.

    services/continuity_extraction.py already refused any obligation that
    mentioned money, so the exclusion holds by construction rather than by a
    second rule here that could drift from the first.
    """
    stretch = find_closed_stretch(_after_a_gap(_exchange(6)))
    episode = build_episode(stretch, creator_id="creator-1", fan_id="fan-1")

    assert not hasattr(episode, "price_cents")
    assert not hasattr(episode, "amount")
    assert "paid" not in episode.summary


# ===========================================================================
# How it ended, read off who spoke last
# ===========================================================================


def test_she_answered_and_he_did_not_come_back():
    stretch = find_closed_stretch(_after_a_gap(_exchange(6)))  # ends on creator

    assert build_episode(
        stretch, creator_id="c", fan_id="f"
    ).ended_with is EpisodeEnding.WENT_QUIET


def test_he_spoke_and_nothing_followed():
    stretch = find_closed_stretch(_after_a_gap(_exchange(7)))  # ends on fan

    assert build_episode(
        stretch, creator_id="c", fan_id="f"
    ).ended_with is EpisodeEnding.INTERRUPTED


# ===========================================================================
# Recording it
# ===========================================================================


def test_a_closed_stretch_is_persisted(db):
    episode = run(
        close_finished_episode(
            creator_id="creator-1", fan_id="fan-1", history=_after_a_gap(_exchange(6))
        )
    )

    assert episode is not None
    assert len(db.tables["conversation_episodes"]) == 1


def test_the_same_turn_processed_twice_writes_one_episode(db):
    """A retry, or a redelivered webhook."""
    history = _after_a_gap(_exchange(6))
    for _ in range(3):
        run(close_finished_episode(creator_id="creator-1", fan_id="fan-1", history=history))

    assert len(db.tables["conversation_episodes"]) == 1


def test_a_live_conversation_persists_nothing(db):
    run(close_finished_episode(creator_id="creator-1", fan_id="fan-1", history=_exchange(20)))

    assert db.tables["conversation_episodes"] == []


def test_recording_never_raises(monkeypatch):
    """Memory must not be able to cost a customer their answer."""
    async def exploding(_episode):
        raise RuntimeError("supabase down")

    monkeypatch.setattr(
        "services.conversation_continuity.record_episode", exploding
    )

    assert run(
        close_finished_episode(
            creator_id="creator-1", fan_id="fan-1", history=_after_a_gap(_exchange(6))
        )
    ) is None


def test_a_closed_episode_reads_back_as_one_line(db):
    episode = run(
        close_finished_episode(
            creator_id="creator-1",
            fan_id="fan-1",
            history=_after_a_gap(_exchange(6)),
            subjects=["he asked whether she gets to chicago"],
        )
    )

    line = episode.render()

    assert "chicago" in line
    assert "he went quiet" in line
    assert "\n" not in line
