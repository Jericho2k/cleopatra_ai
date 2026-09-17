"""Sprint 2 — one budgeted view of the conversation, for everything that reads one.

``docs/autonomy_architecture_review.md`` finding D names three separate problems
in one paragraph, and each has its own section below:

1. the windows counted BUBBLES, so a multipart reply spent the budget several
   times faster than a single one;
2. the analyzer's window was narrower than the writer's, so a decision could be
   made on evidence the reply was not written from;
3. recent chatter evicted unresolved obligations, because they shared one
   window with the transcript.

The builder is pure, so every one of these is testable without a fixture — which
is also what lets §5's replay comparison hold the evidence fixed while the thing
being compared changes.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from ai.prompt_builder import build_prompt
from ai.situation_analyzer import build_analyzer_prompt
from models.schemas import ConversationContext, Fan, Message, Persona, StageType
from services.context_packet import (
    STANDARD_BUDGET,
    ContextBudget,
    build_context_packet,
    group_into_turns,
)

NOW = datetime(2026, 9, 17, 12, 0, tzinfo=timezone.utc)


def _msg(role: str, content: str, minutes: int = 0) -> Message:
    return Message(role=role, content=content, sent_at=NOW + timedelta(minutes=minutes))


# --- 1: bubbles are not turns ----------------------------------------------


def test_a_multipart_reply_is_one_turn_not_three():
    """The bug: three bubbles saying one thing spent three sixteenths of the window."""
    turns = group_into_turns(
        [
            _msg("fan", "hey", 0),
            _msg("creator", "hey you", 1),
            _msg("creator", "how was your day", 2),
            _msg("creator", "i was thinking about you", 3),
            _msg("fan", "long day", 4),
        ]
    )

    assert [turn.speaker for turn in turns] == ["fan", "creator", "fan"]
    assert turns[1].bubbles == ("hey you", "how was your day", "i was thinking about you")


def test_a_multipart_turn_stays_visibly_multipart():
    """Flattening it would teach the writer a rhythm the conversation lacks."""
    turns = group_into_turns([_msg("creator", "hey"), _msg("creator", "you up")])

    assert turns[0].render(creator_name="You") == "You: hey | you up"


def test_the_same_exchange_costs_the_same_however_it_was_sent():
    single = build_context_packet([_msg("creator", "hey you how was your day")])
    split = build_context_packet(
        [_msg("creator", "hey you"), _msg("creator", "how was your day")]
    )

    assert len(single.turns) == len(split.turns) == 1


def test_blank_messages_are_not_things_anybody_said():
    turns = group_into_turns(
        [_msg("fan", "hey"), _msg("creator", "   "), _msg("fan", "you there")]
    )

    assert len(turns) == 1, "the blank must not split one fan turn into two"
    assert turns[0].bubbles == ("hey", "you there")


def test_a_turn_is_stamped_with_when_it_finished():
    """"How long ago did he last say something" means the end of his turn."""
    turns = group_into_turns([_msg("fan", "one", 0), _msg("fan", "two", 5)])

    assert turns[0].at == NOW + timedelta(minutes=5)


def test_the_builder_reads_plain_dicts_too():
    """The evaluation harness replays records that are not Message instances."""
    turns = group_into_turns(
        [{"role": "fan", "content": "hey"}, {"role": "creator", "content": "hi"}]
    )

    assert [turn.speaker for turn in turns] == ["fan", "creator"]


# --- 2: the analyzer and the writer now see the same conversation ----------


def _ctx(history: list[Message], **overrides) -> ConversationContext:
    values = {
        "fan_message": history[-1].content if history else "hey",
        "conversation_history": history,
        "fan_profile": Fan(id="fan-1", display_name="Dave"),
        "creator_persona": Persona(),
        "similar_exchanges": [],
        "conversation_stage": StageType.WARMING_UP,
    }
    values.update(overrides)
    return ConversationContext(**values)


def _long_history(turns: int = 30) -> list[Message]:
    history: list[Message] = []
    for i in range(turns):
        history.append(_msg("fan", f"fan says thing number {i}", i * 2))
        history.append(_msg("creator", f"creator says thing number {i}", i * 2 + 1))
    return history


def test_the_analyzer_no_longer_sees_less_than_the_writer():
    """Finding D's evidence asymmetry, measured on the rendered prompts."""
    history = _long_history()
    ctx = _ctx(history)

    _system, analyzer_user = build_analyzer_prompt(ctx)
    writer_prompt = build_prompt(ctx)
    writer_user = writer_prompt[1]["content"]
    writer_text = (
        writer_user if isinstance(writer_user, str) else str(writer_user)
    )

    analyzer_oldest = min(
        i for i in range(30) if f"fan says thing number {i}" in analyzer_user
    )
    writer_oldest = min(
        i for i in range(30) if f"fan says thing number {i}" in writer_text
    )

    assert analyzer_oldest <= writer_oldest, (
        "the classifier that decides what the turn does must not see less "
        "conversation than the model that writes it"
    )


def test_both_prompts_carry_the_unfinished_business():
    history = _long_history()
    ctx = _ctx(
        history,
        open_threads=["he asked: whether you ever visit Chicago"],
        conversation_episodes=["2026-09-08: talked about his trip (he went quiet)"],
    )

    _system, analyzer_user = build_analyzer_prompt(ctx)
    writer_text = str(build_prompt(ctx)[1]["content"])

    for rendered in (analyzer_user, writer_text):
        assert "whether you ever visit Chicago" in rendered
        assert "talked about his trip" in rendered


def test_the_obligations_block_states_facts_rather_than_instructions():
    """§4: a decision object must not prescribe a fixed sentence shape."""
    packet = build_context_packet(
        [], open_threads=["he asked: whether you ever visit Chicago"]
    )

    rendered = packet.render_continuity()
    assert "not instructions" in rendered
    for imperative in ("you must", "always ", "make sure you"):
        assert imperative not in rendered.lower()


# --- 3: small talk cannot evict an obligation ------------------------------


def test_a_wall_of_small_talk_does_not_drop_an_unanswered_question():
    """§4: do not drop them merely because small talk filled the window."""
    packet = build_context_packet(
        _long_history(60),
        open_threads=["he asked: whether you ever visit Chicago"],
    )

    assert packet.open_threads == ("he asked: whether you ever visit Chicago",)
    assert len(packet.turns) <= packet.budget.turns


def test_obligations_are_allocated_before_the_transcript_is_measured():
    tiny = ContextBudget(turns=1, transcript_chars=50, threads=3, episodes=2)
    packet = build_context_packet(
        _long_history(40),
        open_threads=["a", "b", "c"],
        episodes=["x", "y"],
        budget=tiny,
    )

    assert packet.open_threads == ("a", "b", "c")
    assert packet.episodes == ("x", "y")


def test_the_obligation_allowance_is_itself_bounded():
    packet = build_context_packet(
        [], open_threads=[f"thread {i}" for i in range(20)]
    )

    assert len(packet.open_threads) == packet.budget.threads
    assert packet.dropped_threads == 20 - packet.budget.threads


# --- the transcript budget --------------------------------------------------


def test_the_newest_turns_are_the_ones_kept():
    packet = build_context_packet(_long_history(40), budget=ContextBudget(turns=4))

    rendered = packet.render_transcript()
    assert "thing number 39" in rendered
    assert "thing number 0" not in rendered
    assert packet.dropped_turns > 0


def test_one_pasted_wall_of_text_cannot_push_out_the_exchange_around_it():
    history = [
        _msg("fan", "hey", 0),
        _msg("creator", "hey you", 1),
        _msg("fan", "x" * 9000, 2),
    ]

    packet = build_context_packet(history, budget=ContextBudget(transcript_chars=2000))

    # The wall itself survives — it is the newest thing and the reply answers
    # it — but it took the budget with it, which is visible rather than silent.
    assert len(packet.turns) == 1
    assert packet.dropped_turns == 2


def test_an_empty_conversation_renders_to_nothing_rather_than_a_header():
    packet = build_context_packet([])

    assert packet.render_transcript() == ""
    assert packet.render_continuity() == ""


def test_a_zero_turn_budget_still_keeps_the_obligations():
    packet = build_context_packet(
        _long_history(5),
        open_threads=["he asked something"],
        budget=ContextBudget(turns=0),
    )

    assert packet.turns == ()
    assert packet.open_threads == ("he asked something",)


# --- the fingerprint a reply's provenance records --------------------------


def test_the_fingerprint_says_what_was_dropped_not_just_what_was_kept():
    """"Never mentioned it" and "was never told" must be separable afterwards."""
    packet = build_context_packet(
        _long_history(40),
        open_threads=[f"thread {i}" for i in range(9)],
        budget=ContextBudget(turns=3, threads=2),
    )

    fingerprint = packet.fingerprint()
    assert fingerprint["turns"] == 3
    assert fingerprint["dropped_turns"] > 0
    assert fingerprint["open_threads"] == 2
    assert fingerprint["dropped_threads"] == 7
    assert fingerprint["turn_budget"] == 3


def test_the_fingerprint_carries_counts_and_never_content():
    packet = build_context_packet(
        [_msg("fan", "something private")],
        open_threads=["he asked about something private"],
    )

    assert "something private" not in repr(packet.fingerprint())


def test_the_packet_counts_the_bubbles_its_turns_came_from():
    packet = build_context_packet(
        [_msg("creator", "one"), _msg("creator", "two"), _msg("fan", "three")]
    )

    assert len(packet.turns) == 2
    assert packet.message_count == 3


# --- determinism, which the replay comparison depends on -------------------


def test_the_same_inputs_always_produce_the_same_packet():
    history = _long_history(25)
    first = build_context_packet(history, open_threads=["a", "b"])
    second = build_context_packet(history, open_threads=["a", "b"])

    assert first == second


# ===========================================================================
# A message too big for the budget
# ===========================================================================
#
# The guard was `if kept and used + cost > ceiling`, so the newest turn was
# admitted whole however large it was: a single 10,000-character message
# rendered a 10,005-character transcript against a 6,000 budget. It also took
# the exchange it was pasted into with it, because nothing could follow.
#
# Both extremes are wrong. Dropping it loses the message the reply is
# answering; keeping it whole makes the budget a suggestion. So it is
# shortened, and the shortening is recorded — a turn that was cut is not the
# same as one that arrived whole, and a reply attributed to "the model had the
# message" needs to know which happened.


def _oversized_history(size: int = 10_000) -> list[dict]:
    return [
        {"role": "fan", "content": "hey"},
        {"role": "creator", "content": "hi!"},
        {"role": "fan", "content": "x" * size},
    ]


def test_one_enormous_message_cannot_blow_the_character_budget():
    packet = build_context_packet(history=_oversized_history())

    assert len(packet.render_transcript()) <= STANDARD_BUDGET.transcript_chars


def test_an_enormous_message_is_shortened_rather_than_dropped():
    """It is the message the reply has to answer."""
    packet = build_context_packet(history=_oversized_history())

    assert packet.turns, "the newest turn survives"
    assert packet.truncated_turns == 1
    assert packet.truncated_chars > 0


def test_a_shortened_message_says_it_was_shortened():
    """A reader must not answer an abridgement as though it were the whole."""
    rendered = build_context_packet(history=_oversized_history()).render_transcript()

    assert "characters omitted" in rendered


def test_shortening_keeps_both_ends():
    """A long message often puts its point at the end.

    "…and anyway, can you send me the other one?" is the part a reply must
    answer, and a head-only cut would reliably discard exactly that.
    """
    history = [
        {"role": "fan", "content": "FIRST " + ("x" * 10_000) + " LAST-QUESTION"},
    ]

    rendered = build_context_packet(history=history).render_transcript()

    assert "FIRST" in rendered
    assert "LAST-QUESTION" in rendered


def test_an_ordinary_conversation_is_not_reported_as_shortened():
    packet = build_context_packet(
        history=[{"role": "fan", "content": "hey"}, {"role": "creator", "content": "hi"}]
    )

    assert packet.truncated_turns == 0
    assert packet.truncated_chars == 0
    assert "characters omitted" not in packet.render_transcript()


def test_a_budget_too_small_for_any_message_drops_rather_than_emitting_a_marker():
    """A marker with no message around it is worse than nothing."""
    packet = build_context_packet(
        history=_oversized_history(), budget=ContextBudget(transcript_chars=10)
    )

    assert packet.turns == ()
    assert packet.truncated_turns == 0


def test_obligations_still_survive_an_enormous_message():
    """They are reserved before the transcript is measured, and must stay so.

    This is the "do not silently drop a hard constraint" case: the wall of text
    must not be able to evict what the conversation is still carrying.
    """
    packet = build_context_packet(
        history=_oversized_history(),
        open_threads=["he asked about chicago and nobody answered"],
    )

    assert packet.open_threads == ("he asked about chicago and nobody answered",)


# ===========================================================================
# A fingerprint that can tell two packets apart
# ===========================================================================


def test_two_different_conversations_do_not_share_a_fingerprint():
    """Counts alone could not distinguish them.

    Chicago and Boston are both "1 turn, 1 message, 0 threads, 0 episodes", so
    two provenance records could agree in every field while the model saw
    entirely different evidence — which makes §5's replay comparison
    unverifiable, because nothing could confirm the evidence was held fixed.
    """
    chicago = build_context_packet(history=[{"role": "fan", "content": "i live in chicago"}])
    boston = build_context_packet(history=[{"role": "fan", "content": "i live in boston"}])

    assert chicago.fingerprint() != boston.fingerprint()
    assert chicago.content_digest() != boston.content_digest()


def test_the_same_conversation_has_the_same_fingerprint():
    """What makes "these two replies saw the same input" a statement of fact."""
    history = [{"role": "fan", "content": "hey"}, {"role": "creator", "content": "hi"}]

    assert (
        build_context_packet(history=history).content_digest()
        == build_context_packet(history=list(history)).content_digest()
    )


def test_the_digest_covers_obligations_and_episodes_too():
    """Two packets with an identical transcript are not identical packets."""
    history = [{"role": "fan", "content": "hey"}]

    bare = build_context_packet(history=history)
    carrying = build_context_packet(history=history, open_threads=["chicago, unanswered"])

    assert bare.content_digest() != carrying.content_digest()


def test_the_digest_is_not_the_conversation():
    """A hash stores no message text and cannot be reversed into any.

    Asserted by shape, not by hunting for digits in it. A 16-character hex
    digest contains a given four-hex-digit run about 0.1% of the time by pure
    chance — the first draft of this test asserted `"4111" not in digest` and
    failed on a digest of `4a14111f86701006`, which leaked nothing at all.
    That is the same mistake as the health-redaction assertion this branch
    opened by fixing, made by the person who fixed it.
    """
    secret = "my card number is 4111 1111 1111 1111"
    digest = build_context_packet(
        history=[{"role": "fan", "content": secret}]
    ).content_digest()

    # Fixed-width hex and nothing else: there is no room for content in it.
    assert len(digest) == 16
    assert all(character in "0123456789abcdef" for character in digest)
    assert secret not in digest
