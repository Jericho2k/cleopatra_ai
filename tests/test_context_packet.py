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
