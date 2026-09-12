"""The frozen writer_v1 humanity contract, asserted against the rendered prompt.

These rules used to be asserted against ``inspect.getsource(build_prompt)``.
They now live in ``ai/writer_style.py``, keyed by prompt version, so the
assertions moved with them — and got stronger in the process: the text is read
out of the version the writer would actually be handed, not out of a source file
that happens to contain it.

``writer_v1`` is frozen. ``cleo_legacy_v1`` exists so old and new behaviour can
be compared, and a comparison against a moving baseline is not one, so these
assertions are the thing that stops the legacy voice drifting.
"""

from ai.writer_style import (
    WRITER_V1,
    content_rules,
    response_instructions,
    voice_rules,
)


def _v1_prompt_text() -> str:
    return "\n".join(
        (
            voice_rules(WRITER_V1),
            content_rules(WRITER_V1),
            response_instructions(WRITER_V1),
        )
    )


def test_writer_may_answer_plainly_without_forced_conversation_move():
    source = _v1_prompt_text()

    assert "A plain, vague, or slightly unfinished reaction can be exactly right" in source
    assert "Sometimes answer one relevant thing and stop" in source
    assert "responding only to the part that naturally caught your attention" in source
    assert "do not add one just to keep the fan replying" in source


def test_writer_still_preserves_specificity_and_commercial_direction():
    source = _v1_prompt_text()

    assert "Every reply must contain at least one detail that belongs to this exact conversation" in source
    assert "When a question is required" in source
    assert "Offer paid content only when the conversation actually supports it" in source
