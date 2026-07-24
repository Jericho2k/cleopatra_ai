import inspect

from ai import prompt_builder


def test_writer_may_answer_plainly_without_forced_conversation_move():
    source = inspect.getsource(prompt_builder.build_prompt)

    assert "A plain, vague, or slightly unfinished reaction can be exactly right" in source
    assert "Sometimes answer one relevant thing and stop" in source
    assert "responding only to the part that naturally caught your attention" in source
    assert "do not add one just to keep the fan replying" in source


def test_writer_still_preserves_specificity_and_commercial_direction():
    source = inspect.getsource(prompt_builder.build_prompt)

    assert "Every reply must contain at least one detail that belongs to this exact conversation" in source
    assert "When a question is required" in source
    assert "Offer paid content only when the conversation actually supports it" in source
