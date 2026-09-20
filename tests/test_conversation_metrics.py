"""Objective metrics: the arithmetic, and the line it must not cross.

Two kinds of test. Most of them check a number against a hand-worked example,
because a metric nobody has verified by hand is a number with a name on it.

The last few check the boundary. This module is allowed to count question
marks; it is not allowed to emit anything that reads as a score for
specificity, contribution, initiative or scene quality. Those need a person, and
a plausible-looking number for them would be believed.
"""

from __future__ import annotations

from services.conversation_metrics import (
    NOT_MEASURED_HERE,
    conversation_metrics,
    ends_with_question,
    execution_metrics,
    model_metrics,
    question_metrics,
    repetition_metrics,
    scenario_metrics,
    shape_metrics,
)


def turn(index: int, fan: str, *replies: str, **extra) -> dict:
    record = {
        "turn_index": index,
        "fan_input": fan,
        "creator_output": list(replies),
        "latency_ms": extra.pop("latency_ms", 100),
        "control": extra.pop("control", {"outcome": "replied", "error": ""}),
    }
    record.update(extra)
    return record


# --- questions --------------------------------------------------------------


def test_question_rate_counts_turns_not_question_marks():
    turns = [
        turn(0, "a", "one? two?"),
        turn(1, "b", "no question here"),
        turn(2, "c", "and?"),
        turn(3, "d", "nothing"),
    ]
    metrics = question_metrics(turns)
    assert metrics["creator_turns"] == 4
    assert metrics["turns_containing_question"] == 2
    assert metrics["question_rate"] == 0.5
    assert metrics["question_marks_per_turn"]["max"] == 2


def test_ending_in_a_question_ignores_trailing_emoji_and_whitespace():
    assert ends_with_question(["so what did you do? 🙈  "]) is True
    assert ends_with_question(["so what did you do?"]) is True
    assert ends_with_question(["a question? then a statement."]) is False
    assert ends_with_question(["", "  "]) is False


def test_the_last_bubble_decides_whether_the_turn_ends_on_a_question():
    """Multipart replies are one turn, and the fan reads the last one last."""
    assert ends_with_question(["do you remember?", "anyway. good night x"]) is False
    assert ends_with_question(["anyway.", "what did you end up doing?"]) is True


def test_the_longest_streak_finds_consecutive_interview_turns():
    turns = [
        turn(0, "a", "how was it?"),
        turn(1, "b", "and then?"),
        turn(2, "c", "that sounds nice."),
        turn(3, "d", "what next?"),
        turn(4, "e", "why?"),
        turn(5, "f", "and after that?"),
    ]
    metrics = question_metrics(turns)
    assert metrics["turns_ending_in_question"] == 5
    assert metrics["longest_question_ending_streak"] == 3


def test_a_silent_turn_is_not_counted_as_a_creator_turn():
    metrics = question_metrics([turn(0, "a"), turn(1, "b", "hi?")])
    assert metrics["creator_turns"] == 1
    assert metrics["question_rate"] == 1.0


# --- repetition -------------------------------------------------------------


def test_a_repeated_five_word_phrase_is_counted_once_per_extra_occurrence():
    turns = [
        turn(0, "a", "i keep thinking about that all day"),
        turn(1, "b", "i keep thinking about that constantly"),
    ]
    metrics = repetition_metrics(turns)
    phrases = {row["phrase"] for row in metrics["top_repeated_ngrams"]}
    assert "i keep thinking about that" in phrases
    assert metrics["repeated_ngram_instances"] >= 1


def test_repeated_opening_fragments_are_counted_separately_from_phrases():
    turns = [
        turn(0, "a", "okay but honestly though the weather was unbelievable"),
        turn(1, "b", "okay but honestly though i did not enjoy any of it"),
        turn(2, "c", "something completely different this time around"),
    ]
    metrics = repetition_metrics(turns)
    assert metrics["repeated_opening_fragments"] == 1
    assert metrics["top_repeated_openings"][0]["fragment"] == "okay but honestly though"


def test_short_acknowledgements_repeating_is_not_a_verbatim_repeat():
    turns = [turn(0, "a", "haha yeah"), turn(1, "b", "haha yeah")]
    assert repetition_metrics(turns)["verbatim_repeats"] == 0


def test_the_same_substantial_reply_twice_is_a_verbatim_repeat():
    line = "that is genuinely the best thing i have heard all week honestly"
    turns = [turn(0, "a", line), turn(1, "b", line)]
    assert repetition_metrics(turns)["verbatim_repeats"] == 1


# --- shape ------------------------------------------------------------------


def test_reply_length_reports_a_distribution_not_only_a_mean():
    turns = [turn(0, "a", "x" * 10), turn(1, "b", "y" * 20), turn(2, "c", "z" * 120)]
    shape = shape_metrics(turns)
    assert shape["reply_chars"]["count"] == 3
    assert shape["reply_chars"]["max"] == 120
    assert shape["reply_chars"]["median"] == 20
    # The variance is why the distribution is reported: three runs can share a
    # mean and not share a rhythm.
    assert shape["reply_chars"]["variance"] > 0


def test_uniform_replies_have_zero_variance():
    turns = [turn(index, "a", "x" * 30) for index in range(4)]
    assert shape_metrics(turns)["reply_chars"]["variance"] == 0.0


def test_bubble_counts_are_reported_as_a_distribution():
    turns = [turn(0, "a", "one"), turn(1, "b", "one", "two"), turn(2, "c", "one", "two")]
    assert shape_metrics(turns)["bubbles_per_turn"] == {"1": 1, "2": 2}


# --- execution --------------------------------------------------------------


def test_errors_handoffs_and_freezes_are_counted_separately():
    turns = [
        turn(0, "a", "hi"),
        turn(1, "b", control={"outcome": "human_review", "handoff": True, "error": ""}),
        turn(2, "c", control={"outcome": "", "error": "RuntimeError: boom"}),
        turn(3, "d", control={"outcome": "no_send", "freeze": True, "error": ""}),
    ]
    metrics = execution_metrics(turns)
    assert metrics["errors"] == 1
    assert metrics["handoffs"] == 1
    assert metrics["freezes"] == 1
    assert metrics["error_rate"] == 0.25


def test_an_operation_that_the_validator_refused_is_an_operation_failure():
    turns = [
        turn(
            0,
            "a",
            "here you go",
            operation_proposal={"kind": "present_offer"},
            operation_result={"approved": False, "executed": None},
        ),
        turn(
            1,
            "b",
            "and here",
            operation_proposal={"kind": "present_offer"},
            operation_result={"approved": True, "executed": True},
        ),
    ]
    metrics = execution_metrics(turns)
    assert metrics["operation_proposals"] == 2
    assert metrics["operation_failures"] == 1
    assert metrics["operation_failure_rate"] == 0.5


def test_tokens_and_cost_report_their_coverage_rather_than_a_bare_total():
    """A total over three of forty turns is not a total.

    ``semantic_v2`` records no usage on the reply path, so most runs will report
    nothing here. Reporting a zero would make a run with no data look like a
    free one.
    """
    turns = [
        turn(0, "a", "hi", tokens={"total": 120}, cost_usd=0.002),
        turn(1, "b", "hi"),
        turn(2, "c", "hi"),
    ]
    metrics = execution_metrics(turns)
    assert metrics["tokens"] == {"total": 120.0, "turns_reporting": 1, "turns": 3}
    assert metrics["cost_usd"]["turns_reporting"] == 1
    assert metrics["cost_usd"]["total"] == 0.002


def test_a_run_with_no_usage_recorded_reports_zero_coverage_not_zero_cost():
    metrics = execution_metrics([turn(0, "a", "hi")])
    assert metrics["cost_usd"] == {"total": 0.0, "turns_reporting": 0, "turns": 1}


def test_latency_reports_the_tail():
    turns = [turn(index, "a", "hi", latency_ms=value) for index, value in enumerate([100, 200, 9000])]
    latency = execution_metrics(turns)["latency_ms"]
    assert latency["median"] == 200
    assert latency["max"] == 9000


# --- models -----------------------------------------------------------------


def test_a_fallback_model_answering_is_visible():
    turns = [
        turn(0, "a", "hi", model_requested="kimi", model_served="kimi"),
        turn(1, "b", "hi", model_requested="kimi", model_served="fallback-model"),
    ]
    metrics = model_metrics(turns)
    assert metrics["served_by_requested_model_rate"] == 0.5
    assert metrics["models_served"] == {"kimi": 1, "fallback-model": 1}


# --- the boundary -----------------------------------------------------------


def test_no_metric_claims_to_measure_a_judgement_dimension():
    metrics = conversation_metrics([turn(0, "a", "hi there")])
    flattened = str(metrics).lower()
    for dimension in NOT_MEASURED_HERE:
        assert f'"{dimension}"' not in flattened
    for banned in ("chemistry", "emotional_intelligence", "quality_score", "overall_score"):
        assert banned not in flattened


def test_the_unmeasured_dimensions_are_named_inside_the_metrics_document():
    """So a reader of metrics.json cannot mistake the counts for the rubric."""
    metrics = scenario_metrics([{"scenario_id": "s", "turns": [turn(0, "a", "hi")]}])
    assert "specificity" in metrics["not_measured_here"]
    assert "contribution" in metrics["not_measured_here"]


def test_pooled_metrics_weigh_turns_rather_than_scenarios():
    """A two-turn scenario must not count as much as a twenty-turn one."""
    long_scenario = {
        "scenario_id": "long",
        "turns": [turn(index, "a", "no question here") for index in range(20)],
    }
    short_scenario = {"scenario_id": "short", "turns": [turn(0, "a", "really?")]}
    metrics = scenario_metrics([long_scenario, short_scenario])
    pooled = metrics["pooled"]["questions"]
    assert pooled["creator_turns"] == 21
    assert pooled["question_rate"] == round(1 / 21, 4)


def test_an_empty_run_produces_zeros_rather_than_raising():
    metrics = conversation_metrics([])
    assert metrics["questions"]["creator_turns"] == 0
    assert metrics["execution"]["turns"] == 0
    assert metrics["shape"]["reply_chars"]["count"] == 0
