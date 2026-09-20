"""Countable properties of a finished conversation, and nothing more.

This is the objective half of the Conversational Core v1 comparison. Everything
here is arithmetic over text that was actually sent and over records the
pipeline actually wrote: how often a reply ended in a question, how long the
replies were, which phrases came back, what the turn cost, how often the turn
failed.

WHAT THIS DELIBERATELY DOES NOT MEASURE
---------------------------------------
The rubric the comparison exists to serve names specificity, contribution,
initiative fit, shared-scene continuity, reference handling, direction-change
adaptation and pacing. None of those is a regex. A phrase counter that claimed
to score "contribution" would be measuring its own author's taste and would
carry the authority of a number while doing it, which is worse than not
measuring it at all.

So the dimensions that need judgement live in the blind review
(``services/blind_conversation_review.py``) and, optionally, in a model judge
that is explicitly not a scoreboard. What comes out of THIS module is the
countable layer underneath, and :data:`NOT_MEASURED_HERE` is emitted inside the
metrics document itself so a reader of ``metrics.json`` cannot mistake one for
the other.

The one borderline entry is ``question_discipline``. Rubric item 4 is about
*forced* questions, which needs a person; the counts here — the share of creator
turns ending in a question, the longest consecutive streak — are the evidence a
person reads to answer it, and are named as counts rather than as a verdict.
"""

from __future__ import annotations

import re
import statistics
from collections import Counter
from typing import Any, Iterable, Sequence

#: Named in the metrics document. A dimension listed here is one no number in
#: this module establishes, in either direction.
NOT_MEASURED_HERE: tuple[str, ...] = (
    "specificity",
    "contribution",
    "initiative_fit",
    "shared_scene_continuity",
    "reference_handling",
    "direction_change_adaptation",
    "pacing_naturalness",
    "truthful_presence",
    "commercial_naturalness",
    "post_event_continuation",
    "overall_conversational_coherence",
)

#: Length of the word shingle used for repeated-phrase detection. Five words is
#: long enough that ordinary English collocations ("i want to see more") do not
#: dominate the result and short enough to catch a reused sentence opening.
REPEAT_NGRAM_WORDS = 5

#: How many words of a reply's first bubble count as its opening fragment.
OPENING_FRAGMENT_WORDS = 4

_WORD = re.compile(r"[a-z0-9']+")


def _words(text: str) -> list[str]:
    return _WORD.findall(str(text or "").lower())


def _creator_text(turn: dict[str, Any]) -> list[str]:
    return [str(part) for part in (turn.get("creator_output") or []) if str(part).strip()]


def _spoke(turn: dict[str, Any]) -> bool:
    return bool(_creator_text(turn))


def ends_with_question(bubbles: Sequence[str]) -> bool:
    """Whether the last thing the fan reads is a question.

    Trailing emoji and whitespace are stripped first: "so what did you do? 🙈"
    ends in a question by every reading except a naive ``endswith('?')``, and
    the count exists to describe what the conversation felt like.
    """
    for bubble in reversed(list(bubbles)):
        text = str(bubble or "").strip()
        if not text:
            continue
        trimmed = text.rstrip()
        while trimmed and not (trimmed[-1].isalnum() or trimmed[-1] in ".!?…"):
            trimmed = trimmed[:-1].rstrip()
        return trimmed.endswith("?")
    return False


def _distribution(values: Sequence[float]) -> dict[str, float]:
    """Mean, median, p90, max and variance, or zeros for an empty run.

    Variance is reported next to the mean because a run of identically shaped
    replies and a run that varies are the same average and are not the same
    conversation. Population variance, not sample: this is the whole run, not a
    sample drawn from one.
    """
    if not values:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p90": 0.0, "max": 0.0, "variance": 0.0}
    ordered = sorted(float(value) for value in values)
    index = max(0, min(len(ordered) - 1, int(round(0.9 * (len(ordered) - 1)))))
    return {
        "count": len(ordered),
        "mean": round(statistics.fmean(ordered), 3),
        "median": round(statistics.median(ordered), 3),
        "p90": round(ordered[index], 3),
        "max": round(ordered[-1], 3),
        "variance": round(statistics.pvariance(ordered), 3) if len(ordered) > 1 else 0.0,
    }


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def question_metrics(turns: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """How often this run asked, and how long it kept asking.

    Three separate numbers because they answer three different questions, and
    the one that reads as interview behaviour is the streak: a conversation
    where every reply in a row ends on a question mark is the interrogation
    pattern, and a run with the same overall rate spread out is not.
    """
    spoken = [turn for turn in turns if _spoke(turn)]
    contains = 0
    ending = 0
    streak = 0
    longest = 0
    marks: list[float] = []
    for turn in spoken:
        bubbles = _creator_text(turn)
        joined = " ".join(bubbles)
        marks.append(joined.count("?"))
        if "?" in joined:
            contains += 1
        if ends_with_question(bubbles):
            ending += 1
            streak += 1
            longest = max(longest, streak)
        else:
            streak = 0
    return {
        "creator_turns": len(spoken),
        "turns_containing_question": contains,
        "question_rate": _rate(contains, len(spoken)),
        "turns_ending_in_question": ending,
        "question_ending_rate": _rate(ending, len(spoken)),
        "longest_question_ending_streak": longest,
        "question_marks_per_turn": _distribution(marks),
    }


def repetition_metrics(turns: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Phrases and openings this run reused, counted on the text it sent.

    Three kinds of repetition, kept apart because they mean different things.
    A repeated five-word shingle is a habit of phrasing; a repeated opening
    fragment is the "so, " tic that makes every reply start the same way; a
    verbatim whole reply is the failure mode the supplied excerpts showed.
    """
    shingles: Counter[str] = Counter()
    openings: Counter[str] = Counter()
    whole: Counter[str] = Counter()
    for turn in turns:
        bubbles = _creator_text(turn)
        if not bubbles:
            continue
        words = _words(" ".join(bubbles))
        for start in range(0, max(0, len(words) - REPEAT_NGRAM_WORDS + 1)):
            shingles[" ".join(words[start : start + REPEAT_NGRAM_WORDS])] += 1
        first = _words(bubbles[0])[:OPENING_FRAGMENT_WORDS]
        if first:
            openings[" ".join(first)] += 1
        for bubble in bubbles:
            normalized = " ".join(str(bubble).lower().split())
            # Short acknowledgements repeat in every real conversation. Only a
            # substantial reply arriving twice is the finding.
            if len(normalized) >= 25:
                whole[normalized] += 1
    repeated_shingles = {text: count for text, count in shingles.items() if count > 1}
    repeated_openings = {text: count for text, count in openings.items() if count > 1}
    repeated_whole = sum(count - 1 for count in whole.values() if count > 1)
    return {
        "repeated_ngrams": len(repeated_shingles),
        "repeated_ngram_instances": sum(count - 1 for count in repeated_shingles.values()),
        "top_repeated_ngrams": [
            {"phrase": text, "count": count}
            for text, count in sorted(
                repeated_shingles.items(), key=lambda item: (-item[1], item[0])
            )[:10]
        ],
        "repeated_opening_fragments": len(repeated_openings),
        "top_repeated_openings": [
            {"fragment": text, "count": count}
            for text, count in sorted(
                repeated_openings.items(), key=lambda item: (-item[1], item[0])
            )[:10]
        ],
        "verbatim_repeats": repeated_whole,
    }


def shape_metrics(turns: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Reply length and bubble count, as distributions rather than averages.

    The distribution is the point. "Every reply is 180 characters in two
    bubbles" and "replies run from a word to a paragraph" have the same mean,
    and only one of them reads like a person.
    """
    chars: list[float] = []
    words: list[float] = []
    bubbles: Counter[int] = Counter()
    for turn in turns:
        parts = _creator_text(turn)
        if not parts:
            continue
        joined = " ".join(parts)
        chars.append(len(joined))
        words.append(len(_words(joined)))
        bubbles[len(parts)] += 1
    return {
        "reply_chars": _distribution(chars),
        "reply_words": _distribution(words),
        "bubbles_per_turn": {str(count): total for count, total in sorted(bubbles.items())},
    }


def _sum_optional(turns: Sequence[dict[str, Any]], *path: str) -> tuple[float, int]:
    """Total a field that may simply not be recorded, with its coverage.

    Returns ``(total, turns_that_reported)``. The coverage count is returned
    rather than dropped because a cost total over three of forty turns is not a
    cost total, and a report that prints the number alone invites it to be read
    as one.
    """
    total = 0.0
    seen = 0
    for turn in turns:
        cursor: Any = turn
        for key in path:
            if not isinstance(cursor, dict):
                cursor = None
                break
            cursor = cursor.get(key)
        if isinstance(cursor, (int, float)) and not isinstance(cursor, bool):
            total += float(cursor)
            seen += 1
    return round(total, 6), seen


def execution_metrics(turns: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """What the runtime did, from the control record rather than from prose.

    Errors, handoffs, freezes and refused operations are executions, not
    opinions, so they are counted here and never offset by anything the review
    says about the writing.
    """
    total = len(turns)
    errors = 0
    handoffs = 0
    freezes = 0
    silences = 0
    holds = 0
    proposals = 0
    failed_operations = 0
    unsupported_claims = 0
    for turn in turns:
        control = turn.get("control") or {}
        if control.get("error"):
            errors += 1
        if control.get("handoff"):
            handoffs += 1
        if control.get("freeze"):
            freezes += 1
        if control.get("hold") and str(control.get("hold")) not in {"", "none"}:
            holds += 1
        if not _spoke(turn) and not control.get("error"):
            silences += 1
        operation = turn.get("operation_proposal") or {}
        kind = str(operation.get("kind") or "none")
        if kind and kind != "none":
            proposals += 1
            result = turn.get("operation_result") or {}
            if result.get("executed") is False or result.get("approved") is False:
                failed_operations += 1
        unsupported_claims += len(turn.get("unsupported_claims") or [])
    latency = _distribution([float(turn.get("latency_ms") or 0) for turn in turns])
    tokens_total, tokens_seen = _sum_optional(turns, "tokens", "total")
    cost_total, cost_seen = _sum_optional(turns, "cost_usd")
    return {
        "turns": total,
        "error_rate": _rate(errors, total),
        "errors": errors,
        "handoff_rate": _rate(handoffs, total),
        "handoffs": handoffs,
        "freeze_rate": _rate(freezes, total),
        "freezes": freezes,
        "hold_turns": holds,
        "silent_turns": silences,
        "operation_proposals": proposals,
        "operation_failures": failed_operations,
        "operation_failure_rate": _rate(failed_operations, proposals),
        "unsupported_claim_findings": unsupported_claims,
        "latency_ms": latency,
        # Reported with their coverage, because a total over a subset of turns
        # is not a total. A run where nothing recorded usage says so.
        "tokens": {"total": tokens_total, "turns_reporting": tokens_seen, "turns": total},
        "cost_usd": {"total": cost_total, "turns_reporting": cost_seen, "turns": total},
    }


def model_metrics(turns: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Which models actually answered, and whether the requested one did.

    A comparison of two architectures is only a comparison of two architectures
    if the same model answered on both sides. When half the replies came from a
    fallback, this is where that shows.
    """
    served: Counter[str] = Counter()
    requested: Counter[str] = Counter()
    as_requested = 0
    recorded = 0
    for turn in turns:
        model = str(turn.get("model_served") or "")
        want = str(turn.get("model_requested") or "")
        if model:
            served[model] += 1
        if want:
            requested[want] += 1
        if model and want:
            recorded += 1
            if model == want:
                as_requested += 1
    return {
        "models_served": dict(served),
        "models_requested": dict(requested),
        "served_by_requested_model_rate": _rate(as_requested, recorded),
        "turns_with_model_recorded": recorded,
    }


def conversation_metrics(turns: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Every objective measurement for one conversation."""
    rows = [turn for turn in turns if isinstance(turn, dict)]
    return {
        "questions": question_metrics(rows),
        "repetition": repetition_metrics(rows),
        "shape": shape_metrics(rows),
        "execution": execution_metrics(rows),
        "models": model_metrics(rows),
    }


def scenario_metrics(conversations: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Metrics per scenario for one arm, plus the same metrics pooled.

    Pooled numbers are computed over the concatenated turns rather than by
    averaging per-scenario rates, so a two-turn scenario cannot weigh as much as
    a forty-turn one.
    """
    per_scenario: dict[str, Any] = {}
    pooled: list[dict[str, Any]] = []
    for conversation in conversations:
        scenario_id = str(conversation.get("scenario_id") or conversation.get("name") or "")
        turns = list(conversation.get("turns") or [])
        per_scenario[scenario_id] = conversation_metrics(turns)
        pooled.extend(turns)
    return {
        "per_scenario": per_scenario,
        "pooled": conversation_metrics(pooled),
        "not_measured_here": list(NOT_MEASURED_HERE),
    }


__all__ = [
    "NOT_MEASURED_HERE",
    "conversation_metrics",
    "ends_with_question",
    "execution_metrics",
    "model_metrics",
    "question_metrics",
    "repetition_metrics",
    "scenario_metrics",
    "shape_metrics",
]
