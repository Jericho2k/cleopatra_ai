"""Structural immersion criteria for a long, session-aware trajectory.

The product criterion is qualitative: with the media/PPV cards removed, the
remaining dialogue should still read as ONE continuing interaction, and every
media event should be understandable from the conversation before it rather
than appearing as an isolated catalogue operation.

A deterministic scorer cannot judge prose, so this module scores the
structural properties that make that outcome possible and that a
"chain of PPVs" transcript visibly lacks:

* the interaction premise the writer is given persists across turns;
* the turn after a purchase is not another sale;
* every media event was made appropriate by the interaction (a content beat
  planned on an earlier turn, or the fan's own ask) — content never advances
  merely because another candidate exists;
* the transcript without media still has a creator line for every fan line.

The number of conversational turns between content events is reported as a
DIAGNOSTIC only. Natural pacing owns that gap: sometimes many turns pass,
sometimes one is right because the fan redirects or asks. A short gap is never
a failure by itself, so the metric cannot teach filler.

Live-model runs (``scripts/run_trajectory_eval.py --core conversational_v2``)
feed the same records; blind human/LLM review of ``dialogue_without_media``
remains the qualitative judgement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

PAID_OPERATIONS = frozenset({"present_offer", "send_locked_paid_message"})


@dataclass(frozen=True)
class TurnRecord:
    fan_text: str
    creator_text: str
    operation: str = "none"
    premise_given_to_writer: str = ""
    next_move: str = "converse"
    #: This turn's operation targets content a beat planned on an EARLIER turn.
    content_was_planned_earlier: bool = False
    #: The fan himself asked for content / to buy on this turn.
    fan_asked: bool = False
    #: The ledger showed a purchase/delivery that happened since the last turn.
    follows_content_event: bool = False


@dataclass
class ImmersionReport:
    turns: int
    media_events: int
    #: Diagnostic only; never a failure (natural pacing owns the gap).
    min_conversational_turns_between_media: int | None
    post_event_sales: int
    premise_continuity: float
    unexplained_media_events: int
    dialogue_without_media: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    #: Observations for analysis (e.g. short gaps between content events).
    diagnostics: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures

    def as_dict(self) -> dict[str, Any]:
        return {
            "turns": self.turns,
            "media_events": self.media_events,
            "min_conversational_turns_between_media": (
                self.min_conversational_turns_between_media
            ),
            "post_event_sales": self.post_event_sales,
            "premise_continuity": round(self.premise_continuity, 3),
            "unexplained_media_events": self.unexplained_media_events,
            "failures": list(self.failures),
            "diagnostics": list(self.diagnostics),
            "passed": self.passed,
        }


def score_session_trajectory(
    records: list[TurnRecord],
    *,
    diagnostic_gap: int = 2,
    min_premise_continuity: float = 0.9,
) -> ImmersionReport:
    media_indexes = [
        index for index, row in enumerate(records) if row.operation in PAID_OPERATIONS
    ]
    gaps = [
        later - earlier - 1 for earlier, later in zip(media_indexes, media_indexes[1:])
    ]
    # Only count gaps between DIFFERENT content events; an offer followed by the
    # delivery the fan accepted is one event in two steps.
    event_gaps = [
        gap
        for gap, (earlier, later) in zip(gaps, zip(media_indexes, media_indexes[1:]))
        if not (
            records[earlier].operation == "present_offer"
            and records[later].operation == "send_locked_paid_message"
        )
    ]
    post_event_sales = sum(
        1
        for row in records
        if row.follows_content_event
        and row.operation in PAID_OPERATIONS
        and not row.fan_asked
    )
    premised = [row for row in records if row.creator_text]
    anchor = next(
        (row.premise_given_to_writer for row in premised if row.premise_given_to_writer),
        "",
    )
    continuity = (
        sum(1 for row in premised if anchor and row.premise_given_to_writer)
        / len(premised)
        if premised
        else 0.0
    )
    unexplained = sum(
        1
        for index in media_indexes
        if not (records[index].content_was_planned_earlier or records[index].fan_asked)
    )
    dialogue = [
        line
        for row in records
        for line in (row.fan_text, row.creator_text)
        if line
    ]
    report = ImmersionReport(
        turns=len(records),
        media_events=len(media_indexes),
        min_conversational_turns_between_media=min(event_gaps) if event_gaps else None,
        post_event_sales=post_event_sales,
        premise_continuity=continuity,
        unexplained_media_events=unexplained,
        dialogue_without_media=dialogue,
    )
    short = sum(1 for gap in event_gaps if gap < diagnostic_gap)
    if short:
        # Not a failure: a fan who redirects or asks can make the very next
        # turn the right one. Kept so a live run can be inspected for chains.
        report.diagnostics.append(
            f"{short} content event(s) followed another within fewer than "
            f"{diagnostic_gap} conversational turns"
        )
    if post_event_sales:
        report.failures.append("a purchase or delivery was followed straight by a sale")
    if continuity < min_premise_continuity:
        report.failures.append("the interaction premise did not persist across turns")
    if unexplained:
        report.failures.append(
            "a content event advanced without the interaction making it "
            "appropriate (not planned earlier, not asked for)"
        )
    if any(row.fan_text and not row.creator_text for row in records):
        report.failures.append("a fan line has no creator line once media is removed")
    return report
