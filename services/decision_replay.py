"""Comparing two ways of deciding, on identical evidence, offline.

``docs/autonomy_architecture_review.md`` §6.4 and §5:

    Keep the executor fixed; compare the current controller stack with a single
    semantic decision owner. Change model routing separately so results remain
    attributable.

    Use both fixed-prefix replay and adaptive trajectories. Replay gives
    candidates the same evidence; adaptive trajectories expose the consequences
    of earlier choices.

This is the fixed-prefix half. Every candidate is handed the same
``ContextPacket`` built by the same builder, the same operational facts, and the
same deterministic checks afterwards. Nothing here sends a message, touches the
database or advances any state: a replay must be repeatable, and a comparison
that mutated the thing it measured would be neither.

What it reports, and why each part:

**Disagreements.** Where two owners would do different things with the same
evidence. This is the actual output — §4's instruction is to compare, and the
interesting result is the set of turns where the answer differs, not a score.

**Critical failures, counted separately.** §5 is explicit: *"Count critical
execution failures separately; a high average prose score cannot cancel an
unauthorized transaction."* A decision that proposes an operation it is not
permitted, names a price, or claims something already happened is a critical
failure of that candidate on that turn, and it is reported as a count rather
than folded into anything.

**Missed obligations.** A turn's open threads are known before either owner
runs, so "this candidate would not have addressed the thing he asked about" is
measurable rather than a matter of reading the prose afterwards.

The one thing this deliberately does NOT do is decide which candidate is
better. §4: *"Select it only if complete conversation evaluation establishes a
benefit."* A disagreement count is evidence for a human reading the
disagreements, not a verdict, and the review is explicit that a model grading
its own text is not a substitute for expert review.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Sequence

from models.conversation_decision import (
    ConversationDecision,
    OperationKind,
    deterministic_violations,
)
from services.context_packet import ContextPacket, build_context_packet


@dataclass(frozen=True)
class ReplayTurn:
    """One point in a conversation, with everything a decision may rest on.

    Built from recorded or authored material, never from live state. ``history``
    is the prefix up to and including the message being answered, which is what
    makes this a fixed-prefix replay: every candidate sees the same past and
    none of them can change it.
    """

    name: str
    history: tuple[Any, ...] = ()
    open_threads: tuple[str, ...] = ()
    episodes: tuple[str, ...] = ()
    #: What deterministic code has already established: the analyzer's
    #: situation, the commercial decision, whether the fan is frozen. Handed to
    #: every candidate identically.
    state: dict[str, Any] = field(default_factory=dict)
    #: Which operations this turn is permitted at all. The executor's authority,
    #: held fixed across candidates — that is what "keep the executor fixed"
    #: means in practice.
    authorized_operations: frozenset[OperationKind] = frozenset()
    #: What actually exists for this turn, so an invented subject is detectable.
    known_subjects: frozenset[str] = frozenset()
    #: What a good operator would have had to address, when the scenario asserts
    #: one. Optional: most turns have no single right answer, and pretending
    #: otherwise is how an evaluation starts measuring its own assumptions.
    expected_addresses: tuple[str, ...] = ()

    def packet(self) -> ContextPacket:
        return build_context_packet(
            self.history,
            open_threads=self.open_threads,
            episodes=self.episodes,
        )


@dataclass
class TurnComparison:
    """What each candidate decided about one turn, and what was wrong with it."""

    turn: str
    decisions: dict[str, ConversationDecision] = field(default_factory=dict)
    violations: dict[str, list[str]] = field(default_factory=dict)
    missed_obligations: dict[str, list[str]] = field(default_factory=dict)
    disagreements: list[str] = field(default_factory=list)

    @property
    def agreed(self) -> bool:
        return not self.disagreements

    def as_dict(self) -> dict[str, Any]:
        return {
            "turn": self.turn,
            "agreed": self.agreed,
            "disagreements": list(self.disagreements),
            "decisions": {
                name: decision.as_dict() for name, decision in self.decisions.items()
            },
            "violations": {name: list(v) for name, v in self.violations.items() if v},
            "missed_obligations": {
                name: list(v) for name, v in self.missed_obligations.items() if v
            },
        }


@dataclass
class ReplayReport:
    """The whole comparison. Evidence for a person, not a verdict."""

    comparisons: list[TurnComparison] = field(default_factory=list)
    candidates: tuple[str, ...] = ()

    @property
    def turns(self) -> int:
        return len(self.comparisons)

    @property
    def disagreed(self) -> int:
        return sum(1 for comparison in self.comparisons if not comparison.agreed)

    def critical_failures(self, candidate: str) -> int:
        """Turns where this candidate produced a decision that must be refused.

        Counted separately from everything else, as §5 requires. One of these
        is not offset by any number of agreeable turns.
        """
        return sum(
            1
            for comparison in self.comparisons
            if comparison.violations.get(candidate)
        )

    def missed_obligations(self, candidate: str) -> int:
        return sum(
            1
            for comparison in self.comparisons
            if comparison.missed_obligations.get(candidate)
        )

    def summary(self) -> dict[str, Any]:
        return {
            "turns": self.turns,
            "candidates": list(self.candidates),
            "disagreed_turns": self.disagreed,
            "per_candidate": {
                candidate: {
                    "critical_failures": self.critical_failures(candidate),
                    "missed_obligations": self.missed_obligations(candidate),
                }
                for candidate in self.candidates
            },
        }

    def render(self) -> str:
        """A report a person reads, with the disagreements spelled out.

        Deliberately not a leaderboard. The number that matters is zero critical
        failures; after that, what matters is reading the turns where the two
        candidates would have done different things.
        """
        summary = self.summary()
        lines = [
            f"{summary['turns']} turns, {len(self.candidates)} candidates: "
            f"{', '.join(self.candidates)}",
            f"disagreed on {summary['disagreed_turns']} of {summary['turns']} turns",
            "",
        ]
        for candidate in self.candidates:
            stats = summary["per_candidate"][candidate]
            lines.append(
                f"  {candidate}: {stats['critical_failures']} critical failure(s), "
                f"{stats['missed_obligations']} turn(s) with a missed obligation"
            )
        lines.append("")
        for comparison in self.comparisons:
            if comparison.agreed and not any(comparison.violations.values()):
                continue
            lines.append(f"--- {comparison.turn}")
            for difference in comparison.disagreements:
                lines.append(f"    differs: {difference}")
            for candidate, problems in comparison.violations.items():
                for problem in problems:
                    lines.append(f"    CRITICAL [{candidate}]: {problem}")
            for candidate, missed in comparison.missed_obligations.items():
                for obligation in missed:
                    lines.append(f"    missed [{candidate}]: {obligation}")
        if len(lines) == 5 + len(self.candidates):
            lines.append("every candidate agreed on every turn, with no violations.")
        return "\n".join(lines)


async def compare_owners(
    turns: Sequence[ReplayTurn], owners: Sequence[Any]
) -> ReplayReport:
    """Run every candidate over every turn, on identical evidence.

    Candidates are run sequentially per turn rather than concurrently. A
    semantic owner is a model call and a projection is not; interleaving them
    would make the slow one's latency depend on the fast one's, and the review
    asks for latency to be recorded rather than accidentally measured.
    """
    report = ReplayReport(candidates=tuple(owner.name for owner in owners))

    for turn in turns:
        packet = turn.packet()
        comparison = TurnComparison(turn=turn.name)

        for owner in owners:
            decision = await owner.decide(packet, dict(turn.state))
            comparison.decisions[owner.name] = decision
            comparison.violations[owner.name] = deterministic_violations(
                decision,
                authorized_operations=turn.authorized_operations,
                known_subjects=turn.known_subjects,
            )
            # An obligation the scenario says had to be addressed, that this
            # candidate did not carry. Known before either ran, so it is a fact
            # about the decision rather than a reading of prose.
            required = set(turn.expected_addresses) or set(packet.open_threads)
            comparison.missed_obligations[owner.name] = sorted(
                required - set(decision.must_address)
            )

        names = list(comparison.decisions)
        for i, left in enumerate(names):
            for right in names[i + 1 :]:
                comparison.disagreements.extend(
                    comparison.decisions[left].disagreement_with(
                        comparison.decisions[right]
                    )
                )
        report.comparisons.append(comparison)

    return report


def compare_owners_sync(
    turns: Sequence[ReplayTurn], owners: Sequence[Any]
) -> ReplayReport:
    """The same thing, for a script or a test that has no event loop."""
    return asyncio.run(compare_owners(turns, owners))


def load_turns(scenarios: Sequence[dict[str, Any]]) -> list[ReplayTurn]:
    """Build turns from the JSON scenario format.

    Tolerant by design: a scenario file is authored by hand, and a missing
    optional key should produce a turn with less in it rather than a traceback
    halfway through a replay.
    """
    turns: list[ReplayTurn] = []
    for raw in scenarios:
        if not isinstance(raw, dict):
            continue
        authorized = frozenset(
            OperationKind(value)
            for value in (raw.get("authorized_operations") or [])
            if value in {kind.value for kind in OperationKind}
        )
        turns.append(
            ReplayTurn(
                name=str(raw.get("name") or f"turn-{len(turns) + 1}"),
                history=tuple(raw.get("history") or ()),
                open_threads=tuple(raw.get("open_threads") or ()),
                episodes=tuple(raw.get("episodes") or ()),
                state=dict(raw.get("state") or {}),
                authorized_operations=authorized,
                known_subjects=frozenset(
                    str(value) for value in (raw.get("known_subjects") or ())
                ),
                expected_addresses=tuple(raw.get("expected_addresses") or ()),
            )
        )
    return turns
