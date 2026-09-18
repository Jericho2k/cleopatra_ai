#!/usr/bin/env python3
"""Run whole conversations through the real Full Auto turn, and report what happened.

``docs/autonomy_architecture_review.md`` §5 and §6.5. This is the longitudinal
half: ``scripts/run_model_eval.py`` compares replies, and the review says that
"does not run the complete autonomous orchestration, real state transitions,
delivery, or weeks of interaction". This drives the simulator, which does.

    # Against a real backend, using owner-only simulation test fans.
    python scripts/run_trajectory_eval.py --creator <creator-id> --fan <test-fan-id>

    # Machine-readable.
    python scripts/run_trajectory_eval.py --creator ... --fan ... --json

    # Just check the trajectory file loads and says what it covers.
    python scripts/run_trajectory_eval.py --describe

**It produces no quality score, on purpose.** §5: *"A scripted cooperative
customer and a model grading its own text are insufficient substitutes for
expert human review."* What comes out is the deterministic execution failures,
counted on their own so nothing can offset them; the countable behaviours worth
a person's attention; latency including the tail; which models actually
answered; and the transcript. A human reads the rest.

**The fan must be an owner simulation test fan** (``platform_fan_id`` starting
``test_``). The simulator refuses every remote call regardless, but running a
longitudinal evaluation against a real customer's conversation would write real
state into it, and no report is worth that.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import clock  # noqa: E402
from services.trajectory_fixtures import (  # noqa: E402
    FixtureRefused,
    apply_seed,
    clear_seeded,
)
from services.trajectory_eval import (  # noqa: E402
    TrajectoryReport,
    TurnRecord,
    coverage_gaps,
    load_trajectories,
    render_reports,
    run_trajectory,
)


def _declared_gaps(trajectory, *, use_clock: bool = False):
    """What this trajectory could not cover even if every turn succeeded.

    Built by asking the same function the real run uses, against a report
    describing the best case: every declared turn runs, every declared seed is
    applied, a due-work cycle happens for each unprompted turn, and time
    advances only if this invocation would advance it. One definition of
    "covered", used by both paths, so --describe cannot drift from what a run
    actually reports.

    ``use_clock`` mirrors --simulate-time, because whether the elapsed-time
    claims are covered is a property of the invocation and not of the file.
    """
    elapsed = sum(
        float(disturbance.days_since_previous or 0.0)
        for disturbance in trajectory.disturbances
    )
    best_case = TrajectoryReport(
        trajectory=trajectory.name,
        covers=trajectory.covers,
        turns=[
            TurnRecord(index=index, customer_message=disturbance.message)
            for index, disturbance in enumerate(trajectory.disturbances)
        ],
        due_worker_runs=sum(
            1
            for disturbance in trajectory.disturbances
            if not str(disturbance.message or "").strip()
        ),
        clock_injected=use_clock,
        elapsed_days=elapsed if use_clock else 0.0,
        # The seed has not been applied — nothing has run — but the file
        # declares it, and describing a claim as uncovered when the very next
        # run would cover it is the same class of wrong answer as the labels
        # this whole mechanism exists to fix.
        seeded_purchases=list(trajectory.seed.get("purchases") or []),
    )
    return coverage_gaps(trajectory, best_case)


DEFAULT_TRAJECTORIES = ROOT / "eval" / "trajectories.json"


def _load(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return list(payload.get("trajectories") or [])
    return list(payload or [])


async def _run_due_work(creator_id: str, fan_id: str) -> dict:
    """Run one due-work cycle and report what the customer received from it.

    A turn with no customer message is not "nothing happens". It is the moment
    a queued follow-up becomes due, which is precisely what the goodbye
    trajectory exists to test: a customer asked for no follow-up, and something
    was already scheduled.

    This used to return ``{"outcome": "no_message"}`` without running anything,
    so that trajectory could not detect the behaviour it claimed to cover — the
    one turn that mattered was the one guaranteed to do nothing.

    Messages the cycle sends are read back off the conversation, because the
    worker delivers them rather than returning them.
    """
    from services.suggestions import _recent_creator_message_rows
    from workers.scheduled_actions import process_cycle

    # The same reader the simulated turn uses, so a message the worker sent and
    # a message a turn sent are identified the same way — by row id, which is
    # what keeps multipart replies ordered and complete.
    before = {str(row.get("id")) for row in await _recent_creator_message_rows(fan_id)}
    result = await process_cycle(limit=20)
    fresh = [
        row
        for row in await _recent_creator_message_rows(fan_id)
        if str(row.get("id")) not in before
    ]
    return {
        # Named for what it is. The turn ran queued work; whether that work
        # sent anything is the finding, not the label.
        "outcome": "due_work_ran" if fresh else "due_work_sent_nothing",
        "creator_messages": fresh,
        "due_worker_ran": True,
        "due_work": {
            "claimed": getattr(result, "claimed", 0),
            "processed": getattr(result, "processed", 0),
            "errors": getattr(result, "errors", 0),
        },
    }


async def _run_all(trajectories, creator_id: str, fan_id: str, *, use_clock: bool) -> list:
    from services.suggestions import run_simulated_inbound

    # None when the clock is not enabled, which run_trajectory records as "no
    # clock was injected" and the coverage check reports as an uncovered
    # elapsed-time claim. That is the honest outcome: refusing to advance is
    # not the same as a week having passed, and this harness used to report
    # them identically.
    advance_clock = clock.advance if use_clock else None

    reports = []
    for trajectory in trajectories:
        creator = trajectory.creator_id or creator_id
        fan = trajectory.fan_id or fan_id

        async def send_turn(message: str, _creator=creator, _fan=fan) -> dict:
            if not message.strip():
                return await _run_due_work(_creator, _fan)
            return await run_simulated_inbound(
                fan_id=_fan, creator_id=_creator, message=message, fast=True
            )

        # Fresh state between independent scenarios. A clock carried over
        # from the previous trajectory is state, and so is a purchase the last
        # one seeded — the brief asks for scenarios not to inherit each
        # other's, and the next trajectory would read a leftover delivery as
        # a real one.
        clock.reset()
        seeded: list[dict] = []
        try:
            await clear_seeded(creator, fan)
            if trajectory.seed:
                seeded = await apply_seed(
                    creator_id=creator, fan_id=fan, seed=trajectory.seed
                )
        except FixtureRefused as refused:
            # Refusing to seed is never a reason to fabricate the state
            # anyway, and never a reason to run the trajectory as though the
            # state were there. Say so and move on; the coverage check reports
            # the purchase claim as uncovered.
            print(f"[FIXTURE] {trajectory.name}: {refused}", file=sys.stderr)

        report = await run_trajectory(
            trajectory, send_turn=send_turn, advance_clock=advance_clock
        )
        report.seeded_purchases = seeded
        # Recomputed: the seed is what decides whether the purchase claim is
        # covered, and run_trajectory could not know about it.
        report.coverage_gaps = coverage_gaps(trajectory, report)
        reports.append(report)

    clock.reset()
    return reports


async def _run_with_selected_core(
    trajectories,
    creator_id: str,
    fan_id: str,
    *,
    use_clock: bool,
    core_id: str,
) -> list:
    """Temporarily pin the evaluation fan and restore its prior selection."""
    if not core_id:
        return await _run_all(
            trajectories,
            creator_id,
            fan_id,
            use_clock=use_clock,
        )
    if any(
        trajectory.fan_id and trajectory.fan_id != fan_id
        for trajectory in trajectories
    ):
        raise FixtureRefused(
            "--core requires every trajectory to use the --fan test fan"
        )

    from db.queries import get_fan_by_id
    from services.conversation_core import (
        set_simulation_fan_core_override,
        simulation_fan_core_override,
    )

    fan = await get_fan_by_id(fan_id)
    if fan is None or not str(fan.platform_fan_id or "").startswith("test_"):
        raise FixtureRefused("--core may select only an owner simulation test fan")
    previous = await simulation_fan_core_override(fan_id)
    await set_simulation_fan_core_override(fan_id, core_id)
    print(
        f"[EVAL CORE] fan={fan_id} selected={core_id} "
        f"rollback={previous or 'inherit'}",
        file=sys.stderr,
    )
    try:
        return await _run_all(
            trajectories,
            creator_id,
            fan_id,
            use_clock=use_clock,
        )
    finally:
        await set_simulation_fan_core_override(fan_id, previous)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trajectories", type=Path, default=DEFAULT_TRAJECTORIES,
        help=f"trajectory file (default: {DEFAULT_TRAJECTORIES.relative_to(ROOT)})",
    )
    parser.add_argument("--creator", default="", help="creator id to run against")
    parser.add_argument(
        "--fan", default="", help="owner simulation test fan id (platform_fan_id test_*)"
    )
    parser.add_argument(
        "--describe",
        action="store_true",
        help="list what each trajectory covers and exit, without running anything",
    )
    parser.add_argument("--json", action="store_true", help="emit reports as JSON")
    parser.add_argument(
        "--fail-on-critical",
        action="store_true",
        help="exit non-zero if any trajectory produced a critical execution failure",
    )
    parser.add_argument(
        "--simulate-time",
        action="store_true",
        help=(
            "advance a simulated clock for days_since_previous, so a return "
            "after a day or a week is actually tested. Requires APP_ENV != "
            "production and EVAL_CLOCK_ENABLED=1 (core/clock.py); without it "
            "the elapsed-time claims are reported as uncovered rather than "
            "quietly assumed"
        ),
    )
    parser.add_argument(
        "--fail-on-uncovered",
        action="store_true",
        help=(
            "exit non-zero if any trajectory did not cover what it claims. "
            "Separate from --fail-on-critical: an uncovered claim is not a "
            "system failure, it is a run that did not test what it says"
        ),
    )
    parser.add_argument(
        "--core",
        choices=("legacy", "semantic_v1"),
        default="",
        help=(
            "temporarily pin the test fan to this conversational runtime for "
            "the complete run, then restore its previous override"
        ),
    )
    args = parser.parse_args()

    trajectories = load_trajectories(_load(args.trajectories))
    if not trajectories:
        print(f"no trajectories in {args.trajectories}", file=sys.stderr)
        return 2

    if args.describe:
        for trajectory in trajectories:
            print(f"{trajectory.name} ({len(trajectory.disturbances)} turns)")
            print(f"  covers: {trajectory.covers}")
            # The claim next to what the fixture can actually reach. Printing
            # the claim alone is how "40-80 turns of ordinary conversation"
            # came to sit above a seven-turn script for as long as it did.
            for gap in _declared_gaps(trajectory, use_clock=args.simulate_time):
                print(f"  {gap.render()}")
            for index, disturbance in enumerate(trajectory.disturbances):
                marks = []
                if disturbance.days_since_previous:
                    marks.append(f"+{disturbance.days_since_previous:g}d")
                if disturbance.asks_for_silence:
                    marks.append("asks for silence")
                if disturbance.corrects:
                    marks.append(f"corrects {disturbance.corrects!r}")
                suffix = f" [{', '.join(marks)}]" if marks else ""
                print(f"    {index}. {disturbance.tests}{suffix}")
            print()
        return 0

    if not (args.creator and args.fan):
        print(
            "--creator and --fan are required to run. Use --describe to inspect "
            "the trajectories without a backend.",
            file=sys.stderr,
        )
        return 2

    if args.simulate_time and not clock.movable():
        print(
            "--simulate-time was asked for and this process cannot move its "
            f"clock. It needs APP_ENV != production and {clock.EVAL_CLOCK_FLAG}=1.",
            file=sys.stderr,
        )
        return 2

    try:
        reports = asyncio.run(
            _run_with_selected_core(
                trajectories,
                args.creator,
                args.fan,
                use_clock=args.simulate_time,
                core_id=args.core,
            )
        )
    except FixtureRefused as refused:
        print(f"evaluation refused: {refused}", file=sys.stderr)
        return 2

    if args.json:
        print(
            json.dumps(
                [
                    {
                        "summary": report.summary(),
                        "coverage_gaps": [
                            {
                                "claim": gap.claim,
                                "required": gap.required,
                                "actual": gap.actual,
                            }
                            for gap in report.coverage_gaps
                        ],
                        "findings": [
                            {
                                "severity": finding.severity.value,
                                "kind": finding.kind,
                                "detail": finding.detail,
                                "turn": finding.turn,
                            }
                            for finding in report.findings
                        ],
                        "transcript": [
                            {
                                "turn": turn.index,
                                "customer": turn.customer_message,
                                "outcome": turn.outcome,
                                "replies": turn.replies,
                                "latency_ms": turn.latency_ms,
                                "error": turn.error,
                                "provenance": turn.provenance,
                            }
                            for turn in report.turns
                        ],
                    }
                    for report in reports
                ],
                indent=2,
            )
        )
    else:
        print(render_reports(reports))

    if args.fail_on_critical and any(report.critical for report in reports):
        return 1
    if args.fail_on_uncovered and any(report.coverage_gaps for report in reports):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
