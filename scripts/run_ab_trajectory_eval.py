#!/usr/bin/env python3
"""Run one scripted fan trajectory through two conversational runtimes and compare.

The microscope for Conversational Core v1, not the treatment. It drives the real
simulator (``services.suggestions.run_simulated_inbound``) across whole
conversations twice — once per runtime, each against its own ``test_`` fan —
holding the fan's script, the creator, the stack profile and the starting state
constant, and writes both runs plus the objective metrics into one directory.

    # Baseline only. Works today, before the candidate runtime exists.
    python scripts/run_ab_trajectory_eval.py \\
        --baseline semantic_v2 --creator <creator-id> --provision-fans \\
        --suite conversational --simulate-time

    # A/B, once the candidate runtime registers its core id.
    python scripts/run_ab_trajectory_eval.py \\
        --baseline semantic_v2 --candidate conversational_v1 \\
        --creator <creator-id> --provision-fans --suite all --simulate-time

    # One scenario, against fans that already exist.
    python scripts/run_ab_trajectory_eval.py \\
        --baseline semantic_v2 --candidate conversational_v1 \\
        --creator <creator-id> \\
        --baseline-fan <test-fan-a> --candidate-fan <test-fan-b> \\
        --scenario D_shared_imagined_scene

    # What the suite contains, with no backend at all.
    python scripts/run_ab_trajectory_eval.py --describe

**No quality score comes out of this.** ``metrics.json`` is arithmetic over what
was sent; the rubric dimensions that need judgement are in the blind review
(``scripts/build_conversation_blind_review.py``). Merging the two would produce a
number that looks like an answer and is not one.

**Both arms must be simulator test fans**, and they must be different fans. A
conversation in this product is persistent state; two runtimes writing into one
fan is one conversation with two authors, and every number computed from it
would describe something that never happened.
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
from services.ab_trajectory_eval import (  # noqa: E402
    ArmSpec,
    EvaluationRefused,
    LiveSimulatorBackend,
    METRICS_FILE,
    PAIRED_FILE,
    ROLE_BASELINE,
    ROLE_CANDIDATE,
    RunSpec,
    assert_arm_isolation,
    assert_safe_targets,
    known_core_ids,
    load_scenario_file,
    new_run_id,
    require_known_core,
    run_suite,
    select_trajectories,
    write_artifacts,
)
from services.trajectory_eval import coverage_gaps  # noqa: E402

DEFAULT_SCENARIOS = ROOT / "eval" / "conversational_core_scenarios.json"
DEFAULT_RESULTS = ROOT / "eval" / "results"

#: A fixed default so two people running the same command get the same arm
#: order and the same blind labels. Overridden with --seed when a second
#: independent sample is wanted.
DEFAULT_SEED = 1729


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--baseline",
        default="semantic_v2",
        help="conversation core for the first arm (default: semantic_v2)",
    )
    parser.add_argument(
        "--candidate",
        default="",
        help=(
            "conversation core for the second arm. Omit for a baseline-only "
            "run, which is what to do before the candidate runtime exists"
        ),
    )
    parser.add_argument("--creator", default="", help="creator id to run against")
    parser.add_argument(
        "--baseline-fan", default="", help="existing test fan for the baseline arm"
    )
    parser.add_argument(
        "--candidate-fan", default="", help="existing test fan for the candidate arm"
    )
    parser.add_argument(
        "--provision-fans",
        action="store_true",
        help=(
            "create a fresh test fan per arm for this run. The strongest form "
            "of equivalent initial conditions available: both arms start from a "
            "fan with no history, no commercial state and no learned budget"
        ),
    )
    parser.add_argument(
        "--scenarios",
        type=Path,
        default=DEFAULT_SCENARIOS,
        help=f"scenario file (default: {DEFAULT_SCENARIOS.relative_to(ROOT)})",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        default=[],
        help="run one scenario by id; repeatable. Overrides --suite",
    )
    parser.add_argument("--suite", default="", help="run a named suite from the file")
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"reproducible seed for arm order and blind labels (default: {DEFAULT_SEED})",
    )
    parser.add_argument("--run-id", default="", help="run id (default: generated)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RESULTS,
        help=f"results root (default: {DEFAULT_RESULTS.relative_to(ROOT)}, gitignored)",
    )
    parser.add_argument(
        "--simulate-time",
        action="store_true",
        help=(
            "advance a simulated clock for days_since_previous. Requires "
            f"APP_ENV != production and {clock.EVAL_CLOCK_FLAG}=1; without it "
            "the elapsed-time claims are reported as uncovered rather than "
            "quietly assumed"
        ),
    )
    parser.add_argument(
        "--describe",
        action="store_true",
        help="list the selected scenarios and what they can cover, then exit",
    )
    parser.add_argument(
        "--blind-review",
        action="store_true",
        help="also write blind_review.md and blind_mapping.json for this run",
    )
    parser.add_argument(
        "--fail-on-critical",
        action="store_true",
        help="exit non-zero if any arm produced a critical execution failure",
    )
    return parser.parse_args()


def _describe(trajectories, ids, *, use_clock: bool) -> int:
    from services.ab_trajectory_eval import fan_input_digest
    from services.trajectory_eval import TrajectoryReport, TurnRecord

    for scenario_id, trajectory in zip(ids, trajectories):
        elapsed = sum(
            float(item.days_since_previous or 0.0) for item in trajectory.disturbances
        )
        best_case = TrajectoryReport(
            trajectory=trajectory.name,
            covers=trajectory.covers,
            turns=[
                TurnRecord(index=index, customer_message=item.message)
                for index, item in enumerate(trajectory.disturbances)
            ],
            due_worker_runs=sum(
                1 for item in trajectory.disturbances if not str(item.message or "").strip()
            ),
            clock_injected=use_clock,
            elapsed_days=elapsed if use_clock else 0.0,
            seeded_purchases=list(trajectory.seed.get("purchases") or []),
        )
        print(f"{scenario_id} — {trajectory.name} ({len(trajectory.disturbances)} turns)")
        print(f"  inputs: {fan_input_digest(trajectory)}")
        print(f"  covers: {trajectory.covers}")
        for gap in coverage_gaps(trajectory, best_case):
            print(f"  {gap.render()}")
        print()
    print(f"{len(trajectories)} scenario(s). Known conversation cores: {', '.join(known_core_ids())}")
    return 0


async def _resolve_arms(args, backend) -> tuple[list[ArmSpec], dict[str, dict]]:
    """Build both arms, provisioning fans if asked, and verify every target."""
    roles: list[tuple[str, str, str]] = [
        (ROLE_BASELINE, require_known_core(args.baseline), args.baseline_fan)
    ]
    if args.candidate:
        roles.append((ROLE_CANDIDATE, require_known_core(args.candidate), args.candidate_fan))

    arms: list[ArmSpec] = []
    for role, core_id, fan_id in roles:
        provisioned = False
        if not fan_id:
            if not args.provision_fans:
                raise EvaluationRefused(
                    f"the {role} arm needs --{role}-fan, or --provision-fans to "
                    "create a fresh test fan for it"
                )
            created = await backend.provision_fan(
                args.creator, f"eval {role} {core_id}"
            )
            fan_id = str(created.get("id"))
            provisioned = True
        arms.append(ArmSpec(role=role, core_id=core_id, fan_id=fan_id, provisioned=provisioned))

    assert_arm_isolation(arms)
    rows = {}
    for arm in arms:
        rows[arm.fan_id] = await backend.describe_fan(args.creator, arm.fan_id)
    assert_safe_targets(arms, rows)
    return [
        ArmSpec(
            role=arm.role,
            core_id=arm.core_id,
            fan_id=arm.fan_id,
            provisioned=arm.provisioned,
            platform_fan_id=str((rows.get(arm.fan_id) or {}).get("platform_fan_id") or ""),
        )
        for arm in arms
    ], rows


async def _stack_profile(creator_id: str, fan_id: str) -> str:
    """The profile both arms will resolve, recorded so a run states its stack."""
    try:
        from services.ai_stack import resolve_ai_stack

        resolution = await resolve_ai_stack(creator_id=creator_id, fan_id=fan_id)
        return str(resolution.profile_id)
    except Exception as exc:  # pragma: no cover - never load-bearing
        print(f"[AB EVAL] could not resolve the AI stack profile: {exc}", file=sys.stderr)
        return ""


async def _run(args) -> int:
    payload = load_scenario_file(args.scenarios)
    trajectories, ids = select_trajectories(
        payload, suite=args.suite, scenario_ids=args.scenario
    )
    if not trajectories:
        print(f"no scenarios selected from {args.scenarios}", file=sys.stderr)
        return 2

    if args.describe:
        return _describe(trajectories, ids, use_clock=args.simulate_time)

    if not args.creator:
        print(
            "--creator is required to run. Use --describe to inspect the suite "
            "without a backend.",
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

    backend = LiveSimulatorBackend()
    arms, _rows = await _resolve_arms(args, backend)
    spec = RunSpec(
        run_id=args.run_id or new_run_id(),
        creator_id=args.creator,
        seed=int(args.seed),
        arms=tuple(arms),
        scenario_ids=tuple(ids),
        simulate_time=bool(args.simulate_time),
        stack_profile=await _stack_profile(args.creator, arms[0].fan_id),
        notes="baseline only" if len(arms) == 1 else "",
    )

    for arm in arms:
        print(
            f"[AB EVAL] {arm.role}: core={arm.core_id} fan={arm.fan_id} "
            f"platform_fan={arm.platform_fan_id} "
            f"{'(provisioned for this run)' if arm.provisioned else ''}",
            file=sys.stderr,
        )

    runs = await run_suite(
        spec=spec,
        trajectories=trajectories,
        backend=backend,
        advance_clock=clock.advance if args.simulate_time else None,
        reset_clock=clock.reset,
        scenario_ids=ids,
    )
    root = write_artifacts(
        spec=spec, runs=runs, trajectories=trajectories, results_root=args.output_dir
    )

    if args.blind_review:
        from services.blind_conversation_review import write_review

        paired = json.loads((root / PAIRED_FILE).read_text(encoding="utf-8"))
        write_review(
            paired,
            seed=spec.seed,
            run_id=spec.run_id,
            review_path=root / "blind_review.md",
            mapping_path=root / "blind_mapping.json",
            extra_forbidden=known_core_ids(),
        )

    print(f"run:     {spec.run_id}")
    print(f"results: {root}")
    metrics = json.loads((root / METRICS_FILE).read_text(encoding="utf-8"))
    for role, arm_metrics in metrics.items():
        pooled = arm_metrics["pooled"]
        questions = pooled["questions"]
        execution = pooled["execution"]
        print(
            f"  {role}: {execution['turns']} turns, "
            f"{questions['question_ending_rate']:.0%} end in a question "
            f"(longest streak {questions['longest_question_ending_streak']}), "
            f"{execution['errors']} error(s), {execution['handoffs']} handoff(s), "
            f"median {execution['latency_ms']['median']:.0f}ms"
        )
    print(
        "No quality score is produced. The rubric dimensions are judged in the "
        "blind review, not here."
    )

    critical = sum(
        int(conversation["summary"]["critical_failures"])
        for run in runs.values()
        for conversation in run.conversations
    )
    if args.fail_on_critical and critical:
        print(f"{critical} critical execution failure(s)", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    args = parse_args()
    try:
        return asyncio.run(_run(args))
    except EvaluationRefused as refused:
        print(f"evaluation refused: {refused}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
