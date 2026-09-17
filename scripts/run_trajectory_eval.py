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

from services.trajectory_eval import (  # noqa: E402
    load_trajectories,
    render_reports,
    run_trajectory,
)

DEFAULT_TRAJECTORIES = ROOT / "eval" / "trajectories.json"


def _load(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return list(payload.get("trajectories") or [])
    return list(payload or [])


async def _run_all(trajectories, creator_id: str, fan_id: str) -> list:
    from services.suggestions import run_simulated_inbound

    reports = []
    for trajectory in trajectories:
        creator = trajectory.creator_id or creator_id
        fan = trajectory.fan_id or fan_id

        async def send_turn(message: str, _creator=creator, _fan=fan) -> dict:
            if not message.strip():
                # A turn where the customer says nothing. The system is not
                # asked anything, so nothing should happen — which is exactly
                # what the goodbye trajectory is checking.
                return {"outcome": "no_message", "creator_messages": []}
            return await run_simulated_inbound(
                fan_id=_fan, creator_id=_creator, message=message, fast=True
            )

        reports.append(await run_trajectory(trajectory, send_turn=send_turn))
    return reports


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
    args = parser.parse_args()

    trajectories = load_trajectories(_load(args.trajectories))
    if not trajectories:
        print(f"no trajectories in {args.trajectories}", file=sys.stderr)
        return 2

    if args.describe:
        for trajectory in trajectories:
            print(f"{trajectory.name} ({len(trajectory.disturbances)} turns)")
            print(f"  covers: {trajectory.covers}")
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

    reports = asyncio.run(_run_all(trajectories, args.creator, args.fan))

    if args.json:
        print(
            json.dumps(
                [
                    {
                        "summary": report.summary(),
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
