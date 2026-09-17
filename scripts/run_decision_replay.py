#!/usr/bin/env python3
"""Compare ways of deciding what a turn does, offline, on identical evidence.

``docs/autonomy_architecture_review.md`` §6.4. This runs the fixed-prefix replay:
every candidate gets the same context packet, the same operational facts and the
same executor authority, and the report is the set of turns where they would do
different things.

    # The projection alone — what the current controllers decide, stated once
    # per turn. No model call, no key needed, no cost.
    python scripts/run_decision_replay.py

    # Add the single semantic owner and compare. Costs one model call per turn.
    python scripts/run_decision_replay.py --semantic

    # Machine-readable, for a longer run.
    python scripts/run_decision_replay.py --semantic --json > replay.json

What this does NOT do is pick a winner. §4: *"Select it only if complete
conversation evaluation establishes a benefit."* A disagreement count is
something for a person to read the disagreements behind; the number that has to
be zero is critical failures, and that one is reported on its own because §5
says a prose score must never be allowed to cancel an unauthorized transaction.

Nothing here sends a message, writes to a database or advances any state.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.decision_owners import (  # noqa: E402
    CurrentStackOwner,
    SemanticDecisionOwner,
)
from services.decision_replay import compare_owners_sync, load_turns  # noqa: E402

DEFAULT_SCENARIOS = ROOT / "eval" / "decision_scenarios.json"


def _load(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return list(payload.get("scenarios") or [])
    return list(payload or [])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenarios",
        type=Path,
        default=DEFAULT_SCENARIOS,
        help=f"scenario file (default: {DEFAULT_SCENARIOS.relative_to(ROOT)})",
    )
    parser.add_argument(
        "--semantic",
        action="store_true",
        help="also run the single semantic owner (one model call per turn)",
    )
    parser.add_argument(
        "--json", action="store_true", help="emit the full report as JSON"
    )
    parser.add_argument(
        "--fail-on-critical",
        action="store_true",
        help="exit non-zero if any candidate produced a decision that must be refused",
    )
    args = parser.parse_args()

    turns = load_turns(_load(args.scenarios))
    if not turns:
        print(f"no scenarios in {args.scenarios}", file=sys.stderr)
        return 2

    owners = [CurrentStackOwner()]
    if args.semantic:
        # Imported here so the projection-only run needs no provider
        # configuration at all, and a missing key cannot stop the cheap path.
        from ai.model_providers import complete, get_runtime_target

        owners.append(
            SemanticDecisionOwner(complete, target=get_runtime_target("ANALYZER"))
        )

    report = compare_owners_sync(turns, owners)

    if args.json:
        print(
            json.dumps(
                {
                    "summary": report.summary(),
                    "turns": [c.as_dict() for c in report.comparisons],
                },
                indent=2,
            )
        )
    else:
        print(report.render())

    if args.fail_on_critical:
        failures = sum(
            report.critical_failures(candidate) for candidate in report.candidates
        )
        if failures:
            print(
                f"\n{failures} decision(s) would have to be refused.", file=sys.stderr
            )
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
