#!/usr/bin/env python3
"""Run the Conversational Core v1 owner-stability gate and print its metrics.

    python scripts/run_owner_stability_eval.py --turns 500

The synthetic arm needs no network, no database and no API key: it replaces the
transport with a seeded generator of the response shapes production has
actually produced, and drives every turn through the real owner boundary and
the real deterministic authority. That makes it safe to run in CI and useful
before a deploy, which is the whole point — an intermittent response-format
failure is invisible in one happy-path message.

It reports rates, not a verdict:

    owner_failed_rate               turns that sent nothing because the owner
                                    could not be read, even after one repair
    malformed_first_response_rate   first responses that were not usable
    repair_attempt_rate             turns that needed the bounded repair
    repair_success_rate             repairs that produced a usable answer
    no_send_rate                    turns that deliberately stayed quiet
    operation_rejection_rate        proposals deterministic authority refused
    state_delta_rejection_rate      deltas whose fields were refused
    latency_ms_p50 / p95            owner time per turn

``--json`` writes the full per-turn record, which is what to diff between two
runs. ``--fail-over-owner-failed`` makes this usable as a gate in a pipeline.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.owner_stability_eval import run_synthetic_stability  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--turns", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--repair-success-rate",
        type=float,
        default=0.75,
        help="fraction of bounded repair calls the synthetic owner answers usably",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="write the full per-turn record here",
    )
    parser.add_argument(
        "--fail-over-owner-failed",
        type=float,
        default=None,
        help="exit non-zero when the owner-failed rate exceeds this fraction",
    )
    parser.add_argument(
        "--show-turn-logs",
        action="store_true",
        help="print the per-call diagnostic lines instead of suppressing them",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run = run_synthetic_stability(
        turns=args.turns,
        seed=args.seed,
        repair_success_rate=args.repair_success_rate,
    )
    if args.show_turn_logs:
        report = asyncio.run(run)
    else:
        # The per-call diagnostics are the production log line; thousands of
        # them would bury the numbers this script exists to show.
        with contextlib.redirect_stdout(io.StringIO()):
            report = asyncio.run(run)

    metrics = report.metrics()
    print(json.dumps(metrics, indent=2))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report.as_dict(), indent=2), encoding="utf-8")
        print(f"wrote {args.json}", file=sys.stderr)

    if args.fail_over_owner_failed is not None:
        observed = float(metrics.get("owner_failed_rate") or 0.0)
        if observed > args.fail_over_owner_failed:
            print(
                f"owner_failed_rate {observed} exceeds {args.fail_over_owner_failed}",
                file=sys.stderr,
            )
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
