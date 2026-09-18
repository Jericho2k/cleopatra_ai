#!/usr/bin/env python3
"""Run both complete conversation-core candidates against configured providers.

This is a paid, provider-shadow evaluation: it calls inference providers and
writes a local evidence bundle, but imports no platform/database adapter, sends
no customer message, and mutates no production state.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.model_providers import complete, get_runtime_target  # noqa: E402
from core.build_info import active_flags  # noqa: E402
from services.candidate_provider_eval import (  # noqa: E402
    build_evaluation_bundle,
    git_state,
    run_provider_comparison,
)
from services.decision_replay import load_turns  # noqa: E402

DEFAULT_SCENARIOS = ROOT / "eval" / "decision_scenarios.json"


def _load(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("scenarios") or []) if isinstance(payload, dict) else list(payload)


async def _run(args: argparse.Namespace) -> int:
    raw = _load(args.scenarios)
    turns = load_turns(raw)
    if not turns:
        print(f"no scenarios in {args.scenarios}", file=sys.stderr)
        return 2

    state = git_state(ROOT)
    if state["dirty"] and not args.allow_dirty:
        print(
            "refusing a non-reproducible run from a dirty tree; commit changes or pass --allow-dirty",
            file=sys.stderr,
        )
        return 2

    one_call_target = get_runtime_target(args.one_call_prefix)
    semantic_target = get_runtime_target(args.semantic_prefix)
    writer_target = get_runtime_target(args.writer_prefix)
    report, calls = await run_provider_comparison(
        turns,
        complete=complete,
        one_call_target=one_call_target,
        semantic_target=semantic_target,
        writer_target=writer_target,
    )

    output = args.output or (
        ROOT
        / "evaluation_bundles"
        / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    )
    manifest = build_evaluation_bundle(
        output,
        root=ROOT,
        scenarios_path=args.scenarios,
        scenario_names=[turn.name for turn in turns],
        report=report,
        calls=calls,
        targets={
            "reply_plus_intent": one_call_target,
            "semantic_owner": semantic_target,
            "semantic_owner_writer": writer_target,
        },
        flags=active_flags(),
    )
    print(report.render())
    print(f"\nbundle: {output.resolve()}")
    print(f"source: {manifest['source']['sha']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", type=Path, default=DEFAULT_SCENARIOS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--one-call-prefix", default="CHAT")
    parser.add_argument("--semantic-prefix", default="ANALYZER")
    parser.add_argument("--writer-prefix", default="CHAT")
    parser.add_argument(
        "--confirm-paid-provider-calls",
        action="store_true",
        help="required acknowledgement that this run incurs inference cost",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="allow a bundle whose manifest records dirty=true",
    )
    args = parser.parse_args()
    if not args.confirm_paid_provider_calls:
        parser.error("--confirm-paid-provider-calls is required")
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
