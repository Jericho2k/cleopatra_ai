#!/usr/bin/env python3
"""OPTIONAL: have a model read the blinded conversations from an A/B run.

    python scripts/run_conversation_judge.py eval/results/<run-id> \
        --model <candidate-name-from-config/model_candidates.json> \
        --confirm-paid-provider-calls

Never run automatically, never part of a run, and never a substitute for the
human review. It exists because twelve scenarios run repeatedly is more reading
than a person will do every time, and a consistent first pass that says where to
look is useful.

What it is not allowed to be is the answer. It receives the same blinded
transcripts the human reviewer gets — no runtime names, no "old" and "new" — and
it returns a rating and an observable reason per dimension. There is no total,
no winner and no average, here or in ``judge.json``.

Costs money: it makes real provider calls, so it requires
``--confirm-paid-provider-calls`` the same way
``scripts/run_candidate_provider_eval.py`` does.
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

from ai.model_providers import complete  # noqa: E402
from models.model_runtime import ModelTarget  # noqa: E402
from services.ab_trajectory_eval import METADATA_FILE, PAIRED_FILE, known_core_ids  # noqa: E402
from services.conversation_judge import (  # noqa: E402
    build_prompt,
    parse_judgement,
    unblind_judgement,
)

JUDGE_FILE = "judge.json"
DEFAULT_CATALOG = ROOT / "config" / "model_candidates.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run_dir", type=Path, help="a run directory under eval/results/")
    parser.add_argument("--model", required=True, help="candidate name or model id from the catalog")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--seed", type=int, help="label ordering seed (default: the run's own)")
    parser.add_argument("--max-tokens", type=int, default=2000)
    parser.add_argument(
        "--confirm-paid-provider-calls",
        action="store_true",
        help="required: this command makes real, billable provider calls",
    )
    parser.add_argument(
        "--unblind",
        action="store_true",
        help="also write the judgement re-keyed to the arm roles",
    )
    return parser.parse_args()


def load_target(catalog: Path, selector: str) -> ModelTarget:
    payload = json.loads(catalog.read_text(encoding="utf-8"))
    rows = payload.get("models", payload) if isinstance(payload, dict) else payload
    wanted = selector.strip().lower()
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        target = ModelTarget.from_mapping(row)
        if wanted in {target.name.lower(), target.model.lower()}:
            return target
    raise SystemExit(f"no model named {selector!r} in {catalog}")


async def _run(args) -> int:
    run_dir: Path = args.run_dir
    paired_path = run_dir / PAIRED_FILE
    if not paired_path.exists():
        print(f"no {PAIRED_FILE} in {run_dir}", file=sys.stderr)
        return 2
    paired = json.loads(paired_path.read_text(encoding="utf-8"))
    if not paired.get("paired"):
        print("this run has one arm only; there is nothing to compare", file=sys.stderr)
        return 2

    seed = args.seed
    if seed is None:
        metadata_path = run_dir / METADATA_FILE
        seed = int(json.loads(metadata_path.read_text(encoding="utf-8")).get("seed") or 0)

    target = load_target(args.catalog, args.model)
    results: dict[str, object] = {
        "_about": (
            "An optional model reading of the blinded conversations. Per "
            "dimension, with evidence. No total, no winner, and not a "
            "substitute for the human review."
        ),
        "model": {"name": target.name, "provider": target.provider, "model": target.model},
        "seed": seed,
        "scenarios": {},
    }

    for pair in paired.get("pairs") or []:
        scenario_id = str(pair.get("scenario_id") or "")
        try:
            system, messages, mapping = build_prompt(
                pair, seed=int(seed), extra_forbidden=known_core_ids()
            )
        except ValueError as exc:
            results["scenarios"][scenario_id] = {"skipped": str(exc)}
            continue
        try:
            async with asyncio.timeout(target.timeout_seconds):
                completion = await complete(
                    target, system=system, messages=messages, max_tokens=args.max_tokens
                )
        except Exception as exc:
            results["scenarios"][scenario_id] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"FAIL {scenario_id}: {exc}", file=sys.stderr)
            continue
        judgement = parse_judgement(completion.text)
        entry: dict[str, object] = {"blinded": judgement, "labels_are_blind": True}
        if args.unblind:
            entry["by_arm"] = unblind_judgement(judgement, mapping)
        results["scenarios"][scenario_id] = entry
        print(f"DONE {scenario_id}")

    output = run_dir / JUDGE_FILE
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"judge: {output}")
    print(
        "Per-dimension results only. There is deliberately no overall number, "
        "and this does not replace the human review."
    )
    return 0


def main() -> int:
    args = parse_args()
    if not args.confirm_paid_provider_calls:
        print(
            "This command makes real, billable provider calls. Re-run with "
            "--confirm-paid-provider-calls.",
            file=sys.stderr,
        )
        return 2
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
