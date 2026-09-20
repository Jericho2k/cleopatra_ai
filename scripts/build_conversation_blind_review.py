#!/usr/bin/env python3
"""Turn an A/B trajectory run into a blind conversation review, and unblind it later.

    # Build the review for a run.
    python scripts/build_conversation_blind_review.py eval/results/<run-id>

    # Read the key afterwards.
    python scripts/build_conversation_blind_review.py eval/results/<run-id> --unblind

The reviewer-facing document never names a runtime and never says which one is
expected to be better. The key lands in a separate file, which is the only thing
that makes the review blind rather than a document with the answer further down.

The ordering is drawn from the run's own seed, so rebuilding a review from the
same run directory produces the same labels — a reviewer's notes stay valid, and
two people can review the same document without comparing different ones.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.ab_trajectory_eval import (  # noqa: E402
    METADATA_FILE,
    PAIRED_FILE,
    known_core_ids,
)
from services.blind_conversation_review import (  # noqa: E402
    leaked_terms,
    forbidden_terms,
    unblind_all,
    write_review,
)

REVIEW_FILE = "blind_review.md"
MAPPING_FILE = "blind_mapping.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run_dir", type=Path, help="a run directory under eval/results/")
    parser.add_argument(
        "--seed",
        type=int,
        help="override the label ordering seed (default: the run's own seed)",
    )
    parser.add_argument("--output", type=Path, help=f"review path (default: <run>/{REVIEW_FILE})")
    parser.add_argument("--mapping", type=Path, help=f"key path (default: <run>/{MAPPING_FILE})")
    parser.add_argument(
        "--unblind",
        action="store_true",
        help="print the existing key instead of building a review",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir: Path = args.run_dir
    mapping_path = args.mapping or run_dir / MAPPING_FILE
    review_path = args.output or run_dir / REVIEW_FILE

    if args.unblind:
        if not mapping_path.exists():
            print(f"no key at {mapping_path}", file=sys.stderr)
            return 2
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        for scenario_id, labels in sorted(unblind_all(mapping).items()):
            rendered = ", ".join(
                f"Conversation {label} = {core}" for label, core in sorted(labels.items())
            )
            print(f"{scenario_id}: {rendered}")
        return 0

    paired_path = run_dir / PAIRED_FILE
    if not paired_path.exists():
        print(f"no {PAIRED_FILE} in {run_dir}", file=sys.stderr)
        return 2
    paired = json.loads(paired_path.read_text(encoding="utf-8"))
    if not paired.get("paired"):
        print(
            "this run has one arm only, so there is nothing to compare blindly. "
            "Run it again with --candidate once the second runtime exists.",
            file=sys.stderr,
        )
        return 2

    seed = args.seed
    metadata_path = run_dir / METADATA_FILE
    run_id = ""
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        run_id = str(metadata.get("run_id") or "")
        if seed is None:
            seed = int(metadata.get("seed") or 0)
    if seed is None:
        print("no seed in the run metadata; pass --seed", file=sys.stderr)
        return 2

    document, mapping = write_review(
        paired,
        seed=int(seed),
        run_id=run_id or run_dir.name,
        review_path=review_path,
        mapping_path=mapping_path,
        extra_forbidden=known_core_ids(),
    )
    # Re-read what was written rather than trusting the string in memory: the
    # file is what a reviewer opens, and it is the file that has to be clean.
    leaks = leaked_terms(
        review_path.read_text(encoding="utf-8"), forbidden_terms(known_core_ids())
    )
    if leaks:  # pragma: no cover - build_review already refuses
        print(f"the written review identifies runtimes: {', '.join(leaks)}", file=sys.stderr)
        return 1

    print(f"review: {review_path}")
    print(f"key:    {mapping_path}")
    print(f"seed:   {seed}")
    print(f"scenarios: {len(mapping.get('scenarios') or {})}")
    print("Complete the review before opening the key.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
