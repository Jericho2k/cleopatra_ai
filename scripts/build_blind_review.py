#!/usr/bin/env python3
"""Turn an evaluation result JSON into a model-blind Markdown review sheet."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_json")
    parser.add_argument("--output", default="blind_review.md")
    parser.add_argument("--answer-key", default="blind_review_answer_key.json")
    parser.add_argument("--seed", type=int, default=20260713)
    args = parser.parse_args()

    payload = json.loads(Path(args.result_json).read_text(encoding="utf-8"))
    grouped = defaultdict(list)
    for row in payload["results"]:
        if not row.get("skipped") and not row.get("error"):
            grouped[row["scenario_id"]].append(row)

    rng = random.Random(args.seed)
    answer_key = {}
    lines = ["# Cleopatra Blind Model Review", ""]
    for scenario_id, rows in grouped.items():
        rng.shuffle(rows)
        lines.extend([f"## {scenario_id}", ""])
        answer_key[scenario_id] = {}
        for index, row in enumerate(rows):
            label = chr(ord("A") + index)
            answer_key[scenario_id][label] = {
                "candidate_name": row["candidate_name"],
                "provider": row["provider"],
                "model": row["model"],
            }
            lines.append(f"### Response {label}")
            for reply in row.get("replies", []):
                lines.append(f"- {reply}")
            lines.extend(
                [
                    "",
                    "Fatal issue: [ ] refusal [ ] robotic [ ] context [ ] commercial [ ] repetitive",
                    "",
                ]
            )
        lines.extend(["Best: [ ] A [ ] B [ ] C [ ] D [ ] E [ ] F [ ] None", "", "---", ""])

    Path(args.output).write_text("\n".join(lines), encoding="utf-8")
    Path(args.answer_key).write_text(json.dumps(answer_key, indent=2), encoding="utf-8")
    print(f"Wrote {args.output} and {args.answer_key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
