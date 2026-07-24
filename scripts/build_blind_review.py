#!/usr/bin/env python3
"""Build a randomized blind-review document from Cleopatra model-evaluation results."""

from __future__ import annotations

import argparse
import json
import random
import secrets
import string
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENARIOS = ROOT / "eval" / "scenarios.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results_file",
        help="Combined model-evaluation JSON file.",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "eval" / "results" / "blind_review.md"),
        help="Markdown review file to create.",
    )
    parser.add_argument(
        "--answer-key",
        default=str(ROOT / "eval" / "results" / "blind_review_answer_key.json"),
        help="JSON answer key to create.",
    )
    parser.add_argument(
        "--scenarios",
        default=str(DEFAULT_SCENARIOS),
        help="Scenario catalog used to include fan message and history.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Optional deterministic shuffle seed. A random seed is used when omitted.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_scenario_lookup(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}

    payload = load_json(path)
    rows = payload.get("scenarios", payload) if isinstance(payload, dict) else payload

    if not isinstance(rows, list):
        return {}

    return {
        str(row.get("id")): row
        for row in rows
        if isinstance(row, dict) and row.get("id")
    }


def group_rows(rows: list[dict[str, Any]]) -> "OrderedDict[str, list[dict[str, Any]]]":
    grouped: "OrderedDict[str, list[dict[str, Any]]]" = OrderedDict()

    for row in rows:
        scenario_id = str(row.get("scenario_id") or "unknown")
        grouped.setdefault(scenario_id, []).append(row)

    return grouped


def candidate_label(index: int) -> str:
    """Return A..Z, AA..AZ, BA... for any reasonable candidate count."""

    alphabet = string.ascii_uppercase
    label = ""

    while True:
        index, remainder = divmod(index, 26)
        label = alphabet[remainder] + label

        if index == 0:
            return label

        index -= 1


def format_history_item(item: Any) -> str:
    if isinstance(item, str):
        return item.strip()

    if not isinstance(item, dict):
        return str(item)

    role = (
        item.get("role")
        or item.get("sender")
        or item.get("author")
        or item.get("type")
        or "message"
    )
    content = (
        item.get("content")
        or item.get("text")
        or item.get("body")
        or item.get("message")
        or ""
    )

    return f"{role}: {content}".strip()


def format_scenario_context(
    scenario_id: str,
    scenario_row: dict[str, Any],
    result_rows: list[dict[str, Any]],
) -> list[str]:
    first_result = result_rows[0] if result_rows else {}
    title = (
        scenario_row.get("title")
        or first_result.get("scenario_title")
        or scenario_id
    )

    lines = [
        f"## {title}",
        "",
        f"`scenario_id: {scenario_id}`",
        "",
    ]

    history = scenario_row.get("history") or []
    if isinstance(history, list) and history:
        lines.extend(["### Recent conversation", ""])

        for item in history[-8:]:
            text = format_history_item(item)
            if text:
                lines.append(f"- {text}")

        lines.append("")

    fan_message = scenario_row.get("fan_message")
    if fan_message is not None and str(fan_message).strip():
        lines.extend(
            [
                "### Latest fan message",
                "",
                f"> {str(fan_message).strip()}",
                "",
            ]
        )

    lines.extend(
        [
            "### What to judge",
            "",
            "Read these as messages from a real creator, not as polished copy. Judge whether the "
            "reply feels ordinary and context-aware, whether it avoids obvious AI habits, whether "
            "it matches the creator voice, and whether it still moves the conversation usefully.",
            "",
        ]
    )

    return lines


def render_candidate(row: dict[str, Any], label: str) -> list[str]:
    lines = [f"### Candidate {label}", ""]

    replies = row.get("replies") or []

    if isinstance(replies, list) and replies:
        for index, reply in enumerate(replies, start=1):
            lines.append(f"{index}. {str(reply).strip()}")
    else:
        raw_text = str(row.get("raw_text") or "").strip()
        error = str(row.get("error") or "").strip()
        skip_reason = str(row.get("skip_reason") or "").strip()

        if raw_text:
            lines.extend(
                [
                    "**Invalid structured output — raw model response:**",
                    "",
                    "```text",
                    raw_text,
                    "```",
                ]
            )
        elif "timed out" in error.lower():
            lines.append("**No usable output: request timed out.**")
        elif error:
            lines.append("**No usable output: generation error.**")
        elif skip_reason:
            lines.append("**No usable output: candidate was skipped for this scenario.**")
        else:
            lines.append("**No usable output returned.**")

    lines.extend(
        [
            "",
            "**Human believability (1–5):** ",
            "",
            "**Relevance/context (1–5):** ",
            "",
            "**Creator voice (1–5):** ",
            "",
            "**Commercial usefulness (1–5):** ",
            "",
            "**Would you suspect AI? (yes / maybe / no):** ",
            "",
            "**AI tells, if any:** polished conclusion / forced wit / repeats every point / "
            "unearned expertise / constant question / therapy voice / other",
            "",
            "**Notes:** ",
            "",
        ]
    )

    return lines


def build_answer_key_entry(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_name": row.get("candidate_name"),
        "provider": row.get("provider"),
        "model": row.get("model"),
        "run_id": row.get("run_id"),
        "latency_ms": row.get("latency_ms"),
        "estimated_cost_usd": row.get("estimated_cost_usd"),
        "automatic_checks": row.get("automatic_checks") or {},
        "error": row.get("error"),
        "skip_reason": row.get("skip_reason"),
    }


def main() -> int:
    args = parse_args()

    results_path = Path(args.results_file).resolve()
    output_path = Path(args.output).resolve()
    answer_key_path = Path(args.answer_key).resolve()
    scenarios_path = Path(args.scenarios).resolve()

    payload = load_json(results_path)
    rows = payload.get("results", payload) if isinstance(payload, dict) else payload

    if not isinstance(rows, list) or not rows:
        raise SystemExit("The results file does not contain a non-empty 'results' list.")

    normalized_rows = [row for row in rows if isinstance(row, dict)]
    grouped = group_rows(normalized_rows)
    scenarios = load_scenario_lookup(scenarios_path)

    seed = args.seed if args.seed is not None else secrets.randbelow(2_147_483_647)
    rng = random.Random(seed)

    markdown: list[str] = [
        "# Cleopatra Model Lab — Blind Realism Review",
        "",
        "Do not open the answer key until all rankings are complete.",
        "This is an offline review sheet. It does not score, block, or reroute production replies.",
        "",
        f"Scenarios: {len(grouped)}",
        "",
        "---",
        "",
    ]

    answer_key: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_results_file": str(results_path),
        "scenarios_file": str(scenarios_path) if scenarios_path.exists() else None,
        "shuffle_seed": seed,
        "scenarios": {},
    }

    for scenario_id, scenario_results in grouped.items():
        shuffled = list(scenario_results)
        rng.shuffle(shuffled)

        markdown.extend(
            format_scenario_context(
                scenario_id,
                scenarios.get(scenario_id, {}),
                scenario_results,
            )
        )

        scenario_key: dict[str, Any] = {}

        for index, row in enumerate(shuffled):
            label = candidate_label(index)
            markdown.extend(render_candidate(row, label))
            scenario_key[label] = build_answer_key_entry(row)

        labels = [candidate_label(index) for index in range(len(shuffled))]
        markdown.extend(
            [
                "### Scenario ranking",
                "",
                f"Best → worst: {' > '.join('__' for _ in labels)}",
                "",
                "**Scenario notes:** ",
                "",
                "---",
                "",
            ]
        )

        answer_key["scenarios"][scenario_id] = scenario_key

    output_path.parent.mkdir(parents=True, exist_ok=True)
    answer_key_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text("\n".join(markdown), encoding="utf-8")
    answer_key_path.write_text(
        json.dumps(answer_key, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Blind review: {output_path}")
    print(f"Answer key:   {answer_key_path}")
    print(f"Shuffle seed: {seed}")
    print(f"Scenarios:    {len(grouped)}")
    print(f"Candidates:   {len(normalized_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
