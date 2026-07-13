#!/usr/bin/env python3
"""Run a small, reproducible Cleopatra model comparison from the command line."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.generator import parse_reply_candidates  # noqa: E402
from ai.model_providers import complete  # noqa: E402
from ai.prompt_builder import build_prompt  # noqa: E402
from models.model_runtime import (  # noqa: E402
    ModelTarget,
    ModelTelemetryContext,
    estimate_cost_usd,
)
from models.schemas import (  # noqa: E402
    ConversationContext,
    Fan,
    Message,
    Persona,
    StageType,
)
from services.model_telemetry import record_model_result  # noqa: E402

REFUSAL_MARKERS = (
    "i can't help with that",
    "i cannot help with that",
    "i'm unable to",
    "i cannot engage",
    "i can't engage",
    "as an ai",
    "sexual content policy",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default=str(ROOT / "config" / "model_candidates.json"))
    parser.add_argument("--scenarios", default=str(ROOT / "eval" / "scenarios.json"))
    parser.add_argument("--output-dir", default=str(ROOT / "eval" / "results"))
    parser.add_argument("--models", help="Comma-separated candidate names or model IDs")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Zero-based scenario index to start from.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--persist", action="store_true")
    parser.add_argument(
        "--allow-unverified-adult",
        action="store_true",
        help="Allow adult-required scenarios for targets whose provider policy is unverified.",
    )
    return parser.parse_args()


def load_json(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_targets(path: str, selector: str | None) -> list[ModelTarget]:
    payload = load_json(path)
    rows = payload.get("models", payload)
    targets = [ModelTarget.from_mapping(row) for row in rows if row.get("enabled", True)]
    if not selector:
        return targets
    wanted = {item.strip().lower() for item in selector.split(",") if item.strip()}
    return [
        target
        for target in targets
        if target.name.lower() in wanted or target.model.lower() in wanted
    ]


def build_context(scenario: dict[str, Any]) -> ConversationContext:
    persona = Persona(
        avg_message_length="short",
        sends_multiple_messages=True,
        emoji_usage="rare",
        capitalization="lowercase",
        punctuation_style="casual",
        flirt_style="direct but natural",
        dont_list=["robotic language", "long paragraphs", "generic compliments"],
    )
    fan = Fan(
        id=f"eval-{scenario['id']}",
        display_name="test fan",
        total_spent=int(scenario.get("total_spent") or 0),
        ai_summary=scenario.get("fan_summary") or {},
    )
    history = [Message(**row) for row in scenario.get("history", [])]
    return ConversationContext(
        fan_message=scenario.get("fan_message", ""),
        conversation_history=history,
        fan_profile=fan,
        creator_persona=persona,
        similar_exchanges=[],
        conversation_stage=StageType(scenario.get("stage", "WARMING_UP")),
        creator_name="Maya",
        situation=scenario.get("situation") or {},
        commercial_decision=scenario.get("commercial_decision") or None,
        creator_legend={"location": "Los Angeles", "age": 24},
    )


def automatic_checks(replies: list[str], checks: dict[str, Any]) -> dict[str, Any]:
    joined = " \n ".join(replies).lower()
    result: dict[str, Any] = {
        "has_three_replies": len(replies) == 3,
        "refusal_free": not any(marker in joined for marker in REFUSAL_MARKERS),
    }

    max_words = checks.get("max_words")
    if max_words:
        result["within_word_limit"] = all(len(reply.split()) <= int(max_words) for reply in replies)

    must_not = [str(item).lower() for item in checks.get("must_not_contain", [])]
    if must_not:
        result["forbidden_phrases_absent"] = not any(item in joined for item in must_not)

    must_any = [str(item).lower() for item in checks.get("must_contain_any", [])]
    if must_any:
        result["required_reference_present"] = any(item in joined for item in must_any)

    if checks.get("question_required"):
        result["question_present"] = any("?" in reply for reply in replies)
    if checks.get("question_forbidden"):
        result["question_absent"] = all("?" not in reply for reply in replies)

    max_price = checks.get("max_price_usd")
    if max_price is not None:
        prices = [float(match) for match in re.findall(r"\$(\d+(?:\.\d{1,2})?)", joined)]
        result["price_within_limit"] = all(price <= float(max_price) for price in prices)

    if checks.get("refusal_forbidden", True):
        result["refusal_allowed_for_case"] = result["refusal_free"]

    result["passed"] = all(bool(value) for key, value in result.items() if key != "passed")
    return result


def adult_skip_reason(target: ModelTarget, scenario: dict[str, Any], allow_unverified: bool) -> str | None:
    if not scenario.get("adult_required"):
        return None
    if target.adult_policy == "ineligible":
        return "provider marked ineligible for adult-required scenarios"
    if target.adult_policy == "unverified" and not allow_unverified:
        return "provider adult-use terms unverified; pass --allow-unverified-adult only after review"
    return None


async def persist_run_start(run_id: str, scenario_count: int, model_count: int, config: dict[str, Any]) -> None:
    from core.supabase import get_supabase

    await asyncio.to_thread(
        lambda: get_supabase().table("model_evaluation_runs").insert(
            {
                "id": run_id,
                "scenario_count": scenario_count,
                "model_count": model_count,
                "config": config,
            }
        ).execute()
    )


async def persist_output(run_id: str, row: dict[str, Any]) -> None:
    from core.supabase import get_supabase

    payload = {
        "run_id": run_id,
        "scenario_id": row["scenario_id"],
        "candidate_name": row["candidate_name"],
        "provider": row["provider"],
        "model": row["model"],
        "skipped": row.get("skipped", False),
        "skip_reason": row.get("skip_reason"),
        "replies": row.get("replies", []),
        "automatic_checks": row.get("automatic_checks", {}),
        "latency_ms": row.get("latency_ms"),
        "estimated_cost_usd": row.get("estimated_cost_usd", 0),
        "input_tokens": row.get("usage", {}).get("input_tokens", 0),
        "output_tokens": row.get("usage", {}).get("output_tokens", 0),
        "cache_read_tokens": row.get("usage", {}).get("cache_read_tokens", 0),
        "raw_text": row.get("raw_text"),
        "error": row.get("error"),
    }
    await asyncio.to_thread(
        lambda: get_supabase().table("model_evaluation_outputs").upsert(payload).execute()
    )


async def persist_run_finish(run_id: str, summary: dict[str, Any]) -> None:
    from core.supabase import get_supabase

    await asyncio.to_thread(
        lambda: get_supabase().table("model_evaluation_runs").update(
            {
                "status": "completed",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "summary": summary,
            }
        ).eq("id", run_id).execute()
    )

def build_model_summaries(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}

    for row in rows:
        candidate_name = str(
            row.get("candidate_name") or "unknown"
        )

        summary = summaries.setdefault(
            candidate_name,
            {
                "provider": row.get("provider"),
                "model": row.get("model"),
                "completed": 0,
                "skipped": 0,
                "errors": 0,
                "automatic_passes": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_read_tokens": 0,
                "cache_write_tokens": 0,
                "total_accounted_tokens": 0,
                "estimated_cost_usd": 0.0,
                "latency_ms_total": 0,
            },
        )

        if row.get("skipped"):
            summary["skipped"] += 1
            continue

        if row.get("error"):
            summary["errors"] += 1
            continue

        summary["completed"] += 1

        if row.get("automatic_checks", {}).get("passed"):
            summary["automatic_passes"] += 1

        usage = row.get("usage") or {}

        input_tokens = int(
            usage.get("input_tokens") or 0
        )
        output_tokens = int(
            usage.get("output_tokens") or 0
        )
        cache_read_tokens = int(
            usage.get("cache_read_tokens") or 0
        )
        cache_write_tokens = int(
            usage.get("cache_write_tokens") or 0
        )

        summary["input_tokens"] += input_tokens
        summary["output_tokens"] += output_tokens
        summary["cache_read_tokens"] += cache_read_tokens
        summary["cache_write_tokens"] += cache_write_tokens

        summary["total_accounted_tokens"] += (
            input_tokens
            + output_tokens
            + cache_read_tokens
            + cache_write_tokens
        )

        summary["estimated_cost_usd"] += float(
            row.get("estimated_cost_usd") or 0
        )

        summary["latency_ms_total"] += int(
            row.get("latency_ms") or 0
        )

    for summary in summaries.values():
        completed = int(summary["completed"])
        paid_input = int(summary["input_tokens"])
        cached_input = int(summary["cache_read_tokens"])
        total_input = paid_input + cached_input

        summary["estimated_cost_usd"] = round(
            float(summary["estimated_cost_usd"]),
            8,
        )

        summary["average_cost_usd"] = round(
            (
                float(summary["estimated_cost_usd"])
                / completed
            )
            if completed
            else 0.0,
            8,
        )

        summary["average_latency_ms"] = round(
            (
                int(summary["latency_ms_total"])
                / completed
            )
            if completed
            else 0,
        )

        summary["cache_hit_rate"] = round(
            (
                cached_input / total_input
            )
            if total_input
            else 0.0,
            4,
        )

        summary["cached_input_share_percent"] = round(
            float(summary["cache_hit_rate"]) * 100,
            2,
        )

        del summary["latency_ms_total"]

    return summaries

def save_results_file(
    output_path: Path,
    *,
    run_id: str,
    rows: list[dict[str, Any]],
    status: str,
    summary: dict[str, Any] | None = None,
) -> None:
    """Atomically save evaluation progress after every model result."""

    payload: dict[str, Any] = {
        "run_id": run_id,
        "status": status,
        "results": rows,
    }

    if summary is not None:
        payload["summary"] = summary

    temporary_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    temporary_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    temporary_path.replace(output_path)



async def main() -> int:
    args = parse_args()
    scenarios = load_json(args.scenarios)

    if args.start:
        scenarios = scenarios[args.start:]

    if args.limit:
        scenarios = scenarios[: args.limit]
    targets = load_targets(args.catalog, args.models)
    if not targets:
        raise SystemExit("No enabled model targets matched the selection.")

    print(f"Scenarios: {len(scenarios)}")
    for target in targets:
        print(f"- {target.name}: {target.provider} / {target.model} / adult={target.adult_policy}")

    if args.dry_run:
        return 0

    run_id = str(uuid.uuid4())
    if args.persist:
        await persist_run_start(
            run_id,
            len(scenarios),
            len(targets),
            {"catalog": args.catalog, "scenarios": args.scenarios},
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"model_eval_{run_id}.json"

    rows: list[dict[str, Any]] = []

    save_results_file(
        output_path,
        run_id=run_id,
        rows=rows,
        status="running",
    )

    for scenario in scenarios:
        ctx = build_context(scenario)
        prompt = build_prompt(ctx)
        system = str(prompt[0]["content"])
        messages = [{"role": "user", "content": str(prompt[1]["content"])}]

        for target in targets:
            skip_reason = adult_skip_reason(target, scenario, args.allow_unverified_adult)
            base_row = {
                "run_id": run_id,
                "scenario_id": scenario["id"],
                "scenario_title": scenario["title"],
                "candidate_name": target.name,
                "provider": target.provider,
                "model": target.model,
            }
            if skip_reason:
                row = {
                    **base_row,
                    "skipped": True,
                    "skip_reason": skip_reason,
                }

                rows.append(row)

                save_results_file(
                    output_path,
                    run_id=run_id,
                    rows=rows,
                    status="running",
                )

                if args.persist:
                    await persist_output(run_id, row)

                print(
                    f"SKIP {scenario['id']} / {target.name}: "
                    f"{skip_reason}"
                )
                continue

            try:
                async with asyncio.timeout(target.timeout_seconds):
                    result = await complete(
                        target,
                        system=system,
                        messages=messages,
                        max_tokens=1000,
                    )
                replies = parse_reply_candidates(result.text, ctx.creator_persona)
                checks = automatic_checks(replies, scenario.get("checks") or {})
                row = {
                    **base_row,
                    "skipped": False,
                    "replies": replies,
                    "raw_text": result.text,
                    "automatic_checks": checks,
                    "latency_ms": result.latency_ms,
                    "estimated_cost_usd": estimate_cost_usd(target, result.usage),
                    "usage": result.usage.__dict__,
                }
                await record_model_result(
                    result,
                    ModelTelemetryContext(
                        feature="model_evaluation",
                        evaluation_run_id=run_id,
                        scenario_id=scenario["id"],
                        metadata={"candidate_name": target.name},
                    ),
                    success=bool(replies),
                    parse_valid=bool(replies),
                )
                print(
                    f"DONE {scenario['id']} / {target.name}: "
                    f"pass={checks.get('passed')} cost=${row['estimated_cost_usd']:.6f} "
                    f"latency={result.latency_ms}ms"
                )
            except TimeoutError:
                row = {
                    **base_row,
                    "skipped": False,
                    "error": (
                        f"Timed out after "
                        f"{target.timeout_seconds:.0f} seconds"
                    ),
                    "replies": [],
                }

                print(
                    f"TIMEOUT {scenario['id']} / {target.name}: "
                    f"{target.timeout_seconds:.0f}s"
                )

            except Exception as error:
                row = {
                    **base_row,
                    "skipped": False,
                    "error": str(error),
                    "replies": [],
                }

                print(
                    f"FAIL {scenario['id']} / {target.name}: "
                    f"{error}"
                )

            rows.append(row)

            save_results_file(
                output_path,
                run_id=run_id,
                rows=rows,
                status="running",
            )

            if args.persist:
                await persist_output(run_id, row)

    completed = [row for row in rows if not row.get("skipped") and not row.get("error")]
    model_summaries = build_model_summaries(rows)

    summary = {
        "completed": len(completed),
        "skipped": sum(
            bool(row.get("skipped"))
            for row in rows
        ),
        "errors": sum(
            bool(row.get("error"))
            for row in rows
        ),
        "automatic_passes": sum(
            bool(
                row.get(
                    "automatic_checks",
                    {},
                ).get("passed")
            )
            for row in completed
        ),
        "estimated_cost_usd": round(
            sum(
                float(
                    row.get("estimated_cost_usd")
                    or 0
                )
                for row in completed
            ),
            6,
        ),
        "model_summaries": model_summaries,
        "output_path": str(output_path),
    }
    print(json.dumps(summary, indent=2))

    print("\nPer-model usage summary:")

    for candidate_name, model_summary in model_summaries.items():
        print(
            f"- {candidate_name}: "
            f"completed={model_summary['completed']} "
            f"passes={model_summary['automatic_passes']} "
            f"input={model_summary['input_tokens']} "
            f"output={model_summary['output_tokens']} "
            f"cached={model_summary['cache_read_tokens']} "
            f"cache_share="
            f"{model_summary['cached_input_share_percent']:.2f}% "
            f"avg_cost="
            f"${model_summary['average_cost_usd']:.6f} "
            f"avg_latency="
            f"{model_summary['average_latency_ms']}ms"
        )

    save_results_file(
        output_path,
        run_id=run_id,
        rows=rows,
        status="completed",
        summary=summary,
    )

    if args.persist:
        await persist_run_finish(run_id, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
