#!/usr/bin/env python3
"""Live smoke for the conversational-owner route: is this combination reliable?

Not part of the automated suite: it spends real credits and needs a real
provider key, so CI must never run it.

    OPENROUTER_API_KEY=... python scripts/owner_route_smoke.py --runs 20

WHAT IT ANSWERS
---------------
Whether *this model, on this upstream, with these request parameters* returns
usable structured output repeatedly — not once. The failure it exists to catch
is intermittent by nature: reasoning is mandatory for the owner model and shares
one token budget with the visible answer, so the same request can return clean
JSON, then prose, then ``message.content = null`` with ``finish_reason =
"length"`` after ninety seconds. One successful call proves nothing about that,
which is why this runs the same request many times and reports rates.

For each call it prints the full structural record — finish reason, content
length, whether content was null, reasoning characters and tokens, completion
tokens, which message fields were populated, the upstream that served it, the
response id and the latency — and then whether
``services.owner_contract`` could read a reply, an operation and a delta out of
it. Exits non-zero if any call returned unusable content, or if the reasoning
cap is missing from the resolved target.

No real conversation is sent: the evidence payload is synthetic, and nothing
the model returns is printed except its length and structure.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai.model_providers import complete  # noqa: E402
from ai.stack_profiles import (  # noqa: E402
    STAGE_CONVERSATIONAL_OWNER,
    environment_profile_id,
    get_profile,
)
from models.live_orchestration import EvidenceFact, EvidenceSnapshot, TurnTrigger  # noqa: E402
from models.model_runtime import resolve_cost_usd  # noqa: E402
from services.conversational_core import (  # noqa: E402
    empty_working_state,
    evidence_catalog_view,
    state_fingerprint,
)
from services.live_orchestration import CONVERSATIONAL_V1_SYSTEM  # noqa: E402
from services.owner_contract import extract_owner_result  # noqa: E402

SYNTHETIC_MESSAGES: tuple[str, ...] = (
    "i keep thinking about that story you never finished",
    "what have you been up to",
    "mm fair enough",
    "tell me something i wouldn't guess about you",
    "i wasn't sure how to answer that",
)


def synthetic_payload(index: int) -> str:
    snapshot = EvidenceSnapshot(
        creator_id="smoke-creator",
        fan_id="smoke-fan",
        trigger=TurnTrigger(
            kind="fan_message",
            identity=f"smoke-msg-{index}",
            latest_message=SYNTHETIC_MESSAGES[index % len(SYNTHETIC_MESSAGES)],
        ),
        state_revision=f"smoke-revision-{index}",
        creator_facts=(
            EvidenceFact(
                value="favourite season: late autumn",
                source_ref="creator_legend:favourite_season",
                certainty="creator_confirmed",
            ),
        ),
    )
    state = empty_working_state()
    return json.dumps(
        {
            "evidence_snapshot": snapshot.as_dict(),
            "evidence_catalog": evidence_catalog_view(snapshot),
            "working_state": state.as_dict(),
            "working_state_fingerprint": state_fingerprint(state),
            "legal_operations": ["none"],
        },
        ensure_ascii=False,
        default=str,
    )


def percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))]


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument(
        "--profile",
        default=None,
        help="AI stack profile id (default: the deployment's AI_STACK_PROFILE)",
    )
    args = parser.parse_args()

    profile = get_profile(args.profile or environment_profile_id())
    spec = profile.stage(STAGE_CONVERSATIONAL_OWNER)
    target = spec.primary_target()
    max_tokens = spec.resolved_max_tokens()

    key_env = target.api_key_env or ""
    if not os.getenv(key_env, "").strip():
        print(f"{key_env} is not set; nothing to smoke test.")
        return 2

    reasoning_cap = int((target.metadata or {}).get("reasoning_max_tokens") or 0)
    print(f"profile          : {profile.profile_id}")
    print(f"provider         : {target.provider}")
    print(f"model            : {target.model}")
    print(f"reasoning        : {target.metadata.get('reasoning_enabled')} "
          f"effort={target.metadata.get('reasoning_effort')} cap={reasoning_cap or 'none'}")
    print(f"max_tokens       : {max_tokens}")
    print(f"timeout_seconds  : {target.timeout_seconds}")
    print(f"response_format  : json_object")
    print()

    failures: list[str] = []
    if target.metadata.get("reasoning_enabled") and not reasoning_cap:
        failures.append(
            "reasoning is enabled with no reasoning_max_tokens: the trace can "
            "consume the whole completion budget"
        )

    latencies: list[int] = []
    empty = 0
    unusable = 0
    total_cost = 0.0

    for index in range(int(args.runs)):
        try:
            result = await complete(
                target,
                system=CONVERSATIONAL_V1_SYSTEM,
                messages=[{"role": "user", "content": synthetic_payload(index)}],
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                session_id="owner-smoke",
            )
        except Exception as error:
            print(f"run {index}: FAILED {type(error).__name__}: {error}")
            failures.append(f"run {index}: {type(error).__name__}")
            continue

        diagnostics = result.diagnostics
        extracted = extract_owner_result(
            result.text, source="conversational_owner_v1"
        )
        latencies.append(diagnostics.latency_ms)
        total_cost += resolve_cost_usd(
            target, result.usage, reported_cost_usd=result.reported_cost_usd
        )
        if diagnostics.content_empty:
            empty += 1
        if not extracted.usable:
            unusable += 1
            failures.append(
                f"run {index}: {diagnostics.empty_content_category() or extracted.failure_category}"
            )

        print(f"run {index}: {diagnostics.describe()}")
        print(f"        {extracted.describe()}")

    print()
    print(
        json.dumps(
            {
                "runs": args.runs,
                "empty_content_rate": round(empty / max(args.runs, 1), 4),
                "unusable_rate": round(unusable / max(args.runs, 1), 4),
                "latency_ms_p50": percentile(latencies, 0.50),
                "latency_ms_p95": percentile(latencies, 0.95),
                "latency_ms_max": max(latencies) if latencies else 0,
                "total_cost_usd": round(total_cost, 6),
            },
            indent=2,
        )
    )

    if failures:
        print("\nOWNER ROUTE SMOKE FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        print(
            "\nIf content is empty with completion_tokens at the limit, the trace "
            "consumed the budget: lower reasoning_max_tokens or raise "
            "CONVERSATIONAL_OWNER_MAX_TOKENS. If the provider error names routing, "
            "OPENROUTER_REQUIRE_PARAMETERS=false widens the upstream pool."
        )
        return 1

    print("\nOWNER ROUTE SMOKE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
