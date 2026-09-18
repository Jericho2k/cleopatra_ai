"""Real-provider, side-effect-free comparison of complete conversation cores.

This module deliberately has no database or platform adapter dependency. It
reads authored/recorded replay turns, calls configured inference providers, and
writes local evidence files. "Shadow" therefore means shadow inference only:
no live reply, state transition, entitlement change, or production mutation is
possible through this entry point.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, Sequence

from models.conversation_decision import ConversationDecision
from models.model_runtime import ModelResult, ModelTarget, resolve_cost_usd
from services.candidate_execution import ExecutionReport, compare_candidates
from services.decision_owners import (
    DecideThenWrite,
    ReplyPlusIntentOwner,
    SemanticDecisionOwner,
    build_semantic_prompt,
)

Complete = Callable[..., Awaitable[ModelResult]]

WRITER_SYSTEM = """You are the creator replying to one customer on a paid content platform.

Write only the message itself. Follow the supplied decision exactly. Do not add an operation, price, delivery claim, promise, or subject that the decision does not support. If the decision says to hold, return an empty string."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _target_dict(target: ModelTarget) -> dict[str, Any]:
    return {
        "name": target.name,
        "provider": target.provider,
        "model": target.model,
        "base_url": target.base_url,
        "adult_policy": target.adult_policy,
    }


@dataclass
class ProviderCall:
    candidate: str
    provider: str
    model: str
    latency_ms: int
    gate_wait_ms: int
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    cost_usd: float
    response_id: str | None
    upstream_provider: str | None


@dataclass
class ProviderRecorder:
    """Record actual provider usage without changing candidate behavior."""

    complete: Complete
    calls: list[ProviderCall] = field(default_factory=list)

    def bound(self, candidate: str) -> Complete:
        async def recorded(target: ModelTarget, **kwargs: Any) -> ModelResult:
            result = await self.complete(target, **kwargs)
            usage = result.usage
            self.calls.append(
                ProviderCall(
                    candidate=candidate,
                    provider=result.target.provider,
                    model=result.target.model,
                    latency_ms=result.latency_ms,
                    gate_wait_ms=result.gate_wait_ms,
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    cache_read_tokens=usage.cache_read_tokens,
                    cache_write_tokens=usage.cache_write_tokens,
                    cost_usd=resolve_cost_usd(
                        result.target,
                        usage,
                        reported_cost_usd=result.reported_cost_usd,
                    ),
                    response_id=result.raw_response_id,
                    upstream_provider=result.upstream_provider,
                )
            )
            return result

        return recorded


def _one_call_adapter(complete: Complete) -> Callable[..., Awaitable[ModelResult]]:
    async def call(*, system: str, user: str, target: ModelTarget) -> ModelResult:
        return await complete(
            target,
            system=system,
            messages=[{"role": "user", "content": user}],
            max_tokens=900,
        )

    return call


def _shared_writer(
    complete: Complete, target: ModelTarget
) -> Callable[[ConversationDecision, Any, dict[str, Any]], Awaitable[str]]:
    async def write(
        decision: ConversationDecision, packet: Any, state: dict[str, Any]
    ) -> str:
        if decision.is_hold:
            return ""
        _, evidence = build_semantic_prompt(packet, state)
        user = (
            f"DECISION (typed, must be followed):\n"
            f"{json.dumps(decision.as_dict(), sort_keys=True)}\n\n"
            f"EVIDENCE:\n{evidence}"
        )
        result = await complete(
            target,
            system=WRITER_SYSTEM,
            messages=[{"role": "user", "content": user}],
            max_tokens=500,
        )
        return result.text.strip()

    return write


async def run_provider_comparison(
    turns: Sequence[Any],
    *,
    complete: Complete,
    one_call_target: ModelTarget,
    semantic_target: ModelTarget,
    writer_target: ModelTarget,
) -> tuple[ExecutionReport, list[ProviderCall]]:
    """Run both complete cores through real configured provider call seams."""

    recorder = ProviderRecorder(complete)
    one_call = ReplyPlusIntentOwner(
        _one_call_adapter(recorder.bound("reply_plus_intent")),
        target=one_call_target,
    )
    two_call = DecideThenWrite(
        SemanticDecisionOwner(
            recorder.bound("semantic_owner"), target=semantic_target
        )
    )
    writer = _shared_writer(recorder.bound("semantic_owner_writer"), writer_target)
    report = await compare_candidates(turns, [one_call, two_call], write=writer)
    return report, recorder.calls


def git_state(root: Path) -> dict[str, Any]:
    """Return the exact checked-out source revision and whether it is modified."""

    def run(*args: str) -> str:
        return subprocess.check_output(
            ["git", *args], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()

    try:
        return {
            "sha": run("rev-parse", "HEAD"),
            "branch": run("branch", "--show-current"),
            "dirty": bool(run("status", "--porcelain")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"sha": "unknown", "branch": "unknown", "dirty": None}


def build_evaluation_bundle(
    output_dir: Path,
    *,
    root: Path,
    scenarios_path: Path,
    scenario_names: Sequence[str],
    report: ExecutionReport,
    calls: Sequence[ProviderCall],
    targets: dict[str, ModelTarget],
    flags: dict[str, str],
) -> dict[str, Any]:
    """Write a reproducible local bundle with inputs, outputs, and telemetry."""

    output_dir.mkdir(parents=True, exist_ok=False)
    migration_order = root / "db" / "migration_order.txt"
    source = git_state(root)
    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "mode": "provider_shadow_no_live_sends_no_state_mutations",
        "source": source,
        "flags": flags,
        "scenarios": {
            "path": str(scenarios_path.resolve()),
            "sha256": _sha256(scenarios_path),
            "names": list(scenario_names),
        },
        "database_schema": {
            "migration_order_sha256": _sha256(migration_order),
            "last_migration": [
                line.strip()
                for line in migration_order.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            ][-1],
        },
        "targets": {name: _target_dict(target) for name, target in targets.items()},
        "invariants": {
            "live_messages_sent": 0,
            "production_mutations": 0,
            "candidate_state_isolated_per_turn": True,
            "human_selection_required": True,
        },
    }
    results = {
        "summary": report.summary(),
        "turns": report.turns,
        "provider_calls": [asdict(call) for call in calls],
        "provider_totals": {
            "calls": len(calls),
            "latency_ms": sum(call.latency_ms for call in calls),
            "input_tokens": sum(call.input_tokens for call in calls),
            "output_tokens": sum(call.output_tokens for call in calls),
            "cost_usd": round(sum(call.cost_usd for call in calls), 8),
        },
    }
    disagreements = [turn for turn in report.turns if turn["disagreements"]]

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "disagreements.json").write_text(
        json.dumps(disagreements, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "README.md").write_text(
        "# Cleopatra candidate evaluation bundle\n\n"
        "This is a provider-shadow evaluation. It made inference calls only; "
        "it sent no live messages and performed no production mutations.\n\n"
        "`manifest.json` pins the source, flags, schema ordering, scenario file, "
        "and model targets. `results.json` contains every reply, decision, "
        "dry-run operation outcome, and provider usage record. "
        "`disagreements.json` is the human-review queue. No winner is selected "
        "automatically.\n",
        encoding="utf-8",
    )
    return manifest
