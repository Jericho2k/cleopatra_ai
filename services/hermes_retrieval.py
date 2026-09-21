"""Optional, application-owned retrieval of approved Hermes exemplars.

The production default is deliberately off.  Records are local JSONL, are
eligible only when explicitly marked ``approved``, and are demonstrations for
Kimi rather than evidence about the current creator, fan, inventory, or money.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

HERMES_RETRIEVAL_ENV = "HERMES_RETRIEVAL_ENABLED"
HERMES_INDEX_ENV = "HERMES_EXAMPLE_INDEX"
DEFAULT_INDEX = Path(__file__).resolve().parents[1] / "data" / "hermes_examples.jsonl"
MAX_EXAMPLES = 4


@dataclass(frozen=True)
class HermesExample:
    example_id: str
    source: str
    source_type: str
    quality_status: str
    behavior_tags: tuple[str, ...] = ()
    situation_tags: tuple[str, ...] = ()
    fan_context: str = ""
    creator_response: str = ""
    surrounding_context: tuple[dict[str, str], ...] = ()
    commercial_context: str = ""
    notes: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "HermesExample":
        return cls(
            example_id=str(value.get("example_id") or "").strip(),
            source=str(value.get("source") or "").strip(),
            source_type=str(value.get("source_type") or "").strip(),
            quality_status=str(value.get("quality_status") or "").strip(),
            behavior_tags=tuple(str(v).strip() for v in value.get("behavior_tags") or [] if str(v).strip()),
            situation_tags=tuple(str(v).strip() for v in value.get("situation_tags") or [] if str(v).strip()),
            fan_context=str(value.get("fan_context") or "").strip(),
            creator_response=str(value.get("creator_response") or "").strip(),
            surrounding_context=tuple(
                {"speaker": str(row.get("speaker") or ""), "text": str(row.get("text") or "")}
                for row in value.get("surrounding_context") or []
                if isinstance(row, dict)
            ),
            commercial_context=str(value.get("commercial_context") or "").strip(),
            notes=str(value.get("notes") or "").strip(),
            provenance=dict(value.get("provenance") or {}),
        )

    def eligible(self) -> bool:
        return bool(
            self.example_id
            and self.quality_status == "approved"
            and self.fan_context
            and self.creator_response
        )

    def writer_view(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "behavior_tags": list(self.behavior_tags),
            "situation_tags": list(self.situation_tags),
            "fan_context": self.fan_context,
            "creator_response": self.creator_response,
            "surrounding_context": [dict(row) for row in self.surrounding_context],
            "commercial_context": self.commercial_context,
        }


def retrieval_enabled(override: bool | None = None) -> bool:
    if override is not None:
        return bool(override)
    return os.getenv(HERMES_RETRIEVAL_ENV, "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def index_path(override: str | Path | None = None) -> Path:
    if override is not None:
        return Path(override)
    configured = os.getenv(HERMES_INDEX_ENV, "").strip()
    return Path(configured) if configured else DEFAULT_INDEX


def load_approved_examples(path: str | Path | None = None) -> list[HermesExample]:
    location = index_path(path)
    if not location.exists():
        return []
    examples: list[HermesExample] = []
    for line in location.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except (TypeError, ValueError):
            continue
        if isinstance(value, dict):
            example = HermesExample.from_dict(value)
            if example.eligible():
                examples.append(example)
    return examples


def _terms(values: Iterable[str]) -> set[str]:
    return {
        token
        for value in values
        for token in re.findall(r"[a-z0-9_']{3,}", str(value).lower())
    }


def retrieve_examples(
    *,
    current_text: str,
    behavior_tags: Iterable[str] = (),
    situation_tags: Iterable[str] = (),
    enabled_override: bool | None = None,
    path: str | Path | None = None,
    limit: int = MAX_EXAMPLES,
) -> list[dict[str, Any]]:
    if not retrieval_enabled(enabled_override):
        return []
    wanted_behavior = set(behavior_tags)
    wanted_situation = set(situation_tags)
    query_terms = _terms([current_text, *wanted_behavior, *wanted_situation])

    def score(example: HermesExample) -> tuple[int, str]:
        behavior = len(wanted_behavior.intersection(example.behavior_tags)) * 8
        situation = len(wanted_situation.intersection(example.situation_tags)) * 6
        lexical = len(
            query_terms.intersection(
                _terms(
                    [
                        example.fan_context,
                        example.commercial_context,
                        *example.behavior_tags,
                        *example.situation_tags,
                    ]
                )
            )
        )
        return behavior + situation + lexical, example.example_id

    ranked = sorted(load_approved_examples(path), key=score, reverse=True)
    keep = max(0, min(int(limit), MAX_EXAMPLES))
    return [row.writer_view() for row in ranked[:keep] if score(row)[0] > 0]

