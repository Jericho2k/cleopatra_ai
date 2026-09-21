#!/usr/bin/env python3
"""Normalize explicitly curated Hermes examples into the runtime JSONL index."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ALLOWED_SOURCE_TYPES = {"real_chat", "agency_material", "guide", "synthetic"}
ALLOWED_QUALITY = {"approved", "reference_only", "rejected"}


def _records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        values = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        decoded = json.loads(text)
        values = decoded.get("examples", []) if isinstance(decoded, dict) else decoded
    if not isinstance(values, list) or any(not isinstance(row, dict) for row in values):
        raise ValueError(f"{path} must contain a JSON list (or JSONL) of objects")
    return values


def normalize(row: dict[str, Any], *, source_file: Path, index: int) -> dict[str, Any]:
    source_type = str(row.get("source_type") or "").strip()
    quality = str(row.get("quality_status") or "reference_only").strip()
    if source_type not in ALLOWED_SOURCE_TYPES:
        raise ValueError(f"record {index}: unsupported source_type {source_type!r}")
    if quality not in ALLOWED_QUALITY:
        raise ValueError(f"record {index}: unsupported quality_status {quality!r}")
    fan_context = " ".join(str(row.get("fan_context") or "").split())
    creator_response = " ".join(str(row.get("creator_response") or "").split())
    if quality == "approved" and (not fan_context or not creator_response):
        raise ValueError(f"record {index}: approved examples require context and response")
    seed = f"{source_file}:{index}:{fan_context}:{creator_response}"
    example_id = str(row.get("example_id") or "").strip() or (
        "hermes_" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
    )
    return {
        "example_id": example_id,
        "source": str(row.get("source") or source_file.name),
        "source_type": source_type,
        "quality_status": quality,
        "behavior_tags": sorted({str(v).strip() for v in row.get("behavior_tags") or [] if str(v).strip()}),
        "situation_tags": sorted({str(v).strip() for v in row.get("situation_tags") or [] if str(v).strip()}),
        "fan_context": fan_context,
        "creator_response": creator_response,
        "surrounding_context": [dict(v) for v in row.get("surrounding_context") or [] if isinstance(v, dict)],
        "commercial_context": " ".join(str(row.get("commercial_context") or "").split()),
        "notes": " ".join(str(row.get("notes") or "").split()),
        "provenance": {**dict(row.get("provenance") or {}), "import_file": str(source_file)},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    normalized = [
        normalize(row, source_file=path, index=index)
        for path in args.inputs
        for index, row in enumerate(_records(path))
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in normalized),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "records": len(normalized),
                "approved": sum(row["quality_status"] == "approved" for row in normalized),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
