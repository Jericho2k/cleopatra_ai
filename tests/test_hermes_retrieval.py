from __future__ import annotations

import json

from services.hermes_retrieval import retrieve_examples, retrieval_enabled
from scripts.import_hermes_examples import normalize


def _write_index(path):
    rows = [
        {
            "example_id": "approved-short",
            "source": "curated-chat",
            "source_type": "real_chat",
            "quality_status": "approved",
            "behavior_tags": ["short_reply_continue", "creator_initiative"],
            "situation_tags": ["ongoing_shared_scene"],
            "fan_context": "mm",
            "creator_response": "then stay here with me a second",
            "surrounding_context": [],
            "commercial_context": "",
            "notes": "",
            "provenance": {"source_message_ids": ["external-1"]},
        },
        {
            "example_id": "reference-only",
            "source": "report",
            "source_type": "guide",
            "quality_status": "reference_only",
            "behavior_tags": ["short_reply_continue"],
            "situation_tags": [],
            "fan_context": "okay",
            "creator_response": "not runtime approved",
            "surrounding_context": [],
            "commercial_context": "",
            "notes": "",
            "provenance": {},
        },
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_retrieval_is_disabled_by_default(monkeypatch, tmp_path):
    monkeypatch.delenv("HERMES_RETRIEVAL_ENABLED", raising=False)
    missing = tmp_path / "does-not-exist.jsonl"
    assert retrieval_enabled() is False
    assert retrieve_examples(current_text="mm", path=missing) == []


def test_enabled_retrieval_uses_only_approved_examples(tmp_path):
    index = tmp_path / "examples.jsonl"
    _write_index(index)
    results = retrieve_examples(
        current_text="mm",
        behavior_tags=["short_reply_continue"],
        situation_tags=["ongoing_shared_scene"],
        enabled_override=True,
        path=index,
    )
    assert [row["example_id"] for row in results] == ["approved-short"]
    assert "provenance" not in results[0]


def test_retrieved_examples_are_demonstrations_not_current_facts(tmp_path):
    index = tmp_path / "examples.jsonl"
    _write_index(index)
    [result] = retrieve_examples(
        current_text="mm",
        behavior_tags=["creator_initiative"],
        enabled_override=True,
        path=index,
    )
    assert result["fan_context"] == "mm"
    assert set(result) == {
        "example_id",
        "behavior_tags",
        "situation_tags",
        "fan_context",
        "creator_response",
        "surrounding_context",
        "commercial_context",
    }


def test_importer_requires_explicit_approval_and_provenance(tmp_path):
    source = tmp_path / "curated.json"
    row = normalize(
        {
            "source_type": "real_chat",
            "quality_status": "approved",
            "behavior_tags": ["creator_initiative"],
            "fan_context": "okay",
            "creator_response": "i'm taking that as permission to keep going",
        },
        source_file=source,
        index=0,
    )
    assert row["example_id"].startswith("hermes_")
    assert row["quality_status"] == "approved"
    assert row["provenance"]["import_file"] == str(source)
