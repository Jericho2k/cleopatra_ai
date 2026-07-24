from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_blind_review_module():
    path = ROOT / "scripts" / "build_blind_review.py"
    spec = importlib.util.spec_from_file_location("build_blind_review", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_realism_scenarios_cover_common_ai_tells():
    payload = json.loads((ROOT / "eval" / "realism_scenarios.json").read_text())
    scenario_ids = {row["id"] for row in payload}

    assert len(payload) >= 10
    assert {
        "plain_acknowledgement",
        "selective_reply",
        "no_forced_question",
        "avoid_unearned_expertise",
        "no_therapy_voice",
        "money_transition_stays_human",
    }.issubset(scenario_ids)


def test_blind_review_explicitly_stays_out_of_production():
    module = _load_blind_review_module()
    candidate = module.render_candidate({"replies": ["fair lol"]}, "A")
    rendered = "\n".join(candidate)

    assert "Would you suspect AI?" in rendered
    assert "AI tells, if any" in rendered

    source = (ROOT / "scripts" / "build_blind_review.py").read_text()
    assert "does not score, block, or reroute production replies" in source
