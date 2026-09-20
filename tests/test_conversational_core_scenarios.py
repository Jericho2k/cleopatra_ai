"""The synthetic scenario suite, checked against what it claims to be.

Three kinds of claim, all of which have been wrong in this repository before
and none of which any other test would catch.

*That it is synthetic.* Every message was written for this file. A scenario that
quietly acquired a line from a real conversation would be a private chat log in
a public repository, and no evaluation is worth that.

*That the labels are true.* ``eval/trajectories.json`` once carried "40-80 turns
of ordinary conversation" above a seven-turn script. The coverage machinery in
``services/trajectory_eval.py`` catches that at run time; these tests catch it at
commit time.

*That the suite is not all one thing.* The first Core v1 increment is a
conversational change. A suite where every scenario ended in a sale would
measure the commercial path and report it as a conversational result.
"""

from __future__ import annotations

import json
from pathlib import Path

from services.ab_trajectory_eval import load_scenario_file, select_trajectories
from services.trajectory_eval import TrajectoryReport, TurnRecord, coverage_gaps

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ROOT / "eval" / "conversational_core_scenarios.json"

#: The brief's twelve scenarios, by the letter each one answers to.
REQUIRED_PREFIXES = tuple("ABCDEFGHIJKL")


def _payload() -> dict:
    return load_scenario_file(SCENARIOS)


def _rows() -> list[dict]:
    return _payload()["trajectories"]


def test_every_required_scenario_exists_exactly_once():
    ids = [row["id"] for row in _rows()]
    assert len(ids) == len(set(ids))
    assert [scenario_id.split("_")[0] for scenario_id in ids] == list(REQUIRED_PREFIXES)


def test_every_scenario_declares_what_it_covers_and_what_each_turn_tests():
    for row in _rows():
        assert row.get("covers"), row["id"]
        assert row.get("rubric_focus"), row["id"]
        for index, item in enumerate(row["turns"]):
            assert item.get("tests"), f"{row['id']} turn {index}"


def test_every_coverage_claim_is_reachable_by_the_script_that_declares_it():
    """A 22-turn claim above a 22-turn script, checked rather than asserted.

    Run against the best case: every turn succeeds, the clock is available, the
    seed is applied. Anything the fixture still cannot reach is a label that is
    not true, and would be printed as [NOT COVERED] on every run.
    """
    trajectories, ids = select_trajectories(_payload())
    for scenario_id, trajectory in zip(ids, trajectories):
        elapsed = sum(
            float(item.days_since_previous or 0.0) for item in trajectory.disturbances
        )
        best_case = TrajectoryReport(
            trajectory=trajectory.name,
            covers=trajectory.covers,
            turns=[
                TurnRecord(index=index, customer_message=item.message)
                for index, item in enumerate(trajectory.disturbances)
            ],
            clock_injected=True,
            elapsed_days=elapsed,
            due_worker_runs=sum(
                1 for item in trajectory.disturbances if not str(item.message or "").strip()
            ),
            seeded_purchases=list(trajectory.seed.get("purchases") or []),
        )
        gaps = coverage_gaps(trajectory, best_case)
        assert not gaps, f"{scenario_id}: {[gap.render() for gap in gaps]}"


def test_the_long_trajectory_is_actually_long_and_contains_the_named_disturbances():
    trajectories, ids = select_trajectories(_payload(), scenario_ids=["L_long_trajectory"])
    long_run = trajectories[0]
    assert len(long_run.disturbances) >= 20
    assert any(item.corrects for item in long_run.disturbances), "no correction"
    assert any(item.days_since_previous >= 7 for item in long_run.disturbances), "no week gap"
    assert any(
        len(item.message.split()) <= 3 for item in long_run.disturbances if item.message
    ), "no short reply"
    assert any(item.raises_obligation for item in long_run.disturbances), "no callback obligation"


def test_the_correction_scenario_states_what_replaced_the_corrected_fact():
    trajectories, _ = select_trajectories(_payload(), scenario_ids=["G_correction"])
    corrections = [item for item in trajectories[0].disturbances if item.corrects]
    assert corrections
    for item in corrections:
        # Both halves, so a reply discussing the change is not reported as
        # reasserting it — see find_reasserted_corrections.
        assert item.corrected_to


def test_the_post_event_scenario_seeds_an_authoritative_purchase():
    """A claim of payment is not a purchase.

    The whole point of K is that the conversation starts on the far side of a
    real event, which means the ledger — not a fan message saying "i paid".
    """
    row = next(item for item in _rows() if item["id"].startswith("K_"))
    assert row["requires"]["authoritative_purchase"] is True
    assert row["seed"]["purchases"][0]["price_cents"] > 0
    assert not any("i paid" in turn["message"].lower() for turn in row["turns"])


def test_most_scenarios_are_neither_commercial_nor_explicitly_intimate():
    payload = _payload()
    commercial = set(payload["suites"]["commercial"])
    assert len(commercial) == 3
    assert len(commercial) < len(payload["suites"]["conversational"])


def test_the_suites_only_name_scenarios_that_exist():
    payload = _payload()
    ids = {row["id"] for row in payload["trajectories"]}
    for name, members in payload["suites"].items():
        if name == "_about":
            continue
        assert set(members) <= ids, name


def test_no_scenario_carries_a_creator_or_fan_id():
    """Fixtures name no target.

    A scenario file that pinned a fan id would run against that fan wherever it
    was copied, which is how an evaluation reaches a conversation nobody meant
    to touch.
    """
    for row in _rows():
        assert not row.get("fan_id"), row["id"]
        assert not row.get("creator_id"), row["id"]


def test_the_scenarios_contain_no_identifiers_that_could_come_from_real_data():
    """Synthetic means synthetic: no handles, links, emails or long digit runs."""
    import re

    text = SCENARIOS.read_text(encoding="utf-8")
    messages = " ".join(
        turn["message"] for row in json.loads(text)["trajectories"] for turn in row["turns"]
    )
    assert "@" not in messages
    assert "http" not in messages
    assert not re.search(r"\d{6,}", messages)
    assert "fansly" not in messages.lower()
