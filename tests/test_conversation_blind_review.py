"""The blind conversation review, and the optional judge that reads the same thing.

A blind review that is not blind is worse than no review: it produces a
confident answer from a reviewer who was told what to conclude. So the tests
that matter here are the ones about what the reviewer-facing document does NOT
contain, and about the mapping being the only way back.

The ordering tests exist for a second reason. A reviewer works through a
document over days; if rebuilding it produced different labels, their notes
would silently start referring to the other conversation.
"""

from __future__ import annotations

import pytest

from services.blind_conversation_review import (
    ALWAYS_FORBIDDEN,
    DIMENSIONS,
    LABELS,
    build_review,
    forbidden_terms,
    leaked_terms,
    order_for,
    redact,
    unblind,
    unblind_all,
    write_review,
)
from services.conversation_judge import build_prompt, parse_judgement, unblind_judgement


def _turns(prefix: str, count: int = 3) -> list[dict]:
    return [
        {
            "turn_index": index,
            "fan_input": f"fan line {index}",
            "creator_output": [f"{prefix} reply {index}"],
        }
        for index in range(count)
    ]


def _paired(scenarios=("alpha", "beta", "gamma")) -> dict:
    return {
        "paired": True,
        "scenarios": len(scenarios),
        "input_mismatches": [],
        "fan_inputs_identical": True,
        "pairs": [
            {
                "scenario_id": scenario,
                "name": f"scenario {scenario}",
                "covers": "something",
                "fan_inputs_identical": True,
                "arms": {
                    "baseline": {
                        "conversation_core": "semantic_v2",
                        "fan_id": "fan-a",
                        "turns": _turns("old"),
                    },
                    "candidate": {
                        "conversation_core": "conversational_v1",
                        "fan_id": "fan-b",
                        "turns": _turns("new"),
                    },
                },
            }
            for scenario in scenarios
        ],
    }


# --- 4. deterministic ordering ---------------------------------------------


def test_the_same_seed_produces_the_same_labels_every_time():
    first, first_map = build_review(_paired(), seed=42, run_id="r")
    second, second_map = build_review(_paired(), seed=42, run_id="r")
    assert first == second
    assert unblind_all(first_map) == unblind_all(second_map)


def test_a_different_seed_can_produce_a_different_assignment():
    """Not merely deterministic — actually shuffled.

    A "random" ordering that always produced the same answer would pass a
    determinism test and would hand the same label to the same runtime in every
    run, which is the bias the shuffle exists to remove.
    """
    assignments = {
        tuple(order_for("alpha", seed=seed, roles=["baseline", "candidate"]))
        for seed in range(30)
    }
    assert len(assignments) == 2


def test_labels_are_assigned_per_scenario_rather_than_per_run():
    _, mapping = build_review(_paired(tuple(f"s{index}" for index in range(12))), seed=5, run_id="r")
    by_scenario = unblind_all(mapping)
    first_labels = {scenario: labels["A"] for scenario, labels in by_scenario.items()}
    # If the label were fixed for the run, every scenario would name the same
    # runtime as A, and a reviewer who worked one out would have worked them all
    # out.
    assert len(set(first_labels.values())) == 2


# --- 5. the mapping unblinds -----------------------------------------------


def test_the_mapping_names_the_runtime_behind_each_label():
    _, mapping = build_review(_paired(("alpha",)), seed=3, run_id="r")
    a = unblind(mapping, "alpha", "A")
    b = unblind(mapping, "alpha", "B")
    assert {a["conversation_core"], b["conversation_core"]} == {
        "semantic_v2",
        "conversational_v1",
    }
    assert {a["role"], b["role"]} == {"baseline", "candidate"}
    assert a["conversation_core"] != b["conversation_core"]


def test_unblinding_an_unknown_scenario_is_an_error_not_a_guess():
    _, mapping = build_review(_paired(("alpha",)), seed=3, run_id="r")
    with pytest.raises(KeyError):
        unblind(mapping, "nope", "A")


def test_the_mapping_round_trips_through_files(tmp_path):
    review_path = tmp_path / "blind_review.md"
    mapping_path = tmp_path / "blind_mapping.json"
    document, mapping = write_review(
        _paired(),
        seed=11,
        run_id="run-x",
        review_path=review_path,
        mapping_path=mapping_path,
    )
    import json

    assert review_path.read_text(encoding="utf-8") == document
    assert json.loads(mapping_path.read_text(encoding="utf-8")) == mapping
    # Two files, deliberately. The key is not in the document a reviewer opens.
    assert "semantic_v2" not in review_path.read_text(encoding="utf-8")
    assert "semantic_v2" in mapping_path.read_text(encoding="utf-8")


# --- 6. the reviewer artifact does not identify the runtimes ---------------


def test_the_review_document_names_no_runtime_and_no_role():
    document, _ = build_review(_paired(), seed=9, run_id="r")
    terms = forbidden_terms(["semantic_v2", "conversational_v1", "semantic_v1", "legacy"])
    assert leaked_terms(document, terms) == []
    for word in ALWAYS_FORBIDDEN:
        assert word not in document.lower()
    assert "Conversation A" in document and "Conversation B" in document


def test_the_review_document_carries_no_model_latency_or_fan_identifiers():
    """A fingerprint is an unblinding too.

    Fan ids are constant per arm across the whole suite, so printing one would
    let a reviewer group every scenario by runtime after noticing a single one.
    """
    document, _ = build_review(_paired(), seed=9, run_id="r")
    for leak in ("fan-a", "fan-b", "latency", "model", "cost"):
        assert leak not in document.lower()


def test_a_runtime_name_inside_the_conversation_text_is_redacted_not_leaked():
    paired = _paired(("alpha",))
    paired["pairs"][0]["arms"]["baseline"]["turns"][0]["creator_output"] = [
        "honestly the semantic_v2 of coffee shops"
    ]
    document, _ = build_review(paired, seed=1, run_id="r")
    assert "semantic_v2" not in document
    assert "[redacted]" in document


def test_redaction_matches_whole_words_only():
    assert redact("candidates for baseline", ["baseline"]) == "candidates for [redacted]"
    assert redact("a candidature", ["candidate"]) == "a candidature"


def test_the_review_does_not_tell_the_reviewer_which_side_should_win():
    document, _ = build_review(_paired(), seed=2, run_id="r")
    lowered = document.lower()
    for hint in ("expected", "improved", "new version", "should be better", "we hope"):
        assert hint not in lowered


def test_every_rubric_dimension_appears_with_its_anchors():
    document, _ = build_review(_paired(("alpha",)), seed=2, run_id="r")
    assert len(DIMENSIONS) >= 12
    for dimension in DIMENSIONS:
        assert dimension.title in document
        assert dimension.anchor_low in document
        assert dimension.anchor_high in document
    # Separate dimensions, no single magic number.
    assert "overall score" not in document.lower()
    assert "total score" not in document.lower()


def test_a_single_arm_scenario_is_skipped_rather_than_half_rendered():
    paired = _paired(("alpha",))
    del paired["pairs"][0]["arms"]["candidate"]
    document, mapping = build_review(paired, seed=4, run_id="r")
    assert mapping["scenarios"]["alpha"] == {"skipped": "only one arm ran"}
    assert "Conversation B" not in document


def test_an_input_mismatch_is_flagged_to_the_reviewer():
    paired = _paired(("alpha",))
    paired["pairs"][0]["fan_inputs_identical"] = False
    document, _ = build_review(paired, seed=4, run_id="r")
    assert "did not receive identical fan" in document


# --- the optional judge ----------------------------------------------------


def test_the_judge_prompt_is_blinded_and_carries_the_same_rubric():
    pair = _paired(("alpha",))["pairs"][0]
    system, messages, mapping = build_prompt(pair, seed=8)
    prompt = system + messages[0]["content"]
    terms = forbidden_terms(["semantic_v2", "conversational_v1"])
    assert leaked_terms(prompt, terms) == []
    assert set(mapping) == set(LABELS)
    assert sorted(mapping.values()) == ["baseline", "candidate"]
    for dimension in DIMENSIONS:
        assert dimension.key in prompt


def test_the_judge_is_not_asked_for_hidden_reasoning_or_a_winner():
    system, messages, _ = build_prompt(_paired(("alpha",))["pairs"][0], seed=8)
    lowered = (system + messages[0]["content"]).lower()
    for banned in ("chain of thought", "step by step", "think through", "scratchpad"):
        assert banned not in lowered
    assert "do not produce an overall score" in lowered


def test_the_judge_labels_match_the_human_review_labels_for_one_seed():
    """The judge and the reviewer must be reading the same A and the same B.

    Otherwise comparing the two readings compares a judgement of one runtime
    with a human reading of the other.
    """
    pair = _paired(("alpha",))["pairs"][0]
    _, _, judge_map = build_prompt(pair, seed=21)
    _, mapping = build_review(_paired(("alpha",)), seed=21, run_id="r")
    human = {label: side["role"] for label, side in mapping["scenarios"]["alpha"].items()}
    assert judge_map == human


def test_a_judge_response_is_parsed_and_unknown_dimensions_are_dropped():
    parsed = parse_judgement(
        """Here you go:
        {"A": {"specificity": {"rating": 4, "reason": "picks up the trainers"},
               "vibes": {"rating": 5, "reason": "invented"}},
         "B": {"specificity": {"rating": 2, "reason": "generic"}}}"""
    )
    assert parsed["A"]["specificity"]["rating"] == 4
    assert "vibes" not in parsed["A"]
    assert parsed["B"]["specificity"]["rating"] == 2


def test_an_unparseable_judge_response_is_reported_rather_than_invented():
    parsed = parse_judgement("I would rather not answer in JSON.")
    assert "error" in parsed
    assert "raw" in parsed


def test_a_judgement_is_unblinded_through_its_own_mapping():
    judgement = {"A": {"specificity": {"rating": 4}}, "B": {"specificity": {"rating": 2}}}
    by_arm = unblind_judgement(judgement, {"A": "candidate", "B": "baseline"})
    assert by_arm["candidate"]["specificity"]["rating"] == 4
    assert by_arm["baseline"]["specificity"]["rating"] == 2
