import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_model_eval.py"
spec = importlib.util.spec_from_file_location("run_model_eval", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def test_question_forbidden_and_price_limit():
    checks = module.automatic_checks(
        ["friday isn't far", "i'll remember it", "we can pick this up then"],
        {
            "question_forbidden": True,
            "max_price_usd": 28,
            "refusal_forbidden": True,
        },
    )
    assert checks["passed"] is True


def test_forbidden_phrase_fails():
    checks = module.automatic_checks(
        ["wait until friday", "okay", "fine"],
        {"must_not_contain": ["wait until friday"]},
    )
    assert checks["passed"] is False


def test_build_model_summaries_tracks_cache_and_cost():
    rows = [
        {
            "candidate_name": "Test model",
            "provider": "together",
            "model": "test/model",
            "skipped": False,
            "usage": {
                "input_tokens": 600,
                "output_tokens": 100,
                "cache_read_tokens": 400,
                "cache_write_tokens": 0,
            },
            "estimated_cost_usd": 0.002,
            "latency_ms": 1000,
            "automatic_checks": {
                "passed": True,
            },
        },
        {
            "candidate_name": "Test model",
            "provider": "together",
            "model": "test/model",
            "skipped": False,
            "usage": {
                "input_tokens": 500,
                "output_tokens": 120,
                "cache_read_tokens": 500,
                "cache_write_tokens": 0,
            },
            "estimated_cost_usd": 0.003,
            "latency_ms": 3000,
            "automatic_checks": {
                "passed": False,
            },
        },
    ]

    summaries = module.build_model_summaries(rows)
    summary = summaries["Test model"]

    assert summary["completed"] == 2
    assert summary["automatic_passes"] == 1
    assert summary["input_tokens"] == 1100
    assert summary["output_tokens"] == 220
    assert summary["cache_read_tokens"] == 900
    assert summary["average_cost_usd"] == 0.0025
    assert summary["average_latency_ms"] == 2000
    assert summary["cached_input_share_percent"] == 45.0
