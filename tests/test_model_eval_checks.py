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
