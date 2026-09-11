"""The live smoke must fail when the writer is dead, not report a green tick.

Before this, ``scripts/openrouter_smoke.py`` exercised transport and caching
only. Against the real Kimi route it printed

    attempt 1 output=32 text=''
    attempt 2 output=32 text=''

and exited 0 — a completely non-functional writer passing its own smoke check
twice. The script is never run by CI (it spends real credits), so what is tested
here is its judgement: given a provider response, does it call that a pass?
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_smoke():
    path = Path(__file__).resolve().parents[1] / "scripts" / "openrouter_smoke.py"
    spec = importlib.util.spec_from_file_location("openrouter_smoke", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["openrouter_smoke"] = module
    spec.loader.exec_module(module)
    return module


smoke = _load_smoke()


@pytest.fixture(autouse=True)
def smoke_env(monkeypatch):
    for name in (
        "WRITER_DEFAULT_PROVIDER",
        "WRITER_DEFAULT_MODEL",
        "WRITER_COMPLEX_PROVIDER",
        "WRITER_COMPLEX_MODEL",
        "OPENROUTER_PROVIDERS",
        "OPENROUTER_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("TOGETHER_API_KEY", "test-together-key")
    monkeypatch.setattr(sys, "argv", ["openrouter_smoke.py"])


def _result(text, *, output_tokens=20, cached=0, upstream="Inceptron"):
    return SimpleNamespace(
        text=text,
        usage=SimpleNamespace(
            input_tokens=900,
            output_tokens=output_tokens,
            cache_read_tokens=cached,
            cache_write_tokens=0,
        ),
        latency_ms=120,
        upstream_provider=upstream,
        reported_cost_usd=0.0001,
    )


def _run_with(monkeypatch, results):
    sent = []

    async def fake_complete(target, **kwargs):
        sent.append(kwargs)
        return results[min(len(sent) - 1, len(results) - 1)]

    monkeypatch.setattr(smoke, "complete", fake_complete)
    return asyncio.run(smoke.main()), sent


# --- transport success with no usable text is NOT a pass ---------------------


def test_empty_text_fails_the_smoke(monkeypatch, capsys):
    """The exact production observation: 200 OK, budget spent, nothing said."""
    code, _ = _run_with(monkeypatch, [_result("", output_tokens=200)])

    assert code == 1, "an empty message body must never pass the smoke"
    out = capsys.readouterr().out
    assert "content_empty=true" in out
    assert "SMOKE FAILED" in out
    # The operator is told what to do about it, not just that it broke.
    assert '"reasoning_enabled": false' in out


def test_unparseable_text_fails_the_smoke(monkeypatch, capsys):
    """Transport worked, the model answered — in prose the writer cannot use."""
    code, _ = _run_with(
        monkeypatch,
        [_result("Sure! Here are two lines about the weather.")],
    )

    assert code == 1
    out = capsys.readouterr().out
    assert "content_empty=false" in out
    assert "SMOKE FAILED" in out


def test_wrong_shape_fails_the_smoke(monkeypatch):
    """A JSON object is not the writer's contract; a JSON array of strings is."""
    code, _ = _run_with(monkeypatch, [_result('{"reply": "alpha one"}')])

    assert code == 1


def test_unexpected_upstream_fails_the_smoke(monkeypatch, capsys):
    """A silent provider change alters style, price and cache locality."""
    code, _ = _run_with(
        monkeypatch,
        [_result('["alpha one", "beta two"]', upstream="SomeOtherProvider")],
    )

    assert code == 1
    assert "pinned to" in capsys.readouterr().out


# --- usable final content is the only thing that passes ----------------------


def test_usable_json_array_passes_the_smoke(monkeypatch, capsys):
    code, sent = _run_with(
        monkeypatch,
        [
            _result('["alpha one", "beta two"]'),
            _result('["alpha one", "beta two"]', cached=850),
        ],
    )

    assert code == 0
    out = capsys.readouterr().out
    assert "SMOKE PASSED" in out
    assert "usable=true" in out
    # The cache test is kept, and reports a warm second call.
    assert "prefix cache: OK" in out
    # Both calls share one affinity key and one identical prefix — that is what
    # makes the cache assertion meaningful.
    assert sent[0]["session_id"] == sent[1]["session_id"]
    assert sent[0]["system"] == sent[1]["system"]


def test_a_cold_cache_is_reported_but_does_not_fail(monkeypatch, capsys):
    """A cost regression is not a broken writer, and must not read as one."""
    code, _ = _run_with(monkeypatch, [_result('["alpha one", "beta two"]', cached=0)])

    assert code == 0
    out = capsys.readouterr().out
    assert "cached=0 on attempt 2" in out
    assert "SMOKE PASSED" in out


def test_a_fenced_json_array_still_passes(monkeypatch):
    """The writer's parser strips code fences, so the smoke must agree with it."""
    code, _ = _run_with(
        monkeypatch,
        [_result('```json\n["alpha one", "beta two"]\n```')],
    )

    assert code == 0


# --- the smoke checks the same targets production uses -----------------------


def test_smoke_resolves_the_production_writer_targets():
    from ai.writer_router import (
        COMPLEX_WRITER_MODEL,
        COMPLEX_WRITER_PROVIDER,
        DEFAULT_WRITER_MODEL,
        DEFAULT_WRITER_PROVIDER,
    )

    default_target = smoke._target("default")
    complex_target = smoke._target("complex")

    assert default_target.provider == DEFAULT_WRITER_PROVIDER
    assert default_target.model == DEFAULT_WRITER_MODEL
    assert complex_target.provider == COMPLEX_WRITER_PROVIDER
    assert complex_target.model == COMPLEX_WRITER_MODEL
    # Both carry the reasoning setting the writer depends on, so the smoke is
    # checking the configuration that will actually ship.
    assert default_target.metadata.get("reasoning_enabled") is False
    assert complex_target.metadata.get("reasoning_enabled") is False


def test_smoke_exits_two_when_the_key_for_that_route_is_absent(monkeypatch, capsys):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    assert asyncio.run(smoke.main()) == 2
    assert "OPENROUTER_API_KEY is not set" in capsys.readouterr().out


def test_a_transport_rejection_exits_non_zero(monkeypatch, capsys):
    """A non-serverless or pinned-provider 400 must be loud, never retried away."""

    async def always_fail(_target, **_kwargs):
        raise RuntimeError(
            "Error code: 400 - Unable to access non-serverless model ..."
        )

    monkeypatch.setattr(smoke, "complete", always_fail)

    assert asyncio.run(smoke.main()) == 1
    assert "FAILED RuntimeError" in capsys.readouterr().out
