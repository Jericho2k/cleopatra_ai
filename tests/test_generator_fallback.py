import asyncio

from ai import generator
from models.model_runtime import ModelResult, ModelTarget, ModelUsage
from models.schemas import Persona


def _target(name: str, model: str) -> ModelTarget:
    return ModelTarget(
        name=name,
        provider="together",
        model=model,
        base_url="https://api.together.xyz/v1",
        api_key_env="TOGETHER_API_KEY",
    )


def test_generator_uses_deepseek_after_two_invalid_kimi_outputs(monkeypatch):
    kimi = _target("Kimi", "moonshotai/Kimi-K3")
    deepseek = _target("DeepSeek", "deepseek-ai/DeepSeek-V4-Pro")
    calls: list[str] = []
    telemetry = []

    async def fake_complete(target, **kwargs):
        calls.append(target.model)
        text = (
            '["one", "two", "three"]'
            if target.model == deepseek.model
            else "not json"
        )
        return ModelResult(
            text=text,
            target=target,
            usage=ModelUsage(input_tokens=100, output_tokens=10),
            latency_ms=50,
        )

    async def fake_record_result(result, context, **kwargs):
        telemetry.append(context.metadata)

    async def fake_record_failure(*args, **kwargs):
        raise AssertionError("no provider failure expected")

    monkeypatch.setattr(generator, "complete", fake_complete)
    monkeypatch.setattr(generator, "record_model_result", fake_record_result)
    monkeypatch.setattr(generator, "record_model_failure", fake_record_failure)

    replies = asyncio.run(
        generator.generate_replies(
            [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "user"},
            ],
            Persona(avg_message_length="short"),
            telemetry_context={
                "feature": "auto_reply",
                "writer_route": "default",
                "writer_route_reason": "ordinary_conversation",
            },
            target_override=kimi,
            fallback_target_override=deepseek,
        )
    )

    assert replies == ["one", "two", "three"]
    assert calls == [kimi.model, kimi.model, deepseek.model]
    assert telemetry[-1]["writer_fallback_used"] is True
    assert telemetry[-1]["writer_attempt_role"] == "fallback"
