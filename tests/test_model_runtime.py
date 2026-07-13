from models.model_runtime import ModelTarget, ModelUsage, estimate_cost_usd


def test_estimate_cost_counts_cache_separately():
    target = ModelTarget(
        name="test",
        provider="together",
        model="test-model",
        input_per_million=1.0,
        output_per_million=2.0,
        cache_read_per_million=0.1,
        cache_write_per_million=1.25,
    )
    usage = ModelUsage(
        input_tokens=1_000_000,
        output_tokens=500_000,
        cache_read_tokens=1_000_000,
        cache_write_tokens=100_000,
    )
    assert estimate_cost_usd(target, usage) == 2.225


def test_target_from_mapping_normalizes_provider():
    target = ModelTarget.from_mapping(
        {
            "name": "Candidate",
            "provider": "TOGETHER",
            "model": "some/model",
            "input_per_million": 0.3,
            "output_per_million": 1.2,
        }
    )
    assert target.provider == "together"
    assert target.model == "some/model"
