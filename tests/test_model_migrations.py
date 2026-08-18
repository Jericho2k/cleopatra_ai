from ai.model_migrations import resolve_supported_model


def test_deprecated_kimi_is_redirected_for_together():
    assert (
        resolve_supported_model("together", "moonshotai/Kimi-K2.6")
        == "moonshotai/Kimi-K3"
    )


def test_model_name_is_unchanged_for_other_providers():
    assert (
        resolve_supported_model("self_hosted", "moonshotai/Kimi-K2.6")
        == "moonshotai/Kimi-K2.6"
    )
