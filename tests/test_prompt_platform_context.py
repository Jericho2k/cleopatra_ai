from ai.prompt_builder import PLATFORM_CONTEXT


def test_platform_context_establishes_paid_adult_setting_and_boundaries():
    lowered = PLATFORM_CONTEXT.lower()

    assert "paid adult creator subscription platform" in lowered
    assert "do not assume this is the first message" in lowered
    assert "in-person meeting" in lowered
    assert "on-platform" in lowered
    assert "commercial decision" in lowered
