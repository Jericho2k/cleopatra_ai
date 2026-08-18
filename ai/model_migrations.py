"""Compatibility redirects for retired provider model identifiers."""


_MODEL_REDIRECTS: dict[tuple[str, str], str] = {
    ("together", "moonshotai/Kimi-K2.6"): "moonshotai/Kimi-K3",
}


def resolve_supported_model(provider: str, model: str) -> str:
    """Return the supported successor for a retired configured model."""

    normalized_provider = provider.strip().lower()
    normalized_model = model.strip()
    return _MODEL_REDIRECTS.get(
        (normalized_provider, normalized_model),
        normalized_model,
    )
