from pathlib import Path


def test_proactive_messages_support_the_explicit_local_test_namespace():
    source = (
        Path(__file__).resolve().parents[1] / "services" / "proactive.py"
    ).read_text(encoding="utf-8")
    assert 'startswith("test_")' in source
    assert "[PROACTIVE TEST DELIVERY]" in source
    assert "and not local_test_delivery" in source
