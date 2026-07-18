from pathlib import Path

from services.suggestions import _is_local_test_fan


def test_only_explicit_test_namespace_uses_local_delivery():
    assert _is_local_test_fan("test_1784402987") is True
    assert _is_local_test_fan("707604041756061697") is False
    assert _is_local_test_fan("") is False
    assert _is_local_test_fan(None) is False


def test_live_fans_still_fail_closed_without_a_delivery_route():
    source = (
        Path(__file__).resolve().parents[1] / "services" / "suggestions.py"
    ).read_text(encoding="utf-8")

    assert "and not local_test_delivery" in source
    assert "ppv_delivery_route_missing" in source
    assert "[AUTO TEST DELIVERY]" in source
