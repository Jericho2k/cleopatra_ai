from types import SimpleNamespace

from ai.writer_router import WriterRoute, select_writer_route
from models.schemas import Fan, StageType


def test_vip_lifecycle_routes_to_complex_writer(monkeypatch):
    monkeypatch.setenv("WRITER_ROUTING_ENABLED", "true")
    ctx = SimpleNamespace(
        situation={},
        commercial_decision={},
        fan_profile=Fan(id="fan-1", display_name="A"),
        conversation_stage=StageType.WARMING_UP,
        active_session=None,
        buyer_lifecycle={"stage": "VIP"},
    )
    decision = select_writer_route(ctx)
    assert decision.route == WriterRoute.COMMERCIAL_COMPLEX
    assert decision.reason == "buyer_lifecycle:vip"
