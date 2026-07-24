import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from core import tenancy


class Query:
    def __init__(self, rows):
        self.rows = rows

    def table(self, _name):
        return self

    def select(self, *_args, **_kwargs):
        return self

    def eq(self, column, value):
        self.rows = [row for row in self.rows if str(row.get(column)) == str(value)]
        return self

    def limit(self, _value):
        return self

    def execute(self):
        return SimpleNamespace(data=self.rows)


def request_for(user_id: str):
    return SimpleNamespace(
        state=SimpleNamespace(dashboard_user_id=user_id),
    )


def test_creator_access_allows_only_assigned_creator(monkeypatch):
    monkeypatch.setattr(
        tenancy,
        "get_supabase",
        lambda: Query([
            {"chatter_id": "operator-1", "creator_id": "creator-1"},
            {"chatter_id": "operator-2", "creator_id": "creator-2"},
        ]),
    )
    request = request_for("operator-1")

    asyncio.run(tenancy.require_creator_access(request, "creator-1"))
    with pytest.raises(HTTPException) as exc:
        asyncio.run(tenancy.require_creator_access(request, "creator-2"))
    assert exc.value.status_code == 404


def test_rls_migration_covers_creator_fan_and_list_boundaries():
    root = Path(__file__).resolve().parents[1]
    migration = (root / "db" / "tenant_isolation_v1.sql").read_text()

    assert "can_access_creator" in migration
    assert "can_access_fan" in migration
    assert "can_access_fan_list" in migration
    assert "chatter_id = auth.uid()" in migration
    assert "alter table public.chatter_creators enable row level security" in migration
    assert "service_role" in migration


def test_every_dashboard_resource_route_has_a_tenant_guard():
    root = Path(__file__).resolve().parents[1]
    source = (root / "main.py").read_text()

    assert "authenticated_dashboard_user(" in source
    assert "require_creator_fan_access(request, req.creator_id, req.fan_id)" in source
    assert 'dependencies=[Depends(require_creator_path_access)]' in source
    assert 'dependencies=[Depends(require_fan_path_access)]' in source
    assert 'dependencies=[Depends(require_ppv_approval_path_access)]' in source


def test_webhook_scopes_fans_through_connected_creator():
    root = Path(__file__).resolve().parents[1]
    source = (root / "main.py").read_text()
    webhook = source[source.index("async def fansly_webhook"):source.index(
        "@app.delete(", source.index("async def fansly_webhook")
    )]

    assert '.eq("apifansly_account_id", api_account_id)' in webhook
    assert '.eq("creator_id", creator_id)' in webhook
    assert 'event == "ppv.purchased"' in webhook
    assert 'event == "subscriptions.new"' in webhook
    assert '.eq("platform_fan_id", sender_id)' not in webhook
