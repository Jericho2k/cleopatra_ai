from types import SimpleNamespace

import pytest

import main


class FakeQuery:
    def __init__(self, db, table):
        self.db = db
        self.table = table
        self.operation = "select"
        self.payload = None

    def select(self, *_args, **_kwargs):
        self.operation = "select"
        return self

    def update(self, payload):
        self.operation = "update"
        self.payload = payload
        return self

    def eq(self, *_args):
        return self

    def in_(self, *_args):
        return self

    def limit(self, *_args):
        return self

    def execute(self):
        if self.table == "vault_sets" and self.operation == "select":
            return SimpleNamespace(data=[{
                "id": "set-1",
                "media_ids": ["media-1", "media-2"],
            }])
        if self.table == "creator_vault_media":
            return SimpleNamespace(data=[
                {
                    "fansly_media_id": "media-1",
                    "content_category": "lingerie_photo",
                    "explicitness_level": 2,
                    "scene_location": "bedroom",
                    "scene_outfit": "pink lingerie",
                    "scene_lighting": "warm",
                    "mimetype": "image/jpeg",
                    "tags": ["pink", "bedroom"],
                },
                {
                    "fansly_media_id": "media-2",
                    "content_category": "nude_photo",
                    "explicitness_level": 4,
                    "scene_location": "bedroom",
                    "scene_outfit": "pink lingerie",
                    "scene_lighting": "warm",
                    "mimetype": "image/jpeg",
                    "tags": ["pink", "bedroom"],
                },
            ])
        if self.table == "vault_sets" and self.operation == "update":
            self.db.saved = self.payload
            return SimpleNamespace(data=[{"id": "set-1"}])
        raise AssertionError(f"unexpected query: {self.table} {self.operation}")


class FakeSupabase:
    def __init__(self):
        self.saved = None

    def table(self, name):
        return FakeQuery(self, name)


@pytest.mark.asyncio
async def test_manual_set_description_is_generated_and_saved(monkeypatch):
    db = FakeSupabase()
    monkeypatch.setattr(main, "get_supabase", lambda: db)

    result = await main.generate_vault_set_description("creator-1", "set-1")

    assert result["media_count"] == 2
    assert "bedroom setting" in result["description"]
    assert "progresses from explicitness 2/5 to 4/5" in result["description"]
    assert db.saved["description"] == result["description"]
    assert db.saved["metadata_version"] == main.VAULT_CLASSIFIER_VERSION
