"""The upgrade path: re-analyzing media left behind by an older classifier.

The dashboard has always offered these buttons; the backend rejected the mode
with a 400. Without them a classifier swap only ever reaches new media,
because the initial and new runs both select uncategorized rows only.
"""
import asyncio
from pathlib import Path

import main
from services.vault_classification import CLASSIFIER_VERSION


class _Query:
    """Chainable stand-in for the Supabase query builder."""

    def __init__(self, rows: list[dict]):
        self._rows = rows

    def __getattr__(self, _name):
        return lambda *args, **kwargs: self

    @property
    def not_(self):
        return self

    def execute(self):
        return type("Result", (), {"data": self._rows, "count": len(self._rows)})()


class _FakeDb:
    def __init__(self, tables: dict[str, list[dict]]):
        self.tables = tables
        self.selected: list[str] = []

    def table(self, name: str) -> _Query:
        self.selected.append(name)
        return _Query(self.tables.get(name, []))


MEDIA = [
    {"id": "row-1", "fansly_media_id": "m1"},
    {"id": "row-2", "fansly_media_id": "m2"},
    {"id": "row-3", "fansly_media_id": "m3"},
]


def test_upgrade_scope_all_takes_every_stale_row(monkeypatch):
    monkeypatch.setattr(
        main, "get_supabase", lambda: _FakeDb({"creator_vault_media": MEDIA})
    )
    assert asyncio.run(main._stale_classification_ids("creator-1", "all")) == [
        "row-1",
        "row-2",
        "row-3",
    ]


def test_upgrade_scope_approved_only_touches_media_in_approved_sets(monkeypatch):
    # vault_sets.media_ids holds fansly_media_id values, not row ids.
    monkeypatch.setattr(
        main,
        "get_supabase",
        lambda: _FakeDb(
            {
                "creator_vault_media": MEDIA,
                "vault_sets": [{"media_ids": ["m1", "m3"]}],
            }
        ),
    )
    assert asyncio.run(main._stale_classification_ids("creator-1", "approved")) == [
        "row-1",
        "row-3",
    ]


def test_approved_scope_with_no_approved_sets_selects_nothing(monkeypatch):
    monkeypatch.setattr(
        main,
        "get_supabase",
        lambda: _FakeDb({"creator_vault_media": MEDIA, "vault_sets": []}),
    )
    assert asyncio.run(main._stale_classification_ids("creator-1", "approved")) == []


def test_a_missing_migration_does_not_break_the_overview(monkeypatch):
    class _Exploding(_FakeDb):
        def rpc(self, *args, **kwargs):
            raise RuntimeError("function vault_classification_staleness does not exist")

    monkeypatch.setattr(main, "get_supabase", lambda: _Exploding({}))
    assert asyncio.run(main._classification_staleness("creator-1")) == {
        "stale": 0,
        "stale_approved": 0,
    }


def test_only_the_upgrade_run_rewrites_existing_metadata():
    source = (Path(__file__).resolve().parents[1] / "main.py").read_text(encoding="utf-8")

    # The uncategorized filter is what stops initial/new from touching priced
    # media; upgrade is the single documented exception.
    assert "if not reprocess:" in source
    assert 'reprocess=resolved_mode == "upgrade"' in source
    # An upgrade is billable, so it cannot start by accident.
    assert "if not confirm_upgrade:" in source
    assert "upgrade_scope must be 'all' or 'approved'" in source


def test_the_migration_ships_the_staleness_function_and_version_column():
    migration = (
        Path(__file__).resolve().parents[1] / "db" / "vault_classification_v3.sql"
    ).read_text(encoding="utf-8")

    assert "vault_classification_staleness" in migration
    assert "classification_version" in migration
    assert "classification_evidence" in migration
    # classification_confidence is the dashboard's numeric evidence percentage.
    assert "classification_confidence numeric" in migration
    assert "grant execute on function public.vault_classification_staleness" in migration


def test_every_write_path_stamps_the_current_classifier_version():
    row = main._vault_classification_row(
        {
            "content_category": "nude_photo",
            "ai_description": "x",
            "price_min": 15,
            "price_max": 80,
        }
    )
    assert row["classification_version"] == CLASSIFIER_VERSION
    assert row["classified_at"]
    # Defaults must be storable, not None, for the not-null columns.
    assert row["classification_source"] == "failed"
    assert row["classification_confidence"] == 0.0
