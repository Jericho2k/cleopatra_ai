"""Owner-only test content: rich for the simulator, impossible to deliver.

The owner wants the admin/testing creator to carry a realistic catalog so
simulated Full Auto turns exercise coherence grouping, escalation, media-type
mixes and multi-step allocation — all of which are functions of catalog variety,
and none of which a three-set vault touches.

The danger is precise. ``creator_vault_media.media_id`` is a Fansly media id
belonging to the SOURCE creator's account. A naive copy under a different
creator_id produces rows that every planner and every delivery path reads as
ordinary sellable inventory, pointing at another account's media.

So the mirror is marked rather than disguised, twice over, and both barriers are
independently tested here:

1. ``simulation_only = true``, which live planning filters out and the owner
   simulator includes;
2. a rewritten ``sim:`` media id, which is not a platform id and which every
   delivery path refuses — so a code path that forgot the flag still cannot send
   one.

And throughout: the source creator's vault is never written to.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.apifansly_gate import simulation_scope
from core.simulation_catalog import (
    SIMULATION_MEDIA_PREFIX,
    contains_simulation_media,
    exclude_simulation_only,
    is_simulation_media_id,
    reset_missing_column_warning,
    run_live_catalog_query,
    simulation_catalog_visible,
    simulation_media_id,
)
from services.simulation_catalog import (
    SimulationCatalogError,
    delete_creator_catalog_mirror,
    mirror_creator_catalog,
)
from tests.fake_supabase import FakeSupabase

SOURCE = "11111111-2222-3333-4444-555555555555"
TARGET = "99999999-8888-7777-6666-555555555555"


def run(coro):
    return asyncio.run(coro)


def world() -> FakeSupabase:
    return FakeSupabase(
        {
            "creator_vault_media": [
                {
                    "id": "src-media-1",
                    "creator_id": SOURCE,
                    "media_id": "fansly-media-1",
                    "fansly_media_id": "fansly-media-1",
                    "url": "https://cdn.example/real-1.jpg",
                    "mimetype": "image/jpeg",
                    "album_title": "Bedroom shoot",
                    "content_category": "nude_photo",
                    "ai_description": "Soft bedroom photo.",
                    "price_min": 15,
                    "price_max": 80,
                    "scene_location": "bedroom",
                    "scene_outfit": "black lingerie",
                    "scene_lighting": "warm",
                    "scene_id": "shoot-1",
                    "simulation_only": False,
                },
                {
                    "id": "src-media-2",
                    "creator_id": SOURCE,
                    "media_id": "fansly-media-2",
                    "fansly_media_id": "fansly-media-2",
                    "url": "https://cdn.example/real-2.mp4",
                    "mimetype": "video/mp4",
                    "album_title": "Bedroom shoot",
                    "content_category": "nude_video",
                    "ai_description": "A private bedroom clip.",
                    "price_min": 20,
                    "price_max": 110,
                    "scene_location": "bedroom",
                    "scene_outfit": "black lingerie",
                    "scene_lighting": "warm",
                    "scene_id": "shoot-1",
                    "simulation_only": False,
                },
            ],
            "vault_sets": [
                {
                    "id": "src-set-1",
                    "creator_id": SOURCE,
                    "title": "Bedroom · black lingerie",
                    "description": "Soft bedroom photos.",
                    "location": "bedroom",
                    "outfit": "black lingerie",
                    "explicit_min": 2,
                    "explicit_max": 3,
                    "media_ids": ["fansly-media-1"],
                    "preview_media_id": "fansly-media-1",
                    "suggested_price": 25,
                    "tags": ["nude_photo", "lingerie"],
                    "base_price_cents": 2500,
                    "min_price_cents": 1500,
                    "max_price_cents": 8000,
                    "dynamic_pricing_enabled": True,
                    "metadata_version": 3,
                    "status": "approved",
                    "source": "ai",
                    "simulation_only": False,
                },
                {
                    "id": "src-set-2",
                    "creator_id": SOURCE,
                    "title": "Bedroom · black lingerie · nude video",
                    "description": "A private bedroom clip.",
                    "location": "bedroom",
                    "outfit": "black lingerie",
                    "explicit_min": 5,
                    "explicit_max": 5,
                    "media_ids": ["fansly-media-2"],
                    "preview_media_id": "fansly-media-2",
                    "suggested_price": 45,
                    "tags": ["nude_video", "video", "individual_video"],
                    "base_price_cents": 4500,
                    "min_price_cents": 2000,
                    "max_price_cents": 11000,
                    "dynamic_pricing_enabled": True,
                    "metadata_version": 3,
                    "status": "approved",
                    "source": "ai",
                    "simulation_only": False,
                },
                {
                    "id": "target-own-set",
                    "creator_id": TARGET,
                    "title": "Sophia's own set",
                    "media_ids": ["sophia-media-1"],
                    "status": "approved",
                    "source": "manual",
                    "base_price_cents": 2000,
                    "min_price_cents": 2000,
                    "max_price_cents": 2000,
                    "simulation_only": False,
                },
            ],
        }
    )


@pytest.fixture
def db(monkeypatch):
    fake = world()
    monkeypatch.setattr("core.supabase.get_supabase", lambda: fake)
    monkeypatch.setattr("services.simulation_catalog.get_supabase", lambda: fake)
    reset_missing_column_warning()
    return fake


def mirrored_sets(db) -> list[dict]:
    return [
        row
        for row in db.tables["vault_sets"]
        if row.get("creator_id") == TARGET and row.get("simulation_only")
    ]


def mirrored_media(db) -> list[dict]:
    return [
        row
        for row in db.tables["creator_vault_media"]
        if row.get("creator_id") == TARGET and row.get("simulation_only")
    ]


def source_rows(db) -> tuple[list[dict], list[dict]]:
    return (
        [r for r in db.tables["vault_sets"] if r.get("creator_id") == SOURCE],
        [r for r in db.tables["creator_vault_media"] if r.get("creator_id") == SOURCE],
    )


# ---------------------------------------------------------------------------
# 1. The mirror produces usable test content
# ---------------------------------------------------------------------------


def test_the_mirror_carries_the_metadata_the_simulator_needs(db):
    run(mirror_creator_catalog(source_creator_id=SOURCE, target_creator_id=TARGET))

    sets = mirrored_sets(db)
    assert len(sets) == 2, "photos and video both mirror"
    by_title = {row["title"]: row for row in sets}
    photo = by_title["Bedroom · black lingerie"]

    # Descriptions, tags, explicitness and approved price bounds all survive:
    # they are precisely what makes a simulated plan realistic.
    assert photo["description"] == "Soft bedroom photos."
    assert photo["tags"] == ["nude_photo", "lingerie"]
    assert photo["explicit_min"] == 2 and photo["explicit_max"] == 3
    assert photo["min_price_cents"] == 1500 and photo["max_price_cents"] == 8000
    assert photo["status"] == "approved"

    media = mirrored_media(db)
    assert len(media) == 2
    clip = next(row for row in media if row["content_category"] == "nude_video")
    assert clip["ai_description"] == "A private bedroom clip."
    assert clip["price_min"] == 20 and clip["price_max"] == 110


def test_provenance_is_preserved_internally(db):
    run(mirror_creator_catalog(source_creator_id=SOURCE, target_creator_id=TARGET))

    for row in mirrored_sets(db):
        assert row["source_creator_id"] == SOURCE
        assert row["source_set_id"] in {"src-set-1", "src-set-2"}
        assert row["source"] == "simulation_mirror"
    for row in mirrored_media(db):
        assert row["source_creator_id"] == SOURCE
        assert row["source_media_id"] in {"fansly-media-1", "fansly-media-2"}


def test_set_media_ids_are_remapped_so_the_sets_still_resolve(db):
    run(mirror_creator_catalog(source_creator_id=SOURCE, target_creator_id=TARGET))

    mirrored_ids = {row["media_id"] for row in mirrored_media(db)}
    for row in mirrored_sets(db):
        assert row["media_ids"], "a set with no media cannot be planned"
        assert set(row["media_ids"]) <= mirrored_ids
        assert row["preview_media_id"] in mirrored_ids


# ---------------------------------------------------------------------------
# 2. It can never be delivered
# ---------------------------------------------------------------------------


def test_every_mirrored_media_id_is_rewritten(db):
    run(mirror_creator_catalog(source_creator_id=SOURCE, target_creator_id=TARGET))

    for row in mirrored_media(db):
        assert row["media_id"].startswith(SIMULATION_MEDIA_PREFIX)
        assert is_simulation_media_id(row["media_id"])
        # The source creator's real platform id is not carried over.
        assert row["media_id"] != row["source_media_id"]
    for row in mirrored_sets(db):
        assert all(is_simulation_media_id(mid) for mid in row["media_ids"])


def test_no_live_location_or_platform_identity_travels_with_a_mirror(db):
    run(mirror_creator_catalog(source_creator_id=SOURCE, target_creator_id=TARGET))

    for row in mirrored_media(db):
        assert row["url"] is None
        assert row["fansly_media_id"] is None


def test_the_delivery_path_refuses_simulation_media(monkeypatch):
    """The second barrier, independent of the flag and of planning."""
    from services.ppv_delivery import PPVDeliveryError, send_locked_ppv

    with pytest.raises(PPVDeliveryError, match="simulation-only"):
        run(
            send_locked_ppv(
                creator_id=TARGET,
                fan_id="fan-1",
                media_ids=[simulation_media_id(SOURCE, "fansly-media-1")],
                price_cents=3000,
                message_content="here it is",
                source="operator",
                was_ai_suggested=False,
            )
        )


def test_the_auto_delivery_branch_refuses_simulation_media():
    """The Auto path builds its own platform call, so it enforces the invariant
    itself rather than trusting that planning already did."""
    source = (
        Path(__file__).resolve().parents[1] / "services" / "suggestions.py"
    ).read_text(encoding="utf-8")

    assert "elif ppv_match and contains_simulation_media(media_ids):" in source
    assert "simulation_media_on_live_route" in source


def test_contains_simulation_media_recognises_a_mixed_batch():
    real = "fansly-media-1"
    fake = simulation_media_id(SOURCE, real)
    assert contains_simulation_media([real]) is False
    assert contains_simulation_media([real, fake]) is True
    assert contains_simulation_media(fake) is True
    assert contains_simulation_media(None) is False


# ---------------------------------------------------------------------------
# 3. Live planning excludes it; the simulator includes it
# ---------------------------------------------------------------------------


def test_live_planning_filters_the_catalog_and_the_simulator_does_not():
    assert simulation_catalog_visible() is False
    with simulation_scope():
        assert simulation_catalog_visible() is True


def test_the_live_filter_is_applied_outside_a_simulation(db):
    applied: list[bool] = []

    def build(apply_filter: bool):
        applied.append(apply_filter)
        query = db.table("vault_sets").select("*").eq("creator_id", TARGET)
        if apply_filter:
            query = exclude_simulation_only(query)
        return query.execute()

    run(mirror_creator_catalog(source_creator_id=SOURCE, target_creator_id=TARGET))

    live = run_live_catalog_query(build, label="test.live")
    assert applied == [True]
    assert all(row.get("simulation_only") is not True for row in live.data)
    assert any(row["id"] == "target-own-set" for row in live.data), (
        "the target creator's own vault is untouched by the filter"
    )

    with simulation_scope():
        simulated = run_live_catalog_query(build, label="test.simulated")
    assert any(row.get("simulation_only") for row in simulated.data)


def test_a_missing_column_degrades_instead_of_failing_the_read(db, capsys):
    """Deployed ahead of the migration, the read must still work — and it is
    correct by construction, because no mirrored row can exist yet."""
    attempts: list[bool] = []

    def build(apply_filter: bool):
        attempts.append(apply_filter)
        if apply_filter:
            raise RuntimeError(
                'column vault_sets.simulation_only does not exist (42703)'
            )
        return db.table("vault_sets").select("*").eq("creator_id", TARGET).execute()

    result = run_live_catalog_query(build, label="test.missing_column")

    assert attempts == [True, False]
    assert result.data is not None
    assert "apply db/simulation_catalog_v1.sql" in capsys.readouterr().out


def test_an_unrelated_error_is_not_swallowed(db):
    def build(_apply_filter: bool):
        raise RuntimeError("PostgREST connection terminated")

    with pytest.raises(RuntimeError, match="connection terminated"):
        run_live_catalog_query(build, label="test.other_error")


def test_the_planning_reads_apply_the_filter():
    """Named explicitly so a new planning read cannot quietly skip it."""
    root = Path(__file__).resolve().parents[1]
    for path in (
        "db/commercial_queries.py",
        "services/session_planner.py",
        "main.py",
    ):
        source = (root / path).read_text(encoding="utf-8")
        assert "exclude_simulation_only" in source, f"{path} does not filter"
        assert "run_live_catalog_query" in source, f"{path} is not schema-tolerant"


# ---------------------------------------------------------------------------
# 4. Refreshable, and never destructive to the source
# ---------------------------------------------------------------------------


def test_the_mirror_is_idempotent(db):
    run(mirror_creator_catalog(source_creator_id=SOURCE, target_creator_id=TARGET))
    first_sets = {row["source_set_id"] for row in mirrored_sets(db)}
    first_count = len(mirrored_sets(db))

    result = run(
        mirror_creator_catalog(source_creator_id=SOURCE, target_creator_id=TARGET)
    )

    assert len(mirrored_sets(db)) == first_count, "a refresh must not duplicate"
    assert {row["source_set_id"] for row in mirrored_sets(db)} == first_sets
    assert result.sets_removed == first_count, "the previous mirror was replaced"


def test_a_refresh_drops_rows_whose_source_has_disappeared(db):
    run(mirror_creator_catalog(source_creator_id=SOURCE, target_creator_id=TARGET))
    db.tables["vault_sets"] = [
        row for row in db.tables["vault_sets"] if row.get("id") != "src-set-2"
    ]

    run(mirror_creator_catalog(source_creator_id=SOURCE, target_creator_id=TARGET))

    assert {row["source_set_id"] for row in mirrored_sets(db)} == {"src-set-1"}


def test_rebuilding_a_mirror_never_deletes_the_source(db):
    before_sets, before_media = source_rows(db)
    before_sets = [dict(row) for row in before_sets]
    before_media = [dict(row) for row in before_media]

    run(mirror_creator_catalog(source_creator_id=SOURCE, target_creator_id=TARGET))
    run(mirror_creator_catalog(source_creator_id=SOURCE, target_creator_id=TARGET))
    run(delete_creator_catalog_mirror(source_creator_id=SOURCE, target_creator_id=TARGET))

    after_sets, after_media = source_rows(db)
    assert after_sets == before_sets
    assert after_media == before_media


def test_deleting_a_mirror_leaves_the_targets_own_vault_alone(db):
    run(mirror_creator_catalog(source_creator_id=SOURCE, target_creator_id=TARGET))
    run(delete_creator_catalog_mirror(source_creator_id=SOURCE, target_creator_id=TARGET))

    assert mirrored_sets(db) == []
    assert mirrored_media(db) == []
    assert any(
        row["id"] == "target-own-set" for row in db.tables["vault_sets"]
    ), "the test creator's own content is not part of any mirror"


def test_a_refresh_only_touches_its_own_source_mirror(db):
    """Three predicates scope every delete — target, flag, source — so one
    source's refresh cannot wipe another's mirror or the target's own vault."""
    other_source = "abcdabcd-0000-0000-0000-abcdabcdabcd"
    db.tables["vault_sets"].append(
        {
            "id": "other-mirror-set",
            "creator_id": TARGET,
            "title": "From another source",
            "media_ids": ["sim:abcdabcd:x"],
            "status": "approved",
            "simulation_only": True,
            "source_creator_id": other_source,
            "source_set_id": "other-src-1",
            "base_price_cents": 3000,
            "min_price_cents": 3000,
            "max_price_cents": 3000,
        }
    )

    run(mirror_creator_catalog(source_creator_id=SOURCE, target_creator_id=TARGET))
    run(mirror_creator_catalog(source_creator_id=SOURCE, target_creator_id=TARGET))

    surviving = {row["id"] for row in db.tables["vault_sets"]}
    assert "other-mirror-set" in surviving, "another source's mirror is out of scope"
    assert "target-own-set" in surviving, "the target's own vault is out of scope"
    assert {"src-set-1", "src-set-2"} <= surviving, "the source is out of scope"


def test_a_creator_cannot_mirror_onto_itself(db):
    with pytest.raises(SimulationCatalogError, match="its own catalog"):
        run(mirror_creator_catalog(source_creator_id=SOURCE, target_creator_id=SOURCE))
    assert mirrored_sets(db) == []


# ---------------------------------------------------------------------------
# 5. The migration
# ---------------------------------------------------------------------------


def test_the_migration_is_additive_and_ordered():
    root = Path(__file__).resolve().parents[1]
    migration = (root / "db" / "simulation_catalog_v1.sql").read_text(encoding="utf-8")
    order = (root / "db" / "migration_order.txt").read_text(encoding="utf-8")

    lowered = migration.lower()
    assert "drop table" not in lowered
    assert "drop column" not in lowered
    assert "delete from" not in lowered
    assert "add column if not exists simulation_only" in lowered
    # A mirrored row that is not marked simulation_only would be sellable.
    assert "check (source_creator_id is null or simulation_only = true)" in lowered

    lines = [line.strip() for line in order.splitlines() if line.strip() and not line.startswith("#")]
    assert "simulation_catalog_v1.sql" in lines
    assert lines.index("simulation_catalog_v1.sql") < lines.index("tenant_isolation_v1.sql")
