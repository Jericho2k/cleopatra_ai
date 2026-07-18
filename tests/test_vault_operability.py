from pathlib import Path

from services.vault_operations import (
    MANUAL_RECATEGORIZATION_DAILY_LIMIT,
    categorize_new_batch_enabled,
    manual_recategorization_usage,
    normalize_media_ids,
)


def test_new_media_batch_is_exact_and_never_implies_full_vault():
    assert normalize_media_ids(["a", "", "a", " b "]) == ["a", "b"]
    assert categorize_new_batch_enabled(True, ["a"]) is True
    assert categorize_new_batch_enabled(False, ["a"]) is False
    assert categorize_new_batch_enabled(True, []) is False


def test_manual_ai_reanalysis_has_a_five_per_day_contract():
    assert MANUAL_RECATEGORIZATION_DAILY_LIMIT == 5
    assert manual_recategorization_usage(0) == {
        "used": 0,
        "remaining": 5,
        "daily_limit": 5,
        "allowed": True,
    }
    assert manual_recategorization_usage(5)["allowed"] is False
    assert manual_recategorization_usage(8)["remaining"] == 0


def test_migration_claim_is_atomic_and_creator_scoped():
    migration = (
        Path(__file__).resolve().parents[1] / "db" / "agency_operability_v1.sql"
    ).read_text(encoding="utf-8")
    assert "pg_advisory_xact_lock" in migration
    assert "claim_vault_recategorization" in migration
    assert "vault_initial_categorized_at" in migration
    assert "auto_categorize_new_media" in migration

