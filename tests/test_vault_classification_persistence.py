"""A finished classification is not re-derived on every sync.

Daily vault sync discovers and updates metadata. It does not spend vision or
media credits re-answering a question it already answered. The five reasons a
row IS re-classified are enumerated in services/vault_classification_state.py,
and each gets a test here.
"""
from __future__ import annotations

import pytest

from services.vault_classification_state import (
    REASON_CURRENT,
    REASON_IDENTITY,
    REASON_NEW,
    REASON_OPERATOR,
    REASON_UNFINISHED,
    REASON_VERSION,
    classification_decision,
    media_identity_key,
    needs_classification,
    select_items_for_classification,
)

VERSION = 11


def classified(**overrides) -> dict:
    """A row carrying a complete, current classification."""
    row = {
        "id": "row-1",
        "media_id": "m1",
        "fansly_media_id": "m1",
        "mimetype": "image/jpeg",
        "filename": "a.jpg",
        "content_category": "nude_photo",
        "classified_at": "2026-01-01T00:00:00Z",
        "classification_version": VERSION,
        "classification_status": "complete",
    }
    row["classification_media_key"] = media_identity_key(row)
    row.update(overrides)
    return row


# ---------------------------------------------------------------------------
# The rule: finished work is skipped
# ---------------------------------------------------------------------------


def test_previously_classified_unchanged_media_is_not_reclassified():
    decision = classification_decision(classified(), classifier_version=VERSION)

    assert decision.needed is False
    assert decision.reason == REASON_CURRENT


def test_a_rotated_signed_url_is_not_a_change_of_content():
    """Signed vault links rotate constantly. Keying identity on the URL would
    re-classify the entire vault every time links were refreshed — exactly the
    cost this module exists to prevent."""
    row = classified(
        url="https://cdn3.fansly.com/account/a.jpg?Policy=BRAND-NEW-SIGNATURE",
        thumbnail_url="https://cdn3.fansly.com/account/a-thumb.jpg?Policy=NEW",
    )

    assert needs_classification(row, classifier_version=VERSION) is False


def test_a_renamed_file_is_not_a_change_of_content():
    """Nothing in the classification is derived from the filename."""
    assert needs_classification(
        classified(filename="renamed-by-the-operator.jpg"),
        classifier_version=VERSION,
    ) is False


def test_a_whole_vault_of_finished_work_costs_nothing():
    rows = [classified(id=f"row-{i}") for i in range(500)]
    selected, reasons = select_items_for_classification(
        rows, classifier_version=VERSION
    )

    assert selected == []
    assert reasons == {REASON_CURRENT: 500}


# ---------------------------------------------------------------------------
# 1. New media
# ---------------------------------------------------------------------------


def test_new_media_is_classified():
    brand_new = {
        "id": "row-2",
        "media_id": "m2",
        "fansly_media_id": "m2",
        "mimetype": "video/mp4",
    }
    decision = classification_decision(brand_new, classifier_version=VERSION)

    assert decision.needed is True
    assert decision.reason == REASON_NEW


def test_a_row_with_a_stamp_but_no_category_is_new():
    decision = classification_decision(
        classified(content_category=""), classifier_version=VERSION
    )

    assert decision.needed is True
    assert decision.reason == REASON_NEW


def test_a_row_with_a_category_but_no_stamp_is_new():
    decision = classification_decision(
        classified(classified_at=None), classifier_version=VERSION
    )

    assert decision.needed is True
    assert decision.reason == REASON_NEW


# ---------------------------------------------------------------------------
# 2. Classifier version
# ---------------------------------------------------------------------------


def test_an_ordinary_sync_does_not_reclassify_on_a_version_bump():
    """A version bump across a 10,000-item vault is a paid re-analysis, not
    something a daily job does because a deploy happened."""
    stale = classified(classification_version=VERSION - 1)

    assert needs_classification(stale, classifier_version=VERSION) is False


def test_a_confirmed_upgrade_run_does_reclassify_a_stale_version():
    stale = classified(classification_version=VERSION - 1)
    decision = classification_decision(
        stale, classifier_version=VERSION, allow_version_upgrade=True
    )

    assert decision.needed is True
    assert decision.reason == REASON_VERSION


def test_an_upgrade_run_still_skips_current_rows():
    selected, reasons = select_items_for_classification(
        [classified(id="a"), classified(id="b", classification_version=VERSION - 1)],
        classifier_version=VERSION,
        allow_version_upgrade=True,
    )

    assert [row["id"] for row in selected] == ["b"]
    assert reasons[REASON_CURRENT] == 1


# ---------------------------------------------------------------------------
# 3. Incomplete, stale or errored
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", ["partial", "pending", "error"])
def test_an_unfinished_classification_is_retried(status):
    decision = classification_decision(
        classified(classification_status=status), classifier_version=VERSION
    )

    assert decision.needed is True
    assert decision.reason == REASON_UNFINISHED


def test_a_partial_video_is_retried_so_a_later_sync_can_complete_it():
    """A thumbnail-only result exists because a deep scan was too expensive
    THEN. Direct sampling may succeed on a later run, and it costs nothing to
    find out."""
    partial = classified(
        mimetype="video/mp4",
        classification_status="partial",
        classification_skip_reason="exceeds_size_limit",
    )

    assert needs_classification(partial, classifier_version=VERSION) is True


def test_a_row_predating_the_status_column_is_treated_as_finished():
    """Deployed ahead of its migration, an unknown status must read as
    'not unfinished' — otherwise the migration itself triggers a full
    re-analysis of every vault in the deployment."""
    legacy = classified()
    legacy.pop("classification_status")

    assert needs_classification(legacy, classifier_version=VERSION) is False


# ---------------------------------------------------------------------------
# 4. Operator request
# ---------------------------------------------------------------------------


def test_an_operator_request_overrides_everything():
    decision = classification_decision(
        classified(), classifier_version=VERSION, operator_requested=True
    )

    assert decision.needed is True
    assert decision.reason == REASON_OPERATOR


# ---------------------------------------------------------------------------
# 5. Media identity
# ---------------------------------------------------------------------------


def test_a_changed_platform_media_id_forces_reclassification():
    moved = classified(fansly_media_id="a-completely-different-asset")
    decision = classification_decision(moved, classifier_version=VERSION)

    assert decision.needed is True
    assert decision.reason == REASON_IDENTITY


def test_a_changed_mimetype_forces_reclassification():
    retyped = classified(mimetype="video/mp4")
    decision = classification_decision(retyped, classifier_version=VERSION)

    assert decision.needed is True
    assert decision.reason == REASON_IDENTITY


def test_an_absent_identity_key_does_not_force_reclassification():
    """Rows classified before the key column existed have no key. Absent must
    disable the check, not fail it."""
    legacy = classified()
    legacy.pop("classification_media_key")

    assert needs_classification(legacy, classifier_version=VERSION) is False


def test_the_identity_key_is_stable_and_discriminating():
    row = classified()
    assert media_identity_key(row) == media_identity_key(dict(row))
    assert media_identity_key(row) != media_identity_key(
        {**row, "fansly_media_id": "other"}
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def test_a_run_reports_why_each_item_was_or_was_not_selected():
    """So a run can say "312 new, 4 retried, 9,684 already classified" rather
    than a bare total that hides whether anything was re-spent."""
    rows = [
        classified(id="done-1"),
        classified(id="done-2"),
        {"id": "new-1", "media_id": "n1", "mimetype": "image/jpeg"},
        classified(id="partial-1", classification_status="partial"),
    ]
    selected, reasons = select_items_for_classification(
        rows, classifier_version=VERSION
    )

    assert sorted(row["id"] for row in selected) == ["new-1", "partial-1"]
    assert reasons == {REASON_CURRENT: 2, REASON_NEW: 1, REASON_UNFINISHED: 1}
