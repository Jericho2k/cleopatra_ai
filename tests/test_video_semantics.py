"""Several chronological batches become one honest video-level record.

A twelve-frame sample flattened into one contact sheet gives each moment a
200-pixel thumbnail, and a model asked to describe that square answers about
the square. The record built here is what replaces that: what the clip DOES,
over time, derived deterministically from facts the per-batch classifier
already reported.

Two properties matter most, and both are about NOT inventing:

* every field traces to a batch observation, or is absent;
* an unchanging clip produces an unchanging record, not a narrative.
"""
from __future__ import annotations

from services.video_semantics import (
    DEFAULT_FRAMES_PER_SHEET,
    BatchObservation,
    chronological_batches,
    combine_batch_observations,
    commercial_role,
    describe_video_record,
    observation_from_classification,
)


def observation(index, start, end, **overrides) -> BatchObservation:
    base = dict(
        index=index,
        start_seconds=start,
        end_seconds=end,
        frames=DEFAULT_FRAMES_PER_SHEET,
        scene_location="bedroom",
        nudity="none",
        explicitness=0,
    )
    base.update(overrides)
    return BatchObservation(**base)


# The sprint's worked example: an 8-12 minute clip that escalates.
ESCALATING = [
    observation(
        0, 30, 210,
        description="Standing near a mirror in lingerie",
        scene_outfit="clothed",
        nudity="none",
        explicitness=1,
    ),
    observation(
        1, 270, 450,
        description="Moves onto the bed, increased nudity",
        scene_outfit="lingerie",
        nudity="partial",
        explicitness=3,
    ),
    observation(
        2, 510, 690,
        description="Explicit toy scene on the bed",
        scene_outfit="nude",
        nudity="full",
        explicitness=5,
        sexual_activity=["toy use"],
        props=["toy"],
    ),
]


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------


def test_frames_are_split_into_small_consecutive_groups():
    batches = chronological_batches(
        [b"f"] * 12,
        [30.0 + 60 * i for i in range(12)],
    )

    assert [len(batch.frames) for batch in batches] == [4, 4, 4]
    assert [batch.label for batch in batches] == [
        "0:30–3:30",
        "4:30–7:30",
        "8:30–11:30",
    ]
    # An 8-minute clip is minutes 0-4 then 4-8, exactly as the sprint asks.
    assert batches[0].start_seconds < batches[1].start_seconds < batches[2].start_seconds


def test_the_sheet_count_is_bounded_without_dropping_the_ending():
    """Truncating would throw away the END of the video, which is the part a
    progression most needs. The batches grow instead."""
    batches = chronological_batches(
        [b"f"] * 16, list(range(16)), frames_per_sheet=2, max_sheets=4
    )

    assert len(batches) == 4
    assert sum(len(batch.frames) for batch in batches) == 16
    assert batches[-1].offsets_seconds[-1] == 15


def test_an_empty_sample_produces_no_batches():
    assert chronological_batches([], []) == []
    assert chronological_batches([b"f"], []) == []


def test_a_short_sample_is_a_single_batch():
    batches = chronological_batches([b"f"] * 3, [1.0, 2.0, 3.0])

    assert len(batches) == 1
    assert len(batches[0].frames) == 3


# ---------------------------------------------------------------------------
# Combination
# ---------------------------------------------------------------------------


def test_the_record_describes_the_progression_not_a_moment():
    record = combine_batch_observations(ESCALATING, duration_seconds=720)

    assert record["progression"] == ["clothed", "lingerie", "nude"]
    assert record["nudity_progression"] == ["none", "partial", "full"]
    assert record["setting"] == "bedroom"
    assert record["beginning"] == "Standing near a mirror in lingerie"
    assert record["middle"] == "Moves onto the bed, increased nudity"
    assert record["ending"] == "Explicit toy scene on the bed"
    assert record["duration_label"] == "12:00"
    assert record["sampled_frames"] == 12
    assert record["batches"] == 3


def test_meaningful_changes_carry_the_moment_they_happened():
    record = combine_batch_observations(ESCALATING, duration_seconds=720)
    changes = record["scene_changes"]

    assert [change["at"] for change in changes] == ["4:30", "8:30"]
    assert "more nudity" in changes[0]["changed"]
    assert "wardrobe" in changes[0]["changed"]
    assert "props" in changes[1]["changed"]
    assert changes[1]["new_props"] == ["toy"]


def test_activities_and_props_are_collected_across_the_whole_clip():
    record = combine_batch_observations(ESCALATING, duration_seconds=720)

    assert record["activities"] == ["toy use"]
    assert record["props"] == ["toy"]
    assert record["peak_explicitness"] == 5
    assert record["peak_at"] == "8:30–11:30"


def test_an_unchanging_clip_produces_an_unchanging_record():
    """No narrative is invented where nothing happened."""
    steady = [
        observation(i, 30 + 180 * i, 180 + 180 * i,
                    description="Seated on a sofa in a robe",
                    scene_outfit="robe", nudity="none", explicitness=1)
        for i in range(3)
    ]
    record = combine_batch_observations(steady, duration_seconds=600)

    assert record["progression"] == ["robe"]
    assert record["scene_changes"] == []
    assert record["activities"] == []
    assert "Progression" not in describe_video_record(record)


def test_a_setting_change_is_reported():
    moving = [
        observation(0, 30, 120, scene_location="bedroom", scene_outfit="robe"),
        observation(1, 180, 270, scene_location="bathroom", scene_outfit="robe"),
    ]
    record = combine_batch_observations(moving, duration_seconds=300)

    assert record["setting_progression"] == ["bedroom", "bathroom"]
    assert record["scene_changes"][0]["changed"] == ["setting"]
    assert "bedroom then bathroom" in describe_video_record(record)


def test_nothing_sampled_is_different_from_nothing_found():
    assert combine_batch_observations([], duration_seconds=300) == {}
    assert describe_video_record({}) == ""


def test_batches_are_combined_in_time_order_however_they_arrive():
    record = combine_batch_observations(
        list(reversed(ESCALATING)), duration_seconds=720
    )

    assert record["progression"] == ["clothed", "lingerie", "nude"]
    assert record["beginning"] == "Standing near a mirror in lingerie"


# ---------------------------------------------------------------------------
# Prose and commercial role
# ---------------------------------------------------------------------------


def test_the_prose_reads_as_a_clip_rather_than_a_photo():
    prose = describe_video_record(
        combine_batch_observations(ESCALATING, duration_seconds=720)
    )

    assert "12:00 video" in prose
    assert "12 points" in prose
    assert "clothed → lingerie → nude" in prose
    assert "toy" in prose
    assert "4:30" in prose


def test_an_escalating_clip_is_a_closer_and_a_tame_one_an_opener():
    escalating = combine_batch_observations(ESCALATING, duration_seconds=720)
    assert commercial_role(escalating) == "closer"

    tame = combine_batch_observations(
        [observation(0, 10, 40, scene_outfit="dress", explicitness=0)],
        duration_seconds=60,
    )
    assert commercial_role(tame) == "opener"
    assert commercial_role({}) == "standalone"


# ---------------------------------------------------------------------------
# Reading a classifier result
# ---------------------------------------------------------------------------


def test_a_classification_result_becomes_a_positioned_observation():
    batch = chronological_batches([b"a", b"b"], [30.0, 90.0])[0]
    result = observation_from_classification(
        {
            "description": "A bedroom scene",
            "scene_location": "bedroom",
            "scene_outfit": "lingerie",
            "nudity": "partial",
            "explicitness": 3,
            "props": ["mirror"],
            "sexual_activity": [],
            "confidence": 0.8,
        },
        batch=batch,
    )

    assert result.start_seconds == 30.0
    assert result.end_seconds == 90.0
    assert result.label == "0:30–1:30"
    assert result.nudity_rank == 1
    assert result.props == ["mirror"]


def test_an_unknown_value_never_becomes_a_fact():
    batch = chronological_batches([b"a"], [10.0])[0]
    result = observation_from_classification(
        {
            "description": "unknown",
            "scene_location": "unclear",
            "scene_outfit": "",
            "nudity": "not a real value",
            "explicitness": "nonsense",
        },
        batch=batch,
    )

    assert result.description == ""
    assert result.scene_location == ""
    assert result.nudity == "none"
    assert result.explicitness == 0


def test_a_local_only_batch_still_contributes_its_nudity():
    """NudeNet evidence alone is enough to place a batch on the progression."""
    batch = chronological_batches([b"a"], [10.0])[0]
    result = observation_from_classification(
        {"nudity": "full", "explicitness": 4, "visible_anatomy": ["breasts"]},
        batch=batch,
    )

    assert result.nudity == "full"
    assert result.visible_anatomy == ["breasts"]
    assert result.nudity_rank == 2
