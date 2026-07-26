from services.video_frames import DEFAULT_FRAME_COUNT, frame_sample_offsets


def test_offsets_are_spread_across_the_clip_and_trim_the_edges():
    offsets = frame_sample_offsets(100.0, 4)
    assert len(offsets) == 4
    assert offsets == sorted(offsets)
    # Nothing lands on the cold open or the credits, where black frames live.
    assert offsets[0] > 5.0
    assert offsets[-1] < 95.0


def test_sampling_is_deterministic_so_reanalysis_reads_the_same_frames():
    assert frame_sample_offsets(212.5, 4) == frame_sample_offsets(212.5, 4)


def test_unknown_duration_falls_back_to_the_first_frame():
    assert frame_sample_offsets(0.0, 4) == [0.0]
    assert frame_sample_offsets(-10.0, 4) == [0.0]
    assert frame_sample_offsets("not a number", 4) == [0.0]


def test_single_frame_request_takes_the_middle():
    assert frame_sample_offsets(100.0, 1) == [50.0]


def test_short_clips_never_produce_duplicate_or_out_of_range_offsets():
    offsets = frame_sample_offsets(0.4, DEFAULT_FRAME_COUNT)
    assert len(offsets) == len(set(offsets))
    assert all(0.0 <= offset <= 0.4 for offset in offsets)


def test_a_zero_or_negative_count_still_yields_one_sample():
    assert len(frame_sample_offsets(60.0, 0)) == 1
    assert len(frame_sample_offsets(60.0, -3)) == 1
