from main import _matching_unbound_creator_message


def test_matches_unbound_creator_copy_within_delivery_window() -> None:
    platform = {
        "role": "creator",
        "content": "  hello   there ",
        "sent_at": "2026-07-30T10:00:05+00:00",
    }
    candidate = {
        "id": "local-row",
        "role": "creator",
        "content": "hello there",
        "sent_at": "2026-07-30T10:00:00+00:00",
        "fansly_message_id": None,
    }

    assert _matching_unbound_creator_message(platform, [candidate]) == candidate


def test_does_not_merge_distinct_platform_or_old_messages() -> None:
    platform = {
        "role": "creator",
        "content": "same text",
        "sent_at": "2026-07-30T10:01:00+00:00",
    }
    identified = {
        "id": "identified",
        "role": "creator",
        "content": "same text",
        "sent_at": "2026-07-30T10:00:58+00:00",
        "fansly_message_id": "another-platform-message",
    }
    old = {
        "id": "old",
        "role": "creator",
        "content": "same text",
        "sent_at": "2026-07-30T10:00:00+00:00",
        "fansly_message_id": None,
    }

    assert _matching_unbound_creator_message(
        platform,
        [identified, old],
    ) is None


def test_never_matches_inbound_fan_messages() -> None:
    assert _matching_unbound_creator_message(
        {
            "role": "fan",
            "content": "hello",
            "sent_at": "2026-07-30T10:00:00+00:00",
        },
        [],
    ) is None
