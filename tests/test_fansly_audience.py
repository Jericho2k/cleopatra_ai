from services.fansly_audience import (
    _accounts_from_followers,
    _subscription_rows,
    _timestamp,
)


def test_follower_page_preserves_relationship_and_profile_data():
    follower_ids, accounts = _accounts_from_followers({
        "followers": [
            {"id": "relationship-1", "followerId": "fan-1"},
            {"id": "relationship-2", "followerId": "fan-2"},
        ],
        "aggregationData": {
            "accounts": [
                {"id": "fan-1", "username": "alice"},
                {"id": "fan-2", "displayName": "Bob"},
            ]
        },
        "nextCursor": "next-page",
    })

    assert follower_ids == {"fan-1", "fan-2"}
    assert accounts["fan-1"]["username"] == "alice"
    assert accounts["fan-2"]["displayName"] == "Bob"


def test_subscriptions_and_millisecond_timestamps_use_documented_shape():
    rows = [{"subscriberId": "fan-1", "status": 3, "endsAt": 1_800_000_000_000}]

    assert _subscription_rows({"subscriptions": rows}) == rows
    assert _subscription_rows([]) == []
    assert _timestamp(rows[0]["endsAt"]) == "2027-01-15T08:00:00+00:00"
