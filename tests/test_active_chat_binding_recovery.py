import asyncio

import pytest

import main


@pytest.fixture(autouse=True)
def reset_active_chat_binding_state():
    main._active_chat_binding_retry_after.clear()
    main._active_chat_binding_tasks.clear()
    yield
    main._active_chat_binding_retry_after.clear()
    main._active_chat_binding_tasks.clear()


def test_group_binding_repair_is_coalesced(monkeypatch):
    calls = 0

    async def fake_group_lookup(account_id, platform_fan_id, fan_id):
        nonlocal calls
        calls += 1
        await asyncio.sleep(0)
        assert (account_id, platform_fan_id, fan_id) == (
            "account",
            "platform-fan",
            "fan",
        )
        return "group"

    monkeypatch.setattr(main, "get_or_fetch_group_id", fake_group_lookup)

    async def run():
        return await asyncio.gather(
            main._resolve_active_chat_group_id(
                account_id="account",
                platform_fan_id="platform-fan",
                fan_id="fan",
            ),
            main._resolve_active_chat_group_id(
                account_id="account",
                platform_fan_id="platform-fan",
                fan_id="fan",
            ),
        )

    assert asyncio.run(run()) == [("group", 0), ("group", 0)]
    assert calls == 1


def test_failed_group_binding_repair_is_throttled(monkeypatch):
    calls = 0

    async def missing_group(*_args):
        nonlocal calls
        calls += 1

    monkeypatch.setattr(main, "get_or_fetch_group_id", missing_group)

    async def run():
        first = await main._resolve_active_chat_group_id(
            account_id="account",
            platform_fan_id="platform-fan",
            fan_id="fan",
        )
        second = await main._resolve_active_chat_group_id(
            account_id="account",
            platform_fan_id="platform-fan",
            fan_id="fan",
        )
        return first, second

    first, second = asyncio.run(run())
    assert first[0] is None and first[1] > 800
    assert second[0] is None and second[1] > 800
    assert calls == 1
