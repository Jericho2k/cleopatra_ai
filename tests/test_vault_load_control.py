"""VAULT-001 and VAULT-002 — bounding vault work and stopping it idling.

VAULT-001: vault_autosync_scheduler looped over every due creator and awaited
sync_vault_start, which only *spawns* the run. The await serialised nothing, so
50 creators crossing their 24-hour interval in the same hourly pass started 50
concurrent vault syncs, each running its own media concurrency, inside the
process that also serves chat.

VAULT-002: categorisation ran fixed slices through asyncio.gather. That is a
barrier — a slice of 12 finished only when its slowest item finished, so one
35-second video held eleven idle slots against eleven one-second images. Each
result was then persisted with its own awaited UPDATE.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

import main
from core.vault_gate import VAULT_GATE, configured_vault_concurrency
from services.vault_classifier import VaultClassifierError

CREATOR_ID = "creator-1"


@pytest.fixture(autouse=True)
def clean_gate():
    VAULT_GATE.reset()
    yield
    VAULT_GATE.reset()


# --- VAULT-001: creator-level admission control -----------------------------


def test_the_default_limit_is_conservative(monkeypatch):
    monkeypatch.delenv("VAULT_SYNC_MAX_CONCURRENCY", raising=False)
    assert configured_vault_concurrency() == 2


@pytest.mark.parametrize(
    "raw,expected",
    [("1", 1), ("4", 4), ("0", 1), ("-3", 1), ("banana", 2), ("", 2), ("999", 32)],
)
def test_a_bad_limit_clamps_rather_than_failing_a_deploy(monkeypatch, raw, expected):
    monkeypatch.setenv("VAULT_SYNC_MAX_CONCURRENCY", raw)
    assert configured_vault_concurrency() == expected


@pytest.mark.parametrize("limit", [1, 2, 3])
def test_creator_level_concurrency_never_exceeds_the_limit(monkeypatch, limit):
    """50 creators becoming due at once must not start 50 syncs."""

    monkeypatch.setenv("VAULT_SYNC_MAX_CONCURRENCY", str(limit))

    peak = 0
    active = 0
    completed: list[str] = []

    async def job(creator_id: str):
        nonlocal peak, active
        async with VAULT_GATE.acquire(creator_id=creator_id):
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.01)
            active -= 1
            completed.append(creator_id)

    async def run():
        await asyncio.gather(*[job(f"creator-{index}") for index in range(50)])

    asyncio.run(run())

    assert peak == limit
    # Waiting is lossless: every due creator still ran.
    assert len(completed) == 50


def test_a_waiting_job_is_visible_to_the_operator(monkeypatch):
    monkeypatch.setenv("VAULT_SYNC_MAX_CONCURRENCY", "1")
    observed: dict = {}

    async def holder():
        async with VAULT_GATE.acquire(creator_id="creator-a"):
            await asyncio.sleep(0.05)

    async def waiter():
        await asyncio.sleep(0.01)
        async with VAULT_GATE.acquire(creator_id="creator-b"):
            pass

    async def watcher():
        await asyncio.sleep(0.02)
        observed.update(VAULT_GATE.snapshot())

    async def run():
        await asyncio.gather(holder(), waiter(), watcher())

    asyncio.run(run())

    assert observed["limit"] == 1
    assert observed["active"] == 1
    assert observed["waiting"] == 1
    assert observed["oldest_active_seconds"] >= 0


def test_the_slot_is_released_when_a_job_raises(monkeypatch):
    monkeypatch.setenv("VAULT_SYNC_MAX_CONCURRENCY", "1")

    async def failing():
        async with VAULT_GATE.acquire(creator_id="creator-a"):
            raise RuntimeError("sync blew up")

    async def following():
        async with VAULT_GATE.acquire(creator_id="creator-b"):
            return "ran"

    async def run():
        with pytest.raises(RuntimeError):
            await failing()
        return await following()

    assert asyncio.run(run()) == "ran"
    assert VAULT_GATE.snapshot()["active"] == 0


def test_the_slot_is_released_when_a_job_is_cancelled(monkeypatch):
    monkeypatch.setenv("VAULT_SYNC_MAX_CONCURRENCY", "1")

    async def run():
        started = asyncio.Event()

        async def long_job():
            async with VAULT_GATE.acquire(creator_id="creator-a"):
                started.set()
                await asyncio.sleep(10)

        task = asyncio.create_task(long_job())
        await started.wait()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        async with VAULT_GATE.acquire(creator_id="creator-b"):
            return "ran"

    assert asyncio.run(run()) == "ran"


def test_a_queued_creator_is_not_started_twice(monkeypatch):
    """sync_vault_start must refuse a creator that is already waiting."""

    main._vault_sync_state[CREATOR_ID] = {"status": "queued"}
    try:
        result = asyncio.run(main.sync_vault_start(CREATOR_ID))
        assert result == {"status": "already_running"}
    finally:
        main._vault_sync_state.pop(CREATOR_ID, None)


# --- VAULT-002: worker pool and batched writes ------------------------------


class _RecordingVaultDb:
    """Records every write so the test can count round trips exactly."""

    def __init__(self, rows):
        self.rows = rows
        self.upserts: list[list[dict]] = []
        self.updates: list[dict] = []

    def table(self, name):
        assert name == "creator_vault_media"
        return _RecordingTable(self)


class _RecordingTable:
    def __init__(self, db):
        self._db = db
        self._op = None
        self._payload = None
        self._in = None

    def select(self, *_args, **_kwargs):
        self._op = "select"
        return self

    def eq(self, *_args, **_kwargs):
        return self

    def in_(self, column, values):
        # The real select chunks item ids 250 at a time; a double that ignored
        # the filter would hand back the whole vault on every chunk.
        self._in = (column, {str(value) for value in values})
        return self

    def or_(self, *_args, **_kwargs):
        return self

    def range(self, *_args, **_kwargs):
        return self

    def update(self, payload):
        self._op = "update"
        self._payload = payload
        return self

    def upsert(self, payload, on_conflict=None, **_kwargs):
        self._op = "upsert"
        self._payload = payload
        self._on_conflict = on_conflict
        return self

    def execute(self):
        if self._op == "select":
            rows = list(self._db.rows)
            if self._in is not None:
                column, values = self._in
                rows = [row for row in rows if str(row.get(column)) in values]
            return SimpleNamespace(data=rows)
        if self._op == "upsert":
            self._db.upserts.append(list(self._payload))
            return SimpleNamespace(data=list(self._payload))
        self._db.updates.append(dict(self._payload))
        return SimpleNamespace(data=[])


def _items(count: int, *, videos=()):
    return [
        {
            "id": f"row-{index:05d}",
            "creator_id": CREATOR_ID,
            "media_id": f"media-{index:05d}",
            "fansly_media_id": f"media-{index:05d}",
            "album_id": "album-1",
            "url": "https://cdn.test/a.jpg",
            "thumbnail_url": "https://cdn.test/a-thumb.jpg",
            "mimetype": "video/mp4" if index in videos else "image/jpeg",
            "filename": f"{index}.jpg",
            "album_title": "Album",
        }
        for index in range(count)
    ]


def _classification(item):
    return {
        "id": item["id"],
        "content_category": "nude_photo",
        "ai_description": "a description",
        "price_min": 10,
        "price_max": 40,
        "classification_metadata": {},
    }


@pytest.fixture
def categorize_env(monkeypatch):
    """Run _run_vault_categorization against fakes and record what it did."""

    state: dict = {"peak": 0, "active": 0, "order": []}

    async def noop(*_args, **_kwargs):
        return None

    monkeypatch.setattr(main, "_stamp_vault_op", noop)
    monkeypatch.setattr(main, "_refresh_vault_set_descriptions", lambda _cid: _zero())
    monkeypatch.setenv("VAULT_SEMANTIC_BASE_URL", "https://semantic.test")
    return state


async def _zero():
    return 0


def _run_categorization(items, classify, *, concurrency, monkeypatch):
    db = _RecordingVaultDb(items)
    monkeypatch.setattr(main, "get_supabase", lambda: db)
    monkeypatch.setattr(main, "_categorize_single_item_with_retry", classify)
    monkeypatch.setenv("VAULT_CATEGORIZATION_CONCURRENCY", str(concurrency))

    main._categorize_state[CREATOR_ID] = {
        "status": "running",
        "mode": "upgrade",
        "done": 0,
        "total": len(items),
        "errors": 0,
    }
    asyncio.run(
        main._run_vault_categorization(
            CREATOR_ID,
            item_ids=[item["id"] for item in items],
            upgrade_legacy=True,
        )
    )
    return db, main._categorize_state[CREATOR_ID]


def test_configured_concurrency_is_never_exceeded(categorize_env, monkeypatch):
    tracker = {"active": 0, "peak": 0}

    async def classify(item, **_kwargs):
        tracker["active"] += 1
        tracker["peak"] = max(tracker["peak"], tracker["active"])
        await asyncio.sleep(0.005)
        tracker["active"] -= 1
        return _classification(item)

    _db, state = _run_categorization(
        _items(60), classify, concurrency=6, monkeypatch=monkeypatch
    )

    assert tracker["peak"] == 6
    assert state["done"] == 60


def test_fast_items_keep_moving_while_one_video_is_slow(categorize_env, monkeypatch):
    """The barrier this replaces: eleven images waiting on one video.

    With a slice-and-gather loop, a slow item blocks its whole slice. With a
    worker pool the other workers keep pulling, so nearly every fast item is
    finished before the slow one is.
    """

    finished: list[str] = []

    async def classify(item, **_kwargs):
        if item["mimetype"].startswith("video/"):
            await asyncio.sleep(0.30)
        else:
            await asyncio.sleep(0.002)
        finished.append(item["id"])
        return _classification(item)

    # The video is item 0, so under the old loop it sat in the very first slice
    # and every later slice waited behind it.
    _db, state = _run_categorization(
        _items(48, videos={0}),
        classify,
        concurrency=4,
        monkeypatch=monkeypatch,
    )

    assert state["done"] == 48
    video_position = finished.index("row-00000")
    # The video finishes near the very end despite starting first: the other
    # workers were never blocked on it.
    assert video_position >= 40


def test_every_item_is_written_exactly_once(categorize_env, monkeypatch):
    async def classify(item, **_kwargs):
        return _classification(item)

    db, state = _run_categorization(
        _items(250), classify, concurrency=8, monkeypatch=monkeypatch
    )

    written = [row["id"] for batch in db.upserts for row in batch]
    assert sorted(written) == sorted(item["id"] for item in _items(250))
    assert len(written) == len(set(written))
    assert state["done"] == 250


def test_writes_are_batched_not_one_per_item(categorize_env, monkeypatch):
    async def classify(item, **_kwargs):
        return _classification(item)

    db, _state = _run_categorization(
        _items(1000), classify, concurrency=12, monkeypatch=monkeypatch
    )

    # 1,000 sequential UPDATEs became a bounded number of batched writes.
    assert db.updates == []
    assert len(db.upserts) <= 20
    assert all(len(batch) <= main._CLASSIFICATION_WRITE_BATCH for batch in db.upserts)


def test_a_written_row_carries_its_identity_and_its_classification(
    categorize_env, monkeypatch
):
    async def classify(item, **_kwargs):
        return _classification(item)

    db, _state = _run_categorization(
        _items(3), classify, concurrency=2, monkeypatch=monkeypatch
    )

    row = db.upserts[0][0]
    assert row["creator_id"] == CREATOR_ID
    assert row["media_id"].startswith("media-")
    assert row["content_category"] == "nude_photo"
    assert row["classification_version"] == main.VAULT_CLASSIFIER_VERSION


def test_provider_failure_abort_still_stops_the_run(categorize_env, monkeypatch):
    attempted: list[str] = []

    async def classify(item, **_kwargs):
        attempted.append(item["id"])
        raise VaultClassifierError("provider is down")

    db, state = _run_categorization(
        _items(200), classify, concurrency=4, monkeypatch=monkeypatch
    )

    assert state["status"] == "error"
    assert "three provider failures" in state["error"]
    # It stopped early rather than attempting all 200.
    assert len(attempted) < 200
    assert db.upserts == []


def test_work_completed_before_an_abort_is_still_persisted(
    categorize_env, monkeypatch
):
    """A partial batch must not be discarded when the run stops."""

    calls = {"count": 0}

    async def classify(item, **_kwargs):
        calls["count"] += 1
        if calls["count"] > 10:
            raise VaultClassifierError("provider is down")
        return _classification(item)

    db, state = _run_categorization(
        _items(200), classify, concurrency=2, monkeypatch=monkeypatch
    )

    assert state["status"] == "error"
    written = [row["id"] for batch in db.upserts for row in batch]
    assert len(written) >= 8
    assert state["done"] == len(written)


def test_non_provider_errors_are_counted_but_do_not_abort(
    categorize_env, monkeypatch
):
    async def classify(item, **_kwargs):
        if item["id"].endswith("3"):
            raise ValueError("one bad image")
        return _classification(item)

    _db, state = _run_categorization(
        _items(40), classify, concurrency=4, monkeypatch=monkeypatch
    )

    assert state["status"] == "done"
    assert state["errors"] == 4
    assert state["done"] == 36


def test_progress_never_reports_more_done_than_was_persisted(
    categorize_env, monkeypatch
):
    seen: list[tuple[int, int]] = []

    async def classify(item, **_kwargs):
        state = main._categorize_state[CREATOR_ID]
        seen.append((int(state.get("done") or 0), len(_written(item))))
        return _classification(item)

    def _written(_item):
        return []

    db, _state = _run_categorization(
        _items(300), classify, concurrency=8, monkeypatch=monkeypatch
    )

    persisted = sum(len(batch) for batch in db.upserts)
    for reported, _ in seen:
        assert reported <= persisted


# --- telemetry --------------------------------------------------------------


def test_the_health_document_reports_the_vault_gate():
    """Sprint 2's health surface gains the vault gate, without a new verdict.

    A queue here is the gate working as designed — vault work waiting is what
    stops it competing with chat — so it must be informational and must never
    turn a healthy deployment into a degraded one.
    """

    import services.operational_health as health

    health.reset_cache()
    document = asyncio.run(health.collect(use_cache=False))

    gate = document["vault"]["gate"]
    assert set(gate) >= {
        "limit",
        "active",
        "waiting",
        "oldest_active_seconds",
        "oldest_waiting_seconds",
    }
    assert "vault" not in " ".join(document["degraded_reasons"])
    assert "vault" not in " ".join(document["fatal_reasons"])
