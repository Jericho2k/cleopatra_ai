"""VAULT-002 — what the batch barrier and per-item writes actually cost.

Runs the same synthetic vault two ways against mocked classifiers:

* BEFORE — fixed slices through ``asyncio.gather``, then one awaited UPDATE per
  result. That is what ``_run_vault_categorization`` did before this sprint.
* AFTER  — the worker pool and batched writes it does now.

Classifier timings are simulated (images ~1 s, videos ~35 s, both scaled down by
``--speed`` so the run finishes), and the "database" is an in-process counter
with a fixed simulated latency per round trip. So the wall-clock numbers are a
model of the shape of the work, not a measurement of production. What IS exact
is the write count, the peak media concurrency, and the ordering behaviour —
which is where the finding actually lives.

Usage:
    python scripts/bench_vault_categorization.py
    python scripts/bench_vault_categorization.py --items 1000 --json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Real-world timings the audit measured, in seconds.
IMAGE_SECONDS = 2.5
VIDEO_SECONDS = 35.0
# One database round trip, whatever it carries.
DB_ROUND_TRIP_SECONDS = 0.025
WRITE_BATCH = 100


@dataclass
class Result:
    label: str
    items: int
    duration_seconds: float
    db_writes: int
    max_media_concurrency: int
    rows_held_in_memory_peak: int = 0
    finish_order: list[int] = field(default_factory=list)


def _mixed(items: int) -> list[bool]:
    """True means video. One in five, as in a typical creator vault."""

    return [index % 5 == 4 for index in range(items)]


class _Tracker:
    def __init__(self) -> None:
        self.active = 0
        self.peak = 0

    def enter(self) -> None:
        self.active += 1
        self.peak = max(self.peak, self.active)

    def exit(self) -> None:
        self.active -= 1


async def _classify(is_video: bool, tracker: _Tracker, speed: float) -> None:
    tracker.enter()
    try:
        await asyncio.sleep(
            (VIDEO_SECONDS if is_video else IMAGE_SECONDS) / speed
        )
    finally:
        tracker.exit()


async def run_before(kinds: list[bool], concurrency: int, speed: float) -> Result:
    tracker = _Tracker()
    writes = 0
    order: list[int] = []
    started = time.perf_counter()

    for offset in range(0, len(kinds), concurrency):
        slice_ = list(enumerate(kinds[offset:offset + concurrency], start=offset))
        # The barrier: the whole slice waits for its slowest member.
        await asyncio.gather(
            *[_classify(is_video, tracker, speed) for _index, is_video in slice_]
        )
        for index, _is_video in slice_:
            # One awaited UPDATE per result, sequentially.
            await asyncio.sleep(DB_ROUND_TRIP_SECONDS / speed)
            writes += 1
            order.append(index)

    return Result(
        label="before (slice gather + per-item UPDATE)",
        items=len(kinds),
        duration_seconds=time.perf_counter() - started,
        db_writes=writes,
        max_media_concurrency=tracker.peak,
        rows_held_in_memory_peak=1,
        finish_order=order,
    )


async def run_after(kinds: list[bool], concurrency: int, speed: float) -> Result:
    tracker = _Tracker()
    writes = 0
    order: list[int] = []
    pending: list[int] = []
    peak_pending = 0
    cursor = 0
    cursor_lock = asyncio.Lock()
    write_lock = asyncio.Lock()
    started = time.perf_counter()

    async def flush(force: bool = False) -> None:
        nonlocal pending, writes
        async with write_lock:
            if not pending or (not force and len(pending) < WRITE_BATCH):
                return
            rows, pending = pending, []
            await asyncio.sleep(DB_ROUND_TRIP_SECONDS / speed)
            writes += 1
            order.extend(rows)

    async def take() -> int | None:
        nonlocal cursor
        async with cursor_lock:
            if cursor >= len(kinds):
                return None
            index = cursor
            cursor += 1
            return index

    async def worker() -> None:
        nonlocal peak_pending
        while True:
            index = await take()
            if index is None:
                return
            await _classify(kinds[index], tracker, speed)
            pending.append(index)
            peak_pending = max(peak_pending, len(pending))
            await flush()

    await asyncio.gather(*[worker() for _ in range(concurrency)])
    await flush(force=True)

    return Result(
        label="after (worker pool + batched writes)",
        items=len(kinds),
        duration_seconds=time.perf_counter() - started,
        db_writes=writes,
        max_media_concurrency=tracker.peak,
        rows_held_in_memory_peak=peak_pending,
        finish_order=order,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=int, nargs="*", default=[100, 1000])
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument(
        "--speed",
        type=float,
        default=250.0,
        help="divide simulated durations by this so the benchmark finishes",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    collected = []
    for count in args.items:
        kinds = _mixed(count)
        before = asyncio.run(run_before(kinds, args.concurrency, args.speed))
        after = asyncio.run(run_after(kinds, args.concurrency, args.speed))
        collected.append({
            "items": count,
            "videos": sum(kinds),
            "concurrency": args.concurrency,
            "before": {
                "simulated_seconds": round(before.duration_seconds * args.speed, 1),
                "db_writes": before.db_writes,
                "max_media_concurrency": before.max_media_concurrency,
                "results_held_in_memory_peak": before.rows_held_in_memory_peak,
            },
            "after": {
                "simulated_seconds": round(after.duration_seconds * args.speed, 1),
                "db_writes": after.db_writes,
                "max_media_concurrency": after.max_media_concurrency,
                "results_held_in_memory_peak": after.rows_held_in_memory_peak,
            },
        })

    if args.json:
        print(json.dumps(collected, indent=2))
        return 0

    header = (
        f"{'items':>7}{'videos':>8}{'before s':>11}{'after s':>10}"
        f"{'speedup':>9}{'writes before':>15}{'writes after':>14}"
    )
    print("Simulated vault categorisation (mocked classifiers, modelled DB latency)")
    print()
    print(header)
    print("-" * len(header))
    for row in collected:
        before_s = row["before"]["simulated_seconds"]
        after_s = row["after"]["simulated_seconds"]
        print(
            f"{row['items']:>7}{row['videos']:>8}{before_s:>11.1f}{after_s:>10.1f}"
            f"{before_s / max(after_s, 0.001):>8.2f}x"
            f"{row['before']['db_writes']:>15}{row['after']['db_writes']:>14}"
        )
    print()
    print(
        f"peak media concurrency: before "
        f"{collected[0]['before']['max_media_concurrency']}, after "
        f"{collected[0]['after']['max_media_concurrency']} "
        f"(configured {args.concurrency}) — unchanged, which is the point: the "
        "same budget, no idle slots."
    )
    print(
        "Durations are modelled, not measured against real providers. The write "
        "counts and the concurrency ceiling are exact."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
