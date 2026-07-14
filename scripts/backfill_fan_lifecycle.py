"""Backfill deterministic buyer lifecycle for existing fans.

Run only after db/fan_lifecycle_v1.sql has been applied.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from typing import Any

from core.supabase import get_supabase
from services.fan_lifecycle import refresh_fan_lifecycle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--creator-id", default=None)
    parser.add_argument("--page-size", type=int, default=200)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


async def fetch_fans(
    *, creator_id: str | None, offset: int, page_size: int
) -> list[dict[str, Any]]:
    def _get() -> list[dict[str, Any]]:
        query = (
            get_supabase()
            .table("fans")
            .select("id, creator_id")
            .order("id")
            .range(offset, offset + page_size - 1)
        )
        if creator_id:
            query = query.eq("creator_id", creator_id)
        response = query.execute()
        return list(response.data or [])

    return await asyncio.to_thread(_get)


async def main() -> int:
    args = parse_args()
    os.environ["FAN_LIFECYCLE_ENABLED"] = "true"

    processed = 0
    offset = 0
    page_size = max(1, min(int(args.page_size), 1000))

    while True:
        rows = await fetch_fans(
            creator_id=args.creator_id,
            offset=offset,
            page_size=page_size,
        )
        if not rows:
            break

        for row in rows:
            fan_id = str(row.get("id") or "")
            creator_id = str(row.get("creator_id") or "")
            if not fan_id or not creator_id:
                continue
            context = await refresh_fan_lifecycle(
                creator_id=creator_id,
                fan_id=fan_id,
                trigger_type="backfill",
            )
            processed += 1
            print(
                f"BACKFILL fan={fan_id} stage={context.get('stage', 'unknown')} "
                f"processed={processed}"
            )
            if args.limit and processed >= args.limit:
                print(f"Completed lifecycle backfill for {processed} fans.")
                return 0

        offset += len(rows)
        if len(rows) < page_size:
            break

    print(f"Completed lifecycle backfill for {processed} fans.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
