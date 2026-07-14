"""Backfill confirmed-purchase affordability evidence from existing sales logs."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.supabase import get_supabase
from services.affordability import record_confirmed_purchase


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--creator-id", default=None)
    parser.add_argument("--page-size", type=int, default=200)
    parser.add_argument("--limit", type=int, default=0, help="Maximum fans, not purchases")
    return parser.parse_args()


def _parse_date(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    for fmt in ("%d.%m.%Y", "%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(text[:19], fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


async def fetch_fans(
    *, creator_id: str | None, offset: int, page_size: int
) -> list[dict[str, Any]]:
    def _get() -> list[dict[str, Any]]:
        query = (
            get_supabase()
            .table("fans")
            .select("id, creator_id, sales_log")
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
    os.environ["AFFORDABILITY_ENABLED"] = "true"
    processed_fans = 0
    processed_purchases = 0
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
            processed_fans += 1
            for index, entry in enumerate(row.get("sales_log") or []):
                if not isinstance(entry, dict):
                    continue
                try:
                    amount_cents = max(
                        0, int(round(float(entry.get("amount") or 0) * 100))
                    )
                except (TypeError, ValueError):
                    continue
                media_id = str(entry.get("media_id") or "").strip()
                source_ref = (
                    f"ppv:{media_id}"
                    if media_id
                    else f"legacy:{fan_id}:{entry.get('date')}:{entry.get('item')}:{index}"
                )
                await record_confirmed_purchase(
                    creator_id=creator_id,
                    fan_id=fan_id,
                    amount_cents=amount_cents,
                    source_ref=source_ref,
                    occurred_at=_parse_date(entry.get("date")),
                    metadata={"backfill": True, "legacy_entry": entry},
                )
                processed_purchases += 1
            print(
                f"BACKFILL fan={fan_id} purchases={len(row.get('sales_log') or [])} "
                f"fans_processed={processed_fans}"
            )
            if args.limit and processed_fans >= args.limit:
                print(
                    f"Completed affordability backfill for {processed_fans} fans / "
                    f"{processed_purchases} purchases."
                )
                return 0

        offset += len(rows)
        if len(rows) < page_size:
            break

    print(
        f"Completed affordability backfill for {processed_fans} fans / "
        f"{processed_purchases} purchases."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
