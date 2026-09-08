"""Paginated Supabase reads.

PostgREST caps ordinary selects at ``db-max-rows`` (1,000 by default). A select
without ``.range()`` therefore returns a *silent* prefix of the result, and
without ``.order()`` the particular prefix is not stable between calls.

Two audit findings came from exactly that:

- SCALE-003 — ``services/fansly_lists.py`` built its authoritative fan map from a
  truncated read and then deleted the memberships it could not map.
- SEC-002 — ``preview_auto_audience`` read every agency's list memberships and
  filtered in Python, so past ~1,000 global rows the requesting creator's own
  rows were usually absent.

Callers must supply a deterministic total order. Ordering by a non-unique column
lets Postgres break ties differently between pages, which can drop or duplicate
rows across the page boundary — order by a primary key, or by enough columns to
be unique.
"""
from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

# Supabase's default db-max-rows. Requesting exactly this much per page means a
# short page is an unambiguous end-of-results signal.
PAGE_SIZE = 1000

# A guard against an unbounded loop if a caller's range is ignored. 1,000 pages
# is 1,000,000 rows — far beyond any per-creator table in this product.
MAX_PAGES = 1000


class PaginationIncompleteError(RuntimeError):
    """Raised when a read did not reach the end of the result set.

    Callers that use a read as authoritative evidence — in particular before
    deleting anything — must treat this as "I do not know the full state", never
    as "the extra rows do not exist".
    """


def fetch_all_rows(
    page: Callable[[int, int], Any],
    *,
    page_size: int = PAGE_SIZE,
    max_pages: int = MAX_PAGES,
) -> list[dict]:
    """Read every row by paging until a short page arrives.

    ``page(start, end)`` must execute one inclusive ``.range(start, end)`` query
    carrying a deterministic ``.order(...)`` and return the Supabase response.
    """
    rows: list[dict] = []
    for index in range(max_pages):
        start = index * page_size
        response = page(start, start + page_size - 1)
        batch = list(getattr(response, "data", None) or [])
        rows.extend(batch)
        if len(batch) < page_size:
            return rows
    raise PaginationIncompleteError(
        f"stopped after {max_pages} pages of {page_size} rows without "
        "reaching the end of the result set"
    )


async def fetch_all_rows_async(
    page: Callable[[int, int], Any],
    *,
    page_size: int = PAGE_SIZE,
    max_pages: int = MAX_PAGES,
) -> list[dict]:
    """``fetch_all_rows`` off the event loop; the Supabase client is blocking."""
    return await asyncio.to_thread(
        fetch_all_rows,
        page,
        page_size=page_size,
        max_pages=max_pages,
    )
