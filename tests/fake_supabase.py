"""A Supabase/PostgREST test double that reproduces the 1,000-row cap.

Several audit findings (SCALE-003, SEC-002) are only reachable because PostgREST
silently truncates an unranged select at ``db-max-rows`` and, without an
``order``, returns an unstable prefix. A plain mock that returns every row cannot
reproduce either behaviour, so it would let those bugs pass.

This double therefore:

- caps every response at ``max_rows`` (default 1,000), exactly like PostgREST;
- returns rows in a deliberately *unstable* order when the query carries no
  ``.order()``, so code that paginates without ordering is caught;
- applies ``.range()`` inclusively, after filtering and ordering;
- records every executed query so tests can assert on the filters that reached
  the database rather than on what Python did afterwards.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

DEFAULT_MAX_ROWS = 1000


@dataclass
class ExecutedQuery:
    table: str
    columns: str = ""
    filters: list[tuple] = field(default_factory=list)
    orders: list[tuple] = field(default_factory=list)
    range: tuple[int, int] | None = None

    def filter_values(self, kind: str, column: str) -> list[Any]:
        return [value for k, col, value in self.filters if k == kind and col == column]

    def has_filter_on(self, column: str) -> bool:
        return any(col == column for _kind, col, _value in self.filters)


class FakeSupabase:
    def __init__(self, tables: dict[str, list[dict]], *, max_rows: int = DEFAULT_MAX_ROWS):
        self.tables = {name: list(rows) for name, rows in tables.items()}
        self.max_rows = max_rows
        self.queries: list[ExecutedQuery] = []
        self.writes: list[tuple[str, str, Any]] = []
        self._shuffle = itertools.count()

    def table(self, name: str) -> "_FakeQuery":  # noqa: UP037
        return _FakeQuery(self, name)

    # --- assertions helpers -------------------------------------------------

    def queries_for(self, table: str) -> list[ExecutedQuery]:
        return [query for query in self.queries if query.table == table]


def _like_matches(value: str, pattern: str) -> bool:
    """SQL LIKE, enough of it for the filters this fake sees.

    ``%`` is any run of characters, ``_`` is exactly one, and a backslash
    escapes either. The escape matters here rather than being pedantry: the
    simulation-fan filter is ``test\\_%``, and treating that underscore as a
    wildcard would make the fake agree with a buggy query that also excludes a
    real fan whose platform id merely starts with "test".
    """
    import re

    regex = []
    index = 0
    while index < len(pattern):
        char = pattern[index]
        if char == "\\" and index + 1 < len(pattern):
            regex.append(re.escape(pattern[index + 1]))
            index += 2
            continue
        if char == "%":
            regex.append(".*")
        elif char == "_":
            regex.append(".")
        else:
            regex.append(re.escape(char))
        index += 1
    return re.fullmatch("".join(regex), value) is not None


class _FakeQuery:
    def __init__(self, db: FakeSupabase, table: str):
        self._db = db
        self._record = ExecutedQuery(table=table)
        self._single = False
        self._count_mode: str | None = None
        self._op: str | None = None
        self._payload: Any = None
        self._on_conflict: str | None = None

    # --- read builders ------------------------------------------------------

    def select(self, columns: str = "*", count: str | None = None, **_kwargs):
        self._op = "select"
        self._record.columns = columns
        # PostgREST's count mode. Recorded so a caller that asks for a count and
        # a single row — the cheap way to count without transferring the rows —
        # gets the total here too, rather than the length of the truncated page.
        self._count_mode = count
        return self

    def eq(self, column: str, value: Any):
        self._record.filters.append(("eq", column, value))
        return self

    def neq(self, column: str, value: Any):
        self._record.filters.append(("neq", column, value))
        return self

    def in_(self, column: str, values):
        self._record.filters.append(("in", column, list(values)))
        return self

    def is_(self, column: str, value):
        self._record.filters.append(("is", column, value))
        return self

    def like(self, column: str, pattern: str):
        self._record.filters.append(("like", column, pattern))
        return self

    @property
    def not_(self):
        """PostgREST's negation prefix: ``.not_.is_("col", "null")``."""
        return _NegatedFilters(self)

    def order(self, column: str, desc: bool = False, **_kwargs):
        self._record.orders.append((column, desc))
        return self

    def limit(self, count: int):
        self._record.range = (0, count - 1)
        return self

    def range(self, start: int, end: int):
        self._record.range = (start, end)
        return self

    def single(self):
        self._single = True
        return self

    # --- write builders -----------------------------------------------------

    def insert(self, payload):
        self._op = "insert"
        self._payload = payload
        return self

    def update(self, payload):
        self._op = "update"
        self._payload = payload
        return self

    def upsert(self, payload, on_conflict: str | None = None, **_kwargs):
        self._op = "upsert"
        self._payload = payload
        self._on_conflict = on_conflict
        return self

    def delete(self):
        self._op = "delete"
        return self

    # --- execution ----------------------------------------------------------

    def _matching_rows(self) -> list[dict]:
        rows = list(self._db.tables.get(self._record.table, []))
        for kind, column, value in self._record.filters:
            if kind == "eq":
                rows = [row for row in rows if str(row.get(column)) == str(value)]
            elif kind == "neq":
                rows = [row for row in rows if str(row.get(column)) != str(value)]
            elif kind == "in":
                wanted = {str(item) for item in value}
                rows = [row for row in rows if str(row.get(column)) in wanted]
            elif kind == "is" and value in (None, "null"):
                rows = [row for row in rows if row.get(column) is None]
            elif kind == "not.is" and value in (None, "null"):
                rows = [row for row in rows if row.get(column) is not None]
            elif kind == "not.eq":
                rows = [row for row in rows if str(row.get(column)) != str(value)]
            elif kind == "not.in":
                unwanted = {str(item) for item in value}
                rows = [row for row in rows if str(row.get(column)) not in unwanted]
        return rows

    def execute(self):
        if self._op in {"insert", "update", "upsert", "delete"}:
            return self._execute_write()

        self._db.queries.append(self._record)
        rows = self._matching_rows()
        # Captured before ordering, paging and the row cap, because a count is
        # a statement about the whole match, not about the page.
        matched_total = len(rows)

        if self._record.orders:
            for column, desc in reversed(self._record.orders):
                rows.sort(key=lambda row: str(row.get(column) or ""), reverse=desc)
        else:
            # PostgREST gives no ordering guarantee. Rotating the result makes an
            # unordered paginated read visibly wrong instead of accidentally
            # stable, which is the whole point of SCALE-003.
            offset = next(self._db._shuffle) % (len(rows) or 1)
            rows = rows[offset:] + rows[:offset]

        if self._record.range is not None:
            start, end = self._record.range
            rows = rows[start : end + 1]

        # The cap PostgREST applies regardless of what the caller asked for.
        rows = rows[: self._db.max_rows]

        rows = [dict(row) for row in rows]
        if self._single:
            return SimpleNamespace(data=rows[0] if rows else None)
        if self._count_mode:
            # The count describes every matching row, not the page returned —
            # which is the whole reason a caller pairs count with limit(1).
            return SimpleNamespace(data=rows, count=matched_total)
        return SimpleNamespace(data=rows)

    def _execute_write(self):
        table = self._record.table
        store = self._db.tables.setdefault(table, [])
        self._db.writes.append((self._op, table, self._payload))

        if self._op == "delete":
            remaining, removed = [], []
            for row in store:
                (removed if self._row_matches(row) else remaining).append(row)
            self._db.tables[table] = remaining
            return SimpleNamespace(data=removed)

        if self._op == "update":
            updated = []
            for row in store:
                if self._row_matches(row):
                    row.update(self._payload)
                    updated.append(dict(row))
            return SimpleNamespace(data=updated)

        payloads = (
            self._payload if isinstance(self._payload, list) else [self._payload]
        )
        written = []
        for payload in payloads:
            if self._op == "upsert" and self._on_conflict:
                keys = [key.strip() for key in self._on_conflict.split(",")]
                existing = next(
                    (
                        row
                        for row in store
                        if all(
                            str(row.get(key)) == str(payload.get(key)) for key in keys
                        )
                    ),
                    None,
                )
                if existing is not None:
                    existing.update(payload)
                    written.append(dict(existing))
                    continue
            row = dict(payload)
            row.setdefault("id", f"{table}-{len(store) + 1}")
            store.append(row)
            written.append(dict(row))
        return SimpleNamespace(data=written)

    def _row_matches(self, row: dict) -> bool:
        for kind, column, value in self._record.filters:
            if kind == "eq" and str(row.get(column)) != str(value):
                return False
            if kind == "in" and str(row.get(column)) not in {
                str(item) for item in value
            }:
                return False
            if kind == "like" and not _like_matches(str(row.get(column) or ""), value):
                return False
            if kind == "not.like" and _like_matches(str(row.get(column) or ""), value):
                return False
        return True


class _NegatedFilters:
    """The object PostgREST's ``.not_`` prefix returns.

    It only records the negation and hands the query builder straight back, so
    ``.not_.is_(...)`` reads and chains exactly as it does against the real
    client.
    """

    def __init__(self, query: "_FakeQuery"):  # noqa: UP037
        self._query = query

    def is_(self, column: str, value):
        self._query._record.filters.append(("not.is", column, value))
        return self._query

    def eq(self, column: str, value):
        self._query._record.filters.append(("not.eq", column, value))
        return self._query

    def in_(self, column: str, values):
        self._query._record.filters.append(("not.in", column, list(values)))
        return self._query

    def like(self, column: str, pattern: str):
        self._query._record.filters.append(("not.like", column, pattern))
        return self._query
