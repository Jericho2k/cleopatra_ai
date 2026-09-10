#!/usr/bin/env python3
"""What the provider prompt cache is ACTUALLY doing.

Audit reference: Sprint 4 Part 12.

WHY THIS EXISTS
---------------
Sprint 3 restructured prompts so a long, stable prefix comes first and the
volatile parts come last, which is what makes a provider prompt cache able to
hit. It then reported the improvement as a *cacheability* figure — a property of
the prompts, computed from the prompts.

That is not a cache hit rate. A perfectly cacheable prompt still misses if the
provider evicted the prefix, if the route changed upstream, if traffic is too
sparse to keep anything warm, or if the provider does not cache that model at
all. The only thing that can answer "is the cache working" is the provider's own
accounting, and that has been landing in model_usage_events.cache_read_tokens
since model_lab_v1 without anyone reading it.

So this reads it. It computes nothing about prompts and makes no claim about
what could be cached; every number here is what a provider reported.

WHEN THERE IS NO DATA
---------------------
It says NO DATA. Before real traffic that is the correct answer, and it is a
far more useful one than a theoretical figure presented as a measurement.

USAGE
-----
    SUPABASE_DB_URL='postgresql://...' python scripts/model_cache_report.py
    ... python scripts/model_cache_report.py --window 1h
    ... python scripts/model_cache_report.py --window 7d --by model
    ... python scripts/model_cache_report.py --window 24h --by feature

PRIVACY
-------
Only counters are read: token counts, latency, cost, timestamps, and the
provider/model/feature labels. No prompt, no message content, no fan or creator
identity is selected or printed.
"""

from __future__ import annotations

import argparse
import os
import sys

WINDOWS = {
    "1h": "1 hour",
    "24h": "24 hours",
    "7d": "7 days",
}

# The grouping keys, and the column expression each maps to.
#
# "upstream" is the OpenRouter provider that actually served the request. It
# lives in metadata rather than a column because it only exists for routed
# calls, and it is the single most useful breakdown when a cache rate drops:
# OpenRouter silently moving traffic between upstreams changes cache behaviour
# without anything in our code changing.
GROUPINGS = {
    "provider": "provider",
    "model": "provider || ' / ' || model",
    "upstream": "coalesce(metadata->>'upstream_provider', metadata->>'provider', '(none)')",
    "feature": "feature",
}


def _fetch(connection, schema: str, window: str, group_expression: str) -> list[tuple]:
    sql = f"""
        select
            {group_expression}                       as bucket,
            count(*)                                 as calls,
            sum(input_tokens)                        as input_tokens,
            sum(cache_read_tokens)                   as cached_tokens,
            sum(cache_write_tokens)                  as cache_writes,
            sum(estimated_cost_usd)                  as cost_usd,
            percentile_disc(0.5) within group (order by latency_ms)  as p50_ms,
            percentile_disc(0.95) within group (order by latency_ms) as p95_ms,
            count(*) filter (where not success)      as failures
          from {schema}.model_usage_events
         where created_at >= now() - interval '{window}'
         group by 1
         order by calls desc
    """
    with connection.cursor() as cursor:
        cursor.execute("begin read only")
        try:
            cursor.execute(sql)
            return cursor.fetchall()
        finally:
            cursor.execute("rollback")


def _render(rows: list[tuple], *, window_label: str, group_label: str) -> None:
    if not rows:
        print(f"NO DATA — no model calls recorded in the last {window_label}.")
        print()
        print("This is the honest answer before real traffic. A cache-read rate")
        print("cannot be reported until a provider has reported one.")
        return

    header = (
        f"{group_label:<34}{'calls':>8}{'input':>12}{'cached':>12}"
        f"{'cache %':>9}{'cost $':>10}{'p50':>7}{'p95':>7}{'fail':>6}"
    )
    print(header)
    print("-" * len(header))

    total_calls = total_input = total_cached = total_writes = total_failures = 0
    total_cost = 0.0

    for (
        bucket, calls, input_tokens, cached_tokens, cache_writes,
        cost_usd, p50_ms, p95_ms, failures,
    ) in rows:
        input_tokens = int(input_tokens or 0)
        cached_tokens = int(cached_tokens or 0)
        # Denominator is everything that was sent as prompt: what the provider
        # billed as fresh input PLUS what it served from cache. Dividing by
        # input_tokens alone would report a rate above 100% on a good hit.
        prompt_tokens = input_tokens + cached_tokens
        rate = (cached_tokens / prompt_tokens * 100) if prompt_tokens else 0.0

        print(
            f"{str(bucket)[:33]:<34}{calls:>8}{input_tokens:>12,}"
            f"{cached_tokens:>12,}{rate:>8.1f}%{float(cost_usd or 0):>10.4f}"
            f"{(p50_ms or 0):>7}{(p95_ms or 0):>7}{failures:>6}"
        )

        total_calls += calls
        total_input += input_tokens
        total_cached += cached_tokens
        total_writes += int(cache_writes or 0)
        total_cost += float(cost_usd or 0)
        total_failures += failures

    total_prompt = total_input + total_cached
    total_rate = (total_cached / total_prompt * 100) if total_prompt else 0.0
    print("-" * len(header))
    print(
        f"{'TOTAL':<34}{total_calls:>8}{total_input:>12,}{total_cached:>12,}"
        f"{total_rate:>8.1f}%{total_cost:>10.4f}{'':>7}{'':>7}{total_failures:>6}"
    )
    print()
    print(f"cache writes: {total_writes:,} tokens")

    if total_cached == 0 and total_calls:
        print()
        print("Cache reads are ZERO across every call in this window.")
        print("That is not necessarily a defect — it is the expected reading when")
        print("traffic is too sparse to keep a prefix warm, or when the provider")
        print("serving these calls does not cache this model. Check the upstream")
        print("breakdown (--by upstream) before changing any prompt.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarise ACTUAL provider prompt-cache usage from "
                    "model_usage_events. Reads counters only; never prompts or "
                    "message content.",
    )
    parser.add_argument(
        "--window", choices=sorted(WINDOWS), default="24h",
        help="reporting window (default: 24h)",
    )
    parser.add_argument(
        "--by", choices=sorted(GROUPINGS), default="model",
        help="grouping (default: model)",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="every window and every grouping",
    )
    parser.add_argument(
        "--database-url", default="",
        help="connection string; defaults to $SUPABASE_DB_URL or $TEST_DATABASE_URL",
    )
    parser.add_argument("--schema", default="public")
    args = parser.parse_args(argv)

    url = (
        args.database_url
        or os.environ.get("SUPABASE_DB_URL", "")
        or os.environ.get("TEST_DATABASE_URL", "")
    ).strip()
    if not url:
        print("Set SUPABASE_DB_URL (or pass --database-url).", file=sys.stderr)
        return 2

    try:
        import psycopg
    except ImportError:
        print("psycopg is required: pip install 'psycopg[binary]'", file=sys.stderr)
        return 2

    try:
        connection = psycopg.connect(url, autocommit=True)
    except Exception as exc:
        print(f"Could not connect: {type(exc).__name__}", file=sys.stderr)
        return 2

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "select 1 from information_schema.tables "
                " where table_schema = %s and table_name = 'model_usage_events'",
                (args.schema,),
            )
            if cursor.fetchone() is None:
                print("NO DATA — model_usage_events does not exist in this "
                      "database. Apply db/model_lab_v1.sql.")
                return 0

        combinations = (
            [(w, g) for w in ("1h", "24h", "7d") for g in ("provider", "model",
                                                           "upstream", "feature")]
            if args.all
            else [(args.window, args.by)]
        )

        for index, (window, grouping) in enumerate(combinations):
            if index:
                print()
            label = f"last {WINDOWS[window]}, by {grouping}"
            print(f"=== {label} ===")
            rows = _fetch(connection, args.schema, WINDOWS[window],
                          GROUPINGS[grouping])
            _render(rows, window_label=WINDOWS[window], group_label=grouping)
    finally:
        connection.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
