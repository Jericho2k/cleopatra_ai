# PostgREST transport reliability — `ConnectionTerminated` P1

**Status:** fixed. **Scope:** the Supabase/PostgREST HTTP transport and bounded
retry for reads. No schema change, no business-logic change, no dashboard
change, no change to API Fansly semantics.

---

## 1. Symptom

Four unrelated background loops and inbound API requests failing in the same
instant, with one identical message:

```
[CRON CHAT RECONCILE INFRA ERROR] <ConnectionTerminated error_code:0, last_stream_id:3, additional_data:None>
[SCHEDULED LOOP ERROR]            <ConnectionTerminated error_code:0, last_stream_id:3, additional_data:None>
[PPV SWEEP FATAL]                 <ConnectionTerminated error_code:0, last_stream_id:3, additional_data:None>
[CRON VAULT AUTOSYNC ERROR]       <ConnectionTerminated error_code:0, last_stream_id:3, additional_data:None>
GET /my-creators -> 500   (httpcore/_sync/http2.py, httpcore.RemoteProtocolError)
```

`scripts/production_preflight.py` returned 25 passed / 0 failed / 1 warning
against production Postgres, so the schema, migrations and RLS were healthy.
The failure was entirely in the application transport, as reported.

## 2. Root cause

Three facts compose into the incident.

**(a) HTTP/2 was on, and not by our choice.** `postgrest` builds its session as:

```python
# postgrest/_sync/client.py
self.session = http_client or Client(..., follow_redirects=True, http2=True)
```

`http2=True` is hardcoded, and `h2` is not optional — `postgrest`, `storage3`,
`supabase-auth` and `supabase-functions` all declare `httpx[http2]`, so `h2
4.4.1` is always installed and HTTP/2 is always negotiated. `core/supabase.py`
passed no `http_client`, so production ran on it.

**(b) httpcore does not recover an HTTP/2 GOAWAY that lands on an in-flight
stream.** Supabase's PostgREST edge recycles connections with a graceful
`GOAWAY` (`error_code:0` is NO_ERROR — an orderly shutdown, not a fault).
`httpcore` splits on one condition:

```python
# httpcore/_sync/http2.py, _receive_events
if stream_id and last_stream_id and stream_id > last_stream_id:
    raise ConnectionNotAvailable()                       # pool re-dispatches
raise RemoteProtocolError(self._connection_terminated)   # reaches the caller
```

A GOAWAY seen while the connection is **idle** self-heals — our stream id is
above `last_stream_id`, the connection is discarded and the request is
re-dispatched transparently. A GOAWAY that arrives while requests are **in
flight** does not: those streams raise `RemoteProtocolError` at the caller.

This is exactly the asymmetry HTTP/1.1 does not have. `httpcore`'s HTTP/1.1
connection checks whether an idle socket has become readable before reusing it:

```python
# httpcore/_sync/http11.py, has_expired
server_disconnected = (
    self._state == HTTPConnectionState.IDLE
    and self._network_stream.get_extra_info("is_readable")
)
```

The HTTP/2 path has no equivalent (`has_expired` there only checks the keepalive
timer).

**(c) HTTP/2 multiplexes, and we had exactly one client.** `core/supabase.py`
cached a single `Client` for the process, so every loop and every request shared
one connection pool and, in practice, one TCP connection. One GOAWAY therefore
failed *every* concurrent caller at once — which is why four independent loops
and an HTTP route reported the same `last_stream_id:3` simultaneously.

**Nothing retried it.** `postgrest`'s `send_with_retry` only retries HTTP
*status* 503/520 on GET/HEAD and never sees a transport exception. `httpx`'s
`retries=` only covers connection establishment. Most of our call sites were
bare `asyncio.to_thread(... .execute())`.

### Measured, not assumed

`tests/h2_probe_server.py` runs a real `h2` server over a real socket.
Reproduced against it:

| scenario | result |
|---|---|
| GOAWAY on an **idle** HTTP/2 connection | self-heals; next request succeeds |
| GOAWAY on an **in-flight** HTTP/2 stream | `httpx.RemoteProtocolError: <ConnectionTerminated error_code:0, last_stream_id:3, additional_data:None>` — byte-identical to production |
| 6 concurrent HTTP/2 requests, one GOAWAY | **all 6** fail |
| peer recycling connections, HTTP/1.1 | 6/6 succeed |
| after the failure, is the pool poisoned? | **No** — the pool evicts the dead connection and the next request succeeds |

That last row matters: the cached `Client` was never permanently poisoned. The
outage was not "the client is broken forever", it was "one round trip died and
nobody asked again" — repeated every time the edge recycled a busy connection.

**Exact failing code path:**
`main.py:get_my_creators` → `asyncio.to_thread` → `postgrest` `execute()` →
`send_with_retry` → `httpx.Client.request` → `httpcore` `HTTP2Connection.handle_request`
→ `_receive_events` → `raise RemoteProtocolError(ConnectionTerminated)` →
uncaught → FastAPI 500. Identically for each loop, with the loop's `except`
printing the message and discarding the whole cycle.

**Was HTTP/2 involved?** Yes. It is the root cause, not a contributing factor.

## 3. What changed

### `core/supabase.py` — an explicit transport

- `ClientOptions(httpx_client=...)` — the documented, supported injection point
  in supabase-py 2.31.0. The same client is used by PostgREST and auth.
- **HTTP/2 disabled.** `SUPABASE_HTTP2=1` turns it back on, so reverting is an
  env var rather than a deploy.
- Pool sized against the DB executor: `max_connections=64`,
  `max_keepalive_connections=32`, `keepalive_expiry=30s` (httpx's 5s default
  meant repeated TLS handshakes under burst).
- Timeouts named: `connect=10s`, `pool=30s`. Read/write stay at postgrest's
  120s — tightening those changes which queries succeed, which is not this
  fix's business. All env-tunable.
- `retries=2` on the transport. This only wraps httpcore's `_connect`, which
  catches `ConnectError`/`ConnectTimeout` *before any request bytes are
  written*, so it is safe for writes too.
- `follow_redirects=True` preserved (postgrest sets it on the session it builds;
  an injected client replaces that session wholesale).
- `lru_cache` replaced with a lock-guarded module global plus a **generation
  counter**, `reset_supabase_client()` and `close_supabase_client()`.

### `services/db_reliability.py` — bounded retry, with the safety argument at the call site

- Jittered, capped backoff so callers knocked over together do not re-collide.
- `retry_db_read(...)` — the one-liner for `to_thread(...select().execute())`.
- Structured logs: `transient` / `recovered` / `exhausted`, plus
  `[DB TRANSPORT RESET]`. Healthy calls log nothing.
- `redact()` scrubs `SUPABASE_URL` and `SUPABASE_SERVICE_KEY` from any logged
  message and truncates to 200 chars. No keys, JWTs, URLs or message content.

### Call sites moved onto bounded retry (reads only)

| site | operation |
|---|---|
| `main.py` `/my-creators` | two selects |
| `main.py` `vault_autosync_scheduler` | creators select |
| `main.py` `chat_reconciliation_scheduler` | creators select |
| `services/suggestions.py` `sweep_stale_ppv_checks` | paged fans select |
| `services/operational_health.py` `probe_database` / `probe_queue` | health probes |

220-odd other call sites were deliberately **not** touched. With HTTP/2 off the
blast radius is one request rather than all of them, and widening the retry
surface is where write-safety mistakes come from.

### `main.py` lifespan

Boot logs the resolved transport; shutdown releases the pool.

## 4. Retry policy by operation category

| category | retried? | reasoning |
|---|---|---|
| **Reads** (`select`, paged selects) | Yes, 3 attempts, jittered | Repeating a select cannot apply anything twice. |
| **Health probes** | Yes, 2 attempts, no client reset | One recycled connection is not an outage. Timeouts are *not* retried — a database too slow for a one-row select in 1.5s is a real signal, and repeating it would double health latency during an incident. |
| **Read-only RPCs** | Not changed | None were on the incident path; classifying them is a separate piece of work. |
| **Atomic claim RPC** (`claim_due_actions`) | **No** | A retry cannot double-claim — rows just stamped `PROCESSING` match neither the due nor the stale branch. But it *can* strand the first batch as `PROCESSING` until the 10-minute stale window elapses. The worker re-polls within seconds, so failing the cycle is strictly better than retrying a write to save seconds. Pinned by a test. |
| **Arbitrary writes** (`INSERT`/`UPDATE`/`DELETE`) | **No** | `RemoteProtocolError` after the request is on the wire is ambiguous — the statement may have committed. Ambiguous writes belong to the durable action queue and the reconciliation passes. Pinned by a test on the bare `messages` insert. |
| **Compare-and-set writes** | Yes (pre-existing, unchanged) | e.g. the identity-reconciliation `UPDATE ... WHERE fansly_message_id IS NULL`. A repeat matches no rows because the first application cleared the predicate. Now demonstrated by a test that makes the first attempt commit *and then* lose its response. |
| **Provider sends (API Fansly)** | Untouched | Out of scope, as specified. |

**No new write retry was introduced by this change.**

## 5. How client recovery works

The measured evidence says httpx's pool already evicts a terminated connection,
so a client rebuild is a last resort rather than the recovery path. It is
implemented anyway, for the poisoned-pool case we cannot see, and it is
deliberately hard to trigger:

1. A caller reads `supabase_generation()` **before** each attempt.
2. On a transient failure at or after `reset_after_attempt` (default: the 2nd),
   it calls `reset_supabase_client(reason=..., generation=g)`.
3. Under the lock: if the current generation no longer equals `g`, someone has
   already rebuilt — no-op, return `False`. **This is what prevents a thundering
   herd:** 24 callers failing on one connection produce exactly one reset
   (tested).
4. The winner drops the reference and bumps the generation. The next
   `get_supabase()` builds one new client under the same lock, so N waiting
   callers produce one build, not N (tested).
5. The retired transport is **not** closed synchronously — `httpx.Client.close()`
   closes the whole pool including connections other threads are mid-response
   on, which would turn one failure into several. It is closed on a daemon timer
   after a 130s grace period.
6. Health probes pass `reset_after_attempt=None`: the thing that observes an
   outage must not be a cause of one.

Process-wide reuse in the healthy case is unchanged.

## 6. Health behaviour

`/health` was already un-latched — `collect()` runs live probes behind a 5s
cache, and `evaluate()` derives `database_unreachable` from the current probe
only. The failure was that the *probe itself* was a single unprotected round
trip, so one GOAWAY published `database_unreachable` and put "Cleopatra cannot
reach its database" in front of an operator whose database was fine.

- A transient failure no longer raises the banner (one retry tells it apart from
  an outage — an outage fails both times).
- A genuine outage still reports `unhealthy` + `fatal_reasons`, and
  `/health/ready` still 503s. Not weakened.
- Recovery clears within one cache window. Tested in both directions.
- New diagnostic block `db_transport` in the health document: `http2`,
  `generation`, pool limits. A rising `generation` is the signal that transport
  resets are happening at all. No URLs, no keys.

**The dashboard was inspected and not changed.** `lib/health.ts` and
`components/SystemHealthBanner.tsx` replace state on every successful poll and
return `null` at status `ok`; there is no client-side latch. The "banner
reappears on navigation" symptom was a fresh mount re-polling `/health` and
catching another GOAWAY-killed probe. No independent dashboard bug found.

## 7. Tests added

`tests/h2_probe_server.py` — real h2 and HTTP/1.1 origin servers. These matter
because the bug is in `httpcore`, not in us: a mock of our own client would pass
whatever we did to the transport.

`tests/test_postgrest_transport.py` (6) — the diagnosis, against real sockets:

- the production error string produced by `h2` itself, and classified transient;
- a GOAWAY failing **all 6** concurrent multiplexed requests;
- the same peer behaviour absorbed cleanly over HTTP/1.1;
- the SDK is built on our transport (`http2=False`, shared with auth,
  `follow_redirects` preserved) — pinned because the default lives three
  dependencies down where an upgrade can silently change it back;
- `SUPABASE_HTTP2=1` restores HTTP/2;
- connect retries are configured and cannot replay a write.

`tests/test_db_transport_recovery.py` (18) — behaviour on top of it, using the
real `httpx.RemoteProtocolError(h2.events.ConnectionTerminated)`:

| requirement | tests |
|---|---|
| 1. Read recovery | `/my-creators` returns 200 after a terminated connection; still fails closed when the database is genuinely gone |
| 2. Client recovery | rebuild produces a working client; stale-generation reset is a no-op; a replaced transport is not closed under a live request |
| 3. Bounded retry | stops at the budget and re-raises; non-transient errors are not retried at all; backoff is jittered and capped |
| 4. Non-idempotent write safety | bare `messages` INSERT attempted exactly once; `claim_due_actions` attempted exactly once |
| 5. Idempotent write | compare-and-set UPDATE: first attempt **commits and then loses its response**, retry matches nothing, row updated once |
| 6. Concurrent failure | 24 simultaneous failures → 1 reset; 24 simultaneous users → 1 build; 16 concurrent retries → ≤1 extra build |
| 7. Health recovery | outage → `unhealthy` + `database_unreachable`; recovery → `ok`; a single terminated connection does not raise the banner; the probe never churns the client; the transport snapshot leaks nothing |

## 8. Commands and results

```
$ python -m pytest -q                                    # TEST_DATABASE_URL set
975 passed in 95.77s

$ python -m pytest -q                                    # no database, as CI skips
863 passed, 112 skipped in 74.78s

$ ruff check --select E9,F63,F7,F82,F401,F811,F841 .
All checks passed!

$ python -m compileall -q . -x '(\.git|node_modules)'
OK

$ python scripts/production_preflight.py --env-only
2 passed, 0 failed, 10 warnings, 0 skipped
```

The 975-run includes `tests/test_schema_pipeline.py` (base schema + every
migration in `db/migration_order.txt` applied to a real PostgreSQL 16), the RLS
and tenancy suites, scheduled actions, durable ingestion, purchase identity and
the worker concurrency invariants. No regressions.

## 9. Deployment

**Railway/env changes required: none.** Every new setting has a working default.

Optional knobs, all safe to leave unset:

| variable | default | effect |
|---|---|---|
| `SUPABASE_HTTP2` | off | `1` restores HTTP/2 (the rollback switch) |
| `SUPABASE_MAX_CONNECTIONS` | 64 | pool ceiling |
| `SUPABASE_MAX_KEEPALIVE_CONNECTIONS` | 32 | idle connections kept |
| `SUPABASE_KEEPALIVE_EXPIRY_SECONDS` | 30 | idle connection lifetime |
| `SUPABASE_CONNECT_TIMEOUT_SECONDS` | 10 | handshake budget |
| `SUPABASE_REQUEST_TIMEOUT_SECONDS` | 120 | read/write budget (unchanged from postgrest's default) |
| `SUPABASE_POOL_TIMEOUT_SECONDS` | 30 | wait for a pooled connection |
| `SUPABASE_CONNECT_RETRIES` | 2 | connection-establishment retries only |

**Steps:** merge, deploy as normal, no migration, no downtime, no ordering
constraint with the dashboard.

**Verify after deploy:**

1. `[BOOT] supabase transport: http2=False max_connections=64 ...` in the logs.
2. `GET /health` (with `x-api-key`) → `db_transport.http2 == false`,
   `db_transport.generation == 0`.
3. `ConnectionTerminated` should disappear from `[CRON ...]` / `[SCHEDULED LOOP
   ERROR]` / `[PPV SWEEP FATAL]`.
4. If any residual transient failure occurs, it now logs `[DB RETRY] transient
   ... ` followed by `recovered`, instead of failing a cycle silently.

**Rollback:** set `SUPABASE_HTTP2=1` and restart — that restores the previous
transport behaviour without a code change (the retry and reset layers stay, and
are strictly additive). Full rollback is a normal revert of this commit; nothing
is persisted and no migration was applied.

**Safe to deploy immediately: yes.** No schema change, no business-logic change,
no write-path semantics change, full suite green including the real-database
schema tests.

## 10. Dependency note (`requirements.txt`) — no change made

The reported local `pip install -r requirements.txt` failure was investigated
and is **not** a repository correctness error.

Every pin resolves from PyPI, and the whole file resolves together in a clean
virtualenv:

```
$ python -m venv /tmp/v && /tmp/v/bin/pip install --dry-run --report r.json -r requirements.txt
would install: 54
  fastapi 0.140.7   uvicorn 0.51.0   openai 2.48.0   supabase 2.31.0
  httpx 0.28.1      httpcore 1.0.9   postgrest 2.31.0  h2 4.4.1
```

`fastapi==0.140.7`, `uvicorn==0.51.0` and `openai==2.48.0` all exist and are not
yanked; the same check passes for every other pin including the dev
requirements. These are not typos or future pins, and `requirements.txt` is not
inconsistent with the deployed build. The local failure was an index/mirror that
had not caught up (an offline mirror, a stale cache, or a private index without
those releases) — a local environment problem, not a repository one.

Left unchanged deliberately. Anyone hitting it locally should point pip at PyPI
(`pip install -i https://pypi.org/simple -r requirements.txt`) or clear the pip
cache.

## 11. Still unknown

- **Why the edge GOAWAYs at `last_stream_id:3` specifically.** That is only
  three or four streams into a connection's life, which suggests an aggressive
  stream or lifetime cap on Supabase's HTTP/2 front rather than a normal drain.
  It is Supabase-side and not observable from here. It does not change the fix —
  HTTP/1.1 is unaffected either way — but if HTTP/2 is ever re-enabled, this is
  the thing to ask Supabase about first.
- **Whether disabling HTTP/2 costs measurable throughput.** Under our workload
  (a bounded thread pool of short PostgREST requests) it should not, and the
  keepalive change offsets the loss of multiplexing. Worth watching
  `db_executor.queued` and `database.latency_ms` in the health document for a
  few days.
- **Whether any of the remaining ~220 unprotected read sites matters.** With the
  blast radius reduced from "every concurrent call" to "one call", none should.
  `[DB RETRY]` / `[DB TRANSPORT RESET]` log lines are the evidence that would say
  otherwise.
