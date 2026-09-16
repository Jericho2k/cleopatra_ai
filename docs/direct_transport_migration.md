# Migrating off API Fansly, one operation at a time

## What we are actually buying

API Fansly is not doing anything Fansly's own web client does not do. It holds a
live session per creator and speaks Fansly's private API (`apiv3.fansly.com`)
on our behalf, then bills for the privilege:

| What | Rate |
|---|---|
| ordinary request | 1 credit |
| response over 80 KB | proportional, ~1 credit per 80 KB |
| received webhook events | 1 credit per 80 |
| **media transferred** | **2 credits per MB** |

So the question "can we do this ourselves?" has three separate answers
depending on the operation, and they are not close to each other:

* **Bytes we can move ourselves.** Media transfer is metered at 2 credits/MB
  and involves no cleverness at all. This is the first thing to take back, and
  it is what this document's shipped phase does.
* **Requests we can make ourselves.** Listings, history pages, vault reads. The
  session we already hold is enough. The cost of taking these back is request
  volume against the creator's own account, which is a ban-risk budget rather
  than an invoice.
* **Things the provider genuinely holds for us.** Webhooks, and the session
  lifecycle itself. See *The parts that are not just code* below — these are
  the reason a full migration is not a weekend.

We already hold the sessions: `services/fansly_session_store.py` stores
per-account credentials encrypted, `services/fansly_client.py` speaks the
private API with them, and `services/fansly_poller.py` has been using both to
watch for inbound messages. Half the direct path was already running in
production before this migration started.

## How the switch works

`core/transport_policy.py` answers one question — *who should serve this call?*
— per operation, per account, at runtime:

```
FANSLY_TRANSPORT_<OPERATION>   e.g. FANSLY_TRANSPORT_MEDIA_DOWNLOAD=direct
FANSLY_TRANSPORT_DEFAULT       provider | direct
FANSLY_DIRECT_ACCOUNTS         canary allowlist; empty means every account
FANSLY_DIRECT_FALLBACK         true (default) retries a failed direct call on the provider
```

Unset means `provider` for everything: upgrading changes nothing until someone
opts in. An unrecognised value also means `provider`, so a typo in a deployment
variable leaves a creator on the transport that is known to work.

`FANSLY_DIRECT_FALLBACK` is the load-bearing one. While a direct implementation
is young, a failure that quietly costs a credit is a far better outcome than a
fan waiting on a reply that never comes. Turn it off only when you want a
regression to be loud.

## Measuring it, or it did not happen

`GET /apifansly-usage` now returns both halves over the same 24-hour window:

```json
{
  "...": "provider usage as before",
  "direct_transport": {
    "policy":  { "transports": {...}, "direct_operations": [...] },
    "savings": { "transfers": 12, "credits_saved": 843.2, "by_account": {...} }
  }
}
```

The two ledgers are deliberately separate. `services/apifansly.py`'s means
"what we were billed" and must keep meaning exactly that. The savings ledger in
`services/fansly_direct.py` records what the provider *would* have billed for
each transfer it served, at the same 2 credits/MB rule. Without it the whole
justification stays a theory — a smaller invoice next month could equally be a
quiet week.

## Phase 1, shipped: protected media reads

The expensive one. A 250 MB clip pulled through the metered proxy is ~500
credits on a single call, and vault classification reaches for exactly those
clips.

Reading a protected asset now goes, cheapest first:

1. **Unauthenticated CDN read** — already existed (`_download_direct_cdn`), free,
   works whenever the signed URL is still valid.
2. **Authenticated CDN read** *(new)* — the same fetch carrying the creator's
   session. The common case for a protected asset. Zero credits.
3. **Signed-URL refresh** *(new)* — ask the API for the media item again, take
   the freshly signed location out of the answer, fetch that with no
   credentials at all. This is the route the metered proxy is really selling:
   the signature, not the bytes. Zero credits.
4. **The metered proxy** — unchanged, and now a fallback rather than the plan.

Two consequences worth knowing:

* **The credit guard does not apply to a direct read.** `services/media_cost_guard.py`
  exists *because of* 2 credits/MB; it refuses a 300 MB video on the automatic
  path. Served directly that video costs bandwidth, so refusing it would leave
  the asset unclassified to guard against a cost nobody is incurring. The direct
  path keeps a separate 512 MB in-process ceiling — not paying per megabyte is
  still not a reason to pull an unbounded file into the web process.
* **`billed_bytes` still means billed bytes.** A direct transfer records zero,
  so the existing per-item cost telemetry does not silently start meaning
  "bytes moved".

Session credentials are only ever sent to an HTTPS `*.fansly.com` host, checked
at both layers, including on a location that came back from the API. An API
answer is not a licence to fetch whatever host it names.

### Before switching an account on

Two things in this path are observed from browser traffic rather than published:
whether a protected asset is served to an authenticated session read, and what
the account-media endpoint is called. Neither can be settled by a test suite,
so check them against one real account:

```
python scripts/verify_fansly_direct.py --account <id> --url <cdn url> --media-id <id>
```

It is read-only, takes seconds, and tells you which of the two routes works. If
the endpoint path has moved, `FANSLY_DIRECT_MEDIA_PATH` changes it without a
deploy.

Then:

```
FANSLY_TRANSPORT_MEDIA_DOWNLOAD=direct
FANSLY_DIRECT_ACCOUNTS=<the account you verified>
```

Watch `direct_transport.savings` on `/apifansly-usage`, then widen the
allowlist and finally clear it.

## The order to migrate in, and why

Sorted by credits saved per unit of risk. Read-only work first, sends last,
realtime last of all.

| Phase | Operations | Why here |
|---|---|---|
| 1 ✅ | media download | 2 credits/MB, one call site, no write risk |
| 2 | vault albums, album media, followers, subscribers, lists, chat listing | read-only, high page volume, no fan-visible failure mode |
| 3 | chat message history | the backfill bill — see below |
| 4 | text sends, then PPV sends | writes. Real money and a confused fan if wrong |
| 5 | realtime events | replaces the webhook; hardest, see below |

**Phase 3 deserves its own note.** API Fansly's chat-messages endpoint documents
`limit min=1 max=10`, which is why a 5,000-message fan is 500 round trips
(`docs/historical_memory_and_credits.md`). Fansly's own messenger endpoint is
not obviously subject to that ceiling — `services/fansly_client.py` already asks
it for 50 — so moving history backfill direct could cut it several-fold on top
of removing the per-page credit. That is a large enough prize to be worth
measuring properly before Phase 4, and small enough in risk that it should come
before any write path. **Verify the real page ceiling against the live endpoint
before counting on it**; nothing in this repository has confirmed it.

## The parts that are not just code

Three honest caveats. None is a reason not to migrate; all are reasons to keep
the provider account open while you do.

**`fansly-client-check` is the real moat.** Requests carry `fansly-client-id`,
`fansly-client-ts` and `fansly-client-check`. Our client treats the check value
as a static captured string, which holds only while Fansly does not validate it
strictly per request. It is derived by their frontend JavaScript, and that
JavaScript changes. Maintaining it is the recurring cost you take on in place
of credits — not a one-off port. Budget for re-capturing the header profile
periodically, and expect a day where a Fansly release breaks the direct path
and the provider fallback quietly carries the load. That day is what
`FANSLY_DIRECT_FALLBACK=true` is for.

**There is no direct equivalent of the webhook.** The provider delivers
`ppv.purchased`, `tips.received` and `subscriptions.new` to `/webhook/fansly`.
Fansly's own client learns these over a WebSocket. Until that is implemented,
Phase 5 means polling — and polling every 8–15 seconds per account across N
accounts is a lot of requests against a real session, which is exactly the
signal that gets accounts flagged. Do this one last, do it with a WebSocket,
and keep the provider's webhook until it works.

**Ban risk moves onto you.** The provider absorbs some of it today. The
user-agent rotation, request jitter and per-account proxy support already in
`services/fansly_client.py` exist for this reason. Whether running your own
automation against creator accounts is acceptable under Fansly's terms is a
question for you and your counsel, not one this document can answer — but the
answer changes who carries the consequence of an account being locked, and the
accounts are your creators' livelihoods, not yours.

## Where things live

| File | Role |
|---|---|
| `core/transport_policy.py` | who serves each operation. No network code |
| `services/fansly_direct.py` | direct implementations + the savings ledger |
| `services/fansly_client.py` | the private-API client and its session headers |
| `services/fansly_session_store.py` | encrypted per-creator sessions |
| `services/apifansly.py` | the metered provider client. Unchanged |
| `scripts/verify_fansly_direct.py` | go/no-go check against one real account |
