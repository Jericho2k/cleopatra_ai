# Historical conversation memory and API Fansly credit accounting

Two related systems, shipped together because they are the same problem seen
from two sides: importing an existing fan's archive is the largest discretionary
spend this product can make against API Fansly, and until now neither the cost
nor the progress of that import was visible.

---

## 1. The page-size rule. Read this before "optimising" anything.

**API Fansly's chat-messages endpoint documents `limit min=1 max=10`.**

Ten is the provider's ceiling, not a conservative default we chose. Asking for
50 does not return 50 — it is clamped upstream — so the old `/load-history`,
which requested `limit=50`, believed it was reading 50 messages per page while
actually reading 10. It paid for five times the pages it reported.

The constant lives in exactly one place:

```python
# services/apifansly.py
CHAT_MESSAGE_PAGE_MAX = 10
```

It is not a tunable. **Do not raise it.** The only thing that can change it is
API Fansly publishing a larger documented maximum; if that ever happens, change
the constant there and update the page economics in this document.

`tests/test_apifansly_instrumentation.py` fails the build if any call site asks
`list_chat_messages` for more than ten.

### What that costs

| Conversation | Pages (at max 10) | Credits, ordinary pages | Credits, 240 KB media-heavy pages |
|---|---|---|---|
| 100 messages | 10 | ~10 | ~30 |
| 1,000 messages | 100 | ~100 | ~300 |
| **5,000 messages** | **500** | **~500** | **~1,500** |

A 5,000-message fan is 500 provider round trips, minimum, forever. Every design
decision below follows from refusing to pay that twice, and from refusing to put
it in front of a fan who is waiting for a reply.

---

## 2. Historical conversation memory

### Shape

```
             ┌──────────────── warm resume (bounded, may block a turn) ────────┐
             │  newest ~3 pages, only when local context is too thin           │
             └────────────────────────────────────────────────────────────────┘
                                        │
   provider ──► services/fan_history.py ──► messages (idempotent, platform id)
                         │                         │
                 fan_history_backfill              │
                 (durable page cursor)             ▼
                         │            services/fan_history_memory.py
                         │                   (GLM-5.3-Flash, chunked)
                         ▼                         │
                 durable cost telemetry            ▼
                                     fan_facts / fan_fact_observations
                                     + compact continuity state
```

### Warm resume — why an old fan does not wait for 500 pages

When a previously known fan becomes active, `warm_resume()` asks one question:
is there enough local context to answer naturally? (`HISTORY_WARM_MIN_LOCAL_MESSAGES`,
default 12.)

* **Yes** — the common case. It costs nothing and returns immediately.
* **No** — it fetches the newest few pages (`HISTORY_WARM_MAX_PAGES`, default 3
  = 30 messages), persists them, and returns. Roughly three credits.

That is the *only* history work allowed in front of a live turn. The rest of the
archive is left to the resumable deep pass.

It also never rewinds a deep backfill: a fan already 400 pages in keeps that
cursor, and warm resume reads the newest pages for continuity without touching
it.

### Deep backfill — resumable, restart-safe, bounded

`advance_backfill()` fetches a bounded run of pages from the durable cursor in
`public.fan_history_backfill` and returns. The cursor advances **only after** a
page has been persisted, so a crash between fetch and checkpoint re-fetches that
one page — which writes nothing, because persistence is idempotent on the
`(creator_id, fansly_message_id)` unique index.

This is the whole point of HIST-001: a deploy mid-import used to throw away
every page already paid for.

**Live conversation always wins.** Priority is re-checked *before every page*,
not once per batch, so a reply arriving after page two does not wait out pages
three through twenty. The signal is `services.apifansly.live_work_in_progress()`
— an open live-chat usage scope, or a live-chat provider call in the last 30
seconds.

### Media: metadata only, never a binary

A chat page already carries `accountMedia` for every attachment on it, and we
have already paid for it. History persists the useful lightweight parts —
media id, type, mimetype, price, `is_ppv`, `purchased`, `access` — into the same
`messages.media_context.attachments` shape the live path writes, and links them
to `creator_vault_media` rows we already mirror when the media id matches.

**History makes no media call of any kind.** Media transfer is billed at
2 credits/MB; re-downloading a fan's archive of video would dwarf the entire
import. There is no download path in `services/fan_history.py` and there must
never be one — `tests/test_apifansly_instrumentation.py` asserts it.

### Compaction — a cheap model, not the writer

`services/fan_history_memory.py` reads the imported archive chronologically in
bounded chunks (40 messages) and turns it into durable facts.

* **Model:** Together `zai-org/GLM-5.3-Flash`, $0.15/M input and $0.50/M output
  (snapshot — reverify before a large import). Stage
  `STAGE_HISTORY_EXTRACTION`, identical in every AI Stack Profile.
* **Kimi is untouched.** It remains the conversational writer.
* **Live fan-intelligence extraction is untouched.** It stays on
  `openai/gpt-oss-120b`. Re-pointing a live stage for symmetry with a background
  one is how a profile comparison stops being a comparison.

Facts go into the **existing** `fan_facts` / `fan_fact_observations` tables with
`source_type = 'historical_message'`. There is one fan memory, one set of merge
rules, one evidence table. A second "historical facts" store would mean the
contradiction, explicit-only and money rules all had to be written twice.

#### Evidence discipline

A historical chunk shows the model messages from **both** speakers, which
creates two failure modes live extraction does not have. Both are rejected
deterministically, before the model's output can influence anything:

1. the proposal must name a `source_message_id` that is **in this chunk**;
2. that message must be the **fan's**, not the creator's;
3. the evidence quote must **actually occur** in that message;
4. then every existing live rule applies unchanged — allowed keys, the
   explicit-only set, money parsed from the quote rather than from the model's
   number, confidence floors.

Failure 2 is the one that matters most in practice: it is how
`Creator: "I'm a California girl"` becomes a fact about where the *fan* lives.

#### Historical evidence never overturns current knowledge

`plan_fact_merge(..., historical=True)` changes exactly one outcome: where live
evidence would declare a `CONFLICT` and deactivate the current value, historical
evidence **stands down** (`IGNORE`).

A three-year-old "I live in Chicago" is not news about where he lives today. The
observation is still written to `fan_fact_observations`, so the evidence is not
lost — only the authority to overturn is withheld. Historical evidence can still
create a fact nobody knew, reinforce one, and replace a mere guess.

#### Continuity state

Compaction also keeps a small `continuity` document on the backfill row —
ongoing topics, prior commercial context, a short relationship summary, all
capped. It reaches the writer through the fan-intelligence context every reply
path already reads, rendered as *older context, not current fact*. Structured
facts stay authoritative.

This is what lets a returning fan be treated as an existing relationship rather
than a new lead — without a single archived message reaching a writer prompt.

---

## 3. API Fansly credit accounting

### The model

Published billing rules, as implemented in `services/apifansly.py`:

| Rule | Implementation |
|---|---|
| ordinary request = 1 credit | floor of 1 on every call |
| responses > 80 KB cost proportionally more | `bytes / 81920`, floor 1 |
| 80 webhook events = 1 credit | `events / 80` |
| media upload/download = 2 credits/MB | `2 × MB`, floor 1, **replacing** the response-byte charge |

**Every number this produces is an ESTIMATE. API Fansly's own Usage dashboard is
authoritative.** The estimate exists so an operator can see the *shape* of spend
between dashboard refreshes and attribute it to an operation, an account and a
part of the product.

The 80 KB rule is not academic for history. A chat page carries the full
`accountMedia` metadata for every attachment on it, so a page of ten messages
from a media-heavy conversation routinely exceeds 80 KB. Assuming a flat credit
per page is how a 500-page import gets estimated at 500 credits and bills at
1,500 — which is why credits are computed from **actual response bytes**, never
from a page count.

### Coverage

Not every call goes through `request()`. These raw call sites are now
instrumented explicitly and are enforced by a source-level test:

* typing indicator (`services/suggestions.py`) — one per auto reply
* chat mark-as-read
* account connect and 2FA verification
* media upload (billed on **bytes sent**, not on the small JSON returned)
* media upload status polling
* protected media download (billed on bytes received)
* received webhooks (`/webhook/fansly`, counted after authentication so a
  rejected forgery cannot inflate the estimate)

Accounting is idempotent per response object, so a call site that both raises
through `raise_for_response` and reports its own media bytes is billed once.

A failed call is still accounted. It reached the provider and still cost a
credit; a report that only counted successes would understate spend exactly when
spend is going wrong.

### Categories

Every call is attributed to `live_chat`, `background_history`, `vault`,
`reconciliation`, `account` or `other` — by the scope it runs in, falling back
to the operation name so nothing lands in an unclassified bucket.

The category is also the **priority signal**: an open `live_chat` scope raises
`live_calls_in_flight()`, which deep-history work checks before every page.
Categorisation and priority are one mechanism rather than two that can disagree.

### `GET /apifansly-usage`

Extends the existing snapshot — every field the dashboard already reads is
unchanged — and adds:

```
estimated_credits              total, calls + webhooks
estimated_call_credits         calls only
estimated_webhook_credits      webhooks only
webhook_events                 count in window
media_bytes                    transferred in window
estimated_daily_credits        run-rate from the OBSERVED window
estimated_monthly_credits      run-rate × 30
credits_by_operation           calls / bytes / media_bytes / credits
credits_by_account             same, per API account
credits_by_category            same, per live-vs-background category
webhook_events_by_type
background_history_budget      budget / spent / remaining / exhausted
credit_model                   the constants above, so the arithmetic is legible
```

The run-rate divides by the window this process has actually observed, not a
flat 24 hours: a process that booted ten minutes ago has ten minutes of
evidence.

### Per-fan history telemetry

* `GET /fan-history/{creator_id}/{fan_id}` — pages fetched, messages imported
  and extracted, API calls, response bytes, estimated credits, estimated credits
  **per page**, and whether the provider cursor is exhausted. Needs no provider
  call and no connector: it reads the durable checkpoint.
* `GET /creator-history-usage/{creator_id}` — the same rolled up per creator.

"How much remains" is reported honestly. The provider does not say how long a
conversation is until the cursor runs out, so an unfinished backfill reports
what it has cost and what a page costs on average rather than inventing a total.

### The background history budget

`APIFANSLY_HISTORY_CREDIT_BUDGET_24H` caps estimated credits spent on **optional
deep backfill** in a rolling 24 hours. Unset or `0` means no ceiling.

**It is never consulted on a live conversation, a delivery, or a purchase
reconciliation.** Running out of history budget pauses backfill — it can never
stop a fan being answered or a sale being recorded. That invariant has its own
test.

---

## 4. Configuration

| Variable | Default | Effect |
|---|---|---|
| `HISTORY_BACKFILL_ENABLED` | `false` | The unattended deep-history scheduler. Warm resume and the operator's Load-history button are **not** gated on it. |
| `HISTORY_EXTRACTION_ENABLED` | `false` | Historical compaction. Also requires `FAN_INTELLIGENCE_ENABLED`, so a deployment that declined fan intelligence cannot acquire it through a history import. |
| `HISTORY_WARM_MIN_LOCAL_MESSAGES` | `12` | Below this, a returning fan gets a warm resume. |
| `HISTORY_WARM_MAX_PAGES` | `3` | Newest pages a warm resume may fetch (max 10). |
| `HISTORY_DEEP_PAGES_PER_RUN` | `20` | Pages one deep run may fetch. |
| `APIFANSLY_HISTORY_CREDIT_BUDGET_24H` | unset | Rolling credit ceiling for optional backfill only. |
| `HISTORY_EXTRACTOR_PROVIDER` / `HISTORY_EXTRACTOR_MODEL` / `HISTORY_EXTRACTOR_MAX_TOKENS` | see profile | Re-point the compaction stage without a deploy. |

### Rollout order

1. Apply `db/fan_history_backfill_v1.sql`, then the
   `tenant_isolation_v1` + `browser_least_privilege_v1` pair (never one without
   the other — SEC-001).
2. Deploy. Everything stays off: warm resume works, the scheduler does not run.
3. Set `APIFANSLY_HISTORY_CREDIT_BUDGET_24H` to something you are willing to
   spend.
4. Turn on `HISTORY_BACKFILL_ENABLED`, watch `credits_by_category` and
   `/creator-history-usage`.
5. Turn on `HISTORY_EXTRACTION_ENABLED` once import volume looks right.

The code tolerates step 1 not having happened: `db/fan_history_queries.py`
detects the missing table, logs once, and reports history as unavailable. Live
conversation is unaffected. `scripts/production_preflight.py` reports it as a
warning.
