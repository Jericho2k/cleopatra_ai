# Commercial pricing, probing and session choreography

This is the contract the commercial layer follows after the September realism
sprint. It exists because a real simulator conversation offered "3 pics for $25.
want the link?" and then delivered a $10.63 PPV followed by a $14.37 one.

## 1. The hierarchy

```
CONTENT VALUE / APPROVED PRICE BOUNDS      models/content_pricing.py, models/vault_pricing.py
            |
            v
FAN-SPECIFIC PRICE POSITION / PROBE        models/price_learning.py
            |
            v
ACTUAL APPROVED OFFER                      services/media_packages.py
            |
            v
PER-STEP PPV PRICES                        models/vault_pricing.allocate_step_prices
```

Each layer may only narrow the one above it. A fan-specific recommendation says
*where inside* an approved range to price; it never moves content out of its
range, and a package target never repriced content in the first place.

## 2. Terms

| Term | Meaning | Where it lives |
| --- | --- | --- |
| **content floor** | Cheapest approved price for this exact content. | `price_bounds(row)[1]`, summed by `sequence_bounds` |
| **content base / anchor** | The approved price the creator or classifier settled on. | `price_bounds(row)[0]` (`base_price_cents`, else `suggested_price`) |
| **content ceiling** | Most this content may ever be sold for. | `price_bounds(row)[2]` |
| **fan probe price** | Where inside `[floor, ceiling]` we test this fan *now*. | `probe_price_cents(...)` |
| **explicit current budget ceiling** | "I only have $25 today." A hard cap for the current session. | affordability `current_limit_cents` / `current_available_cents`, and `hard_ceiling_cents` |
| **demonstrated willingness to pay** | The highest *confirmed purchase*. A permanent lower bound on what he will pay for comparable content. | `evidence_summary.demonstrated_willingness_cents` |
| **soft resistance** | A declined offer. Steps the probe down for now. Never stored as a ceiling, never erases demonstrated willingness. | `evidence_summary.latest_soft_resistance_cents` |

## 3. Where a content range comes from

`price_bounds(row)` resolves, most authoritative first:

1. An explicit approved band on the row (`min_price_cents` < `max_price_cents`).
2. `dynamic_pricing_enabled = false` — a deliberate fixed price.
3. The classifier category carried in the row's own metadata. Set generation
   writes `content_category` into a set's tags, so `nude_photo` on a legacy row
   still resolves to the agency's approved $15-$80. This bridge also covers rows
   whose `min`/`max` were backfilled to equal `base` by
   `adaptive_planning_v1.sql`, which read literally would pin a $15-$80 set to
   one price forever.
4. Otherwise the approved price *is* the approved price. Failing closed to a
   fixed price is safer than treating content as unbounded — which is what the
   old code did, and how a global $25 package target became the price of an
   intrinsically $15-$80 set.

The canonical category table is `models/content_pricing.VAULT_CATEGORIES`;
`main.py` re-exports it so the classifier and the pricing layer cannot drift.

## 4. Probing

`probe_price_cents(floor, ceiling, price_learning=..., policy=..., hard_ceiling_cents=...)`
is pure and deterministic:

* **No evidence** — probe at `cold_start_probe_bps` into the *content* range
  (default 2,500 bps). $15-$80 gives ~$31, snapped to $30.
* **Confirmed purchase at X** — the probe never sits below X, and steps up to
  `X × (1 + max_step_up_bps)`, bounded by the ceiling.
* **Repeated effortless purchases** — `effortless_purchase_streak` confirmed
  purchases with no resistance add `repeat_buyer_uplift_bps` on top.
* **Soft resistance at R** — the probe steps down below R, but never below
  demonstrated willingness or the content floor. Nothing is stored.
* **Explicit current ceiling C** — the probe is capped at C. If C is below the
  content floor, the function returns `None`: no offer, rather than a discount
  below approved value.
* **Selected offer** — `mode = EXACT` is authoritative and used verbatim.

The anchor is always taken from the *content* range before caps are applied, so
a fan who says "$25" is offered $25, not the floor.

## 5. Human-facing prices

`human_price_cents` snaps to the agency's grid — $5 by default
(`customer_price_step_cents`), then whole dollars, and only then keeps a raw
value that no clean price can express (an approved fixed $17 stays $17). An
agency that deliberately sets a 1-cent grid gets cent-level prices; nobody gets
them by accident.

## 6. Multi-step sessions

`allocate_step_prices(total, rows, step_cents=...)` returns per-step prices that
simultaneously:

* stay inside each step's own approved bounds,
* sit on a human price grid,
* sum to **exactly** the sold total.

When those cannot all hold it returns `None`. `package_from_sequence` runs the
allocation *before* an offer is presented, so an unsellable structure never
reaches the fan; `plan_session_for_fan` returns `no_valid_allocation` rather
than inventing a distribution. The old weighted division always produced *a*
number, which is how $25 became $10.63 + $14.37.

A package covering more than one set is presented to the writer as a total for
N parts, so "3 pics for $25" cannot describe a two-step session.

## 7. Photo -> video progression

Default order for a generic offer: photo tease -> stronger photo -> video ->
premium video. `build_offer_packages` builds openers from photo sets only and
lets the premium package end on a coherent clip (`choose_video_finale`, which
weights scene continuity above raw explicitness).

Overrides: an explicit request for video (`wants_video`), a vault with only
videos, and an already-selected package — all of which win immediately.

## 8. Session choreography

`session_progress(session)` gives the writer exact, writer-safe state: the step
just purchased, the next planned step, and whether the scene, outfit, asset type
or explicitness continues or escalates. The writer bridges from one to the next
instead of waiting to be asked for more. Nothing here authorizes a send —
purchase gating still decides that, and the cooldown turn is explicitly
`must_not_send_media`.

## 9. Platform semantics

Fansly PPV media is attached directly to a chat message. `services/ppv_language`
is the deterministic backstop for commercial turns: delivery-link phrasing is
rewritten ("want the link?" -> "want it?"), and ordinary conversational uses of
the word "link" are left alone.

## 10. Message shape

`services/message_shape` decides the bubble count outside the model, because
Full Auto sends option 1 verbatim and a model's chosen rhythm otherwise becomes
the creator's whole personality. A 12-turn cycle gives ~67% single, ~25% double
and ~8% triple; the position is keyed off the message being answered, since the
visible history is capped and a counter derived from it would freeze. The
policy only ever *merges* bubbles — it never splits a sentence to hit a target.

## 11. Configuration

No migration is required: `vault_sets` already carries the pricing columns, and
the category bridge is applied at read time. New dials, all optional:

| Environment variable | Default | Meaning |
| --- | --- | --- |
| `PRICE_LEARNING_COLD_START_PROBE_BPS` | `2500` | Position inside a content range for a fan with no evidence. |
| `PRICE_LEARNING_EFFORTLESS_PURCHASE_STREAK` | `2` | Confirmed purchases before repeat-buyer uplift applies. |
| `PRICE_LEARNING_CUSTOMER_PRICE_STEP_CENTS` | `500` | Customer-facing price grid. |
| `PRICE_LEARNING_POLICY_CACHE_SECONDS` | `60` | In-process TTL for scoped pricing settings. |

The same keys are overridable per agency and per creator through
`price_learning_policy_scopes.settings`.
