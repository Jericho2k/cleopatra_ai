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
THE ONE NEXT APPROVED OFFER                services/media_packages.build_next_offer
            |
            v
ITS SINGLE PPV PRICE                       models/vault_pricing.allocate_step_prices
```

Each layer may only narrow the one above it. A fan-specific recommendation says
*where inside* an approved range to price; it never moves content out of its
range, and the content budget never repriced content in the first place.

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

## 6. One unlock, one price

A sold unlock is exactly one approved set. `allocate_step_prices(total, rows,
step_cents=...)` still runs — with one row — so the price is verified to sit
inside that set's own approved bounds and on the human price grid before the
offer is ever presented. When it cannot, `build_next_offer` skips that rung and
`plan_session_for_fan` returns `no_valid_allocation` rather than inventing a
distribution.

This replaces the multi-part prepaid session. A price is never a total split
across deliveries, so the failure it used to produce — $25 becoming $10.63 +
$14.37 — has no shape left to occur in.

## 7. Photo -> video progression

`plan_progression` orders the INTERNAL ladder: photo tease -> stronger photo ->
a coherent clip as the payoff (`choose_video_finale`, which weights scene
continuity above raw explicitness). Only its first rung becomes an offer;
the rest is choreography and is never quoted, counted or promised to the fan.

Overrides: an explicit request for video (`wants_video`) and a vault with only
videos both win immediately.

## 8. Incremental progression

The fan sees the next unlock and its price. He is never told a session total,
how many further pieces exist, or that a sequence exists at all.

After a confirmed purchase the conversation returns to being a conversation.
That used to be a counter — `post_purchase_cooldown_messages` — and the counter
could not survive one-unlock sessions: a single-step plan reaches `completed` ON
the purchase, and the completion branch of `mark_step_purchased` cleared the
counter on the same line that would have set it, so the window never existed in
production. It is retired (`db/retire_post_purchase_cooldown_v1.sql`).

What replaces it is the **Experience Director**
(`services/experience_director.py`, `docs/experience_director_contract.md`): a
persistent SCENE that outlives the commercial session. A confirmed unlock moves
the scene to `AWAIT_REACTION`; the next offer becomes eligible only when the
scene reaches `BRIDGE`, which happens when the dialogue actually produces one —
immediately if he asks for more, and never on a flat reaction however long he
keeps talking. Policy sees this as the single `ctx.experience_allows_new_offer`
flag, and it can only NARROW: an offer already on the table, an acceptance and a
delivery are unaffected.

Once eligible, the NEXT offer is built from approved unsent inventory,
escalating from the piece he just unlocked, at a price probed from its own range
and from his confirmed purchase evidence (`purchase_probe_bonus_bps`).

`session_progress(session)` still gives the writer exact, writer-safe state
about what he just unlocked. Nothing here authorizes a send — purchase gating
decides that.

## 8b. What may be SOLD, and what may be SAID

Two boundaries that used to be one.

**Paid-sellable content.** `models/content_pricing.is_paid_sellable` is the one
predicate every commercial path asks. Teaser inventory — the `teaser_clothed`
and `teaser_bundle` categories the agency prices $0-$0, plus a bare
`tease`/`free`/`not_for_sale` tag on hand-curated rows — is never eligible for
an automatic offer, for pricing, or for paid delivery, whatever price columns it
carries. It is filtered in `services/media_packages.usable_sets`, the single
chokepoint both offer construction and session planning read through. It is NOT
deleted: it stays in the vault for use as a free reward, and the Sets UI marks
it NOT SELLABLE and offers no pricing controls for it.

**Sexual text.** `CommercialDecision.may_be_explicit` answers "may this turn
talk about the media it is selling". It used to be read as the text register
too, which meant an ordinary `CONTINUE_NORMAL_CHAT` turn instructed the writer
to "Keep this response non-explicit" — i.e. whether the creator could speak
sexually depended on whether the engine happened to be selling. The register is
now `services/text_intimacy.py`, decided from fan intent, conversation
intensity, the creator's configured sexting mode and safety state. The agency's
controls are preserved exactly, including that free explicit text SPENDS the
configured allowance (`consumes_free_allowance`) — without that, decoupling
would have created an unlimited free sexting service on chat turns.

## 8a. Delivery is deterministic

`services/ppv_turn.plan_ppv_step_delivery` decides the media, the price and the
asset type from the persisted plan BEFORE the writer runs. The writer is told
only that something is attached and writes the message it arrives with; a
`[PPV:...]` tag in free text is stripped rather than obeyed. Text and attachment
go out as one message, so copy can never claim a delivery that failed.

## 9. Platform semantics

Fansly PPV media is attached directly to a chat message. `services/ppv_language`
is the deterministic backstop for commercial turns: delivery-link phrasing is
rewritten ("want the link?" -> "want it?"), and ordinary conversational uses of
the word "link" are left alone.

## 10. Message shape

`services/message_shape` decides the bubble count outside the model for
`writer_v1` and `writer_v2`, because those versions send option 1 verbatim and a
model's chosen rhythm otherwise becomes the creator's whole personality. A
12-turn cycle gives ~67% single, ~25% double and ~8% triple; the position is
keyed off the message being answered, since the visible history is capped and a
counter derived from it would freeze. The policy only ever *merges* bubbles — it
never splits a sentence to hit a target.

`writer_v3` opts out: its Full Auto contract is one reply returned as
`{"messages": [...]}`, whose entries are the bubbles that reply is actually sent
in. A commercial `max_messages` cap still merges, and a turn that attaches paid
media is always merged to one message.

## 11. Configuration

No migration is required: `vault_sets` already carries the pricing columns, and
the category bridge is applied at read time. New dials, all optional:

| Environment variable | Default | Meaning |
| --- | --- | --- |
| `PRICE_LEARNING_COLD_START_PROBE_BPS` | `2500` | Position inside a content range for a fan with no evidence. |
| `PRICE_LEARNING_EFFORTLESS_PURCHASE_STREAK` | `2` | Confirmed purchases before repeat-buyer uplift applies. |
| `PRICE_LEARNING_CUSTOMER_PRICE_STEP_CENTS` | `500` | Customer-facing price grid. |
| `PRICE_LEARNING_POLICY_CACHE_SECONDS` | `60` | In-process TTL for scoped pricing settings. |
| `WRITER_PRIMARY_RETRY_ATTEMPTS` | `4` | Attempts against the profile's primary writer before any fallback (`cleo_v3` only). |
| `WRITER_PRIMARY_RETRY_WAIT_SECONDS` | `5,30,60` | Waits before primary attempts 2, 3 and 4. |

`db/incremental_offer_v1.sql` drops the two-package policy dials
(`offer_two_packages`, `quick_package_target_cents`, `full_package_target_cents`,
`session_min_steps`, `session_max_steps`) and the ordered offer snapshot
(`fan_commercial_states.offered_packages` and the `selected_package_*` columns),
backfilling the single `pending_offer` and `accepted_offer_*` columns from them
first.

The same keys are overridable per agency and per creator through
`price_learning_policy_scopes.settings`.
