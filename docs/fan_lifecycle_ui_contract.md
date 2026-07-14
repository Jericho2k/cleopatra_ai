# Fan lifecycle UI contract

This sprint deliberately does not redesign the dashboard. It creates a stable backend contract so the future UI can be built without reverse-engineering business logic.

`fan_lifecycle_states` exposes:

- `stage`: `PROSPECT`, `FIRST_PURCHASE_PROSPECT`, `FIRST_TIME_BUYER`, `REPEAT_BUYER`, or `VIP`
- confirmed purchase count and revenue
- total platform spend
- first and latest confirmed purchase timestamps
- temporary first-purchase-intent expiry
- operational flags such as sales pause, active paid session, and human review
- machine-readable reason codes
- `updated_at`

`fan_lifecycle_transitions` provides an immutable timeline for a future fan sidebar or audit panel.

Recommended future UI surfaces:

- lifecycle badge beside the fan name
- purchase count and total spend under the badge
- compact transition timeline in the fan intelligence panel
- warning chips for `sales_paused` and `needs_human_review`
- agency-filterable fan lists by lifecycle stage

The UI must display the lifecycle; it must never calculate or overwrite it.
