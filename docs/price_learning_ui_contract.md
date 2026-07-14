# Price learning UI contract

The future UI should display price learning as explainable guidance, never as an
estimated wealth score.

Required fields:

- mode: `NO_OFFER`, `DISCOVERY`, `RANGE`, or `EXACT`
- confidence: `NONE`, `LOW`, `MEDIUM`, or `HIGH`
- recommended floor / target / ceiling
- lifecycle stage used
- confirmed-purchase and positive-signal counts
- reason codes
- evidence summary
- last update

The UI must distinguish:

- confirmed purchases;
- selected but unpaid offers;
- current explicit limits;
- counteroffers;
- soft resistance from declined offers;
- temporary inability to spend.

It must never label the recommendation as the fan's wealth, income, or permanent
budget. Agency controls should expose creator-level min/max, cold-start target,
uplifts, range width, step size, and lookback.
