# Simulator delivery and reply-length incident — 19 September 2026

## Confirmed defects

The new log identifies `semantic_v1` and a persisted `test_` fan, with mirrored catalog enabled. The final turn plans an unlock, successfully generates text, then completes with `human_review` and zero messages. The reported review reason matches a preflight exception in `send_locked_ppv`.

1. `send_locked_ppv` rejected every `sim:` media identifier before reading the persisted recipient. Its local-test delivery adapter was unreachable for mirrored media. The check now follows recipient ownership and identity verification: test recipients use the local adapter, real recipients still cannot send mirrored IDs to Fansly.
2. The semantic executor saves the accepted session plan before sending. A failed preflight left an unsent plan; a later blanket active-session check blocked it even after review was cleared. Recovery now reuses only the exact accepted initial plan, with matching offer, set and price, no sent/purchased steps, and no pending payment. The delivery journal still blocks an outstanding delivery claim.
3. The shared generator rejected an otherwise valid reply over 25 words whenever `Persona.avg_message_length == 'short'`, the default. This was a rejection/retry condition on the whole response, not a per-message writing preference. That hard gate is removed; JSON, message-array and other content-validity checks remain. The attached `all_rejected` log does not include the discarded text, so that individual rejection cannot be attributed to the length gate with certainty.
4. The semantic writer prompt explicitly preferred 1–2 short bubbles and illustrated two messages. It also omitted the configured creator voice from its payload. The prompt now uses contextual length, a single-message schema illustration, relevant voice settings, and instructions against repeated paraphrases and stock reactions.
5. Unsolicited price omission covered locked captions but left offer text and bare amounts such as `its 30 if ur down` untreated. The same bounded style-rewrite mechanism now covers those cases. A repeated correct approved price remains a style issue, not grounds to freeze a valid transaction; omission is therefore not a universal guarantee. Incorrect amounts remain invalid.
6. The simulator showed a pause but did not surface the existing recovery action. Its companion dashboard PR adds a persistent pause panel and Resume test conversation button using the authorized `resolve-review` route. Resume itself does not replay or send anything.

## What the exports establish

Six paginated conversation exports contain 961 messages after deduplicating IDs within each file. They include direct conversation and promotional traffic; they are not a clean quality benchmark. Using contiguous same-creator messages no more than 120 seconds apart as a message burst, 336 of 400 bursts contain one message (84%). The remaining bursts contain two, three or four messages. This supports varying cadence, not forcing long responses or a fixed two-message pattern.

`augmented_pairs.json` explicitly declares `synthetic: true`; it contains 148 pairs and a separate `real_pair_count: 185` field. That field does not establish 185 observed training pairs in this file. It was not imported into prompts or training. No raw private chat exports are committed with this change.

The useful conversational distinction is responsive substance: answering the particular message, retaining references, varying length with the task, and adding something beyond paraphrasing. More words or more emojis alone are not evidence of higher quality.

## Validation and remaining evidence

Regression coverage includes mirrored media through the semantic executor, local delivery receipt and pending-payment persistence; real-recipient rejection; exact unsent-plan recovery; bare-price validation; creator-voice payload; and valid responses over 25 words under the short persona setting. Model responses and external services are stubbed. These tests establish software behavior, not real-model conversational quality.

After both PRs are deployed, the affected test fan can be resumed in the simulator. Send the next test message and verify an actual locked card, then simulate purchase and verify the existing card changes state. Review the real model's transcript for repetition, appropriate length and truthful content references. Do not describe the product as agency-ready solely because CI is green.
