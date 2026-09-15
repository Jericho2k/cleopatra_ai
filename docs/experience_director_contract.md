# Experience Director — contract

The Experience Director is a persistent, lightweight record of the current
conversational SCENE. It sits beside the commercial layer, not above or below
it, and the division is absolute:

```text
Experience Director   ->  WHEN, conversationally, something may happen
Commercial Policy     ->  WHETHER it may happen at all, and at what price
```

Nothing in the Experience Director can authorize a send, set or move a price,
widen approved content, or lift a pause.

## Why it exists

The commercial model was deliberately reduced to one unlock at a time
(`services/session_planner.plan_session_for_fan` builds a one-step session on
purpose). That fixed checkout — a fan never prepays for a bundle — but it also
meant the only thing with memory across a purchase was the commercial session,
and a one-step session is finished the moment it is paid. The conversation
therefore restarted at offer discovery after every sale.

The scene is the missing half. It survives the session it came from.

## Beats

```text
SETUP → BUILD → OFFER → AWAIT_REACTION → PLAY → BRIDGE → (OFFER | CLOSE)
```

- `SETUP` — reading him; no premise yet.
- `BUILD` — a premise exists and is being built on.
- `OFFER` — one unlock is on the table, unresolved.
- `AWAIT_REACTION` — he unlocked something and has not reacted. **The next move
  is his.** Nothing new is offered from here.
- `PLAY` — he reacted and the conversation is IN the thing he unlocked.
- `BRIDGE` — the scene has produced a natural next direction.
- `CLOSE` — paused, declined, handed off, or simply over.

A confirmed purchase always goes to `AWAIT_REACTION`, never back to offer
discovery. It is recorded at the moment the unlock becomes true
(`record_unlock`, called from the purchase-confirmation path) rather than on his
next message, because the one-step session is cleared immediately afterwards.

## Stored fields

`fan_experience_scenes`, one row per fan (`db/experience_director_v1.sql`):

- `beat`, `previous_beat`
- `scene_key` — stable identity of the active scene/premise
- `premise` — what the scene is about, in approved words
- `last_unlocked_set_id`, `last_unlocked_description`
- `last_fan_reaction` — `NONE | POSITIVE | NEUTRAL | NEGATIVE | WANTS_MORE`
- `reaction_processed` — whether a reply has engaged with it yet
- `intimacy_level`, `tension_level` — 0-5
- `open_hook` — unresolved conversational hook
- `desired_direction` — preferred/desired direction
- `another_unlock_ready` — whether another paid unlock is conversationally ready
- `beats_in_scene`, `turns_since_unlock`, `unlocks_in_scene`
- `transition_reason`

## The gate

`scene_allows_new_offer(scene)` is the only thing the Experience Director tells
commercial policy, and it arrives as one boolean
(`CommercialContext.experience_allows_new_offer`). It is **a veto that can only
narrow**: it gates the DISCOVERY of a new offer and is not consulted for an
offer already on the table, an acceptance, a delivery, or any pause.

It is deliberately **not** a message count. "Wait exactly N messages" is what
was just removed.

- A fan who says "send more / what else do you have" reaches `BRIDGE` on the
  very next turn. He can accelerate his own progression.
- A hot scene that keeps climbing — a positive reaction plus rising intensity,
  or a direction he has named — produces a bridge.
- A flat "ok cool" produces no bridge, however many turns pass.
- A negative reaction never produces one. That gets dealt with instead.

## What the writer is told

`SceneState.writer_context()` is the only projection that reaches the prompt.
It carries the beat, the premise, what he just unlocked, his reaction, whether
that reaction is still owed an answer, the intensity dials, the open hook and
the direction.

It carries **no price, no set id, no counters and no readiness flag** — the fan
is never told future prices, steps, session totals, or that a sequence exists.
`tests/test_experience_director.py` asserts that projection is free of them.

## Media as beats

Set metadata is generated ONCE during catalog classification
(`services/scene_metadata.derive_set_experience`, written by
`db.queries.propose_sets` / `propose_video_ppvs`) rather than asked of the live
writer, which would otherwise describe media it has never seen:

- `scene_key`, `scene_premise` — which scene, and what it is
- `intensity_level` — where it sits on the escalation ladder (0-5)
- `reveals` — what unlocking it actually shows or advances
- `setup_line` — how it can honestly be led into
- `continuation` — where it can honestly lead next (a direction, never a named
  next product)
- `paid_sellable` — explicit authority over automatic selling

Content selection then weighs three things, not two: scene continuity,
explicitness, and `advances_the_interaction` — does this candidate actually move
the interaction he is having right now, or does it repeat the beat he just
unlocked?

## Structural lessons taken from the chatter material

Incorporated: media as the payoff to an ongoing premise; fan participation
between PPVs; pulling back and interacting after a purchase before selling
again; conversational bridges between related media; suspense and pacing over
dumping content; adapting the next beat to the fan's reaction; creator
initiative and improvisation.

Deliberately NOT implemented, and not implementable through this module: fake
relationships, promises to meet, fake shared futures, dependency manipulation,
"enslavement", implanted commands, or lying about real-life facts. The writer
contract already forbids a promised real-life meeting and a fake romantic
future, and nothing here supplies a claim about the creator's life.

## Operability

- Flag: `EXPERIENCE_DIRECTOR_ENABLED`, default **on**. It replaces behaviour
  that no longer works rather than adding an experiment, so shipping it off
  would be shipping nothing. It can still be turned off in an incident.
- Failure is never fatal: an unreadable or unwritable scene costs continuity,
  not a reply. `services/suggestions._scene_and_register` degrades to no SCENE
  and no TEXT INTIMACY block, which leaves the writer on its default voice —
  the safe direction, since the absent block is the one that GRANTS the
  explicit register.
- The Simulator's state panel shows the live scene, including
  `another_unlock_ready` and the transition reason, so an operator can see why
  nothing is being offered.
