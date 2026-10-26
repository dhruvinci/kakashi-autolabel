# E05 results — Astra second opinion and Claude adjudication (2026-10-04 evening)

Sources are files in the private research workspace (`dataset/`), given as `file:line`. Both the second opinion
(Astra) and the adjudication (Claude agents) are AI judgements, not human ground truth.

## Astra second opinion on flagged frames
Source: `reviews/autolabel-v1/REVIEW.md:84-95`; method `src/autolabel_verify.py` (docstring).
All 2,018 trial frames carrying review or risk flags were checked, eight zoomed crops per call (about 16k image
tokens per call).

| result | count |
|---|---|
| both masks correct | 553 frames (27%) |
| missing parts | 1,458 actor-judgements |
| includes the other person | 977 actor-judgements |
| wrong person | 121 actor-judgements |
| merged | 65 actor-judgements |
| position disputed | 250 frames (12%) |

The parts most often missing were legs, head and feet.

## Blind adjudication
Source: `reviews/autolabel-v1/REVIEW.md:97-106`.
Three Claude reviewer agents judged 60 sampled disputes, each shown as A/B claims with the source hidden.

| dispute type | Astra right | pipeline right | both wrong | unsure |
|---|---|---|---|---|
| masks (Astra says fine, pipeline flagged) | 20/30 | 10/30 | - | - |
| position | 8/30 | 6/30 | 6/30 | 10/30 |

- Masks: most disputed pipeline flags (area drops from occlusion, people out of frame, look-alike outfits) were false
  alarms. The real errors were wrong-person tracks and one person split across two masks.
- Positions: mostly genuinely ambiguous kneeling scrambles, takedown entries and inverted tangles. All three reviewers
  independently asked for a scramble/transition class.

## Rules adopted in the export
Source: `reviews/autolabel-v1/REVIEW.md:108-112`; `repo: src/autolabel_export.py:46-63`.
- `quality_tier`: `clean` (no review or risk flags), `astra_verified` (flagged, Astra says both masks are correct),
  `rejected` (Astra confirms a mask problem), `flagged_unverified` (flagged, no second opinion).
- A disputed position becomes `transition_or_uncertain` and is excluded from position training.
- Trial tiers: clean 6,494, Astra-verified 553, rejected 1,465 frames.

The tier rule rests on 30 AI-adjudicated mask disputes; Dhruva's gold review later checked it on human verdicts
([E09](../E09-gold-review/): rejected tier 2/8 mask pairs good vs 10/15 for clean and verified).
