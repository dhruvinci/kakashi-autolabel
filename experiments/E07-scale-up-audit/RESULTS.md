# E07 results — scale-up and full-corpus audit (2026-10-05 to 2026-10-10)

Sources are files in the private research workspace (`dataset/`), given as `file:line`, or this repository
(`repo:`). The audit is the judgement of Claude reviewer agents, not human ground truth. Interval identifiers are
left out.

## Corpus
Source: `repo: docs/export_summary.json:5-28`.
1,526 intervals from 670 videos. By cohort: breadth MMA / no-gi 942 intervals (471 sources), the 100-source trial
200 (100), gi 198 (99), whole matches or rounds 186 chunks of 20 s (15 sources). The orchestrator runs one GPU
job and up to six Astra jobs at a time (`repo: src/autolabel_orchestrate.py:23`). By clip context: competition 635, training
sparring 449, instructional demo 413, training drilling 27, unknown 2.

## Issues found while scaling
Source: `reviews/autolabel-v1/REVIEW.md:114-131`.
- **Primary pair.** `primary_actors()` takes the two largest athletes or instructors over the whole interval. In 241
  of 953 intervals (25%) several pairs or a camera cut mean it can mix one athlete from each pair. The fight stage was
  fixed on 2026-10-06 to outline all athletes and instructors; mask flags and verification still use the
  interval-level pair.
- **Full-match stitching.** The primary pair keeps one identity across all chunks in 7 of 11 matches; three break
  once (probably at camera cuts); one multi-mat roll is unreliable.
- **Power-event hang.** GPU and Astra jobs hung for about 19 h; the orchestrator now reaps network jobs older than
  90 min and GPU jobs older than 4 h. (A 9 h suspend on another night led the watchdog to block system sleep while
  the pipeline runs: `repo: src/autolabel_watchdog.sh:10`.)

## Full-corpus visual audit
Source: `reviews/autolabel-v1/REVIEW.md:162-218`.
Scope: 36 intervals sampled across every cohort (4 trial, 14 breadth, 12 gi, 6 full-match chunks). Four Claude agents
reviewed 9 frames each against `labels.json`, `fight.json` and `smooth.json`: 324 frames, 833 actor-frames.

Interval verdicts: good 21, usable with filtering 13, poor 2 (a wide, low-resolution tournament shot where one mask
swallows both athletes; a P1/P2 swap at a reset between two people in similar black rash guards, with no flag).

| cohort | actor-frames | good | partial | wrong or missing |
|---|---|---|---|---|
| all | 833 | 85.1% | 9.6% | 5.3% |
| breadth | 278 | 84% | 13% | 4% |
| gi | 346 | 84% | 8% | 7% |
| full matches | 125 | 89% | 8% | 3% |
| trial | 84 | 87% | 7% | 6% |

- Identity: 4 errors in 2 intervals. Roles: solid; background competitors in tournament footage are tagged `athlete`.
- Position: 9 of 324 frames wrong (2.8%). Top/dominant actor: 11 errors, mostly dominant-actor flicker in standing
  exchanges.
- Unflagged errors: 58 frames had real errors and no flag (about 19 of 81 frames in the first batch).

Recurring failures:
1. Interlocked bodies: the bottom or behind athlete loses legs or head to nothing or to the partner (half guard, back,
   piggyback), mostly unflagged.
2. Speckled interleaving of the two masks after a merged detection is split by boxes.
3. Smoothing changed 4.7% of positions but kept the raw detail, giving pairs like guard/none (fixed below).
4. Events: striking exchanges often have no actor; guard pulls and level changes sometimes missing; a few events span
   a camera cut in instructionals.
5. Detail rules: a standing piggyback labelled back mount; side ride and kneeling positions sometimes called mount.

## Changes
Source: `reviews/autolabel-v1/REVIEW.md:219-231`.
- `src/autolabel_smooth.py` now writes position detail and top athlete consistent with the smoothed position
  (borrowed from the nearest frame within +-3 with that raw position, else `unknown`; illegal pairs become `unknown`).
  All 1,526 intervals re-smoothed.
- Keypoint mask signals (`src/mask_signals.py`: confident own joints outside the own mask, joints inside the partner's
  mask, largest-component share) tested on the 44 audit-confirmed bad frames vs frames from good intervals: about 30%
  precision at any threshold (33/44 recall with 75 false alarms), no better than the existing flags (24/44, 51 false
  alarms). Exported as uncalibrated per-instance attributes, not used for tiers.

## Re-export, 2026-10-11
Source: `reviews/autolabel-v1/REVIEW.md:290-304`.
Smoothing does not run across camera cuts (164 intervals have cuts); 60 events are marked `crosses_shot_cut`; tiers
before the mask fix: clean 48,056, rejected 12,192, Astra-verified 4,905, flagged-unverified 6 frames.

Operational note (`reviews/autolabel-v1/REVIEW.md:306-308`): every `codex exec` call left a ~33 MB staging folder
behind; 395 of them used 13 GB and pushed free disk below the 30 GB floor. Clear them after large Astra batches.
