# E04 results — first full run on the 200 trial intervals (2026-10-04)

Sources are files in the private research workspace (`dataset/`), given as `file:line`. **No human ground truth:**
every figure below is agreement between models or the visual judgement of Claude reviewer agents.

## Scope and evidence
Source: `reviews/autolabel-v1/REVIEW.md:3-12`.
- All 200 trial intervals: 8,512 frames, 49,794 person instances, 23,406 of them athletes, referees, instructors or
  coaches.
- Corpus metrics from `src/autolabel_review.py`; visual audit by four Claude reviewer agents on 42 intervals
  (460 frames, 1,192 actor-frames): 30 at random (5 per sport/context cell), the 9 worst-scoring, 3 extra.

## Verdict by task
Source: `reviews/autolabel-v1/REVIEW.md:14-25`.

| task | quality (AI audit) | usable for training? |
|---|---|---|
| person detection (every visible person) | matches 98.3% of the earlier agent person boxes at IoU >= 0.5 (median 0.92); 1.2% of confident large SAM detections unlabelled; reviewers found missing people in 0.3% of actor-frames | yes |
| instance masks, standing or separated | near-perfect (broadcast MMA, referees, instructors, drills) | yes |
| instance masks, ground grappling | random sample: 90% good, 8% partial, 1.8% wrong actor-frames; failures cluster in tangles (the bottom player's limbs or head go to the top player, or the masks speckle into each other) | yes, after filtering |
| identity across frames | no swaps in 38 of 42 intervals; rotating IDs in one pad drill with four people in similar dark clothing, now caught as `identity_unstable` | yes, minus the 2 `identity_unstable` intervals |
| role | 0 clear errors after prompt v4; edge cases: pad holders, athletes resting at the wall | yes |
| shot type, clip context | clip context agrees with source metadata in 95% of intervals; 99% of frames are live shots | yes (too few replays to learn that class) |
| position (pair configuration) | 1.5% of random-sample frames clearly wrong (top-in-guard called mount, turtle called half guard); single-frame blips 0.7%; class mix: standing 37%, guard family 22%, then side control, back, mount, leg entanglement | yes, with caution |
| posture | 5.8% of random-sample frames wrong, mostly one interval; face-down kneeling sometimes labelled side-lying or prone | weak label |

## Subsets
Source: `reviews/autolabel-v1/REVIEW.md:27-42`.
1. Clean intervals: drop the 41 intervals with more than 40% of frames flagged or `identity_unstable`; 159 remain.
2. High precision: also drop frames with review flags: 6,247 frames, 33,778 instances (16,213 athletes, referees,
   instructors, coaches). For detection, roles and classification.
3. Strict masks: also drop frames with risk flags (merged detection split by Astra boxes): 5,906 frames. For
   instance-segmentation training on grappling.

Flags rank quality against the audit: 14% of frames carry a review or risk flag in intervals rated good, 36% in
usable-with-filtering, 67% in poor. They are not complete: before the refinements below, reviewers counted 61 errors
without a flag, mostly posture and position errors (no flag checks those) and masks that absorbed the partner.

## Changes made during the review
Source: `reviews/autolabel-v1/REVIEW.md:44-56`.
- Primary pair marked per interval; mask checks run only on that pair, so background people no longer flood the
  flags. Frame flag rate 21.7% -> 12.6%.
- New checks: speckled mask, area drop (same shot only), oversized mask that may include the partner, mask mostly
  lost to overlap, appearance matching another actor (feeds `identity_unstable`).
- Flags split into three tiers: review flags (likely error), risk flags (merged detection split by boxes; about
  two-thirds had ownership errors in the audit) and info flags.
- The three crowded intervals that timed out at 15 minutes completed with a 30-minute timeout.

## Cost
Source: `reviews/autolabel-v1/REVIEW.md:58-65`.

| | |
|---|---|
| Astra time | 22.3 call-hours, 3-6 calls in parallel, about 9 hours wall clock |
| tokens | 6.44M input, 2.08M output (about 32k / 10k per interval) |
| plan usage | 14% of the weekly window, read 2026-10-04 11:05 (0% before calibration; covers calibration, the full run and retries); reset credit not used |
| local GPU | detection about 25 s per interval, mask resolution about 6 s per interval |

## Known limits at this point
Source: `reviews/autolabel-v1/REVIEW.md:67-80`.
Ground-tangle masks are the main weakness. Posture and position are single-pass Astra judgements. No human ground
truth. SAM 3.1 video propagation unused. All trial sources are development data; no source-group-disjoint test split.
