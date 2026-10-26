# E09 results — gold review by Dhruva (2026-10-11)

Sources are files in the private research workspace (`dataset/`), given as `file:line`. The verdicts are Dhruva's
(human). The post-mortem statistics are computed from those verdicts by a Claude agent. Item identifiers, frame
references and the gold labels themselves are not published.

## Gold pack
Source: `reviews/autolabel-v1/REVIEW.md:233-247`; `repo: src/gold_build.py` (docstring).

| item type | count | stratified by |
|---|---|---|
| mask | 300 | quality tier: clean 120, Astra-verified 90, rejected 90 |
| position | 296 | smoothed position class: 37 each for 8 classes |
| identity | 100 | 4 moments of one actor, across cohorts |
| event | 98 | event type: 7 clips each for 14 types, window +-1.5 s |
| strike | 60 | 45 Astra-confirmed strikes by type + 15 rejected candidates |

Items are cropped to the primary pair; Space toggles the raw image. Verdicts are appended to a log, and the page shows
running accuracy per stratum. Items are interleaved in blocks of 10 so a partial session still covers every type.

## First session: 103 verdicts, 47 notes
Source: `reviews/autolabel-v1/REVIEW.md:310-321`; session span from the first and last timestamps in
`reviews/gold-v1/verdicts.jsonl` (07:51 to 09:01).
Dhruva's notes override the keypresses; a Claude agent read them into effective verdicts and corrected labels.

| type | fully correct | notes |
|---|---|---|
| identity | 12/19 | no true identity switch; the misses are the same person with a bad mask (merges in an arm triangle or a choke, lost hooks) |
| event | 12/18 | outcome or completion over-claimed: a finished guillotine labelled `reset`, an arm-triangle escape labelled `reguard`, side control labelled `mount_achieve`, a back-take attempt labelled a success; no end-of-match event; +-1.5 s clips too short to see outcomes |
| strike | 13/20 | rejected candidates 5/6 right; jab, cross, uppercut and kicks right; wrong in ground or clinch context or with combinations |
| position | 11/20 (class 14/20) | 4 of 6 class errors are standing grappling (single leg, double leg, front headlock, takedown attempt) labelled `transition`, though those details exist under `standing`; others: mount vs half guard, half guard vs leg-drag pin; available details (kesa, body triangle, closed guard being opened) not picked |
| mask | 12/23 pairs; 30/46 actors good, 1 wrong | rejected tier 2/8 vs clean and verified 10/15, so the tiering works; failure modes: lost legs or lower body, merges in tight chokes, legs swapped between fighters in entanglements, small holes |

The mask items over-sample the rejected tier (90 of 300) and the position items over-sample rare classes, so these
shares are not corpus-wide accuracy.

## Pipeline bug found
Source: `reviews/autolabel-v1/REVIEW.md:323-326`.
4.0% of position-labelled frames (2,482 of 62,123, plus 608 where neither fighter is visible) have a primary actor
absent from the frame: the pair is chosen once per interval, so in gym or multi-mat clips the frame's position
describes another pair. Fix: a per-frame pair, or `pair_not_visible` in the export.

## Dhruva's suggestions
Source: `reviews/autolabel-v1/REVIEW.md:328-332`.
An anatomical check on legs (in a single leg or 50/50, the leg that is not the defender's belongs to the attacker);
positions are provisional until held for a few seconds; a continuous "progress" measure for passes, pins and
submissions; tag tutorial and demo clips.

## Mask post-mortem
Source: `reviews/gold-v1/mask_postmortem/REPORT.md:3-44`.
Sample: 134 judged actor-frames (94 explicit: 58 good, 34 partial, 2 wrong; 4 unmasked by the pair bug; the rest
weak "good" from identity items). Small sample: read it as direction.
1. Mask provenance is the strongest separator: SAM 3.1 text detections 92 good / 23 not good (20% bad); masks made
   from Astra boxes (used when the detector merged or missed a person) 2 / 13 (87% bad). Those fallback masks are
   fragmented speckle (median 28-60 connected components vs 2) and make up 8.9% of primary actor-frames (11,242 of
   125,794), concentrated in tangled ground positions.
2. Single-signal AUCs for a bad mask: SAM score 0.82, components 0.78, speckle 0.77, blur 0.73, hole fraction 0.71,
   pair-box IoU 0.67, skeleton signals about 0.60, darkness and contrast 0.48-0.51 (the dark-clothing hypothesis is not
   supported).
3. Best single signal: the label-frame mask against the 12.5 fps video mask at the same moment. Median IoU 0.96 for
   good masks, 0.78 for faulted ones; AUC 0.84. IoU < 0.9 catches 17/19 bad with 12/66 false flags, better than the
   `rejected` tier (23/36, 16/58 false flags).
4. Context: frames without contact 0/7 bad. By position: standing 11% bad, guard 47%, mount 64%, leg entanglement 75%.
   By cohort: trial 10%, breadth 36%, gi 62%. Shot type and crowding: no effect.
5. On 8 faulted frames the video mask was clearly better on 5, worse on 2, the same on 1.
6. Failure modes named in Dhruva's notes: lost lower body or legs 7, merged with partner 7, unmasked 4, holes 2, legs
   swapped 2, limb missing 1; about 16 of 23 are limb assignment at contact.

Dhruva's ideas ranked by support in this data: anchor and point tracking over time (strongest); skeleton or 3D-body
part split with anatomical checks; limb appearance fingerprints (weakest, a tie-breaker only).
