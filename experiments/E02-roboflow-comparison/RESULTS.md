# E02 results — comparison with Roboflow's Auto Label (2026-10-01)

Sources are files in the private research workspace (`dataset/`), given as `file:line`. The reviews here are AI
judgements (session agents), not human ground truth. Source identifiers and frame references are left out.

## Scope
Source: `reviews/roboflow-comparison-v1/assessment.json:3-4`, `aggregate-facts.json:3`.
- Comparison target: Roboflow's documented Auto Label workflow, not every dataset hosted on Roboflow and not human
  annotation in general.
- No Roboflow predictions were obtained on these frames. The operating policy was near-zero Roboflow credit use and no
  hosted inference (`docs/PLAN.md:96-105`).
- The analysis was read-only: no labels changed, nothing uploaded.

## What the agent annotation has
Source: `reviews/roboflow-comparison-v1/aggregate-facts.json:4-13`, `assessment.json:29`, `:34-44`.

| measure | value |
|---|---|
| raw claims | 3,213 (1,276 visual, 1,074 temporal, 802 spoken, 61 uncertain) |
| raw geometry records | 1,286 |
| distinct free-text claim kinds | 1,872 |
| distinct keypoint names | 572 |
| core frames with geometry | 647 of 7,926 (a coverage measure, not recall) |
| partial person COCO | 575 images, 1,121 boxes |
| earlier AI audit (27 sources) | 906/909 claims and 319/355 boxes supported |

## Material findings (AI spot checks of original stills)
Source: `reviews/roboflow-comparison-v1/assessment.json:11-31`; `fresh-spotcheck.json:66-80`.
1. A visible head lies outside a retained person box, with a written rationale that the head was hidden: an
   incorrect visible extent in the COCO export.
2. A hip-region landmark falls on the shirted back above the waist.
3. Selected-actor boxes leave out clearly visible referees, observers and background athletes: incomplete by design,
   so naive detector training would learn people as background.
4. Free text is not a fixed taxonomy; a normalised view is needed before fixed-class training.

The fresh spot check covered 12 stills from six sources (one per sport and context cell, fixed seed). It reports no
IoU, precision, recall or error rate.

## Roboflow facts used
Source: `reviews/roboflow-comparison-v1/assessment.json:46-57` (pages accessed 2026-10-01).
- Roboflow Auto Label offers Astra boxes, Astra masks and Astra + SAM 3 polygons. Using the same VLM does not make the
  results equivalent.
- Roboflow's published Astra object-detection results are 82.1 mAP@50 at low effort and 83.6 at high effort on their
  benchmark. They use ground-truth box matching and cannot be compared with claim-support fractions.

## Verdict as recorded
Source: `reviews/roboflow-comparison-v1/assessment.json:3`, `:66`.
"Richer combat-semantic capture than a class-and-box-only output; equal or better annotation accuracy than Roboflow
is not demonstrated." Suitable for filtered VLM training experiments; not for feeding partial COCO into an ordinary
exhaustive detector trainer; no parity or superiority claim without a matched evaluation.

## What it changed
Source: `docs/PLAN.md:61-94`, `docs/autolabel/README.md:3-8`.
Dhruva set the 1 October direction: the dataset must support the core Roboflow tasks (detection, classification,
instance segmentation) and then add combat understanding. The auto-labeller reverses Roboflow's split: SAM 3.1 runs
locally and finds every person in every frame; Astra sees the detections as numbered marks and resolves identity,
role, posture, merges and misses.

## Still open
A matched benchmark: about 200 frames and short clips from 20 source groups, the same images and label rules for
both workflows, independent human references (`assessment.json:58-65`).
