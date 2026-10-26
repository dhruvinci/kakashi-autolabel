# E03 results — auto-labeller calibration (2026-10-03)

Sources are files in the private research workspace (`dataset/`), given as `file:line`. These are model-agreement
and visual-inspection results (the AI agent that ran the calibration looked at the contact sheets). None of
them is a human-verified accuracy measurement.

## Data
Source: `docs/autolabel/CALIBRATION.md:3-5`.
Six sources, one per sport/context cell (MMA and no-gi x competition, training, tutorial), both intervals each:
12 intervals, 508 frames, all development data.

## Prompt versions
Source: `docs/autolabel/CALIBRATION.md:9-13`. Prompts: `docs/autolabel/ASTRA_PROMPT.v1.md` to `.v4.md`.

| version | Astra sees | problem found | change |
|---|---|---|---|
| v1 | 6 full keyframes | identity on the other 35 frames had to be inferred; masks 0.5 s apart overlap at IoU 0.0-0.1 on broadcast footage, and colour histograms do not separate people | label every frame |
| v2 | 6 keyframes + 35 frames as 2x2 half-resolution tiles | every frame labelled, but on entangled grappling box-only prompts mixed ownership (one fighter's mask swallowed the other's) | add ownership points |
| v3 | v2 + two positive points per unmarked person | ownership fixed in most entangled frames; observers sitting on the mat labelled `athlete` | role definitions (v4) |

## Infrastructure findings
Source: `docs/autolabel/CALIBRATION.md:17-23`; `docs/autolabel/README.md:52-58`.
- `sam3.1_multiplex.pt` has no weights for the image model's box-to-mask head; building the image model with
  interactivity silently runs that head untrained. Box prompts are routed through the SAM 3.1 tracker decoder, which
  matches the prompted person at IoU 0.95-0.96 on the prompt frame.
- Video propagation through the tracker-only adapter is broken: from the first propagated frame every object collapses
  onto one region (reproduced on a static three-person scene). Identity across frames therefore comes from Astra, not
  from mask tracking. (The official multiplex video predictor was later made to fit the GPU: [E06](../E06-densify-and-frame-rate/).)

## Results (prompt v3)
Source: `docs/autolabel/CALIBRATION.md:27-48`.
- 508/508 frames labelled; 2,600 person instances, 1,595 of them athletes, referees, instructors or coaches; every
  visible person gets a mask, spectators included. The earlier agent annotation had geometry on 43 of these frames.
- Agreement with the 89 earlier agent boxes: 10 were objects (a grappling dummy, striking pads) and correctly not
  persons here; 77 of the remaining 79 person boxes are matched by an instance at IoU >= 0.5; median best IoU 0.88.
  The two misses are a standing clinch (IoU 0.41 and 0.46).
- Review flags on 118/508 frames (23%), concentrated in the entangled no-gi intervals (37/43 and 32/41 frames in the
  two worst).
- Visual check: clean on standing exchanges, referees, instructors, sparring scrambles, half guard and a leg
  entanglement; weak on fully entangled top-down frames, where one fighter's mask can be partial or mixed. Several
  of those are flagged (`own_point_outside_mask`, `astra_box_vs_mask_iou`), not all.
- Cost: about 33-36k input and 4-15k output tokens per interval; 190-610 s per call with three in parallel. All
  calibration calls moved the weekly plan window from 0% to 1%.

## Limits
Source: `docs/autolabel/CALIBRATION.md:52-54`.
Parity with Roboflow still unmeasured; adjacent frames not yet smoothed over time.
