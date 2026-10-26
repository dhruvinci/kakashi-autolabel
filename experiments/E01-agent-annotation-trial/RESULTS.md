# E01 results — agent annotation pilot and 100-source trial (2026-09-30 to 2026-10-01)

Sources are files in the private research workspace (`dataset/`), given as `file:line`. Every quality figure is an
AI reviewer's judgement (another GPT-6 Astra agent, with identities hidden), not expert or human ground truth.

## Pilot: effort calibration
Source: `docs/PILOT_REPORT.md:49-56`, `:61-66`, `:70-75`.

| Effort, same eight 12 s excerpts | supported claims | salient checklist complete | plausible boxes | landmark-bearing records |
|---|---:|---:|---:|---:|
| low | 90/90 | 50/54 | 43/46 | 0 |
| medium | 102/104 | 52/54 | 45/46 | 45 |
| high | 125/127 | 52/54 | 46/46 | 45 |

- A fresh low-effort repeat on four excerpts supported 47/47 claims but covered only 19/27 checklist items, failing
  the coverage gate. Missing: rear-control configuration, defensive rotation, the favourable outcome.
- Packet size, same minute at low effort: two 30 s packets 41/42 supported, one 60 s packet 39/39; both covered
  19/26 checklist items. No packet-size winner.
- The reviewer froze its checklist from the evidence alone before seeing candidates; writing style could still hint
  at which effort produced a candidate.

Decision (`docs/PILOT_REPORT.md:38-47`): medium effort, 30 s packet cap, a fresh worker every four packets. This is
a conservative policy, not a measured optimum.

## Pilot: delivered labels
Source: `docs/PILOT_REPORT.md:10-26`.
- 447 claims (162 visual, 108 temporal, 168 speech, 9 uncertain hypotheses) over 28 packets covering all 563.668 s.
- 155 geometry records and 343 visible landmarks; the partial COCO view has 149 boxes on 83 images.
- Production review: 314 of 317 production claims supported; two narrowed, one speech-boundary correction.

Runtime (`docs/PILOT_REPORT.md:98-100`): 17.97 worker minutes for 467.668 s of newly labelled video, about 2.31
worker minutes per video minute, excluding evidence preparation and review.

## 100-source trial
Source: `docs/TRIAL100_REPORT.md:3-4`, `:19-22`, `:42-50`, `:60-62`, `:95-100`, `:122-123` (medium effort for
annotation, high effort for the independent AI review).
- 100 sources, 200 intervals of 20 s: 4,000 s out of 12.84 h of full source video; the rest stays unlabelled.
- Raw: 3,213 claims (1,276 visual observations, 1,074 temporal inferences, 802 unverified spoken statements,
  61 uncertain hypotheses) and 1,286 geometry records.
- 8,512 supplied frames; 6,007 core frames referenced by claims; **647 core frames with sparse geometry**; partial
  person COCO view: 575 frames, 1,121 boxes.

AI audit of 27 sources (six context-gate sources, 19 stratified-random, two risk cases):

| unit | supported / complete | other |
|---|---:|---|
| 909 claims | 906 | 1 contradicted, 2 underspecified |
| 426 checklist items | 421 | 5 partial |
| 355 boxes | 319 | 7 underspecified, 28 unresolved, 1 contradicted |
| 918 landmarks | 908 | 10 flagged |

Usage (`docs/PLAN.md:218-220`): the shared weekly window was 26% used at the trial's start and reached 100% after
source 95; Dhruva then continued, and sources 96-100 completed. No reset credit was redeemed by an agent.

## Limits
Source: `docs/TRIAL100_REPORT.md:112-118`.
Sparse frames cannot resolve landed strikes, brief taps, hidden grips or exact hooks. Landmark anatomy and body
ownership need original-image checks. The local speech model is English-only; empty output is not silence.
