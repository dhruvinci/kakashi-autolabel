# E00 results — acquisition and catalogue (2026-09-30 to 2026-10-05)

Sources are files in the private research workspace (`dataset/`), given as `file:line`. They are not published.
Source names, channels and video IDs are left out on purpose.

## Discovery and screening
Source: `docs/EXECUTION.md:40-44`, `:66-68`, `:245`.
- First metadata pass: 2,196 unique candidates; the screen selected 814 provisional sources (MMA 228 competition /
  50 training / 149 tutorial; no-gi 123 / 145 / 119). These are title and channel hypotheses, not visually checked.
- Targeted discovery added 377 candidates (2,573 total) and 109 selections (923). A supplement weighted toward
  training and tutorials brought the MMA / no-gi catalogue to 1,000 entries.
- Gi was added on 2026-10-04 with about 20 GB of extra downloads (`docs/autolabel/PLAN.md:61`, `:76`).

## Catalogue state, 2026-10-26
Source: `manifests/catalog.sqlite` (table `sources`, grouped by sport and status) and
`manifests/discovered-candidates.jsonl` (3,518 lines), queried read-only on 2026-10-26.

| sport | complete | ready | retry | paused (storage) | all | hours (complete) | GB (complete) |
|---|---|---|---|---|---|---|---|
| MMA | 365 | 149 | 10 | 0 | 524 | 58.8 | 34.07 |
| no-gi | 400 | 58 | 17 | 1 | 476 | 61.4 | 33.45 |
| gi | 143 | 84 | 5 | 1 | 233 | 25.6 | 15.03 |
| all | **908** | 291 | 32 | 2 | **1,233** | **145.7** | **82.55** |

Complete masters by context: MMA 134 competition / 84 training / 147 tutorial; no-gi 117 / 133 / 150; gi 74 / 69 / 0.
Intervals in the catalogue (`clips` table): 2,152, of which 2,121 are publisher chapters, 17 visual candidate
windows, 13 calibration selections and 1 scope exclusion.

At the pause before gi (`docs/PLAN.md:225-228`): 765 verified masters, 120.15 hours, 67.52 GB of masters (69.08 GB of
cache); 235 catalogue entries without a master.

Rule used throughout (`README.md:39-40`): the media cache may hold at most 70 GB, the filesystem must keep 30 GB free,
pause at either limit, never delete media or reduce quality to make room.

## Scope audits by sampled frames
Source: `docs/EXECUTION.md:69-80`, `:137-145`, `:234`.
- Six sources inspected with 36 sampled frames plus extras: one source moved from training to tutorial (ground work on
  a bag, not live sparring).
- A no-gi rolling video opens with a gi montage: kept, with a recorded [0, 30.03] s exclusion that derived examples
  must respect. No subclip video is ever written.
- Five more sources: a static-artwork music mix excluded from usable examples (master kept); seven duplicate or scope
  flags recorded for later grouping ("vs" in a title can compare techniques; an MMA athlete can appear in another
  sport).
- A later audit moved two more sources from competition to tutorial or training.
These are sparse triage checks, not full-video review.

## Chapter reconciliation
Source: `docs/EXECUTION.md:81-85`.
Seven early publisher chapter endpoints ran past the verified master duration (publishers round times). A tested
reconciliation keeps the original intervals and archives wholly-outside chapters as events.

## Rights
Every master records `rights_status: recorded_license_only; redistribution_not_cleared` (`repo: src/acquire.py:33`).
Public accessibility is not a redistribution licence, so no media or frames are published.
