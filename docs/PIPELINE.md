# Pipeline

How the code in `src/` fits together. Each script states its environment and usage in its docstring.

## Data flow
```
discover.py / select_sources.py / gi_select.py -> manifests/ (candidates, catalogue.sqlite, intervals)
  -> acquire.py -> media/ (one <=720p master per source, audio kept)
  -> prepare.py / trial_evidence.py / autolabel_select.py / autolabel_matches.py -> evidence/ (2 fps packets)
  -> autolabel_*.py -> autolabel/<run>/<trial>/<interval>/
  -> autolabel_export.py -> exports/<run>/
reviews/: AI audits, gold review, A/B packs, benches
```
- Subclips are **metadata intervals on the source master**; no subclip video is ever written. Exclusion intervals
  (for example a gi montage inside a no-gi video) must be respected by derived examples.
- Storage rule: the media cache holds at most 70 GB and the disk keeps 30 GB free; the downloader pauses at either
  limit and never deletes media or lowers quality.

## Auto-labeller stages
State lives per interval under `autolabel/<run>/<trial>/<interval>/`. A stage is done when its output file exists, so
every stage can be resumed.

| # | stage | script | environment | output |
|---|---|---|---|---|
| 1 | pick a window | `autolabel_select.py` (Astra, overview sheet of 48 thumbnails) | `.venv` | 20 s interval, 2 fps evidence packet |
| 2 | detect | `autolabel_detect.py` (SAM 3.1, text prompt "person", threshold 0.3) | `.venv-sam31`, GPU | `detections.json` |
| 3 | set-of-marks | `autolabel_astra.py` (6 keyframes + 2x2 tile sheets, strict JSON) | `.venv` + Codex CLI | `astra.json`, receipt |
| 4 | fuse: instances | `autolabel_fuse.py instances` | `.venv` | `frame_instances.json` |
| 5 | fuse: resolve | `autolabel_fuse.py resolve` (Astra box + points -> SAM 3.1 tracker decoder; one owner per pixel; flags) | `.venv-sam31`, GPU | `labels.json` |
| 6 | fight | `autolabel_fight.py` (position, detail, top/dominant athlete, events) | `.venv` + Codex CLI | `fight.json` |
| 7 | smooth | `autolabel_smooth.py` (HMM/Viterbi; not across camera cuts) | `.venv` | `smooth.json` |
| 8 | verify | `autolabel_verify.py` (Astra second opinion on flagged frames) | `.venv` + Codex CLI | `verify.json` |
| 9 | densify | `autolabel_densify.py` + `autolabel_sam31_video.py` (2 -> 12.5 fps) | `.venv-sam31`, GPU | `dense/dense.json`, `masks.npz` |
| 10 | 2D pose | `autolabel_pose2d.py` (ViTPose+, mask-guided; `--label-frames` for 2 fps) | `.venv-pose`, GPU | `dense/pose2d.json` or `pose2d_labels.json` |
| 11 | 3D body | `autolabel_body3d.py` (SAM 3D Body, own box + mask) | `.venv-s3d`, GPU | `body3d.json`, `body3d.npz` |
| 12 | strikes | `autolabel_strikes.py` (MMA; speed peaks -> Astra strips) | `.venv` + Codex CLI | `strikes.json` |
| 13 | mask fix | `mask_fix.py` (dense mask replaces fallback masks when coherent) | `.venv-pose` | `labels.json` (backup kept) |
| 14 | export | `autolabel_export.py` | `.venv-sam31` (pycocotools) | `exports/<run>/` ([format](EXPORT.md)) |

Running it:
- `autolabel_orchestrate.py` keeps every interval moving: one GPU job at a time (priority detect > resolve > densify
  > pose > body), up to six Astra jobs, state read from files so it can be stopped and restarted. Stop it with
  `touch autolabel/v1/orchestrator.stop`. `autolabel_watchdog.sh` restarts it from cron and blocks system sleep while
  it runs.
- `autolabel_pipeline.py` runs the first stages by hand for selected intervals.
- Review sheets: `autolabel_render.py`; corpus metrics: `autolabel_review.py`.

## Astra calls
- GPT-6 Astra is called with the logged-in Codex CLI (`codex exec`, ephemeral, no user config), one call per interval
  for set-of-marks and fight labels, eight frames per call for verify, one strip per strike candidate.
- `check_usage()` (`autolabel_astra.py`) reads the plan's rate-limit windows through the local `codex app-server`
  (no model tokens) before every call and stops at 90% used. It never redeems a reset credit. Each receipt records
  usage before the call and token counts.
- Prompts are versioned (`docs/autolabel/*_PROMPT.vN.md`); add a new version instead of editing one in use. The label
  vocabulary is `docs/autolabel/taxonomy.v1.json`; the position ontology (L1A positions, L1B details, T transitions)
  is reused from the earlier Kakashi product.

## Environments
Each GPU model needs its own environment (pinned lists in `env/`).

| environment | used for |
|---|---|
| `.venv` (Python 3.14) | acquisition, catalogue, evidence, Astra calls, smoothing, export summaries, tests |
| `.venv-sam31` (Python 3.12, torch 2.10 + CUDA 12.8, Meta `sam3`) | detect, resolve, densify, mask-fix tracking, export |
| `.venv-pose` | ViTPose+ 2D pose, mask fix, A/B pack building |
| `.venv-s3d` | SAM 3D Body |

Machine limits during these experiments: 14 GB RAM, one 6 GB GPU, one GPU job at a time,
`PYTORCH_ALLOC_CONF=expandable_segments:True`. Peak VRAM: 5.6 GB for detection, about 4.8 GB for video tracking
(sources: workspace `docs/autolabel/README.md:47`; `src/autolabel_sam31_video.py` docstring).

## Known limits
- `sam3.1_multiplex.pt` has no weights for the image model's box-to-mask head; building the image model with
  interactivity runs that head untrained. Box prompts go through the tracker decoder instead.
- Video propagation through the tracker-only adapter used by `fuse resolve` collapses every object onto one region;
  identity across frames therefore comes from Astra. Dense masks use the official multiplex predictor instead.
- The resolve stage imports `build_tracker_only_sam31` from a separate, unpublished project
  (`KAKASHI_SAM31_ADAPTER`); it is not included here.
- The primary pair is chosen once per interval, which mixes pairs in multi-pair clips ([E07](../experiments/E07-scale-up-audit/),
  [E09](../experiments/E09-gold-review/)).
- Entangled grappling remains the hard case ([E11](../experiments/E11-mask-fix/)); the follow-up is the
  [limb-entanglement](https://github.com/dhruvinci/limb-entanglement) repository.

## Changes made for publication
- Absolute paths were made portable (`Path.home()` defaults with `KAKASHI_*` environment overrides).
- `select_sources.py` and `gi_select.py` read channel lists and organisation-specific title terms from a config
  (`src/config/selection-config.example.json`); the original lists named specific channels and events and are not
  published. Discovery queries: `src/config/discovery-queries.example.json` (generic examples only).
- `vlm_bench.py` and `vlm_bench_events.py` read their ground-truth cases from a private `cases.json` instead of
  embedding them.
- `evidence_integrity.py` takes the pilot source key from `KAKASHI_PILOT_SOURCE_KEY`.
- Added `densify_heldout_summary.py` and `export_summary.py` (aggregate counts only) for this write-up.
