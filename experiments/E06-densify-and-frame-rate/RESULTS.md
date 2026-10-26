# E06 results — densify, 2D pose, 3D body and the value of 12.5 fps (2026-10-04 to 2026-10-09)

Sources are files in the private research workspace (`dataset/`), given as `file:line`, or files in this repository
(`repo:`). Every number is model output or agreement between model outputs; none is checked against human labels.

## SAM 3.1 video tracking on a 6 GB GPU
Source: `docs/autolabel/PLAN.md:34-38`, `:42-47`; `repo: src/autolabel_sam31_video.py` (docstring).
- The official multiplex video predictor fits with the detector's Linear/Conv weights in bf16 and everything else in
  fp32 (all-bf16 kills the text-prompt detections; all-fp32 does not fit). Peak about 4.8 GB at 1280x720.
- Three upstream issues fixed: an `init_state` keyword mismatch, `_build_sam2_output` dropping point-prompted objects,
  and 16-frame batches that run out of memory. The detector is skipped during point tracking.
- Densify prompts at clean Astra anchors (interior points of each person's own mask, other people's points as
  negatives), tracks in 6 s chunks, and holds out every third clean anchor. On the first standing broadcast interval:
  held-out IoU 0.94-0.99, median 0.981.

## Held-out anchors across the corpus (new summary, 2026-10-26)
Source: `repo: experiments/E06-densify-and-frame-rate/heldout_summary.json:4-29`, computed by
`repo: src/densify_heldout_summary.py` from every `dense/dense.json` in the workspace.

| | value |
|---|---|
| densified intervals | 1,160 (breadth 631, trial 200, matches 186, gi 143) |
| dense frames | 280,235 |
| held-out actor-anchors | 30,750 |
| IoU median / 25th / 10th percentile | 0.953 / 0.876 / 0.712 |
| share >= 0.9 / < 0.7 | 70.3% / 9.4% |
| median by cohort | breadth 0.952, gi 0.951, matches 0.955, trial 0.955 |

Both masks are model output (the held-out label mask is Astra + SAM 3.1), so this measures agreement between the
video track and the per-frame labeller, not accuracy.

## 2D pose and 3D body
Source: `repo: src/autolabel_pose2d.py`, `repo: src/autolabel_body3d.py` (docstrings).
- ViTPose+ base (COCO-17) per person, from the person's own box with other tracked people's pixels greyed out. Each
  joint is tagged `own`, `partner` (on another person's mask: unreliable) or `off_mask`. Dense frames on densified
  intervals; label frames elsewhere.
- SAM 3D Body (ViT-H) per primary person on every second label frame (about 1 fps), prompted with the person's own
  box and mask; stores 3D/2D keypoints, camera translation and focal length, no meshes.
- In tangles the 3D keypoints of the under-masked person are not reliable: their mean distance to the ViTPose
  keypoints is 0.3-0.73 of body scale, against 0.07-0.29 for the well-masked person, and is often closer to the
  partner's skeleton. Both models are prompted from the same faulty mask (`reviews/autolabel-v1/limb_ownership/README.md:49-56`;
  that thread continues in the limb-entanglement repository).

## Does 12.5 fps add information over 2 fps?
Source: `reviews/autolabel-v1/REVIEW.md:132-160`.
Setup: 720 densified intervals, about 206k joint measurements. Gaps containing camera cuts and frames with a torso
jump (identity switch) are excluded; a first version wrongly counted cuts as strikes.

| measure | result |
|---|---|
| 2D keypoints interpolated from 2 fps vs measured at 12.5 fps, median error | 3.4% of body height |
| same, worst 10% of frames | 10.4% |
| same, worst 10% during transitions or scrambles | 13.2% |
| same, fastest 10% of motion (median) | 8.9% |
| mask halfway between 2 fps frames vs nearest 2 fps mask, IoU median | 0.61; 64% of frames below 0.7 |
| fast wrist peaks (4-15 body heights/s) that fall on a 2 fps frame | 57% of 851, i.e. about 43% missed |

Visual check: a full ground-and-pound punch, from wind-up to impact in 0.24 s, falls entirely between two 2 fps
frames 0.52 s apart. Caveat: the automatic peak counter is approximate; keypoint jitter in tangles also makes peaks.

Decision: 12.5 fps for one window per source plus the full matches (temporal and strike training); 2 fps label-frame
skeletons for the second windows; densify more later if a temporal model is data-limited.
