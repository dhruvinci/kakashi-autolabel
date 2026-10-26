# Findings log

One line per finding, by date. Each links to the experiment that supports it. "AI" marks findings that rest on AI
reviewers (AI references, not human ground truth).

## 2026-09-30
- Search titles and channels are hypotheses: sampled frames moved sources between training and tutorial, found a gi
  montage inside a no-gi video, and a music mix with no action. ([E00](experiments/E00-acquisition-catalogue/))
- Publisher chapter times are rounded and can run past the end of the video; keep the originals and reconcile.
  ([E00](experiments/E00-acquisition-catalogue/))
- AI: low, medium and high effort agents make almost only supported claims (90/90, 102/104, 125/127); what differs is
  coverage. A low-effort repeat covered 19/27 checklist items. Correct claims alone are a weak quality measure.
  ([E01](experiments/E01-agent-annotation-trial/))

## 2026-10-01
- AI: across 100 sources, agent annotation gives rich, mostly supported text (906/909 claims) but geometry on only
  647 of 7,926 core frames, on selected actors. It cannot train a detector. ([E01](experiments/E01-agent-annotation-trial/))
- Parity with Roboflow's Auto Label is not measured: no Roboflow predictions were run on these frames, and its
  published mAP is not comparable with claim-support fractions. ([E02](experiments/E02-roboflow-comparison/))

## 2026-10-03
- The released SAM 3.1 multiplex checkpoint has no trained box-to-mask head; box prompts must go through the tracker
  decoder (IoU 0.95-0.96 on the prompt frame). ([E03](experiments/E03-autolabel-calibration/))
- Reversing Roboflow's split (SAM 3.1 finds everyone, Astra resolves the marks) labels every frame: 508/508 against
  43 for agent annotation, and matches 77/79 earlier person boxes. Two ownership points per person fix most entangled
  frames. ([E03](experiments/E03-autolabel-calibration/))

## 2026-10-04
- AI: on 200 intervals, detection, roles and standing masks are near-perfect; ground masks 90% good / 8% partial /
  1.8% wrong; posture is the weakest label (5.8% wrong). ([E04](experiments/E04-autolabel-first-run/))
- AI: flags rank quality (14% / 36% / 67% of frames flagged in good / usable / poor intervals) but miss posture,
  position and partner-swallowing masks. ([E04](experiments/E04-autolabel-first-run/))
- AI: a second Astra opinion clears 27% of flagged frames; blind Claude adjudication sides with Astra on 20/30 mask
  disputes. Position disputes are mostly genuinely ambiguous scrambles. ([E05](experiments/E05-second-opinion-verify/))
- The official SAM 3.1 video predictor fits a 6 GB GPU with the detector's matrices in bf16 and three upstream fixes.
  ([E06](experiments/E06-densify-and-frame-rate/))

## 2026-10-09
- 2 fps plus interpolation is close for most motion (median keypoint error 3.4% of body height) but not for scrambles
  (worst 10%: 13.2%) or strikes (about 43% of fast wrist peaks fall between 2 fps frames). Masks change a lot within
  0.25 s. ([E06](experiments/E06-densify-and-frame-rate/))

## 2026-10-10
- AI: over 1,526 intervals, masks are 85% good, 10% partial, 5% wrong; the failure is interlocked bodies, and 58
  audited frames had real errors with no flag. ([E07](experiments/E07-scale-up-audit/))
- Skeleton-based mask checks reach about 30% precision, no better than the existing flags.
  ([E07](experiments/E07-scale-up-audit/))
- AI: 3,175 skeleton speed peaks give 1,322 strikes; recall is bounded by the skeleton.
  ([E08](experiments/E08-strike-pass/))

## 2026-10-11
- Dhruva's gold review (103 verdicts): identity 12/19, events 12/18, strikes 13/20, positions 11/20 (class 14/20),
  mask pairs 12/23. Every AI audit with a comparable number was more lenient. ([E09](experiments/E09-gold-review/))
- No true identity switch in the gold items: identity failures were bad masks. ([E09](experiments/E09-gold-review/))
- Masks made from Astra boxes were 87% bad, against 20% for SAM 3.1 text detections; agreement with the 12.5 fps
  video mask is the best single predictor of a good mask (AUC 0.84). ([E09](experiments/E09-gold-review/))
- 4.0% of position frames describe a pair whose primary athlete is not in the frame (pair chosen once per interval).
  ([E09](experiments/E09-gold-review/))
- Astra saturates at 13-14 of 17 gold positions and 13 of 16 gold events with better prompts; the remaining cases
  need context or inference beyond the pixels. Claude Opus trails. ([E10](experiments/E10-vlm-saturation-bench/))
- For events, the whole 20 s interval at 2 fps beats the event window at 4 fps (13/16 vs 9/16): context matters more
  than frame rate. ([E10](experiments/E10-vlm-saturation-bench/))

## 2026-10-12
- Replacing 57% of the fallback masks with 12.5 fps video masks: Dhruva, blind, found 9 better, 4 worse, 13 the same
  and 6 both bad of 32: "only slightly better but not good enough yet". Video masks inherit tangle errors.
  ([E11](experiments/E11-mask-fix/))
