# Experiments

One folder per experiment, in the order they ran (2026-09-30 to 2026-10-12). `README.md` is the short version;
`RESULTS.md` is the full record, with the research-workspace file and line behind every number. These experiments
were not pre-registered: each verdict is the decision taken at the time, stated against what was measured.

Who judged: "AI" means Claude reviewer agents or Astra itself (AI references, not human ground truth); "Dhruva" means
a human verdict.

| ID | Question | Judged by | Verdict |
|---|---|---|---|
| [E00](E00-acquisition-catalogue/) | Collect ~1,000 varied videos as compressed masters, subclips as metadata only | counts | **done**: 1,233 catalogued, 908 verified masters (145.7 h) |
| [E01](E01-agent-annotation-trial/) | Can GPT-6 Astra agents annotate combat video richly, at which effort? | AI | **pass** for rich text, **fail** as a vision dataset (geometry on 8% of frames) |
| [E02](E02-roboflow-comparison/) | Is that as good as Roboflow's Auto Label? | AI | **not measured**: no Roboflow predictions on these frames; parity cannot be claimed |
| [E03](E03-autolabel-calibration/) | Does Astra set-of-marks over local SAM 3.1 give complete masks? Which prompt? | AI | **pass** (exploratory): 508/508 frames, 77/79 earlier boxes matched |
| [E04](E04-autolabel-first-run/) | All 200 trial intervals: which labels are usable? | AI | **usable with filtering**; flags miss posture, position and partner-swallowing masks |
| [E05](E05-second-opinion-verify/) | Can a second Astra pass sort real errors from false alarms? | AI | **adopted**: quality tiers; disputed positions become uncertain |
| [E06](E06-densify-and-frame-rate/) | 12.5 fps masks and skeletons on 6 GB; does 12.5 fps add information? | AI / model agreement | **adopted** where motion is fast; held-out IoU 0.953 (self-consistency) |
| [E07](E07-scale-up-audit/) | Scale to 1,526 intervals from 670 videos; re-audit | AI | **usable with filtering**: masks 85% good; tangles fail, mostly unflagged |
| [E08](E08-strike-pass/) | Individual strikes from 12.5 fps skeletons | AI, then Dhruva | **usable for standing striking**: 1,322 strikes; Dhruva 13/20 |
| [E09](E09-gold-review/) | First human measurement | **Dhruva** | **AI audits were lenient**: 52-70% fully correct by type |
| [E10](E10-vlm-saturation-bench/) | How far can a VLM go on the corrected cases? | Dhruva's labels | **Astra saturates** (14/17 positions, 13/16 events); Claude Opus behind |
| [E11](E11-mask-fix/) | Do 12.5 fps video masks fix the worst masks? | **Dhruva**, blind | **fail on Dhruva's bar**: better 9, worse 4, same 13, both bad 6 of 32 |

Continued elsewhere: tangle tracking and dot-assisted mask correction in
[limb-entanglement](https://github.com/dhruvinci/limb-entanglement); 3D engagement measurement in
[engage3d](https://github.com/dhruvinci/engage3d).
