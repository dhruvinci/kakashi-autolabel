# E08 results — strike pass (2026-10-10 to 2026-10-11)

Sources are files in the private research workspace (`dataset/`), given as `file:line`, or this repository
(`repo:`). Strike labels are Astra's (AI); the only human check is Dhruva's gold review.

## Method
Source: `reviews/autolabel-v1/REVIEW.md:249-258`; `repo: src/autolabel_strikes.py` (docstring).
- Scope: all 269 densified MMA intervals that contain striking events.
- Candidates: speed peaks of wrists, elbows, knees and ankles in the 12.5 fps skeletons, above 3 body heights per
  second, local maxima; camera cuts and identity jumps excluded; peaks within 0.4 s merged.
- Astra sees each candidate as a strip of six frames 80 ms apart with the pair outlined, and judges strike or not,
  striker, type, target and result.

## Results
Source: `reviews/autolabel-v1/REVIEW.md:260-288`; `repo: docs/export_summary.json:67-91`.
- 3,175 candidates: 1,322 strikes, 1,770 non-strikes, 83 unclear (exported as `strikes.jsonl`).
- By type: jab 227, ground punch 193, leg kick 159, hook 116, cross 98, body kick 81, overhand 81, hammerfist 71,
  head kick 71, front kick 67, knee 55, body punch 47, elbow 26, spinning 19, uppercut 11.
- By result: blocked 389, unclear 368, landed partly 203, landed clean 203, missed 159.
- Spot check (Claude, 6 strips): all plausible; it found the ground-and-pound punch that the 2 fps frames missed,
  labelled a kick, an overhand and a hook with the right striker, and correctly rejected a static clinch.
- Limit: recall is bounded by the skeleton detector; strikes thrown by an occluded limb are never proposed. The 15
  rejected-candidate gold items measure precision on the rejections, not misses outside the candidates.
- Cost: Astra plan usage went from 76% to 84% of the weekly window for about 890 calls.

## Human check (Dhruva's gold review, first session)
Source: `reviews/autolabel-v1/REVIEW.md:319`.
13/20 fully correct (`unclear` excluded). Rejected candidates 5/6 right. Jab, cross, uppercut and kicks right; wrong
when the context is ground or clinch (a face push labelled ground punch, an up-kick labelled head kick, a roll-out kick
labelled spinning) or when the strip holds a combination.
