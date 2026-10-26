# E11 results — mask fix (2026-10-11 to 2026-10-12)

Sources are files in the private research workspace (`dataset/`), given as `file:line`. The A/B verdicts are
Dhruva's (human, blind); the 16-frame eye check is a Claude agent's. Frame identifiers are left out.

## Run 1: replace or clean
Source: `reviews/autolabel-v1/maskfix/README.md:1-30`; `repo: src/mask_fix.py`.
- Each touched interval is backed up (1,069 intervals, 294 MB); `mask_fix.py --revert` restores them and was tested
  byte-identical.
- Primary-pair fallback instances (11,464): 3,770 replaced by the dense mask, 7,694 speckle-cleaned. Why the rest were
  cleaned rather than replaced: 4,972 had no dense track (570 of 1,526 intervals had none), 982 dense masks swallowed
  the partner (merge guard: covering more than 35% of the partner's mask), 1,183 were fragmented, the rest failed on
  area or emptiness. Non-primary fallback instances: 12,468 cleaned, 22 replaced.
- Gold check: 3 of the 13 faulted fallback frames visibly fixed; the other 10 only cleaned (speckle gone, ownership
  errors remain); all 6 praised frames untouched. The merge guard also rejected one good replacement because the
  partner's old mask was itself wrong.
- Phase 2 prototype (`repo: src/mask_fix_track.py`): SAM 3.1 tracking over the 2 fps label frames from clean anchors.
  Held-out anchor IoU 0.65-0.98 (median about 0.93), but on tangled intervals without dense tracks the "coherent"
  tracks almost all swallowed the partner. Not applied.

## Run 2: densify the worst intervals first
Source: `reviews/autolabel-v1/maskfix/README.md:32-40`.
- Densified the worst 204 of the 570 un-densified intervals, ordered by fallback-mask count (the remaining 366 hold
  392 of the 4,972 fallback masks; 228 hold none).
- In those intervals 2,795 of 4,580 primary fallback masks (61%) were replaced. All runs: 6,565 of 11,464 (57%)
  replaced, the rest cleaned.
- Export rebuilt with a new tier, `dense_replaced_unverified` (Astra's verify verdict judged the old mask): rejected
  12,192 -> 7,770 frames; `dense_replaced_unverified` 5,120.
- Claude eye check of 16 random replaced frames: 9 better (speckle and holes gone, missing head or limbs regained),
  5 same, 2 worse (holes in a torso; one fighter takes the other's gloves). Tangle ownership errors persist where they
  existed: the dense track inherits them.

## Dhruva's blind A/B review
Source: `reviews/autolabel-v1/maskfix/README.md:42-49`; pack built by `repo: src/maskfix_ab_build.py`.
40 random changed frames (32 dense-replaced, 8 cleaned only); A/B order random per item, the key never served to the
page; notes interpreted over keys.

| group | fix better | original better | same | both bad |
|---|---|---|---|---|
| dense-replaced (32) | 9 (mostly "slightly") | 4 | 13 | 6 |
| cleaned only (8) | 0 | 0 | 7 | 1 |

- Still clearly bad after the fix: about 11 of 40 (both bad, plus "better but neither good enough").
- One interval appeared three times, all both bad: the person inside the closed guard is not masked and a distant
  bystander is the second "primary" person. That is the pair-selection bug ([E09](../E09-gold-review/)), not mask
  shape.
- Dhruva's verdict: "only slightly better but not good enough yet". The 16-frame AI eye check (9/16 better) was more
  lenient.
