"""Corpus summary of the densify stage's held-out anchors (CPU, stdlib only). Added for the public write-up, 2026-10-26.

autolabel_densify.py holds out every third clean label frame and compares the propagated 12.5 fps mask with the label
mask at that moment. Both masks are model output (Astra + SAM 3.1), so this is self-consistency, not accuracy.

  python src/densify_heldout_summary.py --root <workspace> --out experiments/E06-densify-and-frame-rate/heldout_summary.json
"""
import argparse
import json
import statistics
from collections import Counter
from pathlib import Path


def cohort(trial_id):
    return 'trial' if trial_id.startswith('trial') else {'a2s': 'breadth', 'a3m': 'matches', 'a4g': 'gi'}.get(trial_id[:3], 'other')


def quantile(values, q):
    values = sorted(values)
    return round(values[int(q*(len(values) - 1))], 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument('--run', default='v1')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    ious, per_cohort, intervals, frames = [], {}, Counter(), 0
    for path in sorted((a.root/'autolabel'/a.run).glob('*/*/dense/dense.json')):
        doc = json.loads(path.read_text())
        c = cohort(path.parts[-4])
        intervals[c] += 1
        frames += doc['dense_frames']
        vals = [v for row in doc.get('validation', []) for v in row['iou'].values() if v is not None]
        ious += vals
        per_cohort.setdefault(c, []).extend(vals)
    summary = {
        'what': 'IoU between the propagated dense mask and the held-out label-frame mask, per actor and held-out anchor',
        'caveat': 'both masks are model output; agreement, not accuracy',
        'intervals': sum(intervals.values()), 'intervals_by_cohort': dict(intervals), 'dense_frames': frames,
        'actor_anchors': len(ious), 'median': quantile(ious, 0.5), 'p10': quantile(ious, 0.1), 'p25': quantile(ious, 0.25),
        'share_ge_0.9': round(sum(v >= 0.9 for v in ious)/len(ious), 3), 'share_lt_0.7': round(sum(v < 0.7 for v in ious)/len(ious), 3),
        'median_by_cohort': {c: quantile(v, 0.5) for c, v in sorted(per_cohort.items())},
        'p10_by_cohort': {c: quantile(v, 0.1) for c, v in sorted(per_cohort.items())},
        'mean': round(statistics.fmean(ious), 3),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(summary, indent=1) + '\n')
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
