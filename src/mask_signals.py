"""Per actor-frame mask-quality signals for the primary pair (CPU), used by autolabel_requalify.py.

  off_joints      confident (score>0.5) own skeleton joints falling outside the actor's own (dilated) mask
  partner_joints  of those, how many land inside the partner's mask
  lcs             largest connected component share of the mask
  contact         the pair's boxes overlap (IoU > 0.1)
Skeletons: dense/pose2d.npz at label frames (interval01, densified) or pose2d_labels.npz (2 fps intervals).
"""
import json

import numpy as np

from autolabel_common import box_iou, rle_decode
from autolabel_fuse import dilate, largest_component_share


def skeletons(base):
    """{(frame_index, actor): (keypoints[17,2], scores[17])} at label frames."""
    out = {}
    if (base/'dense/pose2d.npz').exists():
        d = json.loads((base/'dense/dense.json').read_text())
        to_label = {f['dense_index']: f['label_index'] for f in d['frames'] if f.get('is_label_frame') and 'label_index' in f}
        if not to_label:
            times = {round(f['time_s'], 2): f['dense_index'] for f in d['frames'] if f.get('is_label_frame')}
            L = json.loads((base/'labels.json').read_text())
            to_label = {}
            for f in L['frames']:
                k = min(times, key=lambda t: abs(t - f['time_s'])) if times else None
                if k is not None and abs(k - f['time_s']) < 0.05:
                    to_label[times[k]] = f['frame_index']
        p = np.load(base/'dense/pose2d.npz')
        for i, (k, a) in enumerate(zip(p['frame_index'], p['actor'])):
            if int(k) in to_label:
                out[(to_label[int(k)], str(a))] = (p['keypoints'][i], p['scores'][i])
    elif (base/'pose2d_labels.npz').exists():
        p = np.load(base/'pose2d_labels.npz')
        for i, (k, a) in enumerate(zip(p['frame_index'], p['actor'])):
            out[(int(k), str(a))] = (p['keypoints'][i], p['scores'][i])
    return out


def frame_pair(frame, roles, k=2):
    """Per-frame primary pair: the two largest athletes/instructors in this frame, preferring the pair in contact."""
    cands = sorted((i for i in frame['instances'] if roles.get(i['actor_id']) in ('athlete', 'instructor') and i['area']),
                   key=lambda i: -i['area'])[:4]
    if len(cands) < 2:
        return [i['actor_id'] for i in cands]
    best = max(((a, b) for x, a in enumerate(cands) for b in cands[x + 1:]),
               key=lambda p: (box_iou(p[0]['box_xyxy'], p[1]['box_xyxy']) > 0.02, p[0]['area'] + p[1]['area']))
    return [best[0]['actor_id'], best[1]['actor_id']]


def signals(base, L=None):
    L = L or json.loads((base/'labels.json').read_text())
    roles = {a['actor_id']: a['role'] for a in L['actors']}
    sk = skeletons(base)
    out = {}
    for f in L['frames']:
        pair = frame_pair(f, roles)
        inst = {i['actor_id']: i for i in f['instances'] if i['actor_id'] in pair}
        masks = {a: rle_decode(i['mask_rle']) for a, i in inst.items()}
        h, w = f['height'], f['width']
        contact = len(pair) == 2 and box_iou(inst[pair[0]]['box_xyxy'], inst[pair[1]]['box_xyxy']) > 0.1
        for a in pair:
            m = masks[a]
            partner = masks[[b for b in pair if b != a][0]] if len(pair) == 2 else np.zeros_like(m)
            row = {'pair': pair, 'contact': contact, 'lcs': largest_component_share(m), 'area': int(m.sum())}
            if (f['frame_index'], a) in sk:
                kp, sc = sk[(f['frame_index'], a)]
                own = dilate(m, 4)
                off = part = conf = 0
                for (x, y), s in zip(kp, sc):
                    x, y = int(round(x)), int(round(y))
                    if s < 0.5 or not (0 <= x < w and 0 <= y < h):
                        continue
                    conf += 1
                    if not own[y, x]:
                        off += 1
                        part += bool(partner[y, x])
                row.update(conf_joints=conf, off_joints=off, partner_joints=part)
            out[(f['frame_index'], a)] = row
    return out


def main():
    import glob
    import sys
    from pathlib import Path
    from autolabel_common import write_json
    bases = sorted(Path(p).parent for p in glob.glob(str(Path(__file__).resolve().parents[1]/'autolabel/v1/*/*/labels.json')))
    for n, base in enumerate(bases):
        if (base/'mask_signals.json').exists() and '--force' not in sys.argv:
            continue
        try:
            sig = signals(base)
        except Exception as error:
            print('failed', base, error, flush=True)
            continue
        write_json(base/'mask_signals.json', {'version': 1, 'signals': {f'{k}:{a}': v for (k, a), v in sig.items()}})
        if n % 100 == 0:
            print(n, len(bases), flush=True)


if __name__ == '__main__':
    main()
