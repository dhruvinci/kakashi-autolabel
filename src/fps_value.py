"""Does 12.5 fps add information over 2 fps? Measured on densified intervals (CPU only).

1. Skeleton interpolation: keep only the label-frame (2 fps) 2D keypoints of each actor, linearly interpolate the dense
   frames in between, compare with the 12.5 fps keypoints actually measured. Error is normalised by person box height.
   Only confident joints on the actor's own mask (score > 0.5, tag own) on both ends and at the target count.
2. Mask staleness: IoU between an actor's mask at a mid-gap dense frame and the nearest label-frame mask.
3. Strike capture (MMA striking): wrist speed peaks from 12.5 fps skeletons; share that fall within one dense frame
   (+-80 ms) of a 2 fps label frame, i.e. would be visible at all at 2 fps.
Writes reviews/autolabel-v1/fps-value.json.
"""
import collections
import glob
import json
import statistics
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'autolabel/v1'
WRISTS = (9, 10)


def load(base):
    d = json.loads((base/'dense/dense.json').read_text())
    p = np.load(base/'dense/pose2d.npz')
    fight = json.loads((base/'fight.json').read_text())['result'] if (base/'fight.json').exists() else None
    return d, p, fight


def tracks(p):
    out = collections.defaultdict(dict)
    for i, (k, a) in enumerate(zip(p['frame_index'], p['actor'])):
        out[str(a)][int(k)] = i
    return out


def torso_jump(kp, ok, idx, a, b, per_frame=0.15):
    """True when the torso centre moves more than per_frame body heights per dense frame between a and b, or when the
    torso joints are not confident at both ends (identity switches and cuts look like this, strikes do not)."""
    if a not in idx or b not in idx:
        return True
    ia, ib = idx[a], idx[b]
    torso = [5, 6, 11, 12]
    if not (ok[ia, torso].sum() >= 3 and ok[ib, torso].sum() >= 3):
        return True
    ca = kp[ia, torso][ok[ia, torso]].mean(0)
    cb = kp[ib, torso][ok[ib, torso]].mean(0)
    h = max(40.0, float(np.ptp(kp[ib][ok[ib], 1])))
    return float(np.linalg.norm(cb - ca))/h > per_frame*max(1, b - a)


def position_at(fight, times, t):
    if not fight:
        return 'unknown'
    k = int(np.argmin(np.abs(np.array(times) - t)))
    return fight['frames'][k]['position']


def main(limit=None):
    interp, stale, strike = collections.defaultdict(list), collections.defaultdict(list), collections.Counter()
    fast = []
    bases = sorted(Path(b).parent.parent for b in glob.glob(str(RUN/'*/*/dense/pose2d.json')))[:limit]
    for base in bases:
        try:
            d, p, fight = load(base)
        except (FileNotFoundError, KeyError):
            continue
        frames = d['frames']
        times = [f['time_s'] for f in frames]
        cuts = set(d.get('shot_cuts', []))
        label_idx = [f['dense_index'] for f in frames if f['is_label_frame']]
        fight_times = [f['time_s'] for f in json.loads((base/'fight.json').read_text())['frames']] if fight else []
        sport = 'mma' if base.parts[-2].startswith(('trial', 'a2s', 'a3m')) and any(
            e['type'] == 'striking_exchange' for e in (fight['events'] if fight else [])) else 'grappling'
        kp, sc, tg = p['keypoints'], p['scores'], p['tags']
        ok = (sc > 0.5) & (tg == 0)
        for actor, idx in tracks(p).items():
            for a, b in zip(label_idx, label_idx[1:]):
                if a not in idx or b not in idx or b - a < 3 or any(a < c <= b for c in cuts):
                    continue
                ia, ib = idx[a], idx[b]
                if torso_jump(kp, ok, idx, a, b):
                    continue
                height = max(1.0, float(np.ptp(kp[ia][ok[ia], 1])) if ok[ia].sum() > 3 else 1.0)
                for k in range(a + 1, b):
                    if k not in idx:
                        continue
                    ik = idx[k]
                    w = (times[k] - times[a])/max(1e-6, times[b] - times[a])
                    pred = (1 - w)*kp[ia] + w*kp[ib]
                    m = ok[ia] & ok[ib] & ok[ik]
                    if m.sum() < 4 or height < 40:
                        continue
                    err = np.linalg.norm(pred[m] - kp[ik][m], axis=1)/height
                    e = float(np.median(err))
                    pos = position_at(fight, fight_times, times[k])
                    group = 'standing' if pos == 'standing' else ('ground' if pos in (
                        'guard', 'side_control', 'mount', 'back', 'turtle', 'leg_entanglement') else 'transition/other')
                    interp[('all',)].append(e)
                    interp[('pos', group)].append(e)
                    interp[('sport', sport)].append(e)
                    speed = float(np.median(np.linalg.norm(kp[ib][m] - kp[ia][m], axis=1)))/height
                    fast.append((speed, e))
        # strikes: wrist speed peaks in MMA striking intervals
        if sport == 'mma':
            label_set = set(label_idx)
            for actor, idx in tracks(p).items():
                ks = sorted(idx)
                for wj in WRISTS:
                    sp = []
                    for k0, k1 in zip(ks, ks[1:]):
                        if k1 - k0 != 1:
                            sp.append((k1, 0.0))
                            continue
                        i0, i1 = idx[k0], idx[k1]
                        if ok[i0, wj] and ok[i1, wj]:
                            dt = max(1e-3, times[k1] - times[k0])
                            h = max(40.0, float(np.ptp(kp[i1][ok[i1], 1])) if ok[i1].sum() > 3 else 40.0)
                            sp.append((k1, float(np.linalg.norm(kp[i1, wj] - kp[i0, wj]))/h/dt))
                        else:
                            sp.append((k1, 0.0))
                    for j in range(1, len(sp) - 1):
                        k, v = sp[j]
                        if any(abs(k - c) <= 2 for c in cuts) or torso_jump(kp, ok, idx, k - 1, k, per_frame=0.08):
                            continue  # camera cut or identity switch, not a strike
                        if 4.0 < v < 15.0 and v >= sp[j - 1][1] and v >= sp[j + 1][1]:  # fast limb, physically plausible
                            strike['peaks'] += 1
                            strike['seen_at_2fps'] += any(abs(k - L) <= 1 for L in label_set)
        # mask staleness at mid-gap
        z = np.load(base/'dense/masks.npz')
        h, w = d['shape']
        for a, b in zip(label_idx, label_idx[1:]):
            if any(a < c <= b for c in cuts):
                continue
            mid = (a + b)//2
            for actor in frames[mid]['actor_areas']:
                ka, km = f'{a}_{actor}', f'{mid}_{actor}'
                if ka in z.files and km in z.files:
                    ma = np.unpackbits(z[ka])[:h*w].astype(bool)
                    mm = np.unpackbits(z[km])[:h*w].astype(bool)
                    stale[('all',)].append(float((ma & mm).sum()/max(1, (ma | mm).sum())))
    fast.sort()
    top = [e for _, e in fast[int(0.9*len(fast)):]]
    summary = {
        'intervals': len(bases),
        'keypoint_interp_error_median_body_heights': {'/'.join(k): round(statistics.median(v), 4) for k, v in interp.items() if v},
        'keypoint_interp_error_p90': {'/'.join(k): round(float(np.percentile(v, 90)), 4) for k, v in interp.items() if v},
        'samples': {'/'.join(k): len(v) for k, v in interp.items()},
        'fastest_10pct_motion_median_error': round(statistics.median(top), 4) if top else None,
        'mask_iou_midgap_vs_nearest_label_frame_median': round(statistics.median(stale[('all',)]), 3) if stale[('all',)] else None,
        'mask_iou_midgap_share_below_0.7': round(float(np.mean(np.array(stale[('all',)]) < 0.7)), 3) if stale[('all',)] else None,
        'wrist_strike_peaks': strike['peaks'],
        'strike_peaks_visible_at_2fps_pct': round(100*strike['seen_at_2fps']/max(1, strike['peaks']), 1)}
    (ROOT/'reviews/autolabel-v1/fps-value.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
