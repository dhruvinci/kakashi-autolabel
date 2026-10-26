"""Stage 9: individual strike labels for MMA striking intervals (Astra on 12.5 fps strips).

Candidates come from the 12.5 fps skeletons of the primary pair: speed peaks of wrists, elbows, knees and ankles
(> 3 body heights/s, local maximum), excluding camera cuts and identity jumps, merged within 0.4 s. Each candidate is
shown to Astra as a 3x2 strip of six consecutive dense frames (-240 ms .. +160 ms around the peak), cropped to the pair,
with the primary actors outlined red/blue from the dense SAM 3.1 masks. Four candidates per call.
Astra answers per candidate: strike thrown?, striker, strike type, target, result, peak tile.
Output: autolabel/<run>/<trial>/<packet>/strikes.json
"""
import argparse
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from autolabel_astra import EFFORT, call_astra, enum, font, obj, outline, read_usage
from autolabel_common import interval_dir, packet, write_json
from autolabel_densify import decode_span

PER_CALL = 4
MAX_CANDIDATES = 32
STOP_AT = 90  # usage percent: leave headroom for Dhruva's post-review requests (hard rule is 99)
JOINTS = {9: 'left_wrist', 10: 'right_wrist', 7: 'left_elbow', 8: 'right_elbow', 13: 'left_knee', 14: 'right_knee',
          15: 'left_ankle', 16: 'right_ankle'}
TYPES = ['jab', 'cross', 'hook', 'uppercut', 'overhand', 'body_punch', 'ground_punch', 'hammerfist', 'elbow', 'knee',
         'leg_kick', 'body_kick', 'head_kick', 'front_kick', 'spinning', 'other']
COLOURS = [(230, 25, 75), (0, 130, 200)]
PROMPT = """You are labelling individual strikes in combat-sports video (MMA / kickboxing). Answer only from the attached images. Do not run commands or read files.

There are {n} images, C1..C{n} (one per candidate, in the order attached). Each image shows 6 consecutive video frames 80 ms apart (tiles 1-6, left-to-right then top-to-bottom, numbered in each tile), about 0.5 s in total. Two fighters are outlined: RED = {a} ("{da}"), BLUE = {b} ("{db}").

An automatic motion detector found a fast limb movement around tile 4 of each image. For each candidate decide:
- strike: "yes" if a punch, kick, knee or elbow is thrown at the opponent within these frames (landed or not); "no" if the motion is something else (feint without extension, stance switch, hand fighting, grappling, adjusting, camera/occlusion artefact); "unclear" if you cannot tell.
- striker: red | blue | none (none when strike != yes).
- type: one of {types}; "other" when strike != yes.
- target: head | body | legs | none.
- result: landed_clean | landed_partial (glancing or partly blocked) | blocked | missed | unclear | none.
- peak_tile: the tile (1-6) where the strike reaches full extension / impact; 0 when strike != yes.
Be conservative: only "yes" when a strike is actually visible. Judge each candidate independently."""


def schema():
    return obj({'candidates': {'type': 'array', 'items': obj({
        'candidate': {'type': 'string'}, 'strike': enum(['yes', 'no', 'unclear']), 'striker': enum(['red', 'blue', 'none']),
        'type': enum(TYPES), 'target': enum(['head', 'body', 'legs', 'none']),
        'result': enum(['landed_clean', 'landed_partial', 'blocked', 'missed', 'unclear', 'none']),
        'peak_tile': {'type': 'integer'}})}, 'notes': {'type': 'string'}})


def candidates(base, pair):
    """Dense indices of fast-limb peaks for the primary pair."""
    from fps_value import torso_jump, tracks
    d = json.loads((base/'dense/dense.json').read_text())
    p = np.load(base/'dense/pose2d.npz')
    times = [f['time_s'] for f in d['frames']]
    cuts = set(d.get('shot_cuts', []))
    kp, sc, tg = p['keypoints'], p['scores'], p['tags']
    ok = (sc > 0.4) & (tg == 0)
    peaks = []
    for actor, idx in tracks(p).items():
        if actor not in pair:
            continue
        ks = sorted(idx)
        for jt in JOINTS:
            sp = []
            for k0, k1 in zip(ks, ks[1:]):
                i0, i1 = idx[k0], idx[k1]
                v = 0.0
                if k1 - k0 == 1 and ok[i0, jt] and ok[i1, jt]:
                    h = max(40.0, float(np.ptp(kp[i1][ok[i1], 1])) if ok[i1].sum() > 3 else 40.0)
                    v = float(np.linalg.norm(kp[i1, jt] - kp[i0, jt]))/h/max(1e-3, times[k1] - times[k0])
                sp.append((k1, v))
            for j in range(1, len(sp) - 1):
                k, v = sp[j]
                if not (3.0 < v < 15.0 and v >= sp[j - 1][1] and v >= sp[j + 1][1]):
                    continue
                if any(abs(k - c) <= 2 for c in cuts) or torso_jump(kp, ok, idx, k - 1, k, per_frame=0.08):
                    continue
                peaks.append((k, v, actor, JOINTS[jt]))
    peaks.sort(key=lambda x: -x[1])
    chosen = []
    for k, v, actor, joint in peaks:
        if all(abs(times[k] - times[c['dense_index']]) > 0.4 for c in chosen):
            chosen.append({'dense_index': k, 'time_s': times[k], 'speed': round(v, 2), 'actor': actor, 'joint': joint})
    return sorted(chosen[:MAX_CANDIDATES], key=lambda c: c['dense_index']), d


def strip(native, d, z, k, pair, label):
    """3x2 sheet of dense frames k-3..k+2, cropped to the pair over the strip, outlines from the dense masks."""
    h, w = d['shape']
    ks = [j for j in range(k - 3, k + 3) if 0 <= j < len(d['frames'])]
    times = np.array([t for _, t in native])
    imgs, boxes = [], []
    for j in ks:
        f = d['frames'][j]
        path = native[int(np.argmin(np.abs(times - f['time_s'])))][0]
        with Image.open(path) as im:
            px = np.asarray(im.convert('RGB')).copy()
        for c, actor in enumerate(pair):
            key = f'{j}_{actor}'
            if key in z.files:
                m = np.unpackbits(z[key])[:h*w].reshape(h, w).astype(bool)
                px[outline(m, 3)] = COLOURS[c]
                ys, xs = np.nonzero(m)
                if len(xs):
                    boxes.append((xs.min(), ys.min(), xs.max(), ys.max()))
        imgs.append(px)
    if boxes:
        x1, y1 = min(b[0] for b in boxes), min(b[1] for b in boxes)
        x2, y2 = max(b[2] for b in boxes), max(b[3] for b in boxes)
        pad = 0.15*max(x2 - x1, y2 - y1)
        box = (int(max(0, x1 - pad)), int(max(0, y1 - pad)), int(min(w, x2 + pad)), int(min(h, y2 + pad)))
    else:
        box = (0, 0, w, h)
    tw, th = 480, 360
    sheet = Image.new('RGB', (tw*3, th*2), 'gray')
    for n, px in enumerate(imgs):
        t = Image.fromarray(px).crop(box)
        t.thumbnail((tw, th))
        tile = Image.new('RGB', (tw, th), 'gray')
        tile.paste(t, ((tw - t.width)//2, (th - t.height)//2))
        dr = ImageDraw.Draw(tile)
        dr.rectangle([0, 0, 70 if n else 110, 26], fill='black')
        dr.text((5, 3), f'{label} {n + 1}' if n == 0 else str(n + 1), fill='white', font=font(20))
        sheet.paste(tile, ((n % 3)*tw, (n//3)*th))
    return sheet


def run(run_name, targets, effort=EFFORT, force=False):
    for trial_id, packet_id in targets:
        base = interval_dir(run_name, trial_id, packet_id)
        if (base/'strikes.json').exists() and not force:
            continue
        used = max((u or {}).get('usedPercent', 0) for u in (read_usage()['primary'], read_usage()['secondary']))
        if used >= STOP_AT:
            raise SystemExit(f'STOP: usage {used}% >= strike-pass budget {STOP_AT}%')
        L = json.loads((base/'labels.json').read_text())
        pair = sorted(a['actor_id'] for a in L['actors'] if a.get('primary'))
        if len(pair) != 2 or not (base/'dense/pose2d.npz').exists():
            continue
        desc = {a['actor_id']: a['description'][:80] for a in L['actors']}
        cands, d = candidates(base, pair)
        if not cands:
            write_json(base/'strikes.json', {'version': 1, 'candidates': [], 'strikes': [], 'note': 'no motion peaks'})
            continue
        _, pkt_path = packet(trial_id, packet_id)
        master = json.loads((pkt_path.parents[1]/'index.json').read_text())['master_path']
        z = np.load(base/'dense/masks.npz')
        tmp = Path(tempfile.mkdtemp(prefix='strikes-'))
        try:
            t0, t1 = cands[0]['time_s'] - 0.6, cands[-1]['time_s'] + 0.6
            native = decode_span(master, max(0, t0), t1, tmp/'native')
            out_dir = base/'strike-images'
            out_dir.mkdir(exist_ok=True)
            for s in range(0, len(cands), PER_CALL):
                group = cands[s:s + PER_CALL]
                paths = []
                for j, c in enumerate(group):
                    path = out_dir/f"c{s + j:02d}.jpg"
                    strip(native, d, z, c['dense_index'], pair, f'C{j + 1}').save(path, quality=85)
                    paths.append(path)
                prompt = PROMPT.format(n=len(group), a=pair[0], b=pair[1], da=desc[pair[0]], db=desc[pair[1]],
                                       types=', '.join(TYPES))
                res, receipt = call_astra(prompt, paths, schema(), effort)
                by = {r['candidate']: r for r in res['candidates']}
                for j, c in enumerate(group):
                    r = by.get(f'C{j + 1}')
                    if r:
                        c.update(answer=r, image=str(paths[j].relative_to(base)))
                print(f'{trial_id}/{packet_id}: candidates {s + 1}-{s + len(group)}/{len(cands)}, {receipt["elapsed_s"]}s', flush=True)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        strikes = []
        for c in cands:
            r = c.get('answer')
            if r and r['strike'] == 'yes' and r['striker'] in ('red', 'blue'):
                tile = r['peak_tile'] if 1 <= r['peak_tile'] <= 6 else 4
                k = min(len(d['frames']) - 1, max(0, c['dense_index'] - 4 + tile))
                strikes.append({'time_s': d['frames'][k]['time_s'], 'dense_index': k,
                                'striker': pair[0] if r['striker'] == 'red' else pair[1],
                                'target_actor': pair[1] if r['striker'] == 'red' else pair[0],
                                'type': r['type'], 'target': r['target'], 'result': r['result'], 'detector_joint': c['joint']})
        write_json(base/'strikes.json', {'version': 1, 'stage': 'astra_strike_labels', 'pair': pair,
                                         'created_at': datetime.now(timezone.utc).isoformat(),
                                         'candidates': cands, 'strikes': strikes})


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='v1')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('targets', nargs='+')
    a = ap.parse_args()
    run(a.run, [tuple(t.split('/')) for t in a.targets], force=a.force)
