"""Mask fix 2 (prototype, GPU, .venv-sam31): re-segment fallback masks by SAM 3.1 video tracking over the 2 fps label frames.

For intervals without a usable dense track (or where the dense mask was not coherent). Anchors are label frames where every
primary actor is a clean text detection (score >= SCORE_MIN, no review flags, no other actor's point inside); every third
anchor is held out to measure tracking IoU. Tracking runs forward and backward from the first prompt anchor
(autolabel_densify.track_chunk). For each primary fallback instance the tracked mask is checked with mask_fix.coherent.
No labels are written: masks go to reviews/autolabel-v1/maskfix/track/<trial>_<packet>.npz + records.

  ../.venv-sam31/bin/python mask_fix_track.py <trial>/<interval> [<trial>/<interval> ...]
"""
import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np

from autolabel_common import ROOT, rle_decode, write_json
from mask_fix import FALLBACK, OUT, RUN, coherent, original_labels

SCORE_MIN = 0.85


def iou(a, b):
    return float((a & b).sum()/max(1, (a | b).sum()))


def track_interval(predictor, base, work):
    import torch
    from autolabel_densify import track_chunk
    L = original_labels(base)
    F = L['frames']
    pair = sorted(a['actor_id'] for a in L['actors'] if a.get('primary'))
    obj_ids = {a: n + 1 for n, a in enumerate(pair)}
    h, w = F[0]['height'], F[0]['width']
    anchors = {}  # per actor: frames where THAT actor is a clean, confident, unfragmented text detection
    for k, f in enumerate(F):
        for i in f['instances']:
            a = i['actor_id']
            if a not in pair or any(fl.startswith(a + ':') for fl in f['review_flags']):
                continue
            if (i['source'] == 'sam31_text_detection' and (i.get('score') or 0) >= SCORE_MIN and not i.get('contains_other_actor_point')
                    and i['area'] > 400 and (i.get('largest_component_share') or 0) >= 0.9):
                anchors.setdefault(k, {})[a] = rle_decode(i['mask_rle'])
    order = sorted(anchors)
    held = set(order[2::3])
    prompts = [k for k in order if k not in held]
    if not prompts:
        return None
    paths = [Path(f['path']) for f in F]
    seeds = {p: anchors[p] for p in prompts}
    fwd = track_chunk(predictor, paths, prompts[0], len(F) - 1, seeds, obj_ids, h, w, work, 'forward', prompts[0])
    torch.cuda.empty_cache()
    bwd = track_chunk(predictor, paths, 0, prompts[-1], seeds, obj_ids, h, w, work, 'backward', prompts[-1]) if prompts[-1] > 0 else {}
    torch.cuda.empty_cache()
    out = {}  # per frame and object: the pass whose nearest seed (for that actor) is closer; forward on ties
    first = {o: min(p for p in prompts if a in seeds[p]) for a, o in obj_ids.items() if any(a in seeds[p] for p in prompts)}
    last = {o: max(p for p in prompts if a in seeds[p]) for a, o in obj_ids.items() if any(a in seeds[p] for p in prompts)}
    for k in range(len(F)):
        ms = {}
        for o in obj_ids.values():
            f_m, b_m = fwd.get(k, {}).get(o), bwd.get(k, {}).get(o)
            use_b = b_m is not None and (f_m is None or k < first.get(o, 0) or (k <= last.get(o, -1) and False))
            m = b_m if use_b else f_m
            if m is not None:
                ms[o] = m
        out[k] = ms
    inv = {v: k for k, v in obj_ids.items()}
    tracked = {k: {inv[o]: m for o, m in ms.items() if o in inv} for k, ms in out.items()}
    held_iou = [iou(tracked.get(k, {}).get(a, np.zeros((h, w), bool)), m) for k in held for a, m in anchors[k].items()]
    dense = None
    if (base/'dense/dense.json').exists():
        d = json.loads((base/'dense/dense.json').read_text())
        z = np.load(base/'dense/masks.npz')
        dense = (np.array([x['time_s'] for x in d['frames']]), z, d['shape'])
    recs, arrays = [], {}
    for k, f in enumerate(F):
        clean = np.zeros((h, w), bool)
        for i in f['instances']:
            if i['source'] == 'sam31_text_detection' and not i.get('contains_other_actor_point'):
                clean |= rle_decode(i['mask_rle'])
        for i in f['instances']:
            if i['actor_id'] not in pair or i['source'] != FALLBACK or not i['area']:
                continue
            tm = tracked.get(k, {}).get(i['actor_id'])
            old = rle_decode(i['mask_rle'])
            r = {'iv': f'{base.parts[-2]}/{base.parts[-1]}', 'fi': k, 'actor': i['actor_id'], 'old_area': int(old.sum()),
                 'gap_to_prompt': min(abs(k - p) for p in prompts)}
            if tm is None:
                r.update(ok=False, reason='no_track')
            else:
                partner = np.zeros((h, w), bool)
                for o in f['instances']:
                    if o['actor_id'] in pair and o['actor_id'] != i['actor_id'] and o.get('mask_rle'):
                        partner |= rle_decode(o['mask_rle'])
                ok, m = coherent(tm, old, clean, partner)
                r.update(m, ok=ok, iou_with_old=round(iou(tm, old), 3))
                arrays[f'{k}_{i["actor_id"]}'] = np.packbits(tm)
                if dense is not None:
                    times, z, shape = dense
                    j = int(np.argmin(np.abs(times - f['time_s'])))
                    key = f'{j}_{i["actor_id"]}'
                    if abs(times[j] - f['time_s']) < 0.06 and key in z.files:
                        dm = np.unpackbits(z[key])[:h*w].reshape(h, w).astype(bool)
                        r['iou_with_dense'] = round(iou(tm, dm), 3)
            recs.append(r)
    return {'pair': pair, 'anchors': len(order), 'prompts': len(prompts), 'held_out': len(held),
            'held_out_iou_median': round(float(np.median(held_iou)), 3) if held_iou else None,
            'held_out_iou': [round(x, 3) for x in held_iou], 'records': recs}, arrays


def main():
    from autolabel_sam31_video import build_predictor, interactive_mode
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument('targets', nargs='+')
    a = ap.parse_args()
    predictor = interactive_mode(build_predictor())
    out_dir = OUT/'track'
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    with tempfile.TemporaryDirectory(prefix='maskfix-track-') as work:
        for t in a.targets:
            t0 = time.time()
            torch.cuda.reset_peak_memory_stats()
            try:
                res = track_interval(predictor, RUN/t, Path(work))
            except Exception as error:
                print('failed', t, repr(error)[:300], flush=True)
                continue
            if res is None:
                print(t, 'no clean anchors', flush=True)
                results[t] = {'skipped': 'no clean anchors'}
                continue
            summary, arrays = res
            np.savez_compressed(out_dir/f"{t.replace('/', '_')}.npz", **arrays)
            summary.update(elapsed_s=round(time.time() - t0, 1), peak_vram_gb=round(torch.cuda.max_memory_allocated()/1e9, 2))
            results[t] = summary
            rs = summary['records']
            print(f"{t}: anchors {summary['anchors']} held-out IoU {summary['held_out_iou_median']} | fallback {len(rs)} "
                  f"coherent {sum(r['ok'] for r in rs)} | {summary['elapsed_s']}s peak {summary['peak_vram_gb']} GB", flush=True)
            write_json(out_dir/'results.json', results)


if __name__ == '__main__':
    main()
